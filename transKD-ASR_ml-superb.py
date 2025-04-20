import os
import sys
import yaml
import types
import numpy as np
import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import Dataset
import pandas as pd
import whisper
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from spec_augment import spec_augment
from utils import (
    add_noise,
    TransASR_collator,
    whisper_optimizer,
    whisper_flamingo_optimizer,
    setup_logging_and_checkpoint_ml_superb,
    wer_cer,
    DistributedSamplerWrapper,
)
from utils_batch_samplers import SortedBatchSampler
from whisper.normalizers.basic import BasicTextNormalizer
import wandb
from pytorch_lightning.loggers import WandbLogger
from transformers import BertModel, BertTokenizer
import copy
import torch.nn.functional as F
import json
# os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DIR"] = "/share/nas169/jerryyang/whisper-flamingo/wandb"
# os.environ["TMPDIR"] = "/share/nas169/jerryyang/whisper-flamingo/tmp"

# my command
# python -u transKD-ASR_ml-superb.py config/distil/ml-superb.yaml

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)

class MLSuperbDataset(Dataset):
    def __init__(self, hf_split, tokenizer, sample_rate, model_name, max_length, lang, translated_json_path=None,
                use_pseudo_labels=False, pseudo_dict=None, is_train=False) -> None:
        super().__init__()
        
        # 直接從 ML-SUPERB 讀取 `eng` 語言數據
        dataset = load_dataset("espnet/ml_superb_hf", split=hf_split)
        self.dataset = [item for item in dataset if item["language"] == "eng"]
        
        # **載入 translated_text JSON**
        self.translated_texts = {}
        if translated_json_path:
            with open(translated_json_path, "r", encoding="utf-8") as f:
                translated_data = json.load(f)
            # **建立 ID 對應 translated_text**
            self.translated_texts = {item["id"]: item["translated_text"] for item in translated_data}
        
        print(f"{hf_split} (eng) size: {len(self.dataset)} samples")

        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.lang = lang
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)

        self.is_train = is_train
        self.use_pseudo_labels = use_pseudo_labels
        self.pseudo_dict = pseudo_dict # { id: (pseudo_text, ground_truth) }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        lang = self.lang
        item = self.dataset[id]
        # 取出原始資料中的 id
        item_id = str(item.get("id", id))
        wav_data = item['audio']['array']
        wav_lens = len(wav_data)

        # **取得翻譯後的文本**
        translated_text = self.translated_texts.get(item["id"], None)
   
        # =========== 決定 text ===========
        if (not self.is_train) or (not self.use_pseudo_labels):
            # dev set 或者根本沒用 pseudo
            text = item["text"][5:].strip()
        else:
            # train set + pseudo
            pseudo_label, _ = self.pseudo_dict[item_id]
            if not isinstance(pseudo_label, str) or not pseudo_label.strip():
                text = item["text"][5:].strip()
            else:
                text = pseudo_label
        # ===============================

        text = self.text_normalizer(text)
        translated_text = self.text_normalizer(translated_text)
        
        audio = wav_data.flatten().astype(np.float32)

        if self.max_length is not None:
            audio = whisper.pad_or_trim(audio.flatten(), length=self.max_length)

        n_mels = 80 if self.model_name != 'large-v3' else 128
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)

        dec_input_ids = [self.tokenizer.sot,
                        self.tokenizer.special_tokens["<|{}|>".format(lang)],
                        self.tokenizer.transcribe,
                        self.tokenizer.no_timestamps] + \
                        self.tokenizer.encode(" " + text)
        labels = dec_input_ids[1:] + [self.tokenizer.eot]

        all_translations = []
        all_translations.append(translated_text)

        return {
            "wav_lens": wav_lens,
            "input_ids": mel,
            "dec_input_ids": dec_input_ids,
            "labels": labels,
            "all_translations": all_translations,
        }

class DistillWhisperModule(LightningModule):
    def __init__(self, cfg, model_name) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_name = model_name
        print("Loading teacher model and weights")
        # ========== Teacher ==========
        self.teacher = whisper.load_model(model_name,
                                        device = 'cpu', # avoid OOM on gpu 0 for distributed
                                        download_root = '/share/nas169/jerryyang/whisper-flamingo/models',
                                        dropout_rate = cfg.dropout_rate,
                                        add_gated_x_attn = cfg.add_gated_x_attn,
                                        num_langs = cfg.num_langs,
                                        )

        if cfg.teacher_ckpt != '': # load audio-only FT ckpt
            checkpoint_root = '/share/nas169/jerryyang/whisper-flamingo/models/checkpoints/'
            state_dict = torch.load(os.path.join(checkpoint_root, cfg.teacher_ckpt), map_location=torch.device('cpu'))
            state_dict = state_dict['state_dict']
            state_dict_updated = {k[6:]: v for k, v in state_dict.items()} # remove 'model.'
            print(state_dict_updated.keys())
            try:
                self.teacher.load_state_dict(state_dict_updated) 
            except BaseException as e: 
                print(str(e))
                print("Loading weights with strict=False")
                self.teacher.load_state_dict(state_dict_updated, strict=False) 

        # Teacher eval & freeze
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # ========== Student ==========
        print("Loading student model")
        self.student = whisper.load_model(model_name,
                                        device="cpu",
                                        download_root="/share/nas169/jerryyang/whisper-flamingo/models",
                                        dropout_rate=cfg.dropout_rate,
                                        )
        
        # # test student
        # if cfg.student_ckpt != '': # load audio-only FT ckpt
        #     checkpoint_root = '/share/nas169/jerryyang/whisper-flamingo/models/checkpoints/'
        #     checkpoint = torch.load(os.path.join(checkpoint_root, cfg.student_ckpt), map_location=torch.device('cpu'))
        #     state_dict = checkpoint["state_dict"] 
        #     student_only_sd = {}
        #     for k, v in state_dict.items():
        #         # 保留 student. 開頭
        #         if k.startswith("student."):
        #             new_k = k[len("student."):]  # 去除前綴
        #             student_only_sd[new_k] = v
        #     # print(student_only_sd.keys())
        #     try:
        #         self.student.load_state_dict(student_only_sd) 
        #     except BaseException as e: 
        #         print(str(e))
        #         print("Loading weights with strict=False")
        #         self.student.load_state_dict(student_only_sd, strict=False) 

        # freeze student encoder gradients for ditil
        if cfg.freeze_encoder != 0:
            for param in self.student.encoder.parameters():
                param.requires_grad = False

        # 部分複製權重 
        partial_init_student_from_teacher(self.teacher, self.student)

        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language=self.cfg.lang, task='transcribe')
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.kd_loss_fn = nn.KLDivLoss(reduction='none') 
        self.special_token_set = set(self.tokenizer.special_tokens.values())
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)

        # 初始化 BERT 分詞器和模型
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        all_translations = batch["all_translations"]
        # all_translations: list of length = batch_size
        # each element is a list of strings for that sample's translations

        # 假設每個 sample 有相同數量的翻譯 texts
        num_samples = len(all_translations)           # batch_size
        num_translations = len(all_translations[0])   # 每個 sample 的翻譯數量

        all_xt = []
        # 迴圈跑 num_translations 次，每次收集 "整個 batch" 對應的某個翻譯索引
        for t_idx in range(num_translations):
            # 收集「batch 中所有樣本」在第 t_idx 個翻譯
            batch_texts_for_this_translation = []
            for s_idx in range(num_samples):
                # 取第 s_idx 個 sample 的第 t_idx 筆翻譯
                text_t = all_translations[s_idx][t_idx]
                batch_texts_for_this_translation.append(text_t)

            # 一次把這批文字 (大小 = batch_size) 丟給 BERT
            tokenized = self.bert_tokenizer(
                batch_texts_for_this_translation,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=448
            ).to(self.device)

            outputs = self.bert_model(**tokenized)
            outputs_last_hidden_state = outputs.last_hidden_state  # shape = [batch_size, seq_len, hidden_size]

            all_xt.append(outputs_last_hidden_state)

        # 這樣 all_xt 就是一個 list，長度 = num_translations
        # all_xt[t] shape = [batch_size, seq_len, hidden_size]

        teacher_feat = self.teacher.encoder(input_ids)
        teacher_out = self.teacher.decoder(dec_input_ids, teacher_feat, xt_list=all_xt)

        # ====== 2) Student ======
        if self.cfg.freeze_encoder:
            student_out = self.student.decoder(dec_input_ids, teacher_feat)
        else:
            student_feat = self.student.encoder(input_ids)
            student_out = self.student.decoder(dec_input_ids, student_feat)

        # ====== 3) CE loss (data loss) ======
        ce_loss = self.ce_loss_fn(student_out.view(-1, student_out.size(-1)), labels.view(-1))

        # ====== 4) KD loss (distribution alignment) ======
        T = self.cfg.temperature
        teacher_probs = F.softmax(teacher_out / T, dim=-1) # [batch, seq_len, vocab_size]
        student_log_probs = F.log_softmax(student_out / T, dim=-1)
        
        # (A) 先算 "逐 token" KL-div, shape = [batch, seq_len, vocab_size]
        kl_all = self.kd_loss_fn(student_log_probs, teacher_probs)

        # (B) 根據 labels 遮蔽要忽略的 token (labels = -100)
        #     通常 labels = -100 表示該位置的 label 無效 (padding / ignore)
        padding_mask = (labels != -100).unsqueeze(-1)  # [batch, seq_len, 1]
        kl_all = kl_all * padding_mask                 # 將無效位置的 KL 設為 0

        # (C) 將剩餘位置的 KL 做 sum，再除以 mask.sum() (有效token總數)，得到「平均」
        kd_loss = kl_all.sum() / padding_mask.sum()
        
        # (D) 別忘了再乘上 (T*T) 
        kd_loss = kd_loss * (T * T)

        # ====== 5) 總損失：alpha * CE + beta * KD ======
        alpha = self.cfg.alpha
        beta  = self.cfg.beta
        loss  = alpha * ce_loss + beta * kd_loss

        # ====== 6) Logging ======
        self.log("train/ce_loss", ce_loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/kd_loss", kd_loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/loss",    loss,    on_step=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        all_translations = batch["all_translations"]
        # all_translations: list of length = batch_size
        # each element is a list of strings for that sample's translations

        # 假設每個 sample 有相同數量的翻譯 texts
        num_samples = len(all_translations)           # batch_size
        num_translations = len(all_translations[0])   # 每個 sample 的翻譯數量

        all_xt = []
        # 迴圈跑 num_translations 次，每次收集 "整個 batch" 對應的某個翻譯索引
        for t_idx in range(num_translations):
            # 收集「batch 中所有樣本」在第 t_idx 個翻譯
            batch_texts_for_this_translation = []
            for s_idx in range(num_samples):
                # 取第 s_idx 個 sample 的第 t_idx 筆翻譯
                text_t = all_translations[s_idx][t_idx]
                batch_texts_for_this_translation.append(text_t)

            # 一次把這批文字 (大小 = batch_size) 丟給 BERT
            tokenized = self.bert_tokenizer(
                batch_texts_for_this_translation,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=448
            ).to(self.device)

            outputs = self.bert_model(**tokenized)
            outputs_last_hidden_state = outputs.last_hidden_state  # shape = [batch_size, seq_len, hidden_size]
            
            all_xt.append(outputs_last_hidden_state)
        
        # 這樣 all_xt 就是一個 list，長度 = num_translations
        # all_xt[t] shape = [batch_size, seq_len, hidden_size]

        teacher_feat = self.teacher.encoder(input_ids)
        teacher_out = self.teacher.decoder(dec_input_ids, teacher_feat, xt_list=all_xt)

        # ====== 2) Student ======
        if self.cfg.freeze_encoder:
            student_out = self.student.decoder(dec_input_ids, teacher_feat)
        else:
            student_feat = self.student.encoder(input_ids)
            student_out = self.student.decoder(dec_input_ids, student_feat)

        # ====== 3) CE loss (data loss) ======
        ce_loss = self.ce_loss_fn(student_out.view(-1, student_out.size(-1)), labels.view(-1))

        # ====== 4) KD loss (distribution alignment) ======
        T = 1.0
        teacher_probs = F.softmax(teacher_out / T, dim=-1) # [batch, seq_len, vocab_size]
        student_log_probs = F.log_softmax(student_out / T, dim=-1)
        
        # (A) 先算 "逐 token" KL-div, shape = [batch, seq_len, vocab_size]
        kl_all = self.kd_loss_fn(student_log_probs, teacher_probs)

        # (B) 根據 labels 遮蔽要忽略的 token (labels = -100)
        #     通常 labels = -100 表示該位置的 label 無效 (padding / ignore)
        padding_mask = (labels != -100).unsqueeze(-1)  # [batch, seq_len, 1]
        kl_all = kl_all * padding_mask                 # 將無效位置的 KL 設為 0
        
        # (C) 將剩餘位置的 KL 做 sum，再除以 mask.sum() (有效token總數)，得到「平均」
        kd_loss = kl_all.sum() / padding_mask.sum()

        # (D) 別忘了再乘上 (T*T) 
        kd_loss = kd_loss * (T * T)

        # ====== 5) 總損失：alpha * CE + beta * KD ======
        alpha = self.cfg.alpha
        beta  = self.cfg.beta
        loss  = alpha * ce_loss + beta * kd_loss

        labels[labels == -100] = self.tokenizer.eot

        mod_list = {"at": student_out}
        for mod, out in mod_list.items():
            # remove all decoder predictions after first eot for proper decoding
            tokens = torch.argmax(out, dim=2)

            # Set all decoder predictions after first eot to eot
            # TODO: fix for large-v3, which predicts <eot> in the beginning
            eot_find = (torch.where(tokens == self.tokenizer.eot, 1, 0))

            # 針對每個序列進行檢查
            for i in range(eot_find.shape[0]):
                if torch.any(eot_find[i] == 1):  # 如果該序列中存在 EOT 標記
                    first_eot = torch.argmax(torch.arange(eot_find.shape[1], 0, -1).cuda() * eot_find[i], dim=0, keepdim=True)
                    tokens[i, torch.arange(eot_find.shape[1]).cuda() > first_eot] = self.tokenizer.eot

            # calculate next token prediction, not include lang tag, task, and no timestamps token
            mask = ~(tokens[:, 3:] == self.tokenizer.eot) # torch.ne fails for some reason
            n_correct = torch.sum(
                tokens[:, 3:].masked_select(mask).eq(labels[:, 3:].masked_select(mask))
            )
            total = torch.sum(mask)
            acc = n_correct.item() / (total.item() + 1e-6)
            acc = acc if acc < 1 else 0

            o_list, l_list = [], []
            for o, l in zip(tokens, labels):
                # 解碼並過濾掉特殊標籤
                decoded_o = self.tokenizer.decode([t for t in o if t.item() not in self.special_token_set])
                decoded_l = self.tokenizer.decode([t for t in l if t.item() not in self.special_token_set])
                
                # 正規化文本
                normalized_o = self.text_normalizer(decoded_o)
                normalized_l = self.text_normalizer(decoded_l)

                # 將正規化的結果添加到列表中
                o_list.append(normalized_o)
                l_list.append(normalized_l)

            wer, cer = wer_cer(hypo=o_list, ref=l_list)

            print("Mod: {}".format(mod))
            for i, (hypo, ref) in enumerate(zip(o_list, l_list)):
                print("="*100)
                print("PRED: {}".format(hypo))
                print("REF:  {}".format(ref))
                if i == 1: break

            log_prefix = 'dev'
            self.log("{}/ce_loss_{}".format(log_prefix, mod), ce_loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/kd_loss_{}".format(log_prefix, mod), kd_loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/loss_{}".format(log_prefix, mod), loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/cer_{}".format(log_prefix, mod), cer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/wer_{}".format(log_prefix, mod), wer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/acc_{}".format(log_prefix, mod), acc, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)

        return

    def configure_optimizers(self):
        optimizer, scheduler = whisper_optimizer(self.student, self.cfg, self.t_total)
        self.optimizer, self.scheduler = optimizer, scheduler
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = self.cfg.num_train_steps

    def train_dataloader(self):
        dataset = MLSuperbDataset('train',
                                self.tokenizer,
                                SAMPLE_RATE,
                                self.model_name,
                                max_length=self.cfg.audio_max_length,
                                lang=self.cfg.lang,
                                translated_json_path=self.cfg.train_translated_json_path,
                                use_pseudo_labels=self.cfg.use_pseudo_labels,
                                pseudo_dict=getattr(self, "pseudo_dict_train", None),
                                is_train=True,
                                )
        batch_sampler = SortedBatchSampler(batch_size = self.cfg.batch_size,
                                        shapes=[(item['wav_lens']) for item in dataset],
                                        sort_in_batch='descending',
                                        sort_batch='descending',
                                        drop_last=True
                                        )
        if self.cfg.num_devices > 1:
            print("Using distributed sampler")
            batch_sampler = DistributedSamplerWrapper(batch_sampler)
        return torch.utils.data.DataLoader(dataset,
                                        batch_sampler=batch_sampler,
                                        num_workers=self.cfg.num_worker,
                                        collate_fn=TransASR_collator()
                                        )

    def val_dataloader(self):
        dataset = MLSuperbDataset('dev',
                                self.tokenizer,
                                SAMPLE_RATE,
                                self.model_name,
                                max_length=self.cfg.audio_max_length,
                                lang=self.cfg.lang,
                                translated_json_path=self.cfg.dev_translated_json_path,
                                )
        batch_sampler = SortedBatchSampler(batch_size = self.cfg.batch_size,
                                        shapes=[(item['wav_lens']) for item in dataset],
                                        sort_in_batch='descending',
                                        sort_batch='descending',
                                        drop_last=False
                                        )
        return torch.utils.data.DataLoader(dataset,
                                        batch_sampler=batch_sampler,
                                        num_workers=self.cfg.num_worker,
                                        collate_fn=TransASR_collator()
                                        )

def partial_init_student_from_teacher(teacher_model, student_model):

    # 1) 複製 encoder
    student_model.encoder.load_state_dict(
        teacher_model.encoder.state_dict(), 
        strict=True
    )

    # 2) 複製 decoder
    student_model.decoder.load_state_dict(
        teacher_model.decoder.state_dict(),
        strict=False
    )

if __name__ == "__main__":
    cfg_yaml = sys.argv[1]
    with open(cfg_yaml, 'r') as file:
        dct = yaml.safe_load(file)
        cfg = types.SimpleNamespace(**dct)

    # 如果要用 pseudo labels，就在這裡讀 CSV 一次
    pseudo_dict_train = {}
    if cfg.use_pseudo_labels:
        if hasattr(cfg, "pseudo_csv_path_train") and cfg.pseudo_csv_path_train:
            df_train = pd.read_csv(cfg.pseudo_csv_path_train)
            for row in df_train.itertuples(index=False):
                # row.id, row.pseudo_text, row.ground_truth
                pseudo_dict_train[row.id] = (row.pseudo_text, row.ground_truth)
    
    print(cfg)
    print("audio max length: {}".format(cfg.audio_max_length))

    # Initialize WandB
    wandb.init(project="TransKD-ASR",
            config=cfg,
            name="transKD-ASR_ml-superb (baseline)",
    )
    
    callback_list = setup_logging_and_checkpoint_ml_superb(cfg.log_output_dir, 
                                                        cfg.check_output_dir, 
                                                        cfg.train_name, 
                                                        cfg.train_id,
                                                        cfg.monitor,
                                                        cfg.filename,
                                                        )
        
    model = DistillWhisperModule(cfg, cfg.model_name)
    model.pseudo_dict_train = pseudo_dict_train
    
    # Create a WandB logger instance
    wandb_logger = WandbLogger()

    strategy = DDPStrategy(find_unused_parameters=True) if cfg.num_devices > 1 else "auto"
    trainer = Trainer(
        precision=cfg.precision,
        strategy=strategy,
        accelerator="gpu",
        max_steps=cfg.num_train_steps,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=wandb_logger,
        callbacks=callback_list,
        num_sanity_val_steps=0, # default is 2 batches, 0 to turn off
        devices=cfg.num_devices,
        val_check_interval=int(cfg.validate_every_n_batches * cfg.gradient_accumulation_steps), # validate after this number batches
        check_val_every_n_epoch=None, # If None, validation will be done solely based on the number of training batches
        reload_dataloaders_every_n_epochs=1, # shuffle the dataloader after an epoch
        use_distributed_sampler=False, # implemented custom distributed trainer
        sync_batchnorm=True,
    )

    # TODO: save config file tp the checkpoint dir, also for pre-trained model
    print(cfg)
    trainer.validate(model=model, dataloaders=model.val_dataloader()) # validate before training
    trainer.fit(model, val_dataloaders=model.val_dataloader())

    # End the WandB run
    wandb.finish()