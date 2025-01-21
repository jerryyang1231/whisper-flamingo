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
import argparse
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from tqdm import tqdm
from spec_augment import spec_augment
from utils import (
    load_wave,
    add_noise,
    WhisperTextCollatorWhithPadding_librispeech_with_bert,
    whisper_optimizer,
    whisper_flamingo_optimizer,
    setup_logging_and_checkpoint_librispeech,
    wer_cer,
    DistributedSamplerWrapper,
)
from utils_batch_samplers import SortedBatchSampler
from whisper.normalizers.basic import BasicTextNormalizer
import wandb 
from pytorch_lightning.loggers import WandbLogger
# os.environ["WANDB_MODE"] = "disabled"
os.environ['WANDB_DIR'] = '/share/nas169/jerryyang/whisper-flamingo/wandb/'
from transformers import BertTokenizer, BertModel
import copy
import torch.nn.functional as F
import pandas as pd

# my command
# python -u distil-whisbert-flamingo_librispeech.py config/audio-text/distil-whisbert-flamingo_en-deu.yaml
# python -u distil-whisbert-flamingo_librispeech.py config/audio-text/distil-whisbert-flamingo_en-cmn+en-deu.yaml

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)

class LibriSpeechTextDataset(Dataset):
    def __init__(self, hf_split, tokenizer, sample_rate, model_name, max_length, 
                spec_augment, noise_prob=0, noise_fn=None, noise_snr=0, translation_base_dirs=None, 
                use_pseudo_labels=False, pseudo_dict=None, is_train=False) -> None:
        super().__init__()
       
        # Hugging Face split 到自定義 split 的映射字典
        self.split_mapping = {
            'train.clean.100': 'train-clean-100',
            'train.clean.360': 'train-clean-360',
            'train.other.500': 'train-other-500',
            'validation.clean': 'dev-clean',
            'validation.other': 'dev-other',
            'test.clean': 'test-clean',
            'test.other': 'test-other'
        }
        
        # 保存 Hugging Face 的 split 名稱
        self.hf_split = hf_split
        self.custom_split_names = [
            self.split_mapping.get(split.strip(), split.strip()) 
            for split in hf_split.split('+')
        ] if 'train' in hf_split else [self.split_mapping.get(hf_split, hf_split)]
        
        # 從 Hugging Face datasets 載入整個 split
        self.dataset = load_dataset("librispeech_asr", split=hf_split)  
        print(f"split {hf_split} size: {len(self.dataset)}")
        self.sample_rate = sample_rate
        
        # 假設只有在 train split & wer_threshold is not None 時才執行 filter
        self.is_train = is_train
        self.use_pseudo_labels = use_pseudo_labels
        self.pseudo_dict = pseudo_dict  # { "id": (pseudo_label, ground_truth, wer), ... }
        if self.is_train and cfg.wer_threshold is not None:
            keep_id_set = set(self.pseudo_dict.keys())
            old_len = len(self.dataset)
            self.dataset = self.dataset.filter(lambda sample: sample["id"] in keep_id_set)
            print(f"Filter train set from {old_len} -> {len(self.dataset)} by pseudo_dict.")
        
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length

        self.spec_augment = spec_augment
        self.noise_prob = noise_prob
        self.noise_fn = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else []

        self.translation_base_dirs = translation_base_dirs
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)  

    def __len__(self):
        return len(self.dataset)
   
    def get_translation_text(self, file_id, translation_base_dir):
        
        # 提取 speaker_id 和 chapter_id
        speaker_id, chapter_id = file_id.split('-')[0], file_id.split('-')[1]
        
        text = ""

        # 遍歷所有可能的 custom_split_names 來查找對應的翻譯文件
        for custom_split_name in self.custom_split_names:
            relative_dir = os.path.join(custom_split_name, speaker_id, chapter_id)
            trans_file_path = os.path.join(translation_base_dir, relative_dir, f"{speaker_id}-{chapter_id}.trans.txt")
            
            # 讀取翻譯文件並提取對應行的翻譯
            if os.path.exists(trans_file_path):
                with open(trans_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)  # 進行拆分
                        if len(parts) == 2:  # 確保拆分出兩個部分
                            line_id, text = parts
                            if line_id == file_id:  # 檢查行ID是否匹配
                                return text  # 返回匹配的文本
                
        return text
    
    def __getitem__(self, id):
        lang = cfg.lang
        item = self.dataset[id]
        file_id = item['id']
        wav_data = item['audio']['array']
        wav_lens = len(wav_data)

        # ====== 決定 text ======
        if (not self.is_train) or (not self.use_pseudo_labels):
            # val/test set 或者根本沒用 pseudo
            text = item["text"]
        else:
            # train set + pseudo
            pseudo_label, ground_truth, wer = self.pseudo_dict[file_id]
            pseudo_label = pseudo_label.strip()
            if not isinstance(pseudo_label, str) or not pseudo_label.strip():
                text = item["text"]
            else:
                text = pseudo_label
        # ===============================
        
        text = self.text_normalizer(text)

        # ====== (可選) 加噪音 ======
        if np.random.rand() > self.noise_prob: # disable noise
            audio = wav_data.flatten().astype(np.float32)
        else: # add noise
            audio = add_noise(wav_data, self.noise_fn, noise_snr=0).flatten().astype(np.float32)

        # ====== pad/truncate ======
        if self.max_length is not None:
            audio = whisper.pad_or_trim(audio.flatten(), length=self.max_length)
        n_mels = 80 if self.model_name != 'large-v3' else 128
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)

        # ====== tokenizer ======
        # Seems like Whisper decode always predicts first token with space, so add a space in the beginning
        # dec_input_ids = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(" " + text)
        dec_input_ids = [self.tokenizer.sot, 
                        self.tokenizer.special_tokens["<|{}|>".format(lang)],
                        self.tokenizer.transcribe, 
                        self.tokenizer.no_timestamps] + \
                        self.tokenizer.encode(" " + text)
        labels = dec_input_ids[1:] + [self.tokenizer.eot]
        
        # ====== (可選) 多翻譯文本 ======
        translation_1 = self.get_translation_text(file_id, self.translation_base_dirs[0])
        translation_1 = self.text_normalizer(translation_1)

        translation_2 = None
        if len(self.translation_base_dirs) > 1:
            translation_2 = self.get_translation_text(file_id, self.translation_base_dirs[1])
            translation_2 = self.text_normalizer(translation_2)

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "translation_1": translation_1,
            "translation_2": translation_2,
            "wav_lens": wav_lens,
        }

class DistillWhisperModule(LightningModule):
    def __init__(self, cfg, model_name, lang, train_split, val_clean_split, val_other_split,
                 test_clean_split, test_other_split, pseudo_dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_name = model_name
        self.pseudo_dict = pseudo_dict
        print("Loading teacher model and weights")
        # ========== Teacher ==========
        self.teacher = whisper.load_model(model_name,
                                        device = 'cpu', # avoid OOM on gpu 0 for distributed
                                        download_root = '/share/nas169/jerryyang/whisper-flamingo/models',
                                        dropout_rate = cfg.dropout_rate,
                                        add_gated_x_attn = cfg.add_gated_x_attn,
                                        bert_encoder = cfg.bert_encoder,
                                        mode = cfg.mode,
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
        
        # 打印 teacher encoder 和 decoder 的層數
        # teacher_enc_layers = len(self.teacher.encoder.blocks)
        # teacher_dec_layers = len(self.teacher.decoder.blocks)

        # ========== Student ==========
        print("Loading student model")
        self.student = whisper.load_model(model_name,
                                        device="cpu",
                                        download_root="/share/nas169/jerryyang/whisper-flamingo/models",
                                        dropout_rate=cfg.dropout_rate,
                                    )
        
        # freeze student encoder gradients for ditil
        if cfg.freeze_encoder != 0:
            for param in self.student.encoder.parameters():
                param.requires_grad = False

        # 砍掉 decoder blocks, 只留 num_student_dec_layers 層
        # old_blocks = list(self.student.decoder.blocks)
        # keep_layers = np.linspace(0, teacher_dec_layers-1, cfg.num_student_dec_layers, dtype=int)
        # self.student.decoder.blocks = nn.ModuleList([old_blocks[i] for i in keep_layers])

        # student encoder 和 decoder 的層數
        # student_enc_layers = len(self.student.encoder.blocks)
        # student_dec_layers = len(self.student.decoder.blocks)
        
        # 部分複製權重 
        # partial_init_student_from_teacher(self.teacher, self.student, student_enc_layers, student_dec_layers, teacher_enc_layers, teacher_dec_layers)
        partial_init_student_from_teacher(self.teacher, self.student)

        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language='en', task='transcribe')
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.kd_loss_fn = nn.KLDivLoss(reduction='none')

        self.train_split = train_split
        self.val_clean_split = val_clean_split
        self.val_other_split = val_other_split
        self.test_clean_split = test_clean_split
        self.test_other_split = test_other_split
        self.special_token_set = set(self.tokenizer.special_tokens.values())
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)

        # 初始化 BERT 分詞器和模型
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        translation_1 = batch["translation_1"] 

        # 檢查 batch 中是否有 translation_2，沒有則設為 None
        bert_hidden_states_2 = None
        translation_2 = batch.get("translation_2", None)

        # ====== 1) Teacher (無梯度) ======
        with torch.no_grad():
            # 取得翻譯 embeddings
            bert_inputs_1 = self.bert_tokenizer(
                translation_1,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=448
            ).to(self.device)
            bert_outputs_1 = self.bert_model(**bert_inputs_1)
            translation_embeddings_1 = bert_outputs_1.last_hidden_state  # [batch_size, seq_len, hidden_size]

            if translation_2 is not None:
                # 取得翻譯 embeddings
                bert_inputs_2 = self.bert_tokenizer(
                    translation_2,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=448
                ).to(self.device)
                bert_outputs_2 = self.bert_model(**bert_inputs_2)
                translation_embeddings_2 = bert_outputs_2.last_hidden_state  # [batch_size, seq_len, hidden_size]
            
                teacher_feat = self.teacher.encoder(input_ids)
                teacher_out = self.teacher.decoder(
                    dec_input_ids, teacher_feat, xt_1=translation_embeddings_1, xt_2=translation_embeddings_2
                )
            else:
                teacher_feat = self.teacher.encoder(input_ids)
                teacher_out = self.teacher.decoder(
                    dec_input_ids, teacher_feat, xt_1=translation_embeddings_1
                ) 

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
            
    def validation_step(self, batch, batch_id, dataloader_idx=None):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        translation_1 = batch["translation_1"]
        wav_lens = batch["wav_lens"] 

        # 檢查 batch 中是否有 translation_2，沒有則設為 None
        bert_hidden_states_2 = None
        translation_2 = batch.get("translation_2", None)
        
        # ====== 1) Teacher (無梯度) ======
        with torch.no_grad():
            # 取得翻譯 embeddings
            bert_inputs_1 = self.bert_tokenizer(
                translation_1,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=448
            ).to(self.device)
            bert_outputs_1 = self.bert_model(**bert_inputs_1)
            translation_embeddings_1 = bert_outputs_1.last_hidden_state  # [batch_size, seq_len, hidden_size]

            if translation_2 is not None:
                # 取得翻譯 embeddings
                bert_inputs_2 = self.bert_tokenizer(
                    translation_2,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=448
                ).to(self.device)
                bert_outputs_2 = self.bert_model(**bert_inputs_2)
                translation_embeddings_2 = bert_outputs_2.last_hidden_state  # [batch_size, seq_len, hidden_size]
            
                teacher_feat = self.teacher.encoder(input_ids)
                teacher_out = self.teacher.decoder(
                    dec_input_ids, teacher_feat, xt_1=translation_embeddings_1, xt_2=translation_embeddings_2
                )
            else:
                teacher_feat = self.teacher.encoder(input_ids)
                teacher_out = self.teacher.decoder(
                    dec_input_ids, teacher_feat, xt_1=translation_embeddings_1
                )

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
                
                # 對解碼結果進行正規化
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

        log_prefix = {0: 'dev-clean', 1: 'dev-other', 2: 'test-clean', 3: 'test-other'}
        self.log("{}/ce_loss".format(log_prefix[dataloader_idx], mod), ce_loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/kd_loss".format(log_prefix[dataloader_idx], mod), kd_loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/loss_{}".format(log_prefix[dataloader_idx], mod), loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/wer_{}".format(log_prefix[dataloader_idx], mod), wer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/acc_{}".format(log_prefix[dataloader_idx], mod), acc, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)

        return
       
    def configure_optimizers(self):
        optimizer, scheduler = whisper_optimizer(self.student, self.cfg, self.t_total)
        self.optimizer, self.scheduler = optimizer, scheduler
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = self.cfg.num_train_steps

    def train_dataloader(self):
        dataset = LibriSpeechTextDataset(self.train_split, 
                                    self.tokenizer, 
                                    SAMPLE_RATE,
                                    self.model_name,
                                    max_length=480000,
                                    spec_augment=self.cfg.spec_augment,
                                    noise_prob=cfg.noise_prob,
                                    noise_snr=cfg.noise_snr_train,
                                    translation_base_dirs=cfg.translation_base_dirs,
                                    use_pseudo_labels=cfg.use_pseudo_labels,
                                    pseudo_dict=self.pseudo_dict,
                                    is_train=True
                                    )  
        batch_sampler = SortedBatchSampler(
                    batch_size = self.cfg.batch_size,
                    shapes=[(item['wav_lens']) for item in dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=True)
        if cfg.num_devices > 1:
            print("Using distributed sampler")
            batch_sampler = DistributedSamplerWrapper(batch_sampler)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=batch_sampler,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperTextCollatorWhithPadding_librispeech_with_bert())

    def val_dataloader_clean(self):
        dataset = LibriSpeechTextDataset(self.val_clean_split,
                                self.tokenizer, 
                                SAMPLE_RATE,
                                self.model_name,
                                max_length=cfg.audio_max_length,
                                spec_augment=False,
                                noise_prob=0,
                                translation_base_dirs=cfg.translation_base_dirs,
                                )
        batch_sampler = SortedBatchSampler(
                    batch_size = self.cfg.batch_size,
                    shapes=[(item['wav_lens']) for item in dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=batch_sampler,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperTextCollatorWhithPadding_librispeech_with_bert())
   
    def val_dataloader_other(self):
        dataset = LibriSpeechTextDataset(self.val_other_split,
                                self.tokenizer, 
                                SAMPLE_RATE,
                                self.model_name,
                                max_length=cfg.audio_max_length,
                                spec_augment=False,
                                noise_prob=0,
                                translation_base_dirs=cfg.translation_base_dirs,
                                )
        batch_sampler = SortedBatchSampler(
                    batch_size = self.cfg.batch_size,
                    shapes=[(item['wav_lens']) for item in dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=batch_sampler,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperTextCollatorWhithPadding_librispeech_with_bert())
    
    def test_dataloader_clean(self):
        dataset = LibriSpeechTextDataset(self.test_clean_split,  
                                self.tokenizer, 
                                SAMPLE_RATE,
                                self.model_name,
                                max_length=cfg.audio_max_length,
                                spec_augment=False,
                                noise_prob=0,
                                translation_base_dirs=cfg.translation_base_dirs,
                                )
        batch_sampler = SortedBatchSampler(
                    batch_size = self.cfg.batch_size,
                    shapes=[(item['wav_lens']) for item in dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=batch_sampler,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperTextCollatorWhithPadding_librispeech_with_bert())
    
    def test_dataloader_other(self):
        dataset = LibriSpeechTextDataset(self.test_other_split, 
                                self.tokenizer, 
                                SAMPLE_RATE,
                                self.model_name,
                                max_length=cfg.audio_max_length,
                                spec_augment=False,
                                noise_prob=0,
                                translation_base_dirs=cfg.translation_base_dirs,
                                )
        batch_sampler = SortedBatchSampler(
                    batch_size = self.cfg.batch_size,
                    shapes=[(item['wav_lens']) for item in dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=batch_sampler,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperTextCollatorWhithPadding_librispeech_with_bert())

# def partial_init_student_from_teacher(teacher_model, student_model, student_enc_layers, student_dec_layers, teacher_enc_layers, teacher_dec_layers):
#     # (A) linspace or manual mapping
#     encoder_mapping = np.linspace(0, teacher_enc_layers-1, student_enc_layers, dtype=int)
#     decoder_mapping = np.linspace(0, teacher_dec_layers-1, student_dec_layers, dtype=int)

#     # (B) 先 load 整體
#     student_model.load_state_dict(teacher_model.state_dict(), strict=False)

#     # (C) 個別block對應
#     for i, teacher_idx in enumerate(encoder_mapping):
#         student_model.encoder.blocks[i].load_state_dict(
#             teacher_model.encoder.blocks[teacher_idx].state_dict(),
#             strict=True
#         )
#     for i, teacher_idx in enumerate(decoder_mapping):
#         student_model.decoder.blocks[i].load_state_dict(
#             teacher_model.decoder.blocks[teacher_idx].state_dict(),
#             strict=False
#         )

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

    pseudo_dict = {}
    if cfg.use_pseudo_labels:
        df = pd.read_csv(cfg.pseudo_csv_path_train)
        if cfg.wer_threshold is not None:
            df = df[df["wer"] <= cfg.wer_threshold]
            print(f"Filtered samples by WER <= {cfg.wer_threshold}, remain: {len(df)}")
        for row in df.itertuples(index=False):
            pseudo_dict[row.id] = (
                row.pseudo_text,
                row.ground_truth,
                row.wer
            )

    print(cfg)
    print("audio max length: {}".format(cfg.audio_max_length))

    # Initialize WandB
    wandb.init(project="whisper-flamingo",
            config=cfg,
            name="distil-whisbert-flamingo_en-cmn+en-deu_small pseudo labels revise(db=12, lr=1.0e-6, bs=4, wt=none)",
            # name="distil-whisbert-flamingo_en-cmn+en-deu_small pseudo labels revise(db=12, lr=1.0e-6, bs=4, wt=0.1)",
            # name="distil-whisbert-flamingo_en-cmn+en-deu_small pseudo labels revise(db=6, lr=1.0e-6, bs=4)",
    )
    
    tflogger, callback_list = setup_logging_and_checkpoint_librispeech(cfg.log_output_dir, 
                                                                        cfg.check_output_dir, 
                                                                        cfg.train_name, 
                                                                        cfg.train_id,
                                                                        cfg.monitor,
                                                                        cfg.filename)
        
    model = DistillWhisperModule(cfg, cfg.model_name, cfg.lang, 
                            'train.clean.100+train.clean.360+train.other.500',
                            'validation.clean',
                            'validation.other',
                            'test.clean',
                            'test.other',
                            pseudo_dict,
                            )
    
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
    resume_ckpt = f"{cfg.check_output_dir}/{cfg.train_id}/last.ckpt"
    if os.path.exists(resume_ckpt) and cfg.resume_training: # resume training, don't validate
        trainer.fit(model, ckpt_path='last', val_dataloaders=[model.val_dataloader_clean(), model.val_dataloader_other(),
                                                model.test_dataloader_clean(), model.test_dataloader_other()])
    else:
        trainer.validate(model=model, dataloaders=[model.val_dataloader_clean(), model.val_dataloader_other(),
                                                model.test_dataloader_clean(), model.test_dataloader_other()]) # validate before training
        trainer.fit(model, val_dataloaders=[model.val_dataloader_clean(), model.val_dataloader_other(),
                                            model.test_dataloader_clean(), model.test_dataloader_other()])

    # End the WandB run
    wandb.finish()