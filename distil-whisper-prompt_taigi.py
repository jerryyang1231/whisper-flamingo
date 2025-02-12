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
from utils import (
    load_wave,
    add_noise,
    DistilPromptCollator,
    whisper_optimizer,
    whisper_flamingo_optimizer,
    setup_logging_and_checkpoint_taigi,
    wer_cer,
    DistributedSamplerWrapper,
)
from utils_batch_samplers import SortedBatchSampler
from whisper.normalizers.basic import BasicTextNormalizer
import wandb 
from pytorch_lightning.loggers import WandbLogger
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DIR"] = "/share/nas169/jerryyang/whisper-flamingo/wandb"
# os.environ["TMPDIR"] = "/share/nas169/jerryyang/whisper-flamingo/tmp"
import copy
import torch.nn.functional as F
import pandas as pd

# my command
# python -u distil-whisper-prompt_taigi.py config/audio-text/distil-whisper-prompt_taigi.yaml

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)
# valid_set_list 包含的前11字符的ID
valid_set_list = ['-d8TlAGYFmc', '3h8m__iwuJ4', '5mPJOkoIu3k', '87omMWX-DTw', 
                'E0-HOPE7_QU', 'EhqcvfaaYu8', 'gDDbnFcvWcQ', 'iy1fPQQSA6c',
                'kGbjIuzvPR8', 'MrwSzSVGiRE', 'yht8d59dCpo']

class YTTDTaigiTRSDataset(Dataset):
    def __init__(self, split, tokenizer, sample_rate, model_name, max_length, 
                use_pseudo_labels=False, pseudo_dict=None, is_train=False) -> None:
        super().__init__()

        if split == 'train':
            dataset = load_dataset("formospeech/yttd_taigi_trs", name='train', split='train')
            self.dataset = dataset.filter(lambda sample: sample['id'][:11] not in valid_set_list)
        elif split == 'val':
            dataset = load_dataset("formospeech/yttd_taigi_trs", name='train', split='train')
            self.dataset = dataset.filter(lambda sample: sample['id'][:11] in valid_set_list)
        else:  # 'test'
            self.dataset = load_dataset("formospeech/yttd_taigi_trs", name='test', split='train')
        print(f"{split} set size: {len(self.dataset)}")

        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)

        self.is_train = is_train
        self.use_pseudo_labels = use_pseudo_labels
        self.pseudo_dict = pseudo_dict # { id: (pseudo_text, ground_truth, cer) }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        lang = cfg.lang
        item = self.dataset[id]
        audio_id = item["id"]
        wav_data = item['audio']['array']
        text_mandarin = item['text_mandarin']
        wav_lens = len(wav_data)

        # =========== 決定 text ===========
        if (not self.is_train) or (not self.use_pseudo_labels):
            # val/test set 或者根本沒用 pseudo
            text = item["text"]
        else:
            # train set + pseudo
            pseudo_label, ground_truth, cer = self.pseudo_dict[audio_id]
            if not isinstance(pseudo_label, str) or not pseudo_label.strip():
                text = item["text"]
            else:
                text = pseudo_label
        # ===============================

        text = self.text_normalizer(text).replace(" ", "")
        text_mandarin = self.text_normalizer(text_mandarin).replace(" ", "")

        audio = wav_data.flatten().astype(np.float32)

        if self.max_length is not None:
            audio = whisper.pad_or_trim(audio.flatten(), length=self.max_length)
            
        n_mels = 80 if self.model_name != 'large-v3' else 128
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels) 

        prompt_ids = [self.tokenizer.sot_prev] + \
                    self.tokenizer.encode(" " + text_mandarin)
        prompt_lens = len(prompt_ids)

        teacher_dec_input_ids = prompt_ids + \
                                [self.tokenizer.sot, 
                                self.tokenizer.special_tokens["<|{}|>".format(lang)],
                                self.tokenizer.transcribe, 
                                self.tokenizer.no_timestamps] + \
                                self.tokenizer.encode(" " + text)
                
        student_dec_input_ids = [self.tokenizer.sot, 
                                self.tokenizer.special_tokens["<|{}|>".format(lang)],
                                self.tokenizer.transcribe, 
                                self.tokenizer.no_timestamps] + \
                                self.tokenizer.encode(" " + text)

        student_labels = student_dec_input_ids[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,
            "wav_lens": wav_lens,
            "prompt_lens": prompt_lens,
            "teacher_dec_input_ids": teacher_dec_input_ids,
            "student_dec_input_ids": student_dec_input_ids,
            "student_labels": student_labels,
        }

class DistillWhisperModule(LightningModule):
    def __init__(self, cfg, model_name, lang, pseudo_dict) -> None:
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
            # print(state_dict_updated.keys())
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

        # 部分複製權重 
        partial_init_student_from_teacher(self.teacher, self.student)
        
        # freeze student encoder gradients for ditil
        if cfg.freeze_encoder != 0:
            for param in self.student.encoder.parameters():
                param.requires_grad = False

        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language='zh', task='transcribe')
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.kd_loss_fn = nn.KLDivLoss(reduction='none') 
       
        self.special_token_set = set(self.tokenizer.special_tokens.values())
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        prompt_lens = batch["prompt_lens"]
        teacher_dec_input_ids = batch["teacher_dec_input_ids"].long()
        student_dec_input_ids = batch["student_dec_input_ids"].long()
        student_labels = batch["student_labels"].long()

        # ====== 1) Teacher (無梯度) ======
        with torch.no_grad():
            teacher_feat = self.teacher.encoder(input_ids)
            teacher_out = self.teacher.decoder(teacher_dec_input_ids, teacher_feat)

        # ====== 2) Student ======
        if self.cfg.freeze_encoder:
            student_out = self.student.decoder(student_dec_input_ids, teacher_feat)
        else:
            student_feat = self.student.encoder(input_ids)
            student_out = self.student.decoder(student_dec_input_ids, student_feat)

        # ====== 3) CE loss (data loss) ======
        ce_loss = self.ce_loss_fn(student_out.view(-1, student_out.size(-1)), student_labels.view(-1))

        # 針對 teacher_out 切除 prompt，並再度 padding
        teacher_out_no_prompt = slice_and_repad_teacher_logits(teacher_out, student_out, prompt_lens)

        # ====== 4) KD loss (distribution alignment) ======
        T = self.cfg.temperature
        teacher_probs = F.softmax(teacher_out_no_prompt / T, dim=-1) # [batch, seq_len, vocab_size]
        student_log_probs = F.log_softmax(student_out / T, dim=-1)
        
        # (A) 先算 "逐 token" KL-div, shape = [batch, seq_len, vocab_size]
        kl_all = self.kd_loss_fn(student_log_probs, teacher_probs)

        # (B) 根據 student_labels 遮蔽要忽略的 token (student_labels = -100)
        #     通常 student_labels = -100 表示該位置的 label 無效 (padding / ignore)
        padding_mask = (student_labels != -100).unsqueeze(-1)  # [batch, seq_len, 1]
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
        prompt_lens = batch["prompt_lens"]
        teacher_dec_input_ids = batch["teacher_dec_input_ids"].long()
        student_dec_input_ids = batch["student_dec_input_ids"].long()
        student_labels = batch["student_labels"].long()

        # ====== 1) Teacher (無梯度) ======
        with torch.no_grad():
            teacher_feat = self.teacher.encoder(input_ids)
            teacher_out = self.teacher.decoder(teacher_dec_input_ids, teacher_feat)

        # ====== 2) Student ======
        if self.cfg.freeze_encoder:
            student_out = self.student.decoder(student_dec_input_ids, teacher_feat)
        else:
            student_feat = self.student.encoder(input_ids)
            student_out = self.student.decoder(student_dec_input_ids, student_feat)

        # ====== 3) CE loss (data loss) ======
        ce_loss = self.ce_loss_fn(student_out.view(-1, student_out.size(-1)), student_labels.view(-1))

        # 針對 teacher_out 切除 prompt，並再度 padding
        teacher_out_no_prompt = slice_and_repad_teacher_logits(teacher_out, student_out, prompt_lens)

        # ====== 4) KD loss (distribution alignment) ======
        T = 1.0
        teacher_probs = F.softmax(teacher_out_no_prompt / T, dim=-1) # [batch, seq_len, vocab_size]
        student_log_probs = F.log_softmax(student_out / T, dim=-1)
        
        # (A) 先算 "逐 token" KL-div, shape = [batch, seq_len, vocab_size]
        kl_all = self.kd_loss_fn(student_log_probs, teacher_probs)

        # (B) 根據 student_labels 遮蔽要忽略的 token (student_labels = -100)
        #     通常 student_labels = -100 表示該位置的 label 無效 (padding / ignore)
        padding_mask = (student_labels != -100).unsqueeze(-1)  # [batch, seq_len, 1]
        kl_all = kl_all * padding_mask                 # 將無效位置的 KL 設為 0
        
        # (C) 將剩餘位置的 KL 做 sum，再除以 mask.sum() (有效token總數)，得到「平均」
        kd_loss = kl_all.sum() / padding_mask.sum()

        # (D) 別忘了再乘上 (T*T) 
        kd_loss = kd_loss * (T * T)

        # ====== 5) 總損失：alpha * CE + beta * KD ======
        alpha = self.cfg.alpha
        beta  = self.cfg.beta
        loss  = alpha * ce_loss + beta * kd_loss

        student_labels[student_labels == -100] = self.tokenizer.eot

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
                tokens[:, 3:].masked_select(mask).eq(student_labels[:, 3:].masked_select(mask))
            )
            total = torch.sum(mask)
            acc = n_correct.item() / (total.item() + 1e-6)
            acc = acc if acc < 1 else 0

            o_list, l_list = [], []
            for o, l in zip(tokens, student_labels):
                # 解碼並過濾掉特殊標籤
                decoded_o = self.tokenizer.decode([t for t in o if t.item() not in self.special_token_set])
                decoded_l = self.tokenizer.decode([t for t in l if t.item() not in self.special_token_set])
                
                # 正規化文本並移除空格
                normalized_o = self.text_normalizer(decoded_o).replace(" ", "")
                normalized_l = self.text_normalizer(decoded_l).replace(" ", "")

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

            log_prefix = {0: 'val', 1: 'test'}
            self.log("{}/ce_loss".format(log_prefix[dataloader_idx], mod), ce_loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/kd_loss".format(log_prefix[dataloader_idx], mod), kd_loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/loss_{}".format(log_prefix[dataloader_idx], mod), loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/cer_{}".format(log_prefix[dataloader_idx], mod), cer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
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
        dataset = YTTDTaigiTRSDataset('train',
                                    self.tokenizer, 
                                    SAMPLE_RATE,
                                    self.model_name,
                                    max_length=self.cfg.audio_max_length,
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
                          collate_fn=DistilPromptCollator())

    def val_dataloader(self):
        dataset = YTTDTaigiTRSDataset('val',
                                    self.tokenizer, 
                                    SAMPLE_RATE,
                                    self.model_name,
                                    max_length=self.cfg.audio_max_length,
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
                          collate_fn=DistilPromptCollator())
       
    def test_dataloader(self):
        dataset = YTTDTaigiTRSDataset('test',  
                                    self.tokenizer, 
                                    SAMPLE_RATE,
                                    self.model_name,
                                    max_length=self.cfg.audio_max_length,
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
                          collate_fn=DistilPromptCollator())

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

def slice_and_repad_teacher_logits(teacher_out, student_out, prompt_lens, pad_value=0.0):
    """
    teacher_out: [B, T_teacher, vocab]
    student_out: [B, T_student, vocab]
    prompt_lens: [B], 每個樣本 prompt token 數量
    pad_value:   slice 後再 padding 的填充值，預設填 0.0 (logits)
    """
    B, T_teacher, vocab = teacher_out.shape
    _, T_student, _     = student_out.shape
    teacher_slices = []
    lengths_after_cut = []

    # 1) 逐樣本砍掉 prompt
    for i in range(B):
        pl = prompt_lens[i].item()
        end_pos = pl + T_student 
        # 如果 pl + T_student 大於 T_teacher，就要避免 out-of-range
        # end_pos = min(end_pos, T_teacher)
        slice_i = teacher_out[i, pl : end_pos, :]
        teacher_slices.append(slice_i)
        lengths_after_cut.append(slice_i.shape[0])

    # 2) 找本 batch 切掉後的最長長度 
    max_len_after_cut = max(lengths_after_cut)

    # 3) 針對不同樣本再度 padding，讓它們都對齊到 max_len_after_cut
    teacher_padded_list = []
    for i, ts in enumerate(teacher_slices):
        seq_len = ts.shape[0]
        pad_needed = max_len_after_cut - seq_len

        if pad_needed > 0:
            # 在 time dim (seq_len) 的後面做 padding
            # F.pad 的填充值順序為 (left, right, top, bottom, ...)
            # 這裡只要在 seq_len 後面補即可 => (0, 0) for vocab dim, (0, pad_needed) for seq dim
            ts_padded = F.pad(ts, (0, 0, 0, pad_needed), value=pad_value)
        else:
            ts_padded = ts
        teacher_padded_list.append(ts_padded)

    # 4) 最終 stack 回 batch => [B, max_len_after_cut, vocab]
    teacher_out_no_prompt = torch.stack(teacher_padded_list, dim=0)
    return teacher_out_no_prompt

if __name__ == "__main__":
    cfg_yaml = sys.argv[1]
    with open(cfg_yaml, 'r') as file:
        dct = yaml.safe_load(file)
        cfg = types.SimpleNamespace(**dct)

    pseudo_dict = {}
    if cfg.use_pseudo_labels:
        df_train = pd.read_csv(cfg.pseudo_csv_path_train)
        for row in df_train.itertuples(index=False):
            pseudo_dict[row.id] = (
                row.pseudo_text,
                row.ground_truth,
                row.cer
            )
    
    print(cfg)
    print("audio max length: {}".format(cfg.audio_max_length))

    # Initialize WandB
    wandb.init(project="whisper-flamingo",
            config=cfg,
            name="distil-whisper-prompt(lr=1.0e-6, bs=16)",
    )
    
    tflogger, checkpoint_callback, callback_list = setup_logging_and_checkpoint_taigi(cfg.log_output_dir, 
                                                                                    cfg.check_output_dir, 
                                                                                    cfg.train_name, 
                                                                                    cfg.train_id,
                                                                                    cfg.monitor,
                                                                                    cfg.filename,
                                                                                    )
        
    model = DistillWhisperModule(cfg, cfg.model_name, cfg.lang, pseudo_dict,)
    
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
        trainer.fit(model, ckpt_path='last', val_dataloaders=[model.val_dataloader(), model.test_dataloader()])
    else:
        trainer.validate(model=model, dataloaders=[model.val_dataloader(), model.test_dataloader()]) # validate before training
        # trainer.fit(model, val_dataloaders=[model.val_dataloader(), model.test_dataloader()])

    # End the WandB run
    wandb.finish()