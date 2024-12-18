import os
import sys
import yaml
import types
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset 
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import whisper
import argparse
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
from spec_augment import spec_augment
from utils import (
    add_noise,
    AdaKWSDataCollatorWhithPadding,
    AdaKWS_optimizer,
    setup_checkpoint_kws,
    DistributedSamplerWrapper,
    get_all_keywords,
)
from utils_batch_samplers import SortedBatchSampler
from whisper.normalizers.basic import BasicTextNormalizer
import wandb 
os.environ["WANDB_MODE"] = "disabled"  # 禁用 WandB
os.environ['WANDB_DIR'] = '/share/nas169/jerryyang/whisper-flamingo/wandb/'
import json

# my command
# python -u kws.py config/kws.yaml

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)

############################################
# Data Loading (Placeholder)
############################################

# valid_set_list 包含的前11字符的ID
valid_set_list = ['-d8TlAGYFmc', '3h8m__iwuJ4', '5mPJOkoIu3k', '87omMWX-DTw', 
                'E0-HOPE7_QU', 'EhqcvfaaYu8', 'gDDbnFcvWcQ', 'iy1fPQQSA6c',
                'kGbjIuzvPR8', 'MrwSzSVGiRE', 'yht8d59dCpo']

class YTTDTaigiTRSDataset(Dataset):
    def __init__(self, split, tokenizer, sample_rate, whisper_size, max_length, 
                 spec_augment, dictionary, noise_prob=0, noise_fn=None) -> None:
        super().__init__()
        
        # 使用 Hugging Face datasets API 加載資料，並進行切分
        if split == 'train':
            dataset = load_dataset("formospeech/yttd_taigi_trs", name='train', split='train')
            # 過濾出不在 valid_set_list 中的資料作為訓練集
            self.dataset = dataset.filter(lambda sample: sample['id'][:11] not in valid_set_list)
            print(f"train set size: {len(self.dataset)}")
        elif split == 'val':
            dataset = load_dataset("formospeech/yttd_taigi_trs", name='train', split='train')
            # 根據 valid_set_list 過濾驗證集
            self.dataset = dataset.filter(lambda sample: sample['id'][:11] in valid_set_list)
            print(f"valid set size: {len(self.dataset)}")
        else:  # 'test'
            self.dataset = load_dataset("formospeech/yttd_taigi_trs", name='test', split='train')
            print(f"test set size: {len(self.dataset)}")
        
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.whisper_size = whisper_size
        self.max_length = max_length
        self.spec_augment = spec_augment
        self.noise_prob = noise_prob
        self.noise_fn = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else []
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)
        self.dictionary = dictionary 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        lang = cfg.lang
        item = self.dataset[id]

        wav_data = item['audio']['array']
        text = item['text']
        mandarin_text = item['text_mandarin']
        wav_lens = len(wav_data)

        text = self.text_normalizer(text)
        text = text.replace(" ", "")

        all_keywords = get_all_keywords(mandarin_text, self.dictionary)
        all_keywords = [self.text_normalizer(word).replace(" ", "") for word in all_keywords]

        # 生成 `keyword_tokens`
        # 將所有關鍵字轉換為 tokens，並使用 tokenizer 編碼
        keyword_tokens = [self.tokenizer.encode(keyword) for keyword in all_keywords]

        # 生成二元 labels
        labels = [keyword in text for keyword in all_keywords]

        if np.random.rand() > self.noise_prob: # 不加噪音
            audio = wav_data.flatten().astype(np.float32)
        else: # 加噪音
            audio = add_noise(wav_data, self.noise_fn, noise_snr=0).flatten().astype(np.float32)
        
        audio_frames = len(audio.flatten()) // 160
        if self.max_length is not None:
            audio = whisper.pad_or_trim(audio.flatten(), length=self.max_length)
            
        n_mels = 80 if self.whisper_size != 'large-v3' else 128
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels) 
        
        return {
            "input_ids": mel,
            "keyword_tokens": keyword_tokens,
            "labels": labels,
            "wav_lens": wav_lens,
            "all_keywords": all_keywords,
        }


############################################
# Model Components
############################################

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=4, batch_first=True)
        self.d_model = 768
        self.fc_mu = nn.Linear(hidden_dim, self.d_model)
        self.fc_sigma = nn.Linear(hidden_dim, self.d_model)

    def forward(self, tokenized_keyword):
        # tokenized_keyword: [batch_size, max_keywords_per_sample, max_keyword_token_len]
        batch_size, max_keywords, max_len = tokenized_keyword.shape
        
        # 初始化輸出
        mu_v_list, sigma_v_list = [], []
        
        for i in range(max_keywords):
            # 1. 取出每個關鍵字序列
            keyword_seq = tokenized_keyword[:, i, :]  # [batch_size, max_keyword_token_len]
            
            # 2. Embedding 層
            emb = self.embedding(keyword_seq)  # [batch_size, max_keyword_token_len, embed_dim]
            
            # 3. LSTM 層
            _, (h, _) = self.lstm(emb)  # h: [num_layers, batch_size, hidden_dim]
            
            # 4. 取出最後一層的隱藏狀態
            h_final = h[-1]  # [batch_size, hidden_dim]
            
            # 5. 生成 mu 和 sigma
            mu_v = self.fc_mu(h_final)  # [batch_size, d_model]
            sigma_v = self.fc_sigma(h_final)  # [batch_size, d_model]

            # 6. 加入列表
            mu_v_list.append(mu_v)
            sigma_v_list.append(sigma_v)
        
        # 7. 堆疊結果
        mu_v = torch.stack(mu_v_list, dim=1)  # [batch_size, max_keywords_per_sample, d_model]
        sigma_v = torch.stack(sigma_v_list, dim=1)  # [batch_size, max_keywords_per_sample, d_model]
        
        return mu_v, sigma_v

class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, z, mu_v, sigma_v):
        # z: [B,T,D]
        # mu_v, sigma_v: [B,1,D] (已針對單一關鍵字)
        mu_z = z.mean(dim=1, keepdim=True)  # [B,1,D]
        sigma_z = z.var(dim=1, keepdim=True, unbiased=False).sqrt() + self.eps  # [B,1,D]
        z_norm = (z - mu_z) / sigma_z
        out = sigma_v * z_norm + mu_v  # 單一關鍵字，不需平均
        return out

class KeywordAdaptiveModule(nn.Module):
    def __init__(self, d_model=768, n_heads=8, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.adain1 = AdaIN()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.adain2 = AdaIN()
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mu_v, sigma_v):
        # x: [B,T,D], mu_v,sigma_v: [B,1,D]
        x_norm = self.adain1(x, mu_v, sigma_v)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)
        x_norm = self.adain2(x, mu_v, sigma_v)
        ff_out = self.fc2(F.relu(self.fc1(x_norm)))
        x = x + self.dropout2(ff_out)
        return x

class AdaKWSModel(LightningModule):
    def __init__(self, cfg, whisper_size, lang, num_classes=2) -> None:
        super().__init__()
        
        self.whisper_size = whisper_size
        print("Loading Whisper model and weights")
        self.whisper = whisper.load_model(whisper_size,
                                        device='cpu', # avoid OOM on gpu 0 for distributed
                                        download_root='/share/nas169/jerryyang/whisper-flamingo/models',
                                        dropout_rate=cfg.dropout_rate,
                                        add_gated_x_attn=cfg.add_gated_x_attn,
                                        bert_encoder = cfg.bert_encoder,
                                        mode = cfg.mode,
                                        sequential_gated_x_attn = cfg.sequential_gated_x_attn,
                                        )

        # Freeze the Whisper encoder
        for param in self.whisper.encoder.parameters():
            param.requires_grad = False
        
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language='zh', task='transcribe')
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.cfg = cfg        
        # self.special_token_set = set(self.tokenizer.special_tokens.values())
        # self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)
        
        # Text encoder (φ)
        self.text_encoder = TextEncoder(vocab_size=self.whisper.dims.n_vocab)

        # Keyword-Adaptive modules (two sequential blocks)
        self.kw_module1 = KeywordAdaptiveModule(d_model=self.text_encoder.d_model)
        self.kw_module2 = KeywordAdaptiveModule(d_model=self.text_encoder.d_model)

        # Classifier head
        # 對每個 keyword 輸出二分類結果，因此維度是 d_model -> 2
        self.classifier = nn.Linear(self.text_encoder.d_model, num_classes)
    
    def forward(self, audio, keyword_tokens):
        """
        audio: [B, n_mels, T]  # Whisper encoder expects [batch_size, n_mels, time]
        keyword_tokens: [B, K, L]
          B: batch_size
          K: 每條語音對應的keyword數量
          L: 該batch中最長的keyword token length
        """
        # 1. Whisper encoder
        audio_features = self.whisper.encoder(audio)  # [B, T, D=768]
        B, T, D = audio_features.shape
        
        # 2. Text encoder
        # mu_v, sigma_v: [B,K,D]
        mu_v, sigma_v = self.text_encoder(keyword_tokens)
        
        # 3. 平行處理，把[ B, T, D ]展開成[ B, K, T, D ]，再reshape成[ B*K, T, D ]
        #   同理mu_v, sigma_v reshape成[ B*K, D ] 
        K = mu_v.size(1)  # keyword數量
        audio_features = audio_features.unsqueeze(1).expand(B, K, T, D) # [B,K,T,D]
        audio_features = audio_features.contiguous().view(B*K, T, D)    # [B*K, T, D]
       
        mu_v = mu_v.view(B*K, D)       # [B*K, D]
        sigma_v = sigma_v.view(B*K, D) # [B*K, D]

        # 4. 通過 KeywordAdaptiveModule (兩層)
        z = self.kw_module1(audio_features, mu_v.unsqueeze(1), sigma_v.unsqueeze(1))  # shape [B*K,T,D]
        z = self.kw_module2(z, mu_v.unsqueeze(1), sigma_v.unsqueeze(1))               # shape [B*K,T,D]
        
        # 5. Pooling
        # z -> [B*K,D]
        z_pooled, _ = torch.max(z, dim=1)
        
        # 6. Classifier
        # logits -> [B*K,2]
        logits_flat = self.classifier(z_pooled)

        # 7. reshape回[B,K,2]
        logits = logits_flat.view(B, K, -1)  # [B,K,2]
        return logits
    
    def training_step(self, batch, batch_idx):
        
        input_ids = batch["input_ids"]
        keyword_tokens = batch["keyword_tokens"].long()
        labels = batch["labels"]
        
        # labels: [B,K] (bool) -> [B,K] (long)
        labels = labels.long()

        logits = self(input_ids, keyword_tokens) # [B,K,2]

        # reshape for loss: logits:[B*K,2], labels:[B*K]
        loss = self.loss_fn(logits.view(-1, 2), labels.view(-1))

        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        
        input_ids = batch["input_ids"]
        keyword_tokens = batch["keyword_tokens"].long()
        labels = batch["labels"].long()
        all_keywords = batch["all_keywords"]

        logits = self(input_ids, keyword_tokens) # [B,K,2]
        loss = self.loss_fn(logits.view(-1, 2), labels.view(-1))
        
        # 計算accuracy
        pred = logits.argmax(dim=-1) # [B,K]
        acc = (pred == labels).float().mean() # 所有keyword平均正確率

        # 計算F1 score
        # 將pred和labels展開成1D向量方便計算整個batch的F1
        pred_flat = pred.view(-1)     # [B*K]
        labels_flat = labels.view(-1) # [B*K]

        # 計算TP、FP、FN
        # True Positive (TP): pred=1, label=1
        # False Positive (FP): pred=1, label=0
        # False Negative (FN): pred=0, label=1
        tp = ((pred_flat == 1) & (labels_flat == 1)).sum().float()
        fp = ((pred_flat == 1) & (labels_flat == 0)).sum().float()
        fn = ((pred_flat == 0) & (labels_flat == 1)).sum().float()

        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        
        # 這裡列印當前 batch 的預測與標籤結果
        # all_keywords: [[kw1, kw2, ...], [kw1, kw2,...], ...]
        # pred, labels: [B,K]
        # 透過 zip 一次處理 batch 中的每一個 sample
        for i, (keywords, p, l) in enumerate(zip(all_keywords, pred, labels)):
            print("="*100)
            print(f"Sample {i}:")
            for kw, pv, lv in zip(keywords, p.tolist(), l.tolist()):
                print(f"Keyword: {kw}, Predicted: {bool(pv)}, Ground Truth: {bool(lv)}")

        log_prefix = {0: 'val', 1: 'test'}
        self.log("{}/loss".format(log_prefix[dataloader_idx]), loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/acc".format(log_prefix[dataloader_idx]), acc, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/f1".format(log_prefix[dataloader_idx]), f1, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)

        return loss

    def configure_optimizers(self):
        optimizer, scheduler = AdaKWS_optimizer(self, self.cfg, self.t_total)        
        self.optimizer, self.scheduler = optimizer, scheduler
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = self.cfg.num_train_steps

    def train_dataloader(self):
        dataset = YTTDTaigiTRSDataset('train',
                                      self.tokenizer, 
                                      SAMPLE_RATE,
                                      self.whisper_size,
                                      max_length=self.cfg.audio_max_length,
                                      spec_augment=self.cfg.spec_augment,
                                      dictionary=dictionary,
                                      noise_prob=cfg.noise_prob)  
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
                          collate_fn=AdaKWSDataCollatorWhithPadding())

    def val_dataloader(self):
        dataset = YTTDTaigiTRSDataset('val',
                                    self.tokenizer, 
                                    SAMPLE_RATE,
                                    self.whisper_size,
                                    max_length=self.cfg.audio_max_length,
                                    spec_augment=False,
                                    dictionary=dictionary,
                                    noise_prob=0)               
        batch_sampler = SortedBatchSampler(
                    batch_size = self.cfg.batch_size,
                    shapes=[(item['wav_lens']) for item in dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=batch_sampler,
                          num_workers=self.cfg.num_worker,
                          collate_fn=AdaKWSDataCollatorWhithPadding())
       
    def test_dataloader(self):
        dataset = YTTDTaigiTRSDataset('test',  
                                    self.tokenizer, 
                                    SAMPLE_RATE,
                                    self.whisper_size,
                                    max_length=self.cfg.audio_max_length,
                                    spec_augment=False,
                                    dictionary=dictionary,
                                    noise_prob=0)                                
        batch_sampler = SortedBatchSampler(
                    batch_size = self.cfg.batch_size,
                    shapes=[(item['wav_lens']) for item in dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=batch_sampler,
                          num_workers=self.cfg.num_worker,
                          collate_fn=AdaKWSDataCollatorWhithPadding())

############################################
# Usage Example (You must customize)
############################################

if __name__ == "__main__":
    cfg_yaml = sys.argv[1]
    with open(cfg_yaml, 'r') as file:
        dct = yaml.safe_load(file)
        cfg = types.SimpleNamespace(**dct)

    print(cfg)
    print("audio max length: {}".format(cfg.audio_max_length))

    with open('mandarin2taibun.json', 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    
    # Initialize WandB
    wandb.init(project="AdaKWS",
            config=cfg,
            name="keyword spotter"
    )
    
    callback_list = setup_checkpoint_kws(cfg.log_output_dir, 
                                        cfg.check_output_dir, 
                                        cfg.train_name, 
                                        cfg.train_id,
                                        cfg.monitor,
                                        cfg.filename)
    
    # 如果cfg中有指定pt_ckpt，則載入已訓練完成的模型
    if hasattr(cfg, 'pt_ckpt') and cfg.pt_ckpt != '':
        print(f"Loading pretrained AdaKWS model from {cfg.pt_ckpt}")
        model = AdaKWSModel.load_from_checkpoint(
            checkpoint_path=cfg.pt_ckpt,
            strict=False,
            cfg=cfg,
            whisper_size=cfg.whisper_size,
            lang=cfg.lang
        )
    else:
        # 若沒有指定預訓練好的 ckpt，則新建一個模型 (用於初次訓練)
        model = AdaKWSModel(cfg, cfg.whisper_size, cfg.lang)

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
