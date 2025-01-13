import os
import sys
import json
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
    KG_WhisperDataCollatorWhithPadding_taigi_translation,
    whisper_optimizer,
    setup_logging_and_checkpoint_taigi,
    wer_cer,
    DistributedSamplerWrapper,
    get_all_keywords,
)
from utils_batch_samplers import SortedBatchSampler
from whisper.normalizers.basic import BasicTextNormalizer
import wandb 
from pytorch_lightning.loggers import WandbLogger
os.environ["WANDB_MODE"] = "disabled"
os.environ['WANDB_DIR'] = '/share/nas169/jerryyang/whisper-flamingo/wandb/'
from transformers import BertModel, BertTokenizer

from collections import defaultdict
import random

# my command
# python -u kg-whisper-flamingo_2stage.py config/audio-text/kg-whisper-flamingo_2stage.yaml

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)
# valid_set_list 包含的前11字符的ID
valid_set_list = ['-d8TlAGYFmc', '3h8m__iwuJ4', '5mPJOkoIu3k', '87omMWX-DTw', 
                'E0-HOPE7_QU', 'EhqcvfaaYu8', 'gDDbnFcvWcQ', 'iy1fPQQSA6c',
                'kGbjIuzvPR8', 'MrwSzSVGiRE', 'yht8d59dCpo']

class YTTDTaigiTRSDataset(Dataset):
    def __init__(self, split, tokenizer, sample_rate, model_name, max_length, 
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
        self.model_name = model_name
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

        # 將所有關鍵字轉換為 tokens，並使用 tokenizer 編碼
        keyword_tokens = [self.tokenizer.encode("|" + keyword) for keyword in all_keywords]
        
        if np.random.rand() > self.noise_prob: # 不加噪音
            audio = wav_data.flatten().astype(np.float32)
        else: # 加噪音
            audio = add_noise(wav_data, self.noise_fn, noise_snr=0).flatten().astype(np.float32)
        
        audio_frames = len(audio.flatten()) // 160
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
        mandarin_text = mandarin_text.replace(" ", "")
        mandarin_text = self.text_normalizer(mandarin_text)
        
        return {
            "input_ids": mel,
            "dec_input_ids": dec_input_ids,
            "labels": labels,
            "wav_lens": wav_lens,
            "keyword_tokens": keyword_tokens,
            "translations": mandarin_text,
        }

class WhisperTextModule(LightningModule):
    def __init__(self, cfg, model_name, lang) -> None:
        super().__init__()
        self.model_name = model_name
        print("Loading Whisper model and weights")
        self.model = whisper.load_model(model_name,
                                        device='cpu', # avoid OOM on gpu 0 for distributed
                                        download_root='/share/nas169/jerryyang/whisper-flamingo/models',
                                        dropout_rate=cfg.dropout_rate,
                                        add_gated_x_attn=cfg.add_gated_x_attn,
                                        bert_encoder = cfg.bert_encoder,
                                        mode = cfg.mode,
                                        adakws_checkpoint = cfg.adakws_checkpoint
                                        )
        
        if cfg.pt_ckpt != '': # load audio-only FT ckpt
            checkpoint_root = '/share/nas169/jerryyang/whisper-flamingo/models/checkpoints/'
            state_dict = torch.load(os.path.join(checkpoint_root, cfg.pt_ckpt), map_location=torch.device('cpu'))
            state_dict = state_dict['state_dict']
            state_dict_updated = {k[6:]: v  for k, v in state_dict.items()} # remove 'model.'
            # print(state_dict_updated.keys())
            try:
                self.model.load_state_dict(state_dict_updated) 
            except BaseException as e: 
                # print(str(e))
                print("Loading weights with strict=False")
                self.model.load_state_dict(state_dict_updated, strict=False) 
        
        if cfg.prompt != 0: # freeze whisper encoder & keyword spotter gradients for prompt
            for param in self.model.encoder.parameters():
                param.requires_grad = False
                
            for param in self.model.keyword_spotter.parameters():
                param.requires_grad = False
        
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language='zh', task='transcribe')
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.cfg = cfg        
        self.special_token_set = set(self.tokenizer.special_tokens.values())
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)
        
        # 初始化 BERT 分詞器和模型
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = BertModel.from_pretrained('bert-base-chinese').to(self.device)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]           # [B, n_mels, T]
        dec_input_ids = batch["dec_input_ids"].long()   # [B, L_dec]
        labels = batch["labels"].long()  # 原本的 label shape: [B, L_dec]
        keyword_tokens = batch["keyword_tokens"].long()  # [B,K,L_kw]
        translations = batch["translations"]

        # 1. Whisper encoder
        audio_features = self.model.encoder(input_ids)  # [B,T,D]
        
        # 2. AdaKWS forward
        kws_logits = self.model.keyword_spotter(audio_features, keyword_tokens) # [B,K,2]
        kws_pred = kws_logits.argmax(dim=-1)  # [B,K]
        
        sop_id = self.tokenizer.sot_prev 
        new_dec_input_ids_list, new_labels_list, prompt_lens_list = self.gather_keywords_and_cat(kws_pred, keyword_tokens, dec_input_ids, labels, sop_id)        

        # 5. 對 new_dec_input_ids_list 和 new_labels_list 做 padding 到同一長度 b
        max_len = max(seq.size(0) for seq in new_dec_input_ids_list)
        padded_new_dec_ids = []
        padded_labels = []
        for dec_ids, lab in zip(new_dec_input_ids_list, new_labels_list):
            pad_len = max_len - dec_ids.size(0)
            
            # dec_ids padding
            dec_ids_padded = torch.cat([dec_ids, torch.full((pad_len,), 50257, device=dec_ids.device, dtype=dec_ids.dtype)], dim=0)
            padded_new_dec_ids.append(dec_ids_padded.unsqueeze(0))

            # labels padding
            lab_padded = torch.cat([lab, torch.full((pad_len,), -100, device=lab.device, dtype=lab.dtype)], dim=0)
            padded_labels.append(lab_padded.unsqueeze(0))

        padded_new_dec_ids = torch.cat(padded_new_dec_ids, dim=0)  # [B, max_len]
        padded_labels = torch.cat(padded_labels, dim=0)            # [B, max_len]
        
        # 使用 BERT 分詞器對文本進行編碼
        bert_inputs = self.bert_tokenizer(
            translations,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512 
        ).to(self.device)
        
        # 通過 BERT 模型獲取輸出
        bert_outputs = self.bert_model(**bert_inputs)
        translation_embeddings = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 6. Whisper decoder forward
        out = self.model.decoder(padded_new_dec_ids, audio_features, xt_1=translation_embeddings)  # [B, max_len, vocab_size]

        loss = self.loss_fn(out.view(-1, out.size(-1)), padded_labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
            
    def validation_step(self, batch, batch_id, dataloader_idx=None):
        input_ids = batch["input_ids"]           # [B, n_mels, T]
        dec_input_ids = batch["dec_input_ids"].long()   # [B, L_dec]
        labels = batch["labels"].long()  # 原本的 label shape: [B, L_dec]
        keyword_tokens = batch["keyword_tokens"].long()  # [B,K,L_kw]
        translations = batch["translations"]

        # 1. Whisper encoder
        audio_features = self.model.encoder(input_ids)
        
        # 2. AdaKWS forward
        kws_logits = self.model.keyword_spotter(audio_features, keyword_tokens)  # [B,K,2]
        print("kws_logits :", kws_logits)
        input("null")
        kws_pred = kws_logits.argmax(dim=-1)  # [B,K]
        print("kws_pred :", kws_pred)
        input("null")
        sop_id = self.tokenizer.sot_prev 
        new_dec_input_ids_list, new_labels_list, prompt_lens_list = self.gather_keywords_and_cat(kws_pred, keyword_tokens, dec_input_ids, labels, sop_id)

        # 5. 對 new_dec_input_ids_list 和 new_labels_list 做 padding 到同一長度 b
        max_len = max(seq.size(0) for seq in new_dec_input_ids_list)
        padded_new_dec_ids = []
        padded_labels = []
        for dec_ids, lab in zip(new_dec_input_ids_list, new_labels_list):
            pad_len = max_len - dec_ids.size(0)

            # dec_ids padding
            dec_ids_padded = torch.cat([dec_ids, torch.full((pad_len,), 50257, device=dec_ids.device, dtype=dec_ids.dtype)], dim=0)
            padded_new_dec_ids.append(dec_ids_padded.unsqueeze(0))

            # labels padding
            lab_padded = torch.cat([lab, torch.full((pad_len,), -100, device=lab.device, dtype=lab.dtype)], dim=0)
            padded_labels.append(lab_padded.unsqueeze(0))
        
        padded_new_dec_ids = torch.cat(padded_new_dec_ids, dim=0)  # [B, max_len]
        padded_labels = torch.cat(padded_labels, dim=0)            # [B, max_len]

        # 也要把 prompt_lens_list 轉為 tensor
        prompt_lens = torch.tensor(prompt_lens_list, dtype=torch.long, device=padded_new_dec_ids.device)

        # 使用 BERT 分詞器對文本進行編碼
        bert_inputs = self.bert_tokenizer(
            translations,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512 
        ).to(self.device)
        
        # 通過 BERT 模型獲取輸出
        bert_outputs = self.bert_model(**bert_inputs)
        translation_embeddings = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 6. Whisper decoder forward
        out_at = self.model.decoder(padded_new_dec_ids, audio_features, xt_1=translation_embeddings)  # [B, max_len, vocab_size]
        
        mod_list = {"at": out_at}
        for mod, out in mod_list.items():         
            # 計算損失，損失函數會自動忽略 -100 的位置
            loss = self.loss_fn(out.view(-1, out.size(-1)), padded_labels.view(-1))

            # remove all decoder predictions after first eot for proper decoding
            tokens = torch.argmax(out, dim=2) 

            for i in range(tokens.size(0)):
                pl = prompt_lens[i].item()  # prompt_lens[i] 是當前樣本的prompt長度
                # 對 tokens[i, pl:] 這段進行 EOT 搜尋
                eot_positions = (tokens[i, pl:] == self.tokenizer.eot).nonzero(as_tuple=False)
                if eot_positions.numel() > 0:
                    # 檢查第一個元素是否為 0
                    if eot_positions[0].item() == 0:
                        if eot_positions.size(0) > 1:
                            # 若有第二個元素，使用第二個位置
                            first_eot = pl + eot_positions[1].item()
                        else:
                            # 若沒有第二個元素，跳過此次處理
                            continue
                    else:
                        # 正常使用第一個元素
                        first_eot = pl + eot_positions[0].item()
                    # 從 first_eot+1 開始填上 eot (50257) 避免 decode 到 padding 區
                    tokens[i, first_eot + 1:] = self.tokenizer.eot

            # 計算準確率，忽略 -100 的位置
            mask = (padded_labels != -100) & (padded_labels != self.tokenizer.eot)
            n_correct = torch.sum(
                tokens.masked_select(mask).eq(padded_labels.masked_select(mask))
            )
            
            total = torch.sum(mask)
            acc = n_correct.item() / (total.item() + 1e-6)
            acc = acc if acc < 1 else 0

            # 計算 WER 和 CER
            o_list, l_list = [], []
            for idx, (o, l, pl) in enumerate(zip(tokens, padded_labels, prompt_lens)): 
                # pl 為當前 sample 在拼接完 keywords 後的 prompt 長度
                pl = pl.item()
                # 排除 prompt_ids 部分
                o = o[pl:]

                # 過濾掉特殊標籤和忽略的標籤
                o_filtered = [t for t in o if t.item() not in self.special_token_set]
                l_filtered = [t for t in l if t.item() not in self.special_token_set and t.item() != -100]
                
                # 解碼
                decoded_o = self.tokenizer.decode(o_filtered)
                decoded_l = self.tokenizer.decode(l_filtered)
                
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
            self.log("{}/loss_{}".format(log_prefix[dataloader_idx], mod), loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/cer_{}".format(log_prefix[dataloader_idx], mod), cer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/wer_{}".format(log_prefix[dataloader_idx], mod), wer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/acc_{}".format(log_prefix[dataloader_idx], mod), acc, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        return

    def gather_keywords_and_cat(
        self,
        kws_pred: torch.Tensor,         # [B, K] with 0/1
        keyword_tokens: torch.Tensor,   # [B, K, L_kw]
        dec_input_ids: torch.Tensor,    # [B, L_dec]
        labels: torch.Tensor,           # [B, L_dec]
        sop_id: int,
    ):
        """
        回傳:
        new_dec_input_ids_list: List[torch.Tensor], 每個sample都拼好 sop + 所有keyword(=1) + 原 dec_input_ids
        new_labels_list: List[torch.Tensor], 每個 sample 的 labels，前面補 a 個 -100
        prompt_lens_list: List[int], 每個sample對應的 prompt 長度
        """
        B, K, L_kw = keyword_tokens.shape
        device = keyword_tokens.device
        max_k = self.cfg.max_keywords_for_prompt  

        # (1) 找出哪些位置為1: shape [N,2]，每行 [b_idx, k_idx]
        indices = (kws_pred == 1).nonzero(as_tuple=False)
        # print("indices :", indices)
        # (2) 以 b_idx 為key，把 k_idx 收集起來
        sample_dict = defaultdict(list)
        for row in indices:
            b_idx, k_idx = row[0].item(), row[1].item()
            sample_dict[b_idx].append(k_idx)
        # print("sample_dict :", sample_dict)
        new_dec_input_ids_list = []
        new_labels_list = []
        prompt_lens_list = []

        for b in range(B):
            # 取出該sample所有 k_idx
            k_idx_list = sample_dict[b]  # 該樣本的有效關鍵字索引
            # print("k_idx_list :", k_idx_list)
            if len(k_idx_list) == 0:
                # 沒有任何keyword
                chosen_tokens = torch.tensor([], dtype=torch.long, device=device)
                # print("chosen_tokens :", chosen_tokens)
            else:
                # 如果超過 max_k，就隨機挑選 max_k 個
                if len(k_idx_list) > max_k:
                    k_idx_list = random.sample(k_idx_list, max_k)
                # print("k_idx_list :", k_idx_list)
                # gather shape [X, L_kw]
                selected_tokens = keyword_tokens[b, k_idx_list, :]  # [X, L_kw]
                # print("selected_tokens :", selected_tokens)
                # 把多個 keyword tokens concat 成一條 => [X * L_kw]
                chosen_tokens = selected_tokens.reshape(-1)
            #     print("chosen_tokens :", chosen_tokens)
            # input("null")
            # 在最前面插入 sop_id
            chosen_tokens = torch.cat([torch.tensor([sop_id], device=device), chosen_tokens], dim=0)
            a = chosen_tokens.size(0)  # prompt長度(含 sop)
            prompt_lens_list.append(a)

            # 與 dec_input_ids 拼接
            new_dec_input_ids = torch.cat([chosen_tokens, dec_input_ids[b]], dim=0)
            new_dec_input_ids_list.append(new_dec_input_ids)
            
            # 對應地修改 labels，前面補 a 個 -100
            new_label = torch.cat([torch.full((a,), -100, device=labels.device, dtype=labels.dtype), labels[b]], dim=0)
            new_labels_list.append(new_label)

        return new_dec_input_ids_list, new_labels_list, prompt_lens_list
    
    def configure_optimizers(self):
        model = self.model
        optimizer, scheduler = whisper_optimizer(model, self.cfg, self.t_total)   
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
                          collate_fn=KG_WhisperDataCollatorWhithPadding_taigi_translation())

    def val_dataloader(self):
        dataset = YTTDTaigiTRSDataset('val',
                                    self.tokenizer, 
                                    SAMPLE_RATE,
                                    self.model_name,
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
                          collate_fn=KG_WhisperDataCollatorWhithPadding_taigi_translation())
       
    def test_dataloader(self):
        dataset = YTTDTaigiTRSDataset('test',  
                                    self.tokenizer, 
                                    SAMPLE_RATE,
                                    self.model_name,
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
                          collate_fn=KG_WhisperDataCollatorWhithPadding_taigi_translation())

if __name__ == "__main__":
    cfg_yaml = sys.argv[1]
    with open(cfg_yaml, 'r') as file:
        dct = yaml.safe_load(file)
        cfg = types.SimpleNamespace(**dct)

    print(cfg)
    print("audio max length: {}".format(cfg.audio_max_length))

    # 讀取您的 JSON 華台辭典
    with open('mandarin2taibun.json', 'r', encoding='utf-8') as f:
        dictionary = json.load(f)

    # Initialize WandB
    wandb.init(project="KG-whisper",
            config=cfg,
            name="kg whisper flamingo 2stage (learning rate=1.0e-6)"
    )
    
    tflogger, checkpoint_callback, callback_list = setup_logging_and_checkpoint_taigi(cfg.log_output_dir, 
                                                                                    cfg.check_output_dir, 
                                                                                    cfg.train_name, 
                                                                                    cfg.train_id,
                                                                                    cfg.monitor,
                                                                                    cfg.filename)
        
    model = WhisperTextModule(cfg, cfg.model_name, cfg.lang)
    
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
        reload_dataloaders_every_n_epochs=1, 
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
    