import os
import sys
import yaml
import types
import numpy as np
import torch
from torch import nn
from datasets import load_dataset  # 載入 Hugging Face 的 datasets
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
    WhisperDataCollatorWhithPadding_taigi,
    whisper_optimizer,
    setup_logging_and_checkpoint_taigi,
    wer_cer,
    DistributedSamplerWrapper,
    get_all_keywords,
)
from utils_batch_samplers import SortedBatchSampler
from whisper.normalizers.basic import BasicTextNormalizer
# os.environ["WANDB_MODE"] = "disabled"  # 禁用 WandB
import wandb 
from pytorch_lightning.loggers import WandbLogger
os.environ['WANDB_DIR'] = '/share/nas169/jerryyang/whisper-flamingo/wandb/'
from transformers import BertModel, BertTokenizer
import json

# my command
# python -u find_best_eot_index.py config/audio-text/at_taigi_small_keyword_prompt_all_keywords.yaml

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
        filtered_keywords = [word for word in all_keywords if word in text]
        
        if np.random.rand() > self.noise_prob: # 不加噪音
            audio = wav_data.flatten().astype(np.float32)
        else: # 加噪音
            audio = add_noise(wav_data, self.noise_fn, noise_snr=0).flatten().astype(np.float32)
        
        audio_frames = len(audio.flatten()) // 160
        if self.max_length is not None:
            audio = whisper.pad_or_trim(audio.flatten(), length=self.max_length)
            
        n_mels = 80 if self.model_name != 'large-v3' else 128
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels) 
            
        if self.spec_augment:
            if self.spec_augment == "ls-double":
                mel = torch.from_numpy(spec_augment(mel.T.numpy(), audio_frames)).T
            elif self.spec_augment == "ls-basic":
                mel = torch.from_numpy(spec_augment(mel.T.numpy(), audio_frames, n_freq_mask=1, n_time_mask=1)).T
            else:
                raise NotImplementedError 
        
        prompt_ids = [self.tokenizer.sot_prev] + \
                        self.tokenizer.encode(" " + " ".join(all_keywords)) 
        
        prompt_lens = len(prompt_ids)

        dec_input_ids = prompt_ids + \
                        [self.tokenizer.sot, 
                        self.tokenizer.special_tokens["<|{}|>".format(lang)],
                        self.tokenizer.transcribe, 
                        self.tokenizer.no_timestamps] + \
                        self.tokenizer.encode(" " + text)

        labels = dec_input_ids[1:] + [self.tokenizer.eot]
        labels[:prompt_lens - 1] = [-100] * (prompt_lens - 1)
        
        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "wav_lens": wav_lens,
            "prompt_lens": prompt_lens,
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
                                        add_resnet= cfg.add_resnet,
                                        num_resnet_layer=cfg.num_resnet_layer,
                                        mode = cfg.mode,
                                        sequential_gated_x_attn = cfg.sequential_gated_x_attn,
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
                
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language='zh', task='transcribe')
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.cfg = cfg        
        self.special_token_set = set(self.tokenizer.special_tokens.values())
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        
        if self.cfg.prompt != 0: # freeze whisper encoder gradients for prompt
            for p in self.model.encoder.parameters():
                p.requires_grad = False

        audio_features = self.model.encoder(input_ids, training=True)

        out = self.model.decoder(dec_input_ids, audio_features)
        
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
            
    def validation_step(self, batch, batch_id, dataloader_idx=None):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        prompt_lens = batch["prompt_lens"]

        audio_features = self.model.encoder(input_ids)
        out_at = self.model.decoder(dec_input_ids, audio_features)
        
        mod_list = {"at": out_at}
        for mod, out in mod_list.items():
            # 計算損失，損失函數會自動忽略 -100 的位置
            loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

            # 獲取模型的預測
            tokens = torch.argmax(out, dim=2)  # [batch_size, seq_len]

            # 初始化 CER 記錄器，key 是填充的 `eot` 編號，value 是 CER 的列表
            cer_records = {}

            # 遍歷批次中的每個樣本
            batch_size = tokens.size(0)
            for i in range(batch_size):
                sample_tokens = tokens[i]
                # print("sample_tokens :", sample_tokens)
                sample_labels = labels[i]
                # print("sample_labels :", sample_labels)
                prompt_len = prompt_lens[i].item()
                # print("prompt_len :", prompt_len)

                # 找到該樣本中所有 `eot` 標記的位置
                eot_positions = (sample_tokens == self.tokenizer.eot).nonzero(as_tuple=False).squeeze()
                # print("eot_positions :", eot_positions)
                
                # 確保 `eot_positions` 是至少 1 維**
                if eot_positions.dim() == 0:
                    eot_positions = eot_positions.unsqueeze(0)

                # 如果存在 `eot` 標記，則進行迭代
                if eot_positions.numel() > 0:
                    num_eot = eot_positions.numel()
                    # print("num_eot :", num_eot)
                    # 遍歷從第 1 個到第 `num_eot` 個 `eot` 標記
                    for n in range(1, num_eot + 1):
                        # 創建 tokens 的副本
                        tokens_modified = sample_tokens.clone()
                        # print("tokens_modified :", tokens_modified)
                        # 獲取第 n 個 `eot` 標記的位置
                        eot_index = eot_positions[n - 1].item()
                        # print("eot_index :", eot_index)
                        # 從第 n 個 `eot` 標記之後，填充 `eot`
                        tokens_modified[eot_index + 1:] = self.tokenizer.eot
                        # print("tokens_modified :", tokens_modified)

                        # 排除 prompt_ids 部分
                        o = tokens_modified[prompt_len:]
                        l = sample_labels

                        # 過濾掉特殊標籤和忽略的標籤
                        o_filtered = [t for t in o if t.item() not in self.special_token_set]
                        l_filtered = [t for t in l if t.item() not in self.special_token_set and t.item() != -100]
                        
                        # 解碼
                        decoded_o = self.tokenizer.decode(o_filtered)
                        decoded_l = self.tokenizer.decode(l_filtered)
                        
                        # 正規化文本並移除空格
                        normalized_o = self.text_normalizer(decoded_o).replace(" ", "")
                        # print("normalized_o :", normalized_o)
                        normalized_l = self.text_normalizer(decoded_l).replace(" ", "")
                        
                        # 計算 CER
                        _, cer = wer_cer(hypo=[normalized_o], ref=[normalized_l])
                        # print("cer :", cer)
                        
                        # 將 CER 記錄到 cer_records 中
                        if n in cer_records:
                            cer_records[n].append(cer)
                        else:
                            cer_records[n] = [cer]
                        # print("cer_records :", cer_records)
                        # input("key")
                else:
                    # 如果沒有 `eot`，可以選擇忽略或處理
                    pass  # 這裡選擇忽略該樣本

            # 計算每個 n 的平均 CER
            avg_cer_records = {n: sum(cer_list) / len(cer_list) for n, cer_list in cer_records.items()}

            log_prefix = {0: 'val', 1: 'test'}
            # 找到平均 CER 最小的 n
            if avg_cer_records:
                best_n = min(avg_cer_records, key=avg_cer_records.get)
                best_cer = avg_cer_records[best_n]
                print(f"the best position to padding eot：{best_n}，average CER：{best_cer}")
                # 您可以記錄或返回這些結果               
                self.log("{}/best_cer_{}".format(log_prefix[dataloader_idx], mod), best_cer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
                self.log("{}/best_eot_index_{}".format(log_prefix[dataloader_idx], mod), best_n, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            else:
                print("this batch do not have any eot token :( ")
            
            # 記錄損失
            self.log("{}/loss_{}".format(log_prefix[dataloader_idx], mod), loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)

        return
       
    def configure_optimizers(self):
        model = self.model
        optimizer, scheduler = whisper_optimizer(model, self.cfg, self.t_total, video=False)        
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
                          collate_fn=WhisperDataCollatorWhithPadding_taigi())

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
                          collate_fn=WhisperDataCollatorWhithPadding_taigi())
       
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
                          collate_fn=WhisperDataCollatorWhithPadding_taigi())

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
            name="whisper taigi small keyword prompt all keywords find best eot index"
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
        # logger=tflogger,
        logger=wandb_logger,
        callbacks=callback_list,
        num_sanity_val_steps=0, # default is 2 batches, 0 to turn off
        devices=cfg.num_devices,
        val_check_interval=int(cfg.validate_every_n_batches * cfg.gradient_accumulation_steps), # validate after this number batches
        # val_check_interval=cfg.validate_every_n_batches, # validate after this number batches
        check_val_every_n_epoch=None, # If None, validation will be done solely based on the number of training batches
        reload_dataloaders_every_n_epochs=1, # shuffle the dataloader after an epoch
        # gradient_clip_val=1, # TODO: add as config variable?
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
    