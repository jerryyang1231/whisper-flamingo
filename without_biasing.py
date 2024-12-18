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
    add_noise,
    WhisperDataCollatorWhithPadding_taigi_no_biasing,
    whisper_optimizer,
    setup_logging_and_checkpoint_taigi,
    wer_cer,
    DistributedSamplerWrapper,
)
from utils_batch_samplers import SortedBatchSampler
from whisper.normalizers.basic import BasicTextNormalizer
os.environ["WANDB_MODE"] = "disabled"  # 禁用 WandB
import wandb 
from pytorch_lightning.loggers import WandbLogger
os.environ['WANDB_DIR'] = '/share/nas169/jerryyang/whisper-flamingo/wandb/'
import json

# my command
# python -u without_biasing.py config/audio-text/at_taigi_small_without_biasing.yaml

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)
# valid_set_list 包含的前11字符的ID
valid_set_list = ['-d8TlAGYFmc', '3h8m__iwuJ4', '5mPJOkoIu3k', '87omMWX-DTw', 
                'E0-HOPE7_QU', 'EhqcvfaaYu8', 'gDDbnFcvWcQ', 'iy1fPQQSA6c',
                'kGbjIuzvPR8', 'MrwSzSVGiRE', 'yht8d59dCpo']

class YTTDTaigiTRSDataset(Dataset):
    def __init__(self, split, tokenizer, sample_rate, model_name, max_length, 
                 spec_augment, noise_prob=0, noise_fn=None) -> None:
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        lang = cfg.lang
        item = self.dataset[id]
        
        # 獲取音頻數據和文本
        wav_data = item['audio']['array']
        text = item['text']
        wav_lens = len(wav_data)

        text = self.text_normalizer(text)
        text = text.replace(" ", "")
        
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

        targets = [self.tokenizer.sot, 
                    self.tokenizer.special_tokens["<|{}|>".format(lang)],
                    self.tokenizer.transcribe, 
                    self.tokenizer.no_timestamps] + \
                    self.tokenizer.encode(" " + text) + \
                    [self.tokenizer.eot]
        
        return {
            "input_ids": mel,
            "wav_lens": wav_lens,
            "targets": targets,
        }

class WhisperModelModule(LightningModule):
    def __init__(self, cfg, model_name, lang) -> None:
        super().__init__()
        self.model_name = model_name
        print("Loading Whisper model and weights")
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language='zh', task='transcribe')
        self.model = whisper.load_model(model_name,
                                        device='cpu', # avoid OOM on gpu 0 for distributed
                                        download_root='/share/nas169/jerryyang/whisper-flamingo/models',
                                        dropout_rate=cfg.dropout_rate,
                                        add_gated_x_attn=cfg.add_gated_x_attn,
                                        bert_encoder = cfg.bert_encoder,
                                        add_resnet= cfg.add_resnet,
                                        num_resnet_layer=cfg.num_resnet_layer,
                                        mode = cfg.mode,
                                        biasing = cfg.biasing,  # 開啟偏置處理
                                        tokenizer = self.tokenizer,  # 傳入 tokenizer
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
        
        self.cfg = cfg        
        self.special_token_set = set(self.tokenizer.special_tokens.values())
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        
        input_ids = batch["input_ids"]
        targets = batch["targets"].long()
        
        targetmask = targets != -100
        targets = targets * targetmask

        loss, _ = self.model.forward(input_ids, targets, sotlen)
        
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
            
    def validation_step(self, batch, batch_id, dataloader_idx=None):
        
        input_ids = batch["input_ids"]
        targets = batch["targets"].long()
        
        targetmask = targets != -100
        targets = targets * targetmask

        loss, output = self.model.forward(input_ids, targets, sotlen)

        targets = targets[:, sotlen:]
        
        output = output.view(targets.size(0), targets.size(1), -1).max(dim=-1)[1]

        # remove all decoder predictions after first eot for proper decoding
        tokens = output
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
            tokens[:, 3:].masked_select(mask).eq(targets[:, 3:].masked_select(mask))
        )
        total = torch.sum(mask)
        acc = n_correct.item() / (total.item() + 1e-6)
        acc = acc if acc < 1 else 0
                        
        o_list, l_list = [], []
        for o, l in zip(tokens, targets):
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

        for i, (hypo, ref) in enumerate(zip(o_list, l_list)):
            print("="*100)
            print("PRED: {}".format(hypo))
            print("REF:  {}".format(ref))
            if i == 1: break
        
        log_prefix = {0: 'val', 1: 'test'}
        self.log("{}/loss".format(log_prefix[dataloader_idx]), loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/cer".format(log_prefix[dataloader_idx]), cer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/wer".format(log_prefix[dataloader_idx]), wer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/acc".format(log_prefix[dataloader_idx]), acc, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        
        return {
            "cer": cer,
            "wer": wer,
            "loss": loss
        }
       
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
                          collate_fn=WhisperDataCollatorWhithPadding_taigi_no_biasing())

    def val_dataloader(self):
        dataset = YTTDTaigiTRSDataset('val',
                                    self.tokenizer, 
                                    SAMPLE_RATE,
                                    self.model_name,
                                    max_length=self.cfg.audio_max_length,
                                    spec_augment=False,
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
                          collate_fn=WhisperDataCollatorWhithPadding_taigi_no_biasing())
       
    def test_dataloader(self):
        dataset = YTTDTaigiTRSDataset('test',  
                                    self.tokenizer, 
                                    SAMPLE_RATE,
                                    self.model_name,
                                    max_length=self.cfg.audio_max_length,
                                    spec_augment=False,
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
                          collate_fn=WhisperDataCollatorWhithPadding_taigi_no_biasing())


if __name__ == "__main__":
    cfg_yaml = sys.argv[1]
    with open(cfg_yaml, 'r') as file:
        dct = yaml.safe_load(file)
        cfg = types.SimpleNamespace(**dct)

    print(cfg)
    print("audio max length: {}".format(cfg.audio_max_length))

    # Initialize WandB
    wandb.init(project="whisper-biasing",
            config=cfg,
            name="whisper-biasing taigi small without biasing"
    )
    
    tflogger, checkpoint_callback, callback_list = setup_logging_and_checkpoint_taigi(cfg.log_output_dir, 
                                                                                    cfg.check_output_dir, 
                                                                                    cfg.train_name, 
                                                                                    cfg.train_id,
                                                                                    cfg.monitor,
                                                                                    cfg.filename)
        
    model = WhisperModelModule(cfg, cfg.model_name, cfg.lang)

    # options = whisper.DecodingOptions(language="zh", fp16=False, without_timestamps=True)
    # tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language="zh")
    # decodetask = whisper.decoding.DecodingTask(model, options)
    # logit_filters = decodetask.logit_filters
    # sot_sequence = decodetask.sot_sequence
    # sotlen = len(sot_sequence)
    sotlen = 4
    
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
        # sync_batchnorm=True,
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
    