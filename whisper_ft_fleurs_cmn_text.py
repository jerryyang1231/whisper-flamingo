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
# from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.strategies import DDPStrategy
from tqdm import tqdm
from spec_augment import spec_augment
from utils import (
    load_wave,
    add_noise,
    WhisperTextCollatorWhithPadding_librispeech,
    whisper_optimizer,
    whisper_flamingo_optimizer,
    setup_logging_and_checkpoint,
    wer_cer,
    DistributedSamplerWrapper,
)
from utils_batch_samplers import SortedBatchSampler
from whisper.normalizers.basic import BasicTextNormalizer
import wandb 
from pytorch_lightning.loggers import WandbLogger
os.environ['WANDB_DIR'] = '/share/nas169/jerryyang/whisper-flamingo/wandb/'

# my command
# python -u whisper_ft_fleurs_cmn_text.py config/audio-text/at_en_tiny_fleurs_cmn.yaml

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)

class fleursTextDataset(Dataset):
    def __init__(self, split, tokenizer, sample_rate, model_name, max_length, 
                 spec_augment, noise_prob=0, noise_fn=None, translation_base_dir=None) -> None:
        super().__init__()

        # 加載英文資料集
        self.datasets = load_dataset("google/fleurs", "en_us", split=split)
        print(f"English ({split}) set size: {len(self.datasets)}")
        self.split = split  # 保存當前 split (train, validation, test)
        
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.spec_augment = spec_augment
        self.noise_prob = noise_prob
        self.noise_fn = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else []
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)
        self.translation_base_dir = translation_base_dir  # 本地翻譯文本資料夾路徑

        # 加載對應split的transcription.txt
        self.translations = self._load_translations()

        print(f"Dataloader max length : {max_length}")

    def _load_translations(self):
        # 確定當前 split 的 transcription.txt 文件路徑
        transcription_file_path = os.path.join(self.translation_base_dir, self.split, "transcriptions.txt")
        # print("transcription_file_path :", transcription_file_path)
        
        # 檢查文件是否存在
        if not os.path.exists(transcription_file_path):
            raise FileNotFoundError(f"Transcription file not found for split: {self.split}")
        
        # 讀取 transcription.txt 文件，並將其內容保存為字典格式
        translations = {}
        with open(transcription_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(":")
                if len(parts) == 2:  # 確保行格式正確，應為 sample_id \t 翻譯文本
                    sample_id, translation = parts[0].strip(), parts[1].strip()  # 去除多餘空白
                    translations[sample_id] = translation
        
        return translations
    
    def __len__(self):
        return len(self.datasets)
    
    def _get_translation_text(self, sample_id):
        # 從已加載的翻譯字典中查找對應的 sample_id 翻譯
        translated_text = self.translations.get(str(sample_id), None)
        
        if translated_text is None:
            print(f"Translation not found for sample ID: {sample_id}")
        
        return translated_text
    
    def __getitem__(self, id):
        item = self.datasets[id]
        
        # 獲取英文音頻數據與文本
        wav_data = item['audio']['array']
        text = item['transcription']
        wav_lens = len(wav_data)
        sample_id = item['id']
        
        # 正規化英文文本
        text = self.text_normalizer(text)
        
        # 處理噪音或無噪音的音頻
        if np.random.rand() > self.noise_prob:
            audio = wav_data.flatten().astype(np.float32)
        else:
            audio = add_noise(wav_data, self.noise_fn, noise_snr=0).flatten().astype(np.float32)

        audio_frames = len(audio.flatten()) // 160
        if self.max_length is not None:
            audio = whisper.pad_or_trim(audio.flatten(), length=self.max_length)

        # 計算梅爾頻譜
        n_mels = 80 if self.model_name != 'large-v3' else 128
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)

        # 設置頻譜增強
        if self.spec_augment:
            mel = torch.from_numpy(spec_augment(mel.T.numpy(), audio_frames)).T

        dec_input_ids = [self.tokenizer.sot, 
                        self.tokenizer.special_tokens["<|{}|>".format('en')],
                        self.tokenizer.transcribe, 
                        self.tokenizer.no_timestamps] + \
                        self.tokenizer.encode(" " + text)

        labels = dec_input_ids[1:] + [self.tokenizer.eot]

        # 獲取對應的翻譯文本
        translated_text = self._get_translation_text(sample_id)
        
        # 檢查翻譯文本是否為 None，並印出沒有翻譯文本的 id
        if translated_text is None:
            print(f"Missing Chinese transcription for ID: {sample_id}")
            translated_text = ""
        
        # 使用 BasicTextNormalizer 正規化文本
        translated_text = self.text_normalizer(translated_text)
        translated_text = [self.tokenizer.sot,
                        self.tokenizer.special_tokens["<|{}|>".format('zh')],
                        self.tokenizer.transcribe, 
                        self.tokenizer.no_timestamps] + \
                        self.tokenizer.encode(" " + translated_text)
        
        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "translated_text": translated_text,
            "wav_lens": wav_lens,
            "audio": audio
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
                                        add_gated_x_attn=cfg.add_gated_x_attn,)
        
        if cfg.pt_ckpt != '': # load audio-only FT ckpt
            checkpoint_root = '/share/nas169/jerryyang/whisper-flamingo/models/checkpoints/'
            state_dict = torch.load(os.path.join(checkpoint_root, cfg.pt_ckpt), map_location=torch.device('cpu'))
            state_dict = state_dict['state_dict']
            state_dict_updated = {k[6:]: v  for k, v in state_dict.items()} # remove 'model.'
            print(state_dict_updated.keys())
            try:
                self.model.load_state_dict(state_dict_updated) 
            except BaseException as e: 
                print(str(e))
                print("Loading weights with strict=False")
                self.model.load_state_dict(state_dict_updated, strict=False) 
                
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language='en', task='transcribe')

        # if 'large' in self.model_name: # only decoder training
        #     for p in self.model.encoder.parameters():
        #         p.requires_grad = False

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
        translated_text = batch["translated_text"].long()

        if self.cfg.add_gated_x_attn != 0: # freeze whisper encoder gradients for x-attn
            for p in self.model.encoder.parameters():
                p.requires_grad = False

        # if 'large' in self.model_name: # only decoder training, NOTE: be careful with linear layer here
        #     with torch.no_grad():
        #         features, x_v = self.model.encoder(input_ids, video, training=True)
        # else:
        features = self.model.encoder(input_ids, training=True)

        out = self.model.decoder(dec_input_ids, features, xt=translated_text)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
            
    def validation_step(self, batch, batch_id, dataloader_idx=None):
        
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        translated_text = batch["translated_text"].long()
        audio = batch["audio"]
        wav_lens = batch["wav_lens"]  

        # 初始化要存儲結果的列表
        o_list, l_list = [], []

        if any(length > 480000 for length in wav_lens):
            # print("There is at least one sample with a length greater than 30 sec.")
            for i, sample in enumerate(audio):
                
                # 修剪音訊數據根據 wav_lens 長度
                trimmed_audio = sample[:wav_lens[i]]
                
                # 使用模型進行音頻解碼
                decode_result = whisper.transcribe(self.model, trimmed_audio)
                predicted_text = decode_result['text']
                
                # 解碼 ground truth labels
                label = labels[i].tolist()  # 轉換為 list
                label = [t for t in label if t != -100]  # 過濾掉 -100（忽略的標記）
                decoded_label = self.tokenizer.decode([t for t in label if t not in self.special_token_set])
                
                # 正規化預測文本和 ground truth 文本
                normalized_o = self.text_normalizer(predicted_text)
                normalized_l = self.text_normalizer(decoded_label)
                
                # 將正規化的結果添加到列表中
                o_list.append(normalized_o)
                l_list.append(normalized_l) 
        else:
            # print("No sample exceeds 30 sec.")
            features_a, x_norm= self.model.encoder(input_ids, track_norm=True)
        
            out_at = self.model.decoder(dec_input_ids, features_a, xt=translated_text)

            labels[labels == -100] = self.tokenizer.eot

            # mod_list = {"at": out_at, "a": out_a, "t": out_t} if cfg.add_gated_x_attn == 0 else {"at": out_at}
            mod_list = {"at": out_at}
            for mod, out in mod_list.items():
                loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
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
           
        # print("Mod: {}".format(mod))
        for i, (hypo, ref) in enumerate(zip(o_list, l_list)):
            print("="*100)
            print("PRED: {}".format(hypo))
            print("REF:  {}".format(ref))
            if i == 1: break
        
        log_prefix = {0: 'validation', 1: 'test'}
        self.log("{}/cer".format(log_prefix[dataloader_idx]), cer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/wer".format(log_prefix[dataloader_idx]), wer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
                
        return
       
    def configure_optimizers(self):
        model = self.model
        if self.cfg.add_gated_x_attn != 0:
            optimizer, scheduler = whisper_flamingo_optimizer(model, self.cfg, self.t_total)
        else:
            optimizer, scheduler = whisper_optimizer(model, self.cfg, self.t_total)
        self.optimizer, self.scheduler = optimizer, scheduler
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = self.cfg.num_train_steps

    def train_dataloader(self):
        dataset = fleursTextDataset('train', 
                                    self.tokenizer, 
                                    SAMPLE_RATE,
                                    self.model_name,
                                    max_length=None,
                                    spec_augment=self.cfg.spec_augment,
                                    noise_prob=cfg.noise_prob,
                                    translation_base_dir=cfg.translation_base_dir)  
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
                          collate_fn=WhisperTextCollatorWhithPadding_librispeech())

    def val_dataloader(self):
        dataset = fleursTextDataset('validation',
                                self.tokenizer, 
                                SAMPLE_RATE,
                                self.model_name,
                                max_length=self.cfg.audio_max_length,
                                spec_augment=False,
                                noise_prob=0,
                                translation_base_dir=cfg.translation_base_dir)
        batch_sampler = SortedBatchSampler(
                    batch_size = self.cfg.batch_size,
                    shapes=[(item['wav_lens']) for item in dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=batch_sampler,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperTextCollatorWhithPadding_librispeech())
    
    def test_dataloader(self):
        dataset = fleursTextDataset('test',  
                                self.tokenizer, 
                                SAMPLE_RATE,
                                self.model_name,
                                max_length=self.cfg.audio_max_length,
                                spec_augment=False,
                                noise_prob=0,
                                translation_base_dir=cfg.translation_base_dir)
        batch_sampler = SortedBatchSampler(
                    batch_size = self.cfg.batch_size,
                    shapes=[(item['wav_lens']) for item in dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=batch_sampler,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperTextCollatorWhithPadding_librispeech())

if __name__ == "__main__":
    cfg_yaml = sys.argv[1]
    with open(cfg_yaml, 'r') as file:
        dct = yaml.safe_load(file)
        cfg = types.SimpleNamespace(**dct)

    print(cfg)
    print("audio max length: {}".format(cfg.audio_max_length))

    # Initialize WandB
    wandb.init(project="whisper-flamingo",
            config=cfg,
            name="whisper-flamingo fleurs cmn (learning_rate = 5.0e-5 )",
    )
    
    tflogger, checkpoint_callback, callback_list = setup_logging_and_checkpoint(cfg.log_output_dir, 
                                                                                cfg.check_output_dir, 
                                                                                cfg.train_name, 
                                                                                cfg.train_id,
                                                                                cfg.monitor,)
        
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
        trainer.fit(model, val_dataloaders=[model.val_dataloader(), model.test_dataloader()])

    # End the WandB run
    wandb.finish()