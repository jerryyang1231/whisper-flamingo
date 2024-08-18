import os
import sys
import yaml
import types
import numpy as np
import torch
from torch import nn
# from scipy.io import wavfile
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
    load_librispeech_data_,  # 使用專門為 LibriSpeech 設計的數據加載函數
    load_wave,
    add_noise,
    WhisperTextCollatorWhithPadding,
    whisper_optimizer,
    # whisper_video_projection_optimizer,
    # whisper_flamingo_projection_optimizer,
    setup_logging_and_checkpoint,
    wer_cer,
    DistributedSamplerWrapper,
)
from utils_batch_samplers import LengthBatchSampler
import librosa
from whisper.normalizers.basic import BasicTextNormalizer
import wandb 
from pytorch_lightning.loggers import WandbLogger
os.environ['WANDB_DIR'] = '/share/nas169/jerryyang/whisper-flamingo/wandb/'

# my command
# python -u whisper_ft_librispeech_text_new.py config/audio-text/at_en_tiny_new.yaml

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)

class LibriSpeechTextDataset(torch.utils.data.Dataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate, model_name, max_length, 
                 spec_augment, noise_prob=0, noise_fn=None, train=False, noise_snr=0, translation_base_dir=None) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.spec_augment = spec_augment
        self.noise_prob = noise_prob
        self.noise_fn = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else []
        self.train = train
        self.noise_snr = noise_snr
        self.translation_base_dir = translation_base_dir
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)  
        
        print("Dataloader max length : {}".format(max_length))
        # print("Loaded {} noise wavs".format(len(self.noise_fn)))

    def __len__(self):
        return len(self.audio_info_list)

    def _get_translation_text(self, audio_path):
        # 提取音檔的目錄和文件名
        file_dir = os.path.dirname(audio_path)
        file_id = os.path.basename(audio_path).replace('.flac', '')

        # 構建翻譯文件的路徑
        relative_dir = os.path.relpath(file_dir, '/share/nas169/jerryyang/corpus/LibriSpeech/LibriSpeech')
        trans_file_path = os.path.join(self.translation_base_dir, relative_dir, "{}-{}.trans.txt".format(file_id.split('-')[0], file_id.split('-')[1]))
        
        # 讀取翻譯文件並提取對應行的翻譯
        translated_text = ""
        if os.path.exists(trans_file_path):
            with open(trans_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) < 2:  # 如果沒有text，跳過該行
                        continue
                    line_id, text = parts
                    if line_id == file_id:
                        translated_text = text
                        break
                    
        return translated_text
    
    def __getitem__(self, id):
        lang = 'en'
        lang_tr = 'zh'
        audio_path, text, _ = self.audio_info_list[id]  # LibriSpeech 使用音頻和文本的二元組
        
        # 使用 BasicTextNormalizer 正規化文本
        text = self.text_normalizer(text)
        
        # 使用 librosa 讀取音檔
        wav_data, sample_rate = librosa.load(audio_path, sr=self.sample_rate)
        
        if np.random.rand() > self.noise_prob: # disable noise
            audio = wav_data.flatten().astype(np.float32)
        else: # add noise
            audio = add_noise(wav_data, self.noise_fn, noise_snr=0).flatten().astype(np.float32)

        audio_frames = len(audio.flatten()) // 160
        # pad audio to cfg.audio_max_length (longer samples filtered out already)
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

        # Seems like Whisper decode always predicts first token with space, so add a space in the beginning
        # dec_input_ids = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(" " + text)
        dec_input_ids = [self.tokenizer.sot, 
                        self.tokenizer.special_tokens["<|{}|>".format(lang)],
                        self.tokenizer.transcribe, 
                        self.tokenizer.no_timestamps] + \
                        self.tokenizer.encode(" " + text)
        labels = dec_input_ids[1:] + [self.tokenizer.eot]
        
        # 獲取對應的翻譯文本
        translated_text = self._get_translation_text(audio_path)
        translated_text = [self.tokenizer.sot,
                           self.tokenizer.special_tokens["<|{}|>".format(lang_tr)],
                           self.tokenizer.transcribe, 
                           self.tokenizer.no_timestamps] + \
                           self.tokenizer.encode(" " + translated_text)
        
        # 將 translated_text 轉換為 NumPy array 並且轉換為 float32
        translated_text = np.array(translated_text).astype(np.float32)
               
        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "translated_text": translated_text
        }

class WhisperTextModule(LightningModule):
    def __init__(self, cfg, model_name, lang, train_dataset, val_dataset_clean, val_dataset_other,
                test_dataset_clean, test_dataset_other) -> None:
        super().__init__()
        self.model_name = model_name
        print("Loading Whisper model and weights")
        self.model = whisper.load_model(model_name,
                                        device='cpu', # avoid OOM on gpu 0 for distributed
                                        download_root='/share/nas169/jerryyang/whisper-flamingo/experiments/whisper',
                                        dropout_rate=cfg.dropout_rate,
                                        add_gated_x_attn=cfg.add_gated_x_attn,)
               
        if cfg.pt_ckpt != '': # load audio-only FT ckpt
            checkpoint_root = '/share/nas169/jerryyang/whisper-flamingo/models/checkpoints/'
            state_dict = torch.load(os.path.join(checkpoint_root, cfg.pt_ckpt, 'last.ckpt'), map_location=torch.device('cpu'))
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
        self.__train_dataset = train_dataset
        self.train_dataset = train_dataset
        self.__val_dataset_clean = val_dataset_clean
        self.__val_dataset_other = val_dataset_other
        self.__test_dataset_clean = test_dataset_clean
        self.__test_dataset_other = test_dataset_other
        self.special_token_set = set(self.tokenizer.special_tokens.values())

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

        features_a, x_norm= self.model.encoder(input_ids, track_norm=True)
        
        out_at = self.model.decoder(dec_input_ids, features_a, xt=translated_text)

        # skip
        # if cfg.add_gated_x_attn == 0:
        #     features_a = self.model.encoder(input_ids, test_a=True)
        #     out_a = self.model.decoder(dec_input_ids, features_a)

        #     features_t = self.model.encoder(input_ids, test_v=True)
        #     out_t = self.model.decoder(dec_input_ids, features_t)

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
            
            o_list, o_list_full, l_list, l_list_full = [], [], [], []
            for o, l in zip(tokens, labels):
                o_list.append(self.tokenizer.decode([t for t in o if t.item() not in self.special_token_set]))
                # o_list_full.append(self.tokenizer.decode(o))
                l_list.append(self.tokenizer.decode([t for t in l if t.item() not in self.special_token_set]))
                # l_list_full.append(self.tokenizer.decode(l))
            wer, cer = wer_cer(hypo=o_list, ref=l_list)
        
            # for i, (hypo, hypo_full, ref, ref_full) in enumerate(zip(o_list, o_list_full, l_list, l_list_full)):
            print("Mod: {}".format(mod))
            for i, (hypo, ref) in enumerate(zip(o_list, l_list)):
                print("-"*10)
                print("PRED: {}".format(hypo))
                # print(hypo_full)
                print("REF:  {}".format(ref))
                # print(ref_full)
                if i == 1: break
            
            # log_prefix = 'val' if dataloader_idx == 1 else 'val_noisy_en_babble'
            log_prefix = {0: 'dev-clean', 1: 'dev-other', 2: 'test-clean', 3: 'test-other'}
            self.log("{}/loss_{}".format(log_prefix[dataloader_idx], mod), loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/cer_{}".format(log_prefix[dataloader_idx], mod), cer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/wer_{}".format(log_prefix[dataloader_idx], mod), wer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/acc_{}".format(log_prefix[dataloader_idx], mod), acc, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            # if self.cfg.add_gated_x_attn != 0 and dataloader_idx == 1: # only log for clean
            #     for i in range(0, 24, 6):
            #         self.log("val/attn_gate_layer_{}".format(i), self.model.decoder.blocks[i].attn_gate.tanh().item(), on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            #         self.log("val/ff_gate_layer_{}".format(i), self.model.decoder.blocks[i].ff_gate.tanh().item(), on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)         
        if dataloader_idx == 3: # only log for val,clean
            self.log("val/x_norm", x_norm, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("val/x_v_norm_pre", x_v_norm_pre, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("val/x_v_norm_post", x_v_norm_post, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        
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
        dataset = LibriSpeechTextDataset(self.__train_dataset, 
                                      self.tokenizer, 
                                      SAMPLE_RATE,
                                      self.model_name,
                                      max_length=None,
                                      spec_augment=self.cfg.spec_augment,
                                      noise_prob=cfg.noise_prob,
                                      train=True,
                                      noise_snr=cfg.noise_snr_train,
                                      translation_base_dir=cfg.translation_base_dir)  
        length_sorter = LengthBatchSampler(batch_bins=int(self.cfg.audio_max_length * self.cfg.batch_size),
                            shapes=[i[2] for i in self.__train_dataset],
                            sort_in_batch='descending',
                            sort_batch='shuffle',
                            drop_last=True,)
        if cfg.num_devices > 1:
            print("Using distributed sampler")
            length_sorter = DistributedSamplerWrapper(length_sorter)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=length_sorter,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperTextCollatorWhithPadding())

    def val_dataloader_clean(self):
        dataset = LibriSpeechTextDataset(self.__val_dataset_clean,
                                self.tokenizer, 
                                SAMPLE_RATE,
                                self.model_name,
                                max_length=None,
                                spec_augment=False,
                                noise_prob=0,
                                train=False,
                                translation_base_dir=cfg.translation_base_dir)
        length_sorter = LengthBatchSampler(batch_bins=int(self.cfg.audio_max_length * 16),
                            shapes=[i[2] for i in self.__val_dataset_clean],
                            sort_in_batch='descending',
                            sort_batch='descending',
                            drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=length_sorter,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperTextCollatorWhithPadding())
   
    def val_dataloader_other(self):
        dataset = LibriSpeechTextDataset(self.__val_dataset_other,
                                self.tokenizer, 
                                SAMPLE_RATE,
                                self.model_name,
                                max_length=None,
                                spec_augment=False,
                                noise_prob=0,
                                train=False,
                                translation_base_dir=cfg.translation_base_dir)
        length_sorter = LengthBatchSampler(batch_bins=int(self.cfg.audio_max_length * 16),
                            shapes=[i[2] for i in self.__val_dataset_other],
                            sort_in_batch='descending',
                            sort_batch='descending',
                            drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=length_sorter,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperTextCollatorWhithPadding())
    
    def test_dataloader_clean(self):
        dataset = LibriSpeechTextDataset(self.__test_dataset_clean, 
                                self.tokenizer, 
                                SAMPLE_RATE,
                                self.model_name,
                                max_length=None,
                                spec_augment=False,
                                noise_prob=0,
                                train=False,
                                translation_base_dir=cfg.translation_base_dir)
        length_sorter = LengthBatchSampler(batch_bins=int(self.cfg.audio_max_length * 16),
                            shapes=[i[2] for i in self.__test_dataset_clean],
                            sort_in_batch='descending',
                            sort_batch='descending',
                            drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=length_sorter,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperTextCollatorWhithPadding())
    
    def test_dataloader_other(self):
        dataset = LibriSpeechTextDataset(self.__test_dataset_other, 
                                self.tokenizer, 
                                SAMPLE_RATE,
                                self.model_name,
                                max_length=None,
                                spec_augment=False,
                                noise_prob=0,
                                train=False,
                                translation_base_dir=cfg.translation_base_dir)
        length_sorter = LengthBatchSampler(batch_bins=int(self.cfg.audio_max_length * 16),
                            shapes=[i[2] for i in self.__test_dataset_other],
                            sort_in_batch='descending',
                            sort_batch='descending',
                            drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=length_sorter,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperTextCollatorWhithPadding())

if __name__ == "__main__":
    cfg_yaml = sys.argv[1]
    with open(cfg_yaml, 'r') as file:
        dct = yaml.safe_load(file)
        cfg = types.SimpleNamespace(**dct)

    print(cfg)
    print("audio max length: {}".format(cfg.audio_max_length))

    # Initialize WandB
    # wandb.init(project="whisper-flamingo",
    #         config=cfg,
    #         # name="whisper-flamingo train librispeech with text",
    #         name="debug"
    # )
    
    tflogger, checkpoint_callback, callback_list = setup_logging_and_checkpoint(cfg.log_output_dir, 
                                                                                cfg.check_output_dir, 
                                                                                cfg.train_name, 
                                                                                cfg.train_id,
                                                                                cfg.monitor,)
        
    # 使用 LibriSpeech 數據集
    audio_transcript_pair_list = load_librispeech_data_(cfg.audio_max_length, cfg.text_max_length, include_audio_lens=True)

    model = WhisperTextModule(cfg, cfg.model_name, cfg.lang, 
                            audio_transcript_pair_list['train'], 
                            audio_transcript_pair_list['dev-clean'],
                            audio_transcript_pair_list['dev-other'],
                            audio_transcript_pair_list['test-clean'],
                            audio_transcript_pair_list['test-other'])

    # Create a WandB logger instance
    # wandb_logger = WandbLogger()
    
    strategy = DDPStrategy(find_unused_parameters=True) if cfg.num_devices > 1 else "auto"
    trainer = Trainer(
        precision=cfg.precision,
        strategy=strategy,
        accelerator="gpu",
        max_steps=cfg.num_train_steps,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=tflogger,
        # logger=[tflogger, wandb_logger],
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
        trainer.fit(model, ckpt_path='last', val_dataloaders=[model.val_dataloader_clean(), model.val_dataloader_other(),
                                                model.test_dataloader_clean(), model.test_dataloader_other()])
    else:
        trainer.validate(model=model, dataloaders=[model.val_dataloader_clean(), model.val_dataloader_other(),
                                                model.test_dataloader_clean(), model.test_dataloader_other()]) # validate before training
        # trainer.fit(model, val_dataloaders=[model.val_dataloader_clean(), model.val_dataloader_other(),
                                            # model.test_dataloader_clean(), model.test_dataloader_other()])

    # End the WandB run
    # wandb.finish()