import os
import sys
import yaml
import types
import numpy as np
import torch
from torch import nn
from datasets import load_dataset, DatasetDict
from torch.utils.data import Dataset
import whisper
from pytorch_lightning import LightningModule, Trainer, seed_everything
from tqdm import tqdm
from spec_augment import spec_augment
from utils import (
    add_noise,
    WhisperDataCollatorWhithPadding_librispeech,
    whisper_optimizer,
    setup_logging_and_checkpoint_librispeech,
    wer_cer,
    DistributedSamplerWrapper,
)
from utils_batch_samplers import SortedBatchSampler
from whisper.normalizers.basic import BasicTextNormalizer
import wandb 
from pytorch_lightning.loggers import WandbLogger
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
import time
os.environ["WANDB_MODE"] = "disabled"
os.environ['WANDB_DIR'] = '/share/nas169/jerryyang/whisper-flamingo/wandb/'

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)

# my command
# python -u whisper_ft_librispeech.py config/audio/audio_en-x_tiny.yaml
# python -u whisper_ft_librispeech.py config/audio/audio_en-x_small.yaml

class LibriSpeechDataset(Dataset):
    def __init__(self, hf_split, tokenizer, sample_rate, model_name, model, max_length, 
                 spec_augment, noise_prob=0, noise_fn=None) -> None:
        super().__init__()

        # 直接使用 Hugging Face datasets API 加載數據
        self.dataset = load_dataset("librispeech_asr", split=hf_split)
        print(f"{hf_split} size: {len(self.dataset)} samples")
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
        
        # 使用 BasicTextNormalizer 正規化文本
        text = self.text_normalizer(text)
        
        if np.random.rand() > self.noise_prob: # disable noise
            audio = wav_data.flatten().astype(np.float32)
        else: # add noise
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

        dec_input_ids = [self.tokenizer.sot, 
                        self.tokenizer.special_tokens["<|{}|>".format(lang)],
                        self.tokenizer.transcribe, 
                        self.tokenizer.no_timestamps] + \
                        self.tokenizer.encode(" " + text)       
        labels = dec_input_ids[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "wav_lens": wav_lens,
            "audio": audio
        }

class WhisperModelModule(LightningModule):
    def __init__(self, cfg, model_name, lang) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_name = model_name
        print("Loading Whisper model and weights")
        self.model = whisper.load_model(model_name, 
                                        device='cpu', # avoid OOM on gpu 0 for distributed
                                        download_root='/share/nas169/jerryyang/whisper-flamingo/models',
                                        dropout_rate=cfg.dropout_rate
                                        )
        
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

        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language=cfg.lang, task='transcribe')
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.special_token_set = set(self.tokenizer.special_tokens.values())
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)
    
    def forward(self, mel, tokens):
        encoder_out = self.model.encoder(mel)
        decoder_out = self.model.decoder(tokens, encoder_out)
        return decoder_out
    
    def training_step(self, batch, batch_id):
        
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        
        audio_features = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_id, dataloader_idx=None):
               
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        audio = batch["audio"]
        wav_lens = batch["wav_lens"] 
        
        # 初始化要存儲結果的列表
        o_list, l_list = [], []

        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        
        labels[labels == -100] = self.tokenizer.eot
        tokens = torch.argmax(out, dim=2)

        eot_find = (torch.where(tokens == self.tokenizer.eot, 1, 0))

        # 針對每個序列進行檢查
        for i in range(eot_find.shape[0]):
            if torch.any(eot_find[i] == 1):  # 如果該序列中存在 EOT 標記
                first_eot = torch.argmax(torch.arange(eot_find.shape[1], 0, -1).cuda() * eot_find[i], dim=0, keepdim=True)
                tokens[i, torch.arange(eot_find.shape[1]).cuda() > first_eot] = self.tokenizer.eot

        mask = ~(tokens[:, 3:] == self.tokenizer.eot)      
        n_correct = torch.sum(
            tokens[:, 3:].masked_select(mask).eq(labels[:, 3:].masked_select(mask))
        )
        total = torch.sum(mask)
        acc = n_correct.item() / (total.item() + 1e-8)
        acc = acc if acc < 1 else 0
    
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

        for i, (hypo, ref) in enumerate(zip(o_list, l_list)):
            print("="*100)
            print("PRED: {}".format(hypo))
            print("REF:  {}".format(ref))
            if i == 1: break
        
        log_prefix = {0: 'dev-clean', 1: 'dev-other', 2: 'test-clean', 3: 'test-other'}
        self.log("{}/loss".format(log_prefix[dataloader_idx]), loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/cer".format(log_prefix[dataloader_idx]), cer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/wer".format(log_prefix[dataloader_idx]), wer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
                
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
        dataset = LibriSpeechDataset('train.clean.100+train.clean.360+train.other.500', 
                                      self.tokenizer, 
                                      SAMPLE_RATE,
                                      self.model_name,
                                      self.model,
                                      max_length=480000,
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
                        collate_fn=WhisperDataCollatorWhithPadding_librispeech())

    def val_dataloader_clean(self):
        dataset = LibriSpeechDataset('validation.clean', 
                                      self.tokenizer, 
                                      SAMPLE_RATE,
                                      self.model_name,
                                      self.model,
                                      max_length=cfg.audio_max_length,
                                      spec_augment=False,
                                      noise_prob=0
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
                          collate_fn=WhisperDataCollatorWhithPadding_librispeech()
                          )

    def val_dataloader_other(self):
        dataset = LibriSpeechDataset('validation.other', 
                                      self.tokenizer, 
                                      SAMPLE_RATE,
                                      self.model_name,
                                      self.model,
                                      max_length=cfg.audio_max_length,
                                      spec_augment=False,
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
                          collate_fn=WhisperDataCollatorWhithPadding_librispeech()
                          )

    def test_dataloader_clean(self):
        dataset = LibriSpeechDataset('test.clean', 
                                    self.tokenizer, 
                                    SAMPLE_RATE,
                                    self.model_name,
                                    self.model,
                                    max_length=cfg.audio_max_length,
                                    spec_augment=False,
                                    noise_prob=0
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
                          collate_fn=WhisperDataCollatorWhithPadding_librispeech()
                          )

    def test_dataloader_other(self):
        dataset = LibriSpeechDataset('test.other', 
                                    self.tokenizer, 
                                    SAMPLE_RATE,
                                    self.model_name,
                                    self.model,
                                    max_length=cfg.audio_max_length,
                                    spec_augment=False,
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
                          collate_fn=WhisperDataCollatorWhithPadding_librispeech()
                          )

cfg_yaml = sys.argv[1]
with open(cfg_yaml, 'r') as file:
    dct = yaml.safe_load(file)
    cfg = types.SimpleNamespace(**dct)

print(cfg)
print("audio max length: {}".format(cfg.audio_max_length))

# # Initialize WandB
wandb.init(project="whisper-flamingo",
           config=cfg,
        #    name="whisper tiny finetune on librispeech",
           name="whisper small finetune on librispeech",
)

tflogger, callback_list = setup_logging_and_checkpoint_librispeech(cfg.log_output_dir, 
                                                                    cfg.check_output_dir, 
                                                                    cfg.train_name, 
                                                                    cfg.train_id,
                                                                    cfg.monitor,
                                                                    cfg.filename)

model = WhisperModelModule(cfg, cfg.model_name, cfg.lang)
model.to("cuda")  # 確保所有權重都在 GPU

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model)
print(f"Total Trainable Parameters: {num_params / 1e6:.2f}M")

# 建立測試輸入
dummy_input = torch.randn(1, 80, 3000).to("cuda")  # (Batch, Mel, Time Frames)
dummy_tokens = torch.randint(0, 51865, (1, 30)).to("cuda")  # (Batch, Token Length)

flop_analyzer = FlopCountAnalysis(model, (dummy_input, dummy_tokens))
print(f"Total FLOPs: {flop_analyzer.total() / 1e9:.2f} GFLOPs")

def measure_inference_time(model, dummy_input, dummy_tokens, num_trials=10):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input, dummy_tokens)
            if device == "cuda":
                torch.cuda.synchronize()
    
    # 開始計時 (使用 torch.cuda.Event 進行更精確計時)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for _ in range(num_trials):
            _ = model(dummy_input, dummy_tokens)
            if device == "cuda":
                torch.cuda.synchronize()
    end_event.record()
    
    # 等待所有事件完成
    if device == "cuda":
        torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time = total_time_ms / num_trials
    print(f"Average Inference Time per sample: {avg_time:.2f} ms")
    
# 假設 dummy_input 和 dummy_tokens 已正確準備
measure_inference_time(model, dummy_input, dummy_tokens)

# # Create a WandB logger instance
# wandb_logger = WandbLogger()

# trainer = Trainer(
#     precision=cfg.precision,
#     accelerator="gpu",
#     max_steps=cfg.num_train_steps,
#     accumulate_grad_batches=cfg.gradient_accumulation_steps,
#     logger=wandb_logger,
#     callbacks=callback_list,
#     num_sanity_val_steps=0, # default is 2 batches, 0 to turn off
#     devices=cfg.num_devices,
#     val_check_interval=int(cfg.validate_every_n_batches * cfg.gradient_accumulation_steps), # validate after this number batches
#     check_val_every_n_epoch=None, # If None, validation will be done solely based on the number of training batches
#     reload_dataloaders_every_n_epochs=1, # shuffle the dataloader after an epoch
#     use_distributed_sampler=False, # implemented custom distributed trainer
# )

# print(cfg)
# resume_ckpt = f"{cfg.check_output_dir}/{cfg.train_id}/last.ckpt"
# if os.path.exists(resume_ckpt) and cfg.resume_training: # resume training, don't validate
#     trainer.fit(model, ckpt_path='last', val_dataloaders=[model.val_dataloader_clean(), model.val_dataloader_other()])
# else:
#     trainer.validate(model=model, dataloaders=[model.val_dataloader_clean(), model.val_dataloader_other(),
#                                                 model.test_dataloader_clean(), model.test_dataloader_other()]) # validate before training
#     # trainer.fit(model, val_dataloaders=[model.val_dataloader_clean(), model.val_dataloader_other(),
#     #                                     model.test_dataloader_clean(), model.test_dataloader_other()])

# # End the WandB run
# wandb.finish()