import os
import sys
import yaml
import types
import numpy as np
import torch
from torch import nn
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
import whisper
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from spec_augment import spec_augment
from utils import (
    add_noise,
    kloka_crawled_collator,
    whisper_optimizer,
    setup_logging_and_checkpoint_kloka_crawled,
    wer_cer,
    DistributedSamplerWrapper,
)
from utils_batch_samplers import SortedBatchSampler
from whisper.normalizers.basic import BasicTextNormalizer
import wandb
from pytorch_lightning.loggers import WandbLogger

# os.environ["WANDB_MODE"] = "disabled"
os.environ['WANDB_DIR'] = '/share/nas169/jerryyang/whisper-flamingo/wandb/'

# my command
# python -u whisper_ft_kloka_without_prompt.py config/audio/truku.yaml
# python -u whisper_ft_kloka_without_prompt.py config/audio/seediq.yaml
# python -u whisper_ft_kloka_without_prompt.py config/audio/amis.yaml

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)
HF_TOKEN = "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"

class KlokaCrawledDataset(Dataset):
    def __init__(self, split, tokenizer, sample_rate, model_name, max_length, 
                config_names, noise_prob=0, noise_fn=None) -> None:
        super().__init__()

        # 根據 split 選擇不同的 dataset
        dataset_name = "formospeech/kloka_crawled_asr_train" if split == 'train' else "formospeech/kloka_crawled_asr_eval"

        # 這裡若 config_names 為字串，會先分割，再依序下載
        if isinstance(config_names, str):
            config_names = config_names.split("+")

        # 若只需要單一 config 就不做合併
        if len(config_names) == 1:
            config_name = config_names[0].strip()
            ds = load_dataset(
                dataset_name,
                name=config_name,
                split='train',
                use_auth_token=HF_TOKEN,
            )

            # 取得原始數據筆數
            original_count = len(ds)
            ds = ds.filter(lambda example: example.get("chinese", "").strip() != "")

            # 取得過濾後的數據筆數
            filtered_count = len(ds)
            self.dataset = ds
            # print(f"Config '{config_name}': original count = {original_count}, filtered count = {filtered_count}")
            print(f"{split} set for '{config_name}' size: {len(self.dataset)}")
        else:
            # 訓練時使用多個 config 的合併
            all_datasets = []
            for config_name in config_names:
                config_name = config_name.strip()
                ds = load_dataset(
                    dataset_name,
                    name=config_name,
                    split='train',
                    use_auth_token=HF_TOKEN,
                )

                # 取得原始數據筆數
                original_count = len(ds)
                ds = ds.filter(lambda example: example.get("chinese", "").strip() != "")

                # 取得過濾後的數據筆數
                filtered_count = len(ds)
                # print(f"Config '{config_name}': original count = {original_count}, filtered count = {filtered_count}")
                all_datasets.append(ds)
            self.dataset = concatenate_datasets(all_datasets)
            print(f"{split} set (merged) size: {len(self.dataset)}")

        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.noise_prob = noise_prob
        self.noise_fn = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else []
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        lang = cfg.lang
        item = self.dataset[id]

        wav_data = item['audio']['array']
        wav_lens = len(wav_data)

        text = item['text']
        language = item['language']
        dialect = item['dialect']

        # 使用 BasicTextNormalizer 正規化文本
        text = self.text_normalizer(text)
        
        if np.random.rand() > self.noise_prob: # 不加噪音
            audio = wav_data.flatten().astype(np.float32)
        else: # 加噪音
            audio = add_noise(wav_data, self.noise_fn, noise_snr=0).flatten().astype(np.float32)

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

        return {
            "wav_lens": wav_lens,
            "input_ids": mel,
            "dec_input_ids": dec_input_ids,
            "labels": labels,
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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

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

        for i, (hypo, ref) in enumerate(zip(o_list, l_list)):
            print("="*100)
            print("PRED: {}".format(hypo))
            print("REF:  {}".format(ref))
            if i == 1: break

        log_prefix = "eval"
        self.log("{}/loss".format(log_prefix), loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/cer".format(log_prefix), cer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/wer".format(log_prefix), wer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/acc".format(log_prefix), acc, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
                
        return {
            "cer": cer,
            "wer": wer,
            "loss": loss
        }

    def configure_optimizers(self):
        model = self.model
        optimizer, scheduler = whisper_optimizer(model, self.cfg, self.t_total)
        self.optimizer, self.scheduler = optimizer, scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = self.cfg.num_train_steps

    def train_dataloader(self):
        dataset = KlokaCrawledDataset('train',
                                    self.tokenizer,
                                    SAMPLE_RATE,
                                    self.model_name,
                                    max_length=self.cfg.audio_max_length,
                                    config_names=self.cfg.config_names,
                                    noise_prob=self.cfg.noise_prob
                                    )
        batch_sampler = SortedBatchSampler(
                    batch_size = self.cfg.batch_size,
                    shapes=[(item['wav_lens']) for item in dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=True
                    )
        if cfg.num_devices > 1:
            print("Using distributed sampler")
            batch_sampler = DistributedSamplerWrapper(batch_sampler)
        return torch.utils.data.DataLoader(dataset,
                        batch_sampler=batch_sampler,
                        num_workers=self.cfg.num_worker,
                        collate_fn=kloka_crawled_collator()
                        )

    def val_dataloader(self):
        dataset = KlokaCrawledDataset('eval',
                                      self.tokenizer,
                                      SAMPLE_RATE,
                                      self.model_name,
                                      max_length=self.cfg.audio_max_length,
                                      config_names=self.cfg.config_names,
                                      noise_prob=0
                                    )
        batch_sampler = SortedBatchSampler(
                            batch_size = self.cfg.batch_size,
                            shapes=[(item['wav_lens']) for item in dataset],
                            sort_in_batch='descending',
                            sort_batch='descending',
                            drop_last=False
                            )
        return torch.utils.data.DataLoader(dataset,
                        batch_sampler=batch_sampler,
                        num_workers=self.cfg.num_worker,
                        collate_fn=kloka_crawled_collator()
                        )

cfg_yaml = sys.argv[1]
with open(cfg_yaml, 'r') as file:
    dct = yaml.safe_load(file)
    cfg = types.SimpleNamespace(**dct)

print(cfg)
print("audio max length: {}".format(cfg.audio_max_length))

# Initialize WandB
wandb.init(project="whisper-flamingo",
            config=cfg,
            # name="whisper finetune truku",
            # name="whisper finetune seediq",
            name="whisper finetune amis_siwkulan"
            )

callback_list = setup_logging_and_checkpoint_kloka_crawled(cfg.log_output_dir, 
                                                            cfg.check_output_dir, 
                                                            cfg.train_name, 
                                                            cfg.train_id,
                                                            cfg.monitor,
                                                            cfg.filename
                                                            )

model = WhisperModelModule(cfg, cfg.model_name, cfg.lang)

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
)

print(cfg)
trainer.validate(model=model, dataloaders=model.val_dataloader()) # validate before training
trainer.fit(model, val_dataloaders=model.val_dataloader())

# End the WandB run
wandb.finish()