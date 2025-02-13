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
from tqdm import tqdm
from utils import (
    add_noise,
    prompt_collator,
    whisper_optimizer,
    setup_logging_and_checkpoint_kloka_crawled,
    wer_cer,
    DistributedSamplerWrapper,
)
from utils_batch_samplers import SortedBatchSampler
import wandb
from pytorch_lightning.loggers import WandbLogger
from whisper.normalizers.basic import BasicTextNormalizer
from datasets.download.download_config import DownloadConfig
# os.environ["WANDB_MODE"] = "disabled"
os.environ['WANDB_DIR'] = '/share/nas169/jerryyang/whisper-flamingo/wandb/'
# os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "86400"

# my command
# python -u whisper_ft_kloka_crawled.py config/audio/kloka_crawled.yaml

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)
HF_TOKEN = "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"
# download_config = DownloadConfig(timeout=86400)

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
                # download_config=download_config
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
                    # download_config=download_config
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
        prompt = "_".join([language, dialect])

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
        
        # 建立方言 prompt
        prompt_ids = [self.tokenizer.sot_prev] + \
                    self.tokenizer.encode(" " + prompt)
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
            "wav_lens": wav_lens,
            "input_ids": mel,
            "prompt": prompt,
            "prompt_lens": prompt_lens,
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

        # if cfg.pt_ckpt != '': # load audio-only FT ckpt
        #     checkpoint_root = '/share/nas169/jerryyang/whisper-flamingo/models/checkpoints/'
        #     state_dict = torch.load(os.path.join(checkpoint_root, cfg.pt_ckpt), map_location=torch.device('cpu'))
        #     state_dict = state_dict['state_dict']
        #     state_dict_updated = {k[6:]: v  for k, v in state_dict.items()} # remove 'model.'
        #     print(state_dict_updated.keys())
        #     try:
        #         self.model.load_state_dict(state_dict_updated) 
        #     except BaseException as e: 
        #         print(str(e))
        #         print("Loading weights with strict=False")
        #         self.model.load_state_dict(state_dict_updated, strict=False)

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

    def on_validation_epoch_start(self) -> None:
        # 使用一個字典來儲存每個 dataloader 的結果，key 為 dataloader_idx
        self._val_outputs = {}

    def validation_step(self, batch, batch_id, dataloader_idx=None):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        prompt_lens = batch["prompt_lens"]
        prompt = batch["prompt"][0]

        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        
        # labels[labels == -100] = self.tokenizer.eot
        tokens = torch.argmax(out, dim=2)

        # Set all decoder predictions after first eot to eot
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
        mask = (labels != -100) & (labels != self.tokenizer.eot)
        n_correct = torch.sum(
            tokens.masked_select(mask).eq(labels.masked_select(mask))
        )
        total = torch.sum(mask)
        acc = n_correct.item() / (total.item() + 1e-6)
        acc = acc if acc < 1 else 0
       
        o_list, l_list = [], []
        for idx, (o, l, pl) in enumerate(zip(tokens, labels, prompt_lens)):
            pl = pl.item()
            o = o[pl:]

            # 過濾掉特殊標籤和忽略的標籤
            o_filtered = [t for t in o if t.item() not in self.special_token_set]
            l_filtered = [t for t in l if t.item() not in self.special_token_set and t.item() != -100]

            # 解碼
            decoded_o = self.tokenizer.decode(o_filtered)
            decoded_l = self.tokenizer.decode(l_filtered)
            
            # 正規化文本並移除空格
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

        log_prefix = f"eval_{prompt}"
        self.log(f"{log_prefix}/loss", loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log(f"{log_prefix}/cer", cer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log(f"{log_prefix}/wer", wer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log(f"{log_prefix}/acc", acc, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)

        # 組成一個結果 dict（你也可以根據需要加入其他指標）
        result = {
                "loss": loss,
                "wer": wer,
                "cer": cer,
                "acc": acc,
            }
        
        # 將結果存到 _val_outputs 中，依據 dataloader_idx 進行區分
        if dataloader_idx not in self._val_outputs:
            self._val_outputs[dataloader_idx] = []
        self._val_outputs[dataloader_idx].append(result)
        
        return result

    def on_validation_epoch_end(self) -> None:
        total_wer = 0.0
        num_dataloaders = 0

        # 遍歷每個 dataloader 的驗證結果
        for dl_idx, outputs in self._val_outputs.items():
            if len(outputs) == 0:
                continue
            # 計算該 dataloader 的平均 WER
            avg_wer_dl = sum(x["wer"] for x in outputs) / len(outputs)
            total_wer += avg_wer_dl
            num_dataloaders += 1

        overall_avg_wer = total_wer / num_dataloaders if num_dataloaders > 0 else 0.0

        # 記錄整體平均 WER，作為 checkpoint 監控指標
        self.log("avg_eval/wer", overall_avg_wer, prog_bar=True, logger=True, sync_dist=True)

        # 若有需要，也可在此清除 self._val_outputs
        self._val_outputs = {}

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
                        collate_fn=prompt_collator()
                        )
    
    def val_dataloaders(self):
        # 假設 self.cfg.eval_config_names 是一個加號分隔的字串，每個項目為單一配置名稱
        eval_configs = self.cfg.eval_config_names.split("+")
        dataloaders = {}
        for config_name in eval_configs:
            config_name = config_name.strip()
            # 使用 KlokaCrawledDataset，但只傳入一個配置名稱
            dataset = KlokaCrawledDataset('eval',
                                        self.tokenizer,
                                        SAMPLE_RATE,
                                        self.model_name,
                                        max_length=self.cfg.audio_max_length,
                                        config_names=config_name,  # 傳入單一配置
                                        noise_prob=0
                                        )
            batch_sampler = SortedBatchSampler(batch_size=self.cfg.batch_size,
                                            shapes=[(item['wav_lens']) for item in dataset],
                                            sort_in_batch='descending',
                                            sort_batch='descending',
                                            drop_last=False
                                            )
            dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_sampler=batch_sampler,
                                                num_workers=self.cfg.num_worker,
                                                collate_fn=prompt_collator()
                                                )
            dataloaders[config_name] = dataloader
        return dataloaders

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
                name="whisper finetune kloka crawled (sota)"
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
        sync_batchnorm=True,
    )

    print(cfg)
    # eval_dataloaders = model.val_dataloaders()
    # for config_name, dl in eval_dataloaders.items():
    #     print(f"Evaluating config: {config_name}")
    #     trainer.validate(model=model, dataloaders=dl)
    eval_dataloaders = list(model.val_dataloaders().values())
    trainer.validate(model=model, dataloaders=eval_dataloaders)
    trainer.fit(model, val_dataloaders=eval_dataloaders)

    # End the WandB run
    wandb.finish()