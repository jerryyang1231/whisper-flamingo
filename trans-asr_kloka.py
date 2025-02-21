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
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from tqdm import tqdm
from spec_augment import spec_augment
from utils import (
    add_noise,
    kloka_crawled_collator_with_trans,
    whisper_optimizer,
    whisper_flamingo_optimizer,
    setup_logging_and_checkpoint_kloka_crawled,
    wer_cer,
    DistributedSamplerWrapper,
)
from utils_batch_samplers import SortedBatchSampler
from whisper.normalizers.basic import BasicTextNormalizer
import wandb
from pytorch_lightning.loggers import WandbLogger
from transformers import BertModel, BertTokenizer
import pandas as pd
# os.environ["WANDB_MODE"] = "disabled"
os.environ['WANDB_DIR'] = '/share/nas169/jerryyang/whisper-flamingo/wandb/'

# my command
# python -u trans-asr_kloka.py config/audio-text/trans-asr_truku.yaml
# python -u trans-asr_kloka.py config/audio-text/flamingo_seediq.yaml
# python -u trans-asr_kloka.py config/audio-text/flamingo_amis.yaml

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)
HF_TOKEN = "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"

class KlokaCrawledDataset(Dataset):
    def __init__(self, split, tokenizer, sample_rate, model_name, lang, max_length, 
                spec_augment, config_names, noise_prob=0, noise_fn=None, translation_csv=None) -> None:
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
                use_auth_token=HF_TOKEN
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
                    use_auth_token=HF_TOKEN
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
        self.spec_augment = spec_augment
        self.lang = lang
        self.max_length = max_length
        self.noise_prob = noise_prob
        self.noise_fn = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else []
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)
        
        # 若提供了 translation_csv 則讀取印尼文翻譯對應表（CSV需包含欄位：id, chinese, translation）
        self.translation_dict = {}
        if translation_csv is not None:
            try:
                df = pd.read_csv(translation_csv)
                # 以 id 為 key，translation 為 value
                self.translation_dict = dict(zip(df["id"].astype(str), df["translation"]))
                print(f"Loaded {len(self.translation_dict)} translation entries from {translation_csv}")
            except Exception as e:
                print(f"Failed to load translation CSV: {e}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        lang = self.lang
        item = self.dataset[id]

        # 取出原始資料中的 id
        item_id = str(item.get("id", id))
        wav_data = item['audio']['array']
        wav_lens = len(wav_data)
        text = item['text']
        chinese = item['chinese']

        # 這裡原本的 chinese 欄位為中文翻譯，但我們希望使用印尼文翻譯
        # 如果 translation_dict 中有對應的翻譯，則使用；否則 fallback 為原有的 chinese 欄位
        if item_id in self.translation_dict:
            translation_text = self.translation_dict[item_id]
        else:
            translation_text = ""

        # 如果翻譯不是字串（例如是 NaN 或 float），則轉換為字串或設定為空字串
        if not isinstance(translation_text, str):
            translation_text = "" if np.isnan(translation_text) else str(translation_text)    

        # 使用 BasicTextNormalizer 正規化文本
        text = self.text_normalizer(text)
        chinese = self.text_normalizer(chinese)
        translation_text = self.text_normalizer(translation_text)

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
        dec_input_ids = [self.tokenizer.sot, 
                        self.tokenizer.special_tokens["<|{}|>".format(lang)],
                        self.tokenizer.transcribe, 
                        self.tokenizer.no_timestamps] + \
                        self.tokenizer.encode(" " + text)
        labels = dec_input_ids[1:] + [self.tokenizer.eot]
        
        all_translations = []
        all_translations.append(chinese)
        all_translations.append(translation_text)

        return {
            "wav_lens": wav_lens,
            "all_translations": all_translations,
            "input_ids": mel,
            "dec_input_ids": dec_input_ids,
            "labels": labels,
        }

class WhisperTextModule(LightningModule):
    def __init__(self, cfg, model_name, lang) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_name = model_name
        print("Loading Whisper model and weights")
        self.model = whisper.load_model(model_name,
                                        device='cpu', # avoid OOM on gpu 0 for distributed
                                        download_root='/share/nas169/jerryyang/whisper-flamingo/models',
                                        dropout_rate=cfg.dropout_rate,
                                        add_gated_x_attn=cfg.add_gated_x_attn,
                                        num_langs = cfg.num_langs,
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

        if self.cfg.add_gated_x_attn != 0: # freeze whisper encoder gradients for x-attn
            for p in self.model.encoder.parameters():
                p.requires_grad = False

        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language=cfg.lang, task='transcribe')
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.special_token_set = set(self.tokenizer.special_tokens.values())
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)
        
        # 初始化 BERT 分詞器和模型
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        all_translations = batch["all_translations"]
        # all_translations: list of length = batch_size
        # each element is a list of strings for that sample's translations

        # 假設每個 sample 有相同數量的翻譯 texts
        num_samples = len(all_translations)           # batch_size
        num_translations = len(all_translations[0])   # 每個 sample 的翻譯數量

        all_xt = []
        # 迴圈跑 num_translations 次，每次收集 "整個 batch" 對應的某個翻譯索引
        for t_idx in range(num_translations):
            # 收集「batch 中所有樣本」在第 t_idx 個翻譯
            batch_texts_for_this_translation = []
            for s_idx in range(num_samples):
                # 取第 s_idx 個 sample 的第 t_idx 筆翻譯
                text_t = all_translations[s_idx][t_idx]
                batch_texts_for_this_translation.append(text_t)

            # 一次把這批文字 (大小 = batch_size) 丟給 BERT
            tokenized = self.bert_tokenizer(
                batch_texts_for_this_translation,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=448
            ).to(self.device)

            outputs = self.bert_model(**tokenized)
            outputs_last_hidden_state = outputs.last_hidden_state  # shape = [batch_size, seq_len, hidden_size]

            all_xt.append(outputs_last_hidden_state)

        # 這樣 all_xt 就是一個 list，長度 = num_translations
        # all_xt[t] shape = [batch_size, seq_len, hidden_size]

        features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, features, xt_list=all_xt)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        all_translations = batch["all_translations"]
        # all_translations: list of length = batch_size
        # each element is a list of strings for that sample's translations

        # 假設每個 sample 有相同數量的翻譯 texts
        num_samples = len(all_translations)           # batch_size
        num_translations = len(all_translations[0])   # 每個 sample 的翻譯數量

        all_xt = []
        # 迴圈跑 num_translations 次，每次收集 "整個 batch" 對應的某個翻譯索引
        for t_idx in range(num_translations):
            # 收集「batch 中所有樣本」在第 t_idx 個翻譯
            batch_texts_for_this_translation = []
            for s_idx in range(num_samples):
                # 取第 s_idx 個 sample 的第 t_idx 筆翻譯
                text_t = all_translations[s_idx][t_idx]
                batch_texts_for_this_translation.append(text_t)

            # 一次把這批文字 (大小 = batch_size) 丟給 BERT
            tokenized = self.bert_tokenizer(
                batch_texts_for_this_translation,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=448
            ).to(self.device)

            outputs = self.bert_model(**tokenized)
            outputs_last_hidden_state = outputs.last_hidden_state  # shape = [batch_size, seq_len, hidden_size]
            
            all_xt.append(outputs_last_hidden_state)
        
        # 這樣 all_xt 就是一個 list，長度 = num_translations
        # all_xt[t] shape = [batch_size, seq_len, hidden_size]

        features_a = self.model.encoder(input_ids)
        out_at = self.model.decoder(dec_input_ids, features_a, xt_list=all_xt)

        labels[labels == -100] = self.tokenizer.eot

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
                
                # 正規化文本
                normalized_o = self.text_normalizer(decoded_o)
                normalized_l = self.text_normalizer(decoded_l)

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

            log_prefix = "eval"
            self.log("{}/loss_{}".format(log_prefix, mod), loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/cer_{}".format(log_prefix, mod), cer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/wer_{}".format(log_prefix, mod), wer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/acc_{}".format(log_prefix, mod), acc, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)

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
        dataset = KlokaCrawledDataset('train',
                                    self.tokenizer,
                                    SAMPLE_RATE,
                                    self.model_name,
                                    self.cfg.lang,
                                    max_length=self.cfg.audio_max_length,
                                    spec_augment=self.cfg.spec_augment,
                                    config_names=self.cfg.config_names,
                                    noise_prob=self.cfg.noise_prob,
                                    translation_csv=self.cfg.translation_csv_train
                                    )
        batch_sampler = SortedBatchSampler(
                                    batch_size = self.cfg.batch_size,
                                    shapes=[(item['wav_lens']) for item in dataset],
                                    sort_in_batch='descending',
                                    sort_batch='descending',
                                    drop_last=True
                                    )
        if self.cfg.num_devices > 1:
            print("Using distributed sampler")
            batch_sampler = DistributedSamplerWrapper(batch_sampler)
        return torch.utils.data.DataLoader(dataset,
                        batch_sampler=batch_sampler,
                        num_workers=self.cfg.num_worker,
                        collate_fn=kloka_crawled_collator_with_trans()
                        )

    def val_dataloader(self):
        dataset = KlokaCrawledDataset('eval',
                                    self.tokenizer,
                                    SAMPLE_RATE,
                                    self.model_name,
                                    self.cfg.lang,
                                    max_length=self.cfg.audio_max_length,
                                    spec_augment=False,
                                    config_names=self.cfg.config_names,
                                    noise_prob=0,
                                    translation_csv=self.cfg.translation_csv_eval
                                    )
        batch_sampler = SortedBatchSampler(
                                        batch_size=self.cfg.batch_size,
                                        shapes=[(item['wav_lens']) for item in dataset],
                                        sort_in_batch='descending',
                                        sort_batch='descending',
                                        drop_last=False
                                        )
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=batch_sampler,
                          num_workers=self.cfg.num_worker,
                          collate_fn=kloka_crawled_collator_with_trans()
                          )

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
            name="trans-asr_truku_bilingual_spec_augment",
            # name="Trans-ASR_seediq_Tgdaya bilingual",
            # name="Trans-ASR_amis_siwkulan bilingual"
    )

    callback_list = setup_logging_and_checkpoint_kloka_crawled(cfg.log_output_dir, 
                                                            cfg.check_output_dir, 
                                                            cfg.train_name, 
                                                            cfg.train_id,
                                                            cfg.monitor,
                                                            cfg.filename
                                                            )
        
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
        reload_dataloaders_every_n_epochs=1, # shuffle the dataloader after an epoch
        use_distributed_sampler=False, # implemented custom distributed trainer
        sync_batchnorm=True,
    )

    # TODO: save config file tp the checkpoint dir, also for pre-trained model
    print(cfg)
    trainer.validate(model=model, dataloaders=model.val_dataloader()) # validate before training
    trainer.fit(model, val_dataloaders=model.val_dataloader())

    # End the WandB run
    wandb.finish()