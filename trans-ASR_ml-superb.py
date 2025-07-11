import os
import sys
import yaml
import types
import numpy as np
import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import Dataset
import whisper
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from spec_augment import spec_augment
from utils import (
    add_noise,
    TransASR_collator,
    whisper_optimizer,
    whisper_flamingo_optimizer,
    setup_logging_and_checkpoint_ml_superb,
    wer_cer,
    DistributedSamplerWrapper,
)
from utils_batch_samplers import SortedBatchSampler
from whisper.normalizers.basic import BasicTextNormalizer
import wandb
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizer, BertModel
import json

# os.environ["WANDB_MODE"] = "disabled"
os.environ['WANDB_DIR'] = '/share/nas169/jerryyang/whisper-flamingo/wandb/'

# my command
# python -u trans-ASR_ml-superb.py config/audio-text/ml-superb.yaml

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)

class MLSuperbDataset(Dataset):
    def __init__(self, hf_split, tokenizer, sample_rate, model_name, max_length, lang,
                spec_augment, noise_prob=0, noise_fn=None, noise_snr=0, translated_json_path=None) -> None:
        super().__init__()
        
        # 直接從 ML-SUPERB 讀取 `eng` 語言數據
        dataset = load_dataset("espnet/ml_superb_hf", split=hf_split)
        self.dataset = [item for item in dataset if item["language"] == "eng"]
        
        # **載入 translated_text JSON**
        self.translated_texts = {}
        if translated_json_path:
            with open(translated_json_path, "r", encoding="utf-8") as f:
                translated_data = json.load(f)
            # **建立 ID 對應 translated_text**
            self.translated_texts = {item["id"]: item["translated_text"] for item in translated_data}
        
        print(f"{hf_split} (eng) size: {len(self.dataset)} samples")

        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.lang = lang
        self.spec_augment = spec_augment
        self.noise_prob = noise_prob
        self.noise_fn = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else []
        self.noise_snr = noise_snr
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        lang = self.lang
        item = self.dataset[id]
        wav_data = item['audio']['array']
        wav_lens = len(wav_data)

        # **移除 text 前 5 個字元**
        text = item["text"][5:].strip()

        # **取得翻譯後的文本**
        translated_text = self.translated_texts.get(item["id"], None)

        # **文本正規化**
        text = self.text_normalizer(text)
        translated_text = self.text_normalizer(translated_text)

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

        # 針對每一個翻譯資料夾，取出翻譯文字
        all_translations = []
        all_translations.append(translated_text)

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "all_translations": all_translations,
            "wav_lens": wav_lens,
        }

class WhisperTextModule(LightningModule):
    def __init__(self, cfg, model_name) -> None:
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
            state_dict_updated = {k[6:]: v for k, v in state_dict.items()} # remove 'model.'
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

        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language=self.cfg.lang, task='transcribe')
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.special_token_set = set(self.tokenizer.special_tokens.values())
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)

        # 初始化 BERT 分詞器和模型
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
    
    def forward(self, mel, tokens, xt_list=None):
        encoder_out = self.model.encoder(mel)
        if xt_list is not None:
            decoder_out = self.model.decoder(tokens, encoder_out, xt_list=xt_list)
        else:
            decoder_out = self.model.decoder(tokens, encoder_out)
        return decoder_out

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

        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features, xt_list=all_xt)

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

        audio_features = self.model.encoder(input_ids)
        out_at = self.model.decoder(dec_input_ids, audio_features, xt_list=all_xt)

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
                
                # 對解碼結果進行正規化
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

        log_prefix = "dev"
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
        dataset = MLSuperbDataset('train',
            self.tokenizer,
            SAMPLE_RATE,
            self.model_name,
            max_length=self.cfg.audio_max_length,
            lang=self.cfg.lang,
            spec_augment=self.cfg.spec_augment,
            noise_prob=self.cfg.noise_prob,
            noise_snr=self.cfg.noise_snr_train,
            translated_json_path=self.cfg.train_translated_json_path
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
                          collate_fn=TransASR_collator()
                          )

    def val_dataloader(self):
        dataset = MLSuperbDataset('dev',
                self.tokenizer,
                SAMPLE_RATE,
                self.model_name,
                max_length=self.cfg.audio_max_length,
                lang=self.cfg.lang,
                spec_augment=False,
                noise_prob=0,
                translated_json_path=self.cfg.dev_translated_json_path
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
                            collate_fn=TransASR_collator()
                            )


if __name__ == "__main__":
    cfg_yaml = sys.argv[1]
    with open(cfg_yaml, 'r') as file:
        dct = yaml.safe_load(file)
        cfg = types.SimpleNamespace(**dct)

    print(cfg)
    print("audio max length: {}".format(cfg.audio_max_length))

    # Initialize WandB
    wandb.init(project="TransKD-ASR",
            config=cfg,
            name="transASR ml-superb",
    )
    
    callback_list = setup_logging_and_checkpoint_ml_superb(cfg.log_output_dir, 
                                                            cfg.check_output_dir, 
                                                            cfg.train_name, 
                                                            cfg.train_id,
                                                            cfg.monitor,
                                                            cfg.filename)
        
    model = WhisperTextModule(cfg, cfg.model_name)
    
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
    trainer.validate(model=model, dataloaders=model.val_dataloader()) # validate before training
    trainer.fit(model, val_dataloaders=model.val_dataloader())

    # End the WandB run
    wandb.finish()