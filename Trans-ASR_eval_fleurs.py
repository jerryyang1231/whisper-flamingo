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
from tqdm import tqdm
from spec_augment import spec_augment
from utils import (
    add_noise,
    Multiple_language_collator,
    whisper_optimizer,
    whisper_flamingo_optimizer,
    setup_logging_and_checkpoint_librispeech,
    wer_cer,
    DistributedSamplerWrapper,
)
from utils_batch_samplers import SortedBatchSampler
from whisper.normalizers.basic import BasicTextNormalizer
import wandb
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizer, BertModel
import time

os.environ["WANDB_MODE"] = "disabled"
os.environ['WANDB_DIR'] = '/share/nas169/jerryyang/whisper-flamingo/wandb/'

# my command
# python -u Trans-ASR_eval_fleurs.py config/Trans-ASR_eval_fleurs.yaml

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)

###############################################################################
# Fleurs 資料集：Evaluation 用的 Dataset (取代 LibriSpeech)
###############################################################################
class FleursTextDataset(Dataset):
    def __init__(self, hf_split, tokenizer, sample_rate, model_name, max_length, 
                 spec_augment, noise_prob=0, noise_fn=None, translation_configs=None) -> None:
        super().__init__()
        # 載入 Fleurs 資料集，使用 Hugging Face 的 "google/fleurs"
        self.dataset = load_dataset("google/fleurs", 'en_us', split=hf_split)
        print(f"Fleurs {hf_split} (en_us) size: {len(self.dataset)} samples")
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.spec_augment = spec_augment
        self.noise_prob = noise_prob
        self.noise_fn = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else []
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)
        
        # 如果提供 translation_configs（例如 ["cmn_hans_cn"]），則額外載入這些語言的 Fleurs 資料
        self.translation_dicts = {}  # key: language code, value: {id: transcription}
        if translation_configs is not None:
            for lang_config in translation_configs:
                # 載入對應語言的資料集
                ds_trans = load_dataset("google/fleurs", lang_config, split=hf_split)
                print(f"Fleurs {hf_split} ({lang_config}) size: {len(ds_trans)} samples")
                # 建立 id -> transcription 的字典（假設欄位名稱為 "transcription"）
                trans_dict = {str(item["id"]): item["transcription"] for item in ds_trans}
                self.translation_dicts[lang_config] = trans_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 假設 cfg.lang（全域變數）已設定，例如 "en"、"id" 等
        lang = cfg.lang
        item = self.dataset[idx]
        # Fleurs 資料集的音頻存於 'audio'，轉錄存於 'transcription'
        wav_data = item['audio']['array']
        transcription = item['transcription']
        wav_lens = len(wav_data)

        # 正規化轉錄文本
        transcription = self.text_normalizer(transcription)
        
        # 依據 noise_prob 決定是否加噪
        if np.random.rand() > self.noise_prob:
            audio = wav_data.flatten().astype(np.float32)
        else:
            audio = add_noise(wav_data, self.noise_fn, noise_snr=0).flatten().astype(np.float32)
        
        # Pad 或 trim 音訊至指定長度
        if self.max_length is not None:
            audio = whisper.pad_or_trim(audio.flatten(), length=self.max_length)
        
        n_mels = 80 if self.model_name != 'large-v3' else 128
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)
        
        if self.spec_augment:
            audio_frames = len(audio.flatten()) // 160
            if self.spec_augment == "ls-double":
                mel = torch.from_numpy(spec_augment(mel.T.numpy(), audio_frames)).T
            elif self.spec_augment == "ls-basic":
                mel = torch.from_numpy(spec_augment(mel.T.numpy(), audio_frames, n_freq_mask=1, n_time_mask=1)).T
            else:
                raise NotImplementedError
        
        # 建立 decoder 輸入：注意，Whisper decode 常以空白起始，因此在文本前加上空格
        dec_input_ids = [self.tokenizer.sot,
                         self.tokenizer.special_tokens["<|{}|>".format(lang)],
                         self.tokenizer.transcribe,
                         self.tokenizer.no_timestamps] + \
                         self.tokenizer.encode(" " + transcription)
        labels = dec_input_ids[1:] + [self.tokenizer.eot]
        
        # 處理額外的翻譯：依照 translation_dicts 中的每個語言，查找對應 sample id 的翻譯
        # 假設每個 sample 的 id 存在於 item['id']
        sample_id = str(item["id"])
        all_translations = []
        for lang_config, trans_dict in self.translation_dicts.items():
            trans_text = trans_dict.get(sample_id, "")  # 若找不到則為空字串
            trans_text = self.text_normalizer(trans_text)
            all_translations.append(trans_text)

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "wav_lens": wav_lens,
            "audio": audio,
            "all_translations": all_translations
        }

###############################################################################
# 模型部分：與原始 Trans-ASR 基本相同
###############################################################################
class WhisperModelModule(LightningModule):
    def __init__(self, cfg, model_name, lang) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_name = model_name
        print("Loading Whisper model and weights")
        self.model = whisper.load_model(model_name,
                                        device='cpu',  # 避免在分散式中 GPU 0 記憶體不足
                                        download_root='/share/nas169/jerryyang/whisper-flamingo/models',
                                        dropout_rate=cfg.dropout_rate,
                                        add_gated_x_attn=cfg.add_gated_x_attn,
                                        num_langs=cfg.num_langs)
        if cfg.pt_ckpt != '':
            checkpoint_root = '/share/nas169/jerryyang/whisper-flamingo/models/checkpoints/'
            state_dict = torch.load(os.path.join(checkpoint_root, cfg.pt_ckpt), map_location=torch.device('cpu'))
            state_dict = state_dict['state_dict']
            state_dict_updated = {k[6:]: v for k, v in state_dict.items()}  # 移除前綴 'model.'
            print(state_dict_updated.keys())
            try:
                self.model.load_state_dict(state_dict_updated)
            except BaseException as e:
                print(str(e))
                print("Loading weights with strict=False")
                self.model.load_state_dict(state_dict_updated, strict=False)
        if cfg.add_gated_x_attn != 0:
            for p in self.model.encoder.parameters():
                p.requires_grad = False
        
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language=cfg.lang, task='transcribe')
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.special_token_set = set(self.tokenizer.special_tokens.values())
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)
        
        # 初始化 BERT 分詞器與模型 (多語版本)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
    
    def forward(self, mel, tokens, xt_list=None):
        encoder_out = self.model.encoder(mel)
        if xt_list is not None:
            decoder_out = self.model.decoder(tokens, encoder_out, xt_list=xt_list)
        else:
            decoder_out = self.model.decoder(tokens, encoder_out)
        return decoder_out
    
    # evaluation 版本只需要實作 validation_step
    def validation_step(self, batch, batch_id, dataloader_idx=None):
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
        
        # 將忽略索引 -100 改成 eot token（以便於後續 decode）
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
        
        # 這裡用一個 dict 來對應各個 split（你可根據需求調整）
        log_prefix = {0: 'validation', 1: 'test'}
        
        self.log(f"{log_prefix[dataloader_idx]}/loss", loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{log_prefix[dataloader_idx]}/wer", wer, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{log_prefix[dataloader_idx]}/acc", acc, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss, "wer": wer, "acc": acc}

    def configure_optimizers(self):
        model = self.model
        optimizer, scheduler = whisper_optimizer(model, self.cfg, self.t_total, video=False)
        self.optimizer, self.scheduler = optimizer, scheduler
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = self.cfg.num_train_steps
    
    # evaluation 版本只需要一個 val_dataloader
    def val_dataloader(self):
        dataset = FleursTextDataset(
            hf_split="validation",  # 使用 Fleurs 的 validation split
            tokenizer=self.tokenizer,
            sample_rate=SAMPLE_RATE,
            model_name=self.model_name,
            max_length=self.cfg.audio_max_length,
            spec_augment=False,
            noise_prob=0,
            translation_configs=cfg.translation_configs
        )
        batch_sampler = SortedBatchSampler(
            batch_size=self.cfg.batch_size,
            shapes=[(item['wav_lens']) for item in dataset],
            sort_in_batch='descending',
            sort_batch='descending',
            drop_last=False
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.num_worker,
            collate_fn=Multiple_language_collator()
        )

    def test_dataloader(self):
        # 這裡以 Fleurs 的 validation split 為例，你也可以加入 test split 等
        dataset = FleursTextDataset(
            hf_split="test",  # 請根據 Fleurs 資料集實際 split 名稱做調整
            tokenizer=self.tokenizer,
            sample_rate=SAMPLE_RATE,
            model_name=self.model_name,
            max_length=self.cfg.audio_max_length,
            spec_augment=False,
            noise_prob=0,
            translation_configs=cfg.translation_configs
        )
        batch_sampler = SortedBatchSampler(
            batch_size=self.cfg.batch_size,
            shapes=[(item['wav_lens']) for item in dataset],
            sort_in_batch='descending',
            sort_batch='descending',
            drop_last=False
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.num_worker,
            collate_fn=Multiple_language_collator()  # 若需要，也可以針對 Fleurs 實作新的 collator
        )
    
###############################################################################
# 主程式：只進行 evaluation（validate）
###############################################################################
if __name__ == "__main__":
    cfg_yaml = sys.argv[1]
    with open(cfg_yaml, 'r') as file:
        dct = yaml.safe_load(file)
        cfg = types.SimpleNamespace(**dct)
    
    print(cfg)
    print("audio max length: {}".format(cfg.audio_max_length))
    
    wandb.init(project="whisper-flamingo",
               config=cfg,
               name="TransASR Evaluation on Fleurs")
    
    tflogger, callback_list = setup_logging_and_checkpoint_librispeech(
        cfg.log_output_dir,
        cfg.check_output_dir,
        cfg.train_name,
        cfg.train_id,
        cfg.monitor,
        cfg.filename
    )
    
    model = WhisperModelModule(cfg, cfg.model_name, cfg.lang)
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
        num_sanity_val_steps=0,
        devices=cfg.num_devices,
        val_check_interval=int(cfg.validate_every_n_batches * cfg.gradient_accumulation_steps),
        check_val_every_n_epoch=None,
        reload_dataloaders_every_n_epochs=1,
        use_distributed_sampler=False,
        sync_batchnorm=True,
    )
    
    print(cfg)
    # 如果需要 resume，這裡也可以做 checkpoint 載入；否則直接進行 evaluation
    resume_ckpt = f"{cfg.check_output_dir}/{cfg.train_id}/last.ckpt"
    if os.path.exists(resume_ckpt) and cfg.resume_training:
        trainer.fit(model, ckpt_path='last', val_dataloaders=[model.val_dataloader(), model.test_dataloader()])
    else:
        trainer.validate(model=model, dataloaders=[model.val_dataloader(), model.test_dataloader()])
    
    wandb.finish()
