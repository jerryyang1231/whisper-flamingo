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
    WhisperTextCollatorWhithPadding_taigi_cls,
    whisper_optimizer,
    whisper_flamingo_optimizer,
    setup_logging_and_checkpoint_taigi,
    wer_cer,
    DistributedSamplerWrapper,
    get_all_keywords
)
from utils_batch_samplers import SortedBatchSampler
from whisper.normalizers.basic import BasicTextNormalizer
os.environ["WANDB_MODE"] = "disabled"  # 禁用 WandB
import wandb 
from pytorch_lightning.loggers import WandbLogger
os.environ['WANDB_DIR'] = '/share/nas169/jerryyang/whisper-flamingo/wandb/'
from transformers import BertModel, BertTokenizer
import json

# my command
# python -u keyword_reprogram_per_word_mean_pooling.py config/audio-text/at_taigi_small_keyword_reprogram_m2.yaml
# python -u keyword_reprogram_per_word_mean_pooling.py config/audio-text/at_taigi_small_keyword_reprogram_m2_keyword_only.yaml
# python -u keyword_reprogram_per_word_mean_pooling.py config/audio-text/at_taigi_small_keyword_reprogram_m2_translation_only.yaml

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
        self.dictionary = dictionary  # 儲存辭典以便後續使用

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        lang = cfg.lang
        item = self.dataset[id]
        
        # 獲取音頻數據和文本
        wav_data = item['audio']['array']
        text = item['text']
        mandarin_text = item['text_mandarin']
        wav_lens = len(wav_data)

        text = self.text_normalizer(text)
        text = text.replace(" ", "")

        all_keywords = get_all_keywords(mandarin_text, self.dictionary, separate=True)
        
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

        translations = mandarin_text.replace(" ", "")
        translations = self.text_normalizer(translations)
        
        mandarin_words = mandarin_text.split()
        
        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "keywords": all_keywords,
            "translations": translations,
            "wav_lens": wav_lens,
            "mandarin_words": mandarin_words
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

        # if 'large' in self.model_name: # only decoder training
        #     for p in self.model.encoder.parameters():
        #         p.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.cfg = cfg        
        self.special_token_set = set(self.tokenizer.special_tokens.values())
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)
        
        # 初始化 BERT 分詞器和模型
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = BertModel.from_pretrained('bert-base-chinese').to(self.device)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        keywords_list = batch["keywords"] # List[List[str]]
        translations = batch["translations"]  # List[str]
        mandarin_words_list = batch["mandarin_words"]  # List[List[str]]
        
        # 將關鍵詞列表轉換為字符串，然後使用 Whisper tokenizer 獲取 token IDs
        # keywords_token_ids = [self.tokenizer.encode(' '.join(kw)) if kw else [] for kw in keywords_list]
        keywords_token_ids = [self.tokenizer.encode(''.join(kw)) if kw else [] for kw in keywords_list]

        # 將列表轉換為張量，並進行填充
        max_seq_len = max([len(ids) for ids in keywords_token_ids])
        padded_keywords_token_ids = []
        for ids in keywords_token_ids:
            if ids:
                ids_tensor = torch.tensor(ids + [self.tokenizer.eot] * (max_seq_len - len(ids))).to(self.device)
            else:
                ids_tensor = torch.tensor([self.tokenizer.eot] * max_seq_len).to(self.device)
            padded_keywords_token_ids.append(ids_tensor)
        xt_1_token_ids = torch.stack(padded_keywords_token_ids, dim=0)  # [batch_size, max_seq_len]

        # 為每個樣本的中文詞彙列表獲取嵌入
        source_embeddings_list = []
        for mandarin_words in mandarin_words_list:
            if mandarin_words:
                # 使用 BERT 分詞器對中文詞彙進行編碼
                bert_inputs_1 = self.bert_tokenizer(
                    mandarin_words,
                    add_special_tokens=False,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).to(self.device)
                # 通過 BERT 模型獲取嵌入
                with torch.no_grad():
                    bert_outputs = self.bert_model(**bert_inputs_1)
                # 獲取每個詞的嵌入，取最後一層隱狀態
                embeddings = bert_outputs.last_hidden_state  # [num_words, seq_len, hidden_size]
                # 對 seq_len 維度取平均（因為每個詞可能被分成多個子詞）
                embeddings = embeddings.mean(dim=1)  # [num_words, hidden_size]
                source_embeddings_list.append(embeddings)
            else:
                # 如果沒有中文詞彙，添加一個零向量
                embeddings = torch.zeros(1, self.bert_model.config.hidden_size).to(self.device)
                source_embeddings_list.append(embeddings)

        # 對每個樣本的 source_embeddings 進行填充，確保形狀一致
        max_num_words = max([emb.shape[0] for emb in source_embeddings_list])
        padded_source_embeddings = []
        for emb in source_embeddings_list:
            num_words = emb.shape[0]
            if num_words < max_num_words:
                pad_size = max_num_words - num_words
                pad_emb = torch.zeros(pad_size, emb.shape[1]).to(self.device)
                emb = torch.cat([emb, pad_emb], dim=0)
            padded_source_embeddings.append(emb)
        # 將列表轉換為張量，形狀為 [batch_size, S, hidden_size]
        source_embedding = torch.stack(padded_source_embeddings, dim=0)  # [batch_size, S, hidden_size]
        value_embedding = source_embedding  # 在這種情況下，value_embedding 與 source_embedding 相同

        # 使用 BERT 分詞器對翻譯文本進行編碼
        bert_inputs_2 = self.bert_tokenizer(
            translations,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512  # 根據需要調整最大長度
        ).to(self.device)
        
        # 通過 BERT 模型獲取輸出
        bert_outputs_2 = self.bert_model(**bert_inputs_2)
        translation_embeddings = bert_outputs_2.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        if self.cfg.add_gated_x_attn != 0: # freeze whisper encoder gradients for x-attn
            for p in self.model.encoder.parameters():
                p.requires_grad = False

        # if 'large' in self.model_name: # only decoder training, NOTE: be careful with linear layer here
        #     with torch.no_grad():
        #         features, x_v = self.model.encoder(input_ids, video, training=True)
        # else:
        audio_features = self.model.encoder(input_ids, training=True)

        # 將 BERT 輸出作為 xt 傳遞給解碼器
        # out = self.model.decoder(dec_input_ids, audio_features, xt_1=xt_1_token_ids, xt_2=translation_embeddings,
        #                             source_embedding=source_embedding, value_embedding=value_embedding)
        
        # keyword only
        out = self.model.decoder(dec_input_ids, audio_features, xt_1=xt_1_token_ids,
                                 source_embedding=source_embedding, value_embedding=value_embedding)
        
        # translation only
        # out = self.model.decoder(dec_input_ids, audio_features, xt_1=translation_embeddings)
        
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
            
    def validation_step(self, batch, batch_id, dataloader_idx=None):
        
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        keywords_list = batch["keywords"] # List[List[str]]
        translations = batch["translations"]  # List[str]
        mandarin_words_list = batch["mandarin_words"]  # List[List[str]]

        # 將關鍵詞列表轉換為字符串，然後使用 Whisper tokenizer 獲取 token IDs
        # keywords_token_ids = [self.tokenizer.encode(' '.join(kw)) if kw else [] for kw in keywords_list]
        keywords_token_ids = [self.tokenizer.encode(''.join(kw)) if kw else [] for kw in keywords_list]

        # 將列表轉換為張量，並進行填充
        max_seq_len = max([len(ids) for ids in keywords_token_ids])
        padded_keywords_token_ids = []
        for ids in keywords_token_ids:
            if ids:
                ids_tensor = torch.tensor(ids + [self.tokenizer.eot] * (max_seq_len - len(ids))).to(self.device)
            else:
                ids_tensor = torch.tensor([self.tokenizer.eot] * max_seq_len).to(self.device)
            padded_keywords_token_ids.append(ids_tensor)
        xt_1_token_ids = torch.stack(padded_keywords_token_ids, dim=0)  # [batch_size, max_seq_len]

        # 為每個樣本的中文詞彙列表獲取嵌入
        source_embeddings_list = []
        for mandarin_words in mandarin_words_list:
            if mandarin_words: # 也是list
                # 使用 BERT 分詞器對中文詞彙進行編碼
                bert_inputs_1 = self.bert_tokenizer(
                    mandarin_words,
                    add_special_tokens=False,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).to(self.device)
                # 通過 BERT 模型獲取嵌入
                with torch.no_grad():
                    bert_outputs = self.bert_model(**bert_inputs_1)
                # 獲取每個詞的嵌入，取最後一層隱狀態
                embeddings = bert_outputs.last_hidden_state  # [num_words, seq_len, hidden_size]
                # 對 seq_len 維度取平均（因為每個詞可能被分成多個子詞）
                embeddings = embeddings.mean(dim=1)  # [num_words, hidden_size]
                source_embeddings_list.append(embeddings)
            else:
                # 如果沒有中文詞彙，添加一個零向量
                embeddings = torch.zeros(1, self.bert_model.config.hidden_size).to(self.device)
                source_embeddings_list.append(embeddings)

        # 對每個樣本的 source_embeddings 進行填充，確保形狀一致
        max_num_words = max([emb.shape[0] for emb in source_embeddings_list])
        padded_source_embeddings = []
        for emb in source_embeddings_list:
            num_words = emb.shape[0]
            if num_words < max_num_words:
                pad_size = max_num_words - num_words
                pad_emb = torch.zeros(pad_size, emb.shape[1]).to(self.device)
                emb = torch.cat([emb, pad_emb], dim=0)
            padded_source_embeddings.append(emb)
        # 將列表轉換為張量，形狀為 [batch_size, S, hidden_size]
        # S = max_num_words
        source_embedding = torch.stack(padded_source_embeddings, dim=0)  # [batch_size, S, hidden_size]
        value_embedding = source_embedding  # 在這種情況下，value_embedding 與 source_embedding 相同

        # 使用 BERT 分詞器對文本進行編碼
        bert_inputs_2 = self.bert_tokenizer(
            translations,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512  # 根據需要調整最大長度
        ).to(self.device)
        
        # 通過 BERT 模型獲取輸出
        bert_outputs_2 = self.bert_model(**bert_inputs_2)
        translation_embeddings = bert_outputs_2.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        audio_features = self.model.encoder(input_ids)

        # 將 BERT 輸出作為 xt 傳遞給解碼器
        # out_at = self.model.decoder(dec_input_ids, audio_features, xt_1=xt_1_token_ids, xt_2=translation_embeddings,
        #                             source_embedding=source_embedding, value_embedding=value_embedding)
        
        # keyword only
        out_at = self.model.decoder(dec_input_ids, audio_features, xt_1=xt_1_token_ids,
                                 source_embedding=source_embedding, value_embedding=value_embedding)
        
        # translation only
        # out_at = self.model.decoder(dec_input_ids, audio_features, xt_1=translation_embeddings)

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
                           
            o_list, o_list_full, l_list, l_list_full = [], [], [], []
            for o, l in zip(tokens, labels):
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

            print("Mod: {}".format(mod))
            for i, (hypo, ref) in enumerate(zip(o_list, l_list)):
                print("="*100)
                print("PRED: {}".format(hypo))
                print("REF:  {}".format(ref))
                if i == 1: break
            
            log_prefix = {0: 'val', 1: 'test'}
            self.log("{}/loss_{}".format(log_prefix[dataloader_idx], mod), loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/cer_{}".format(log_prefix[dataloader_idx], mod), cer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/wer_{}".format(log_prefix[dataloader_idx], mod), wer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/acc_{}".format(log_prefix[dataloader_idx], mod), acc, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        
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
                          collate_fn=WhisperTextCollatorWhithPadding_taigi_cls())

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
                          collate_fn=WhisperTextCollatorWhithPadding_taigi_cls())
       
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
                          collate_fn=WhisperTextCollatorWhithPadding_taigi_cls())

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
    wandb.init(project="whisper-flamingo",
            config=cfg,
            # name="whisbert-flamingo taigi small keyword (reprogram per word mean pooling)"
            name="whisbert-flamingo taigi small keyword (reprogram per word mean pooling keyword only)"
            # name="whisbert-flamingo taigi small keyword (reprogram per word mean pooling translation only)"
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
    