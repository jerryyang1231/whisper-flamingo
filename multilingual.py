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
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
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
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
import time
os.environ["WANDB_MODE"] = "disabled"
os.environ['WANDB_DIR'] = '/share/nas169/jerryyang/whisper-flamingo/wandb/'

# my command
# python -u multilingual.py config/audio-text/monolingual.yaml
# python -u multilingual.py config/audio-text/monolingual_deu.yaml
# python -u multilingual.py config/audio-text/monolingual_fra.yaml
# python -u multilingual.py config/audio-text/monolingual_ita.yaml
# python -u multilingual.py config/audio-text/bilingual.yaml
# python -u multilingual.py config/audio-text/bilingual_top2.yaml
# python -u multilingual.py config/audio-text/trilingual.yaml
# python -u multilingual.py config/audio-text/trilingual_top3.yaml
# python -u multilingual.py config/audio-text/quadrilingual.yaml
# python -u multilingual.py config/audio-text/pentalingual.yaml

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)

class LibriSpeechTextDataset(Dataset):
    def __init__(self, hf_split, tokenizer, sample_rate, model_name, max_length, 
                spec_augment, noise_prob=0, noise_fn=None, noise_snr=0,
                translation_base_dirs=None) -> None:
        super().__init__()
       
        # Hugging Face split 到自定義 split 的映射字典
        self.split_mapping = {
            'train.clean.100': 'train-clean-100',
            'train.clean.360': 'train-clean-360',
            'train.other.500': 'train-other-500',
            'validation.clean': 'dev-clean',
            'validation.other': 'dev-other',
            'test.clean': 'test-clean',
            'test.other': 'test-other'
        }
        
        # 保存 Hugging Face 的 split 名稱
        self.hf_split = hf_split
        self.custom_split_names = [
            self.split_mapping.get(split.strip(), split.strip()) 
            for split in hf_split.split('+')
        ] if 'train' in hf_split else [self.split_mapping.get(hf_split, hf_split)]
        
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
        self.noise_snr = noise_snr

        self.translation_base_dirs = translation_base_dirs
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)

    def __len__(self):
        return len(self.dataset)
   
    def get_translation_text(self, file_id, translation_base_dir):
        
        # 提取 speaker_id 和 chapter_id
        speaker_id, chapter_id = file_id.split('-')[0], file_id.split('-')[1]
        
        text = ""

        # 遍歷所有可能的 custom_split_names 來查找對應的翻譯文件
        for custom_split_name in self.custom_split_names:
            relative_dir = os.path.join(custom_split_name, speaker_id, chapter_id)
            trans_file_path = os.path.join(translation_base_dir, relative_dir, f"{speaker_id}-{chapter_id}.trans.txt")
            
            # 讀取翻譯文件並提取對應行的翻譯
            if os.path.exists(trans_file_path):
                with open(trans_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)  # 進行拆分
                        if len(parts) == 2:  # 確保拆分出兩個部分
                            line_id, text = parts
                            if line_id == file_id:  # 檢查行ID是否匹配
                                return text  # 返回匹配的文本
                
        return text
    
    def __getitem__(self, id):
        lang= cfg.lang
        item = self.dataset[id]
        file_id = item['id']
        wav_data = item['audio']['array']
        text = self.text_normalizer(item['text'])
        wav_lens = len(wav_data)

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
        for base_dir in self.translation_base_dirs:
            t = self.get_translation_text(file_id, base_dir)
            t = self.text_normalizer(t)  # 正規化
            all_translations.append(t)

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "all_translations": all_translations,
            "wav_lens": wav_lens,
            "audio": audio
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

        features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, features, xt_list=all_xt)
        
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
            
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

        log_prefix = {0: 'dev-clean', 1: 'dev-other', 2: 'test-clean', 3: 'test-other'}
        self.log("{}/loss_{}".format(log_prefix[dataloader_idx], mod), loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
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
        dataset = LibriSpeechTextDataset('train.clean.100+train.clean.360+train.other.500', 
                                    self.tokenizer, 
                                    SAMPLE_RATE,
                                    self.model_name,
                                    max_length=self.cfg.audio_max_length,
                                    spec_augment=self.cfg.spec_augment,
                                    noise_prob=cfg.noise_prob,
                                    noise_snr=cfg.noise_snr_train,
                                    translation_base_dirs=cfg.translation_base_dirs
                                    )  
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
                          collate_fn=Multiple_language_collator())

    def val_dataloader_clean(self):
        dataset = LibriSpeechTextDataset('validation.clean',
                                        self.tokenizer, 
                                        SAMPLE_RATE,
                                        self.model_name,
                                        max_length=cfg.audio_max_length,
                                        spec_augment=False,
                                        noise_prob=0,
                                        translation_base_dirs=cfg.translation_base_dirs
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
                          collate_fn=Multiple_language_collator())
   
    def val_dataloader_other(self):
        dataset = LibriSpeechTextDataset('validation.other',
                                self.tokenizer, 
                                SAMPLE_RATE,
                                self.model_name,
                                max_length=cfg.audio_max_length,
                                spec_augment=False,
                                noise_prob=0,
                                translation_base_dirs=cfg.translation_base_dirs
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
                          collate_fn=Multiple_language_collator())
    
    def test_dataloader_clean(self):
        dataset = LibriSpeechTextDataset('test.clean',  
                                self.tokenizer, 
                                SAMPLE_RATE,
                                self.model_name,
                                max_length=cfg.audio_max_length,
                                spec_augment=False,
                                noise_prob=0,
                                translation_base_dirs=cfg.translation_base_dirs
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
                          collate_fn=Multiple_language_collator())
    
    def test_dataloader_other(self):
        dataset = LibriSpeechTextDataset('test.other', 
                                self.tokenizer, 
                                SAMPLE_RATE,
                                self.model_name,
                                max_length=cfg.audio_max_length,
                                spec_augment=False,
                                noise_prob=0,
                                translation_base_dirs=cfg.translation_base_dirs
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
                          collate_fn=Multiple_language_collator())

if __name__ == "__main__":
    cfg_yaml = sys.argv[1]
    with open(cfg_yaml, 'r') as file:
        dct = yaml.safe_load(file)
        cfg = types.SimpleNamespace(**dct)

    print(cfg)
    print("audio max length: {}".format(cfg.audio_max_length))

    # API Key
    wandb.login(key="643af3c5874c573aab4168b3d9d9a4c23fa49463")
    # Initialize WandB
    wandb.init(project="whisper-flamingo",
            config=cfg,
            # name="monolingual",
            # name="bilingual",
            # name="trilingual",
            # name="quadrilingual",
            # name="pentalingual",
            # name="monolingual_fra",
            # name="monolingual_ita",
            # name="trilingual_top3"
            name="bilingual_top2"
    )
    
    tflogger, callback_list = setup_logging_and_checkpoint_librispeech(cfg.log_output_dir, 
                                                                            cfg.check_output_dir, 
                                                                            cfg.train_name, 
                                                                            cfg.train_id,
                                                                            cfg.monitor,
                                                                            cfg.filename)
        
    model = WhisperTextModule(cfg, cfg.model_name, cfg.lang)
    model.to("cuda")  # 確保所有權重都在 GPU

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_params = count_parameters(model)
    print(f"Total Trainable Parameters: {num_params / 1e6:.2f}M")

    # 建立測試輸入
    dummy_input = torch.randn(1, 80, 3000).to("cuda")  # (Batch, Mel, Time Frames)
    dummy_tokens = torch.randint(0, 51865, (1, 30)).to("cuda")  # (Batch, Token Length)
    # 假設有 4 種額外語言，每個 xt 的形狀假設為 [1, 50, 768] (batch_size, seq_len, hidden_size)
    num_languages = 4
    dummy_xt_list = [torch.randn(1, 50, 768).to("cuda") for _ in range(num_languages)]
    
    flop_analyzer = FlopCountAnalysis(model, (dummy_input, dummy_tokens, dummy_xt_list))
    print(f"Total FLOPs: {flop_analyzer.total() / 1e9:.2f} GFLOPs")

    def measure_inference_time(model, dummy_input, dummy_tokens, dummy_xt_list,  num_trials=10):
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        # Warm-up
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input, dummy_tokens, dummy_xt_list)
                if device == "cuda":
                    torch.cuda.synchronize()
        
        # 開始計時 (使用 torch.cuda.Event 進行更精確計時)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for _ in range(num_trials):
                _ = model(dummy_input, dummy_tokens, dummy_xt_list)
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
    measure_inference_time(model, dummy_input, dummy_tokens, dummy_xt_list)
    
    # # Create a WandB logger instance
    # wandb_logger = WandbLogger()
    
    # strategy = DDPStrategy(find_unused_parameters=True) if cfg.num_devices > 1 else "auto"
    # trainer = Trainer(
    #     precision=cfg.precision,
    #     strategy=strategy,
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
    #     sync_batchnorm=True,
    # )

    # # TODO: save config file tp the checkpoint dir, also for pre-trained model
    # print(cfg)
    # resume_ckpt = f"{cfg.check_output_dir}/{cfg.train_id}/last.ckpt"
    # if os.path.exists(resume_ckpt) and cfg.resume_training: # resume training, don't validate
    #     trainer.fit(model, ckpt_path='last', val_dataloaders=[model.val_dataloader_clean(), model.val_dataloader_other(),
    #                                             model.test_dataloader_clean(), model.test_dataloader_other()])
    # else:
    #     # trainer.validate(model=model, dataloaders=[model.val_dataloader_clean(), model.val_dataloader_other(),
    #                                             # model.test_dataloader_clean(), model.test_dataloader_other()]) # validate before training
    #     trainer.fit(model, val_dataloaders=[model.val_dataloader_clean(), model.val_dataloader_other(),
    #                                         model.test_dataloader_clean(), model.test_dataloader_other()])

    # # End the WandB run
    # wandb.finish()