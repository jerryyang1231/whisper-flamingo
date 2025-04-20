import os
import sys
import yaml
import types
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import whisper
from utils import wer_cer
from whisper.normalizers.basic import BasicTextNormalizer
from transformers import BertModel, BertTokenizer
from utils_batch_samplers import SortedBatchSampler

# my command
# python generate_pseudo_labels_kloka_crawled.py config/generate_pseudo_labels_kloka_crawled.yaml train
# python generate_pseudo_labels_kloka_crawled.py config/generate_pseudo_labels_kloka_crawled.yaml eval

################################################################################
# 1. Dataset 定義 
################################################################################
HF_TOKEN = "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"

class KlokaCrawledDataset(Dataset):
    def __init__(self, cfg, split, config_names,
                ) -> None:
        super().__init__()

        # 根據 split 選擇不同的 dataset
        dataset_name = "formospeech/kloka_crawled_asr_train" if split == 'train' else "formospeech/kloka_crawled_asr_eval"
        translation_csv = cfg.translation_csv_train if split =='train' else cfg.translation_csv_eval
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

        self.sample_rate = 16000
        self.model_name = cfg.model_name
        self.max_length = cfg.audio_max_length
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language=cfg.lang, task='transcribe')
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
        lang = cfg.lang
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
        
        # language = item['language']
        # dialect = item['dialect']
        # prompt = "_".join([language, dialect])
        text = self.text_normalizer(text)
        chinese = self.text_normalizer(chinese)

        audio = wav_data.flatten().astype(np.float32)

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
        
        all_translations = []
        all_translations.append(chinese)
        all_translations.append(translation_text)

        return {
            "ids": item_id,
            "wav_lens": wav_lens,
            "all_translations": all_translations,
            "input_ids": mel,
            "dec_input_ids": dec_input_ids,
            "labels": labels,
        }

################################################################################
# 2. collate_fn: 用於 batch 推理
################################################################################

class Collator_kloka_crawled_pseudo:
    def __call__(self, features):
        # 收集各項
        ids, input_ids, labels, dec_input_ids, all_translations, wav_lens = [], [], [], [], [], []

        for f in features:
            ids.append(f["ids"])
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            all_translations.append(f["all_translations"])
            wav_lens.append(f["wav_lens"])

        # pad audio (input_ids) with 0
        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len = max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0) for audio, audio_len in zip(input_ids, audio_lengths)]

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)]

        # 整理 batch
        batch = {
            "ids": ids,
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "all_translations": all_translations,
            "wav_lens": wav_lens,
        }

        # 3) 轉成 torch.Tensor
        # 只將數值類型的項目轉 tensor
        for key in ["input_ids", "labels", "dec_input_ids", "wav_lens"]:
            batch[key] = torch.tensor(np.array(batch[key]), requires_grad=False)

        return batch

################################################################################
# 3. Offline 推理 (Teacher = Whisper Flamingo) + Batch decode
################################################################################

def generate_pseudo_labels(cfg, split):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Teacher Model ...")
    teacher = whisper.load_model(cfg.model_name,
                                device="cpu",  # 先載到 CPU
                                download_root='/share/nas169/jerryyang/whisper-flamingo/models',
                                dropout_rate = cfg.dropout_rate,
                                add_gated_x_attn = cfg.add_gated_x_attn,
                                num_langs = cfg.num_langs,
                                )

    # 如果有 teacher_ckpt
    if cfg.teacher_ckpt != '':
        checkpoint_root = '/share/nas169/jerryyang/whisper-flamingo/models/checkpoints/'
        state_dict = torch.load(os.path.join(checkpoint_root, cfg.teacher_ckpt), map_location='cpu')
        state_dict = state_dict['state_dict']
        state_dict_updated = {k[6:]: v for k, v in state_dict.items()}
        teacher.load_state_dict(state_dict_updated, strict=False)
        print("Teacher ckpt loaded:", cfg.teacher_ckpt)

    teacher.to(device)
    teacher.eval()

    # 載入 BERT (for xt_list)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert_model = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)
    bert_model.eval()

    # 準備 Dataset & DataLoader
    dataset = KlokaCrawledDataset(cfg,
                                split=split,
                                config_names=cfg.config_names,
                                )
    batch_sampler = SortedBatchSampler(
                    batch_size = cfg.batch_size,
                    shapes=[(item['wav_lens']) for item in dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=False
                    )
    loader = DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    num_workers=cfg.num_worker,
                    collate_fn=Collator_kloka_crawled_pseudo(),
                    )

    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language=cfg.lang, task='transcribe')
    special_token_set = set(tokenizer.special_tokens.values())
    text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)

    results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Generating pseudo-labels ({split})"):
            ids = batch["ids"]
            input_ids = batch["input_ids"].to(device)  
            labels = batch["labels"].long().to(device)
            dec_input_ids = batch["dec_input_ids"].long().to(device) 
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
                tokenized = bert_tokenizer(
                    batch_texts_for_this_translation,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=448
                ).to(device)

                outputs = bert_model(**tokenized)
                outputs_last_hidden_state = outputs.last_hidden_state  # shape = [batch_size, seq_len, hidden_size]

                all_xt.append(outputs_last_hidden_state)
            
            # 這樣 all_xt 就是一個 list，長度 = num_translations
            # all_xt[t] shape = [batch_size, seq_len, hidden_size]

            # 2) Teacher Forward
            teacher_feat = teacher.encoder(input_ids)  # [B, T', n_audio_state]
            teacher_out = teacher.decoder(dec_input_ids, teacher_feat, xt_list=all_xt)  # => shape [B, dec_len_max, vocab_size]

            labels[labels == -100] = tokenizer.eot

            # 3) decode for each sample in batch
            tokens = torch.argmax(teacher_out, dim=2)  # [B, dec_len_max]
            eot_find = (torch.where(tokens == tokenizer.eot, 1, 0))

            # 針對每個序列進行檢查
            for i in range(eot_find.shape[0]):
                if torch.any(eot_find[i] == 1): 
                    first_eot = torch.argmax(torch.arange(eot_find.shape[1], 0, -1).cuda() * eot_find[i], dim=0, keepdim=True)
                    tokens[i, torch.arange(eot_find.shape[1]).cuda() > first_eot] = tokenizer.eot

            for i, (o, l) in enumerate(zip(tokens, labels)):
                decoded_o = tokenizer.decode([t for t in o if t.item() not in special_token_set])
                decoded_l = tokenizer.decode([t for t in l if t.item() not in special_token_set])

                normalized_o = text_normalizer(decoded_o)
                normalized_l = text_normalizer(decoded_l)

                wer, _ = wer_cer(hypo=[normalized_o], ref=[normalized_l])

                results.append({
                    "id": ids[i],
                    "pseudo_text": normalized_o,
                    "ground_truth": normalized_l,
                    "wer": wer
                })

    # 寫入 CSV
    file_name = cfg.file_name
    csv_name = os.path.join(cfg.output_dir, f"{file_name}_{split}.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_name, index=False, encoding="utf-8")
    print(f"Done! Pseudo labels saved to: {csv_name}")

################################################################################
# 4. Main
################################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_pseudo_labels.py config.yaml [train|val|test]")
        sys.exit(1)

    cfg_yaml = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else 'train'

    with open(cfg_yaml, 'r') as file:
        dct = yaml.safe_load(file)
        cfg = types.SimpleNamespace(**dct)

    generate_pseudo_labels(cfg, split=split)
