import os
import sys
import yaml
import types
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import whisper
from utils import wer_cer
from whisper.normalizers.basic import BasicTextNormalizer
from transformers import BertModel, BertTokenizer
from utils_batch_samplers import SortedBatchSampler
import json

# my command
# python generate_pseudo_labels_ml-superb.py config/generate_pseudo_labels_ml-superb.yaml train

################################################################################
# 1. Dataset 
################################################################################

class MLSuperbDataset(Dataset):
    def __init__(self, cfg, hf_split) -> None:
        super().__init__()
       
        translated_json_path = cfg.train_translated_json_path if hf_split =='train' else cfg.dev_translated_json_path
       
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

        self.sample_rate = 16000
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language=cfg.lang, task='transcribe')
        self.model_name = cfg.model_name
        self.audio_max_length = cfg.audio_max_length
        self.lang = cfg.lang
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)  

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        lang = self.lang
        item = self.dataset[id]
        item_id = item['id']
        wav_data = item['audio']['array']
        wav_lens = len(wav_data)

        text = item["text"][5:].strip()        
        translated_text = self.translated_texts.get(item["id"], None)

        text = self.text_normalizer(text)
        translated_text = self.text_normalizer(translated_text)

        audio = wav_data.flatten().astype(np.float32)

        # pad audio to cfg.audio_max_length (longer samples filtered out already)
        if self.audio_max_length is not None:
            audio = whisper.pad_or_trim(audio.flatten(), length=self.audio_max_length)

        n_mels = 80 if self.model_name != 'large-v3' else 128
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels) 

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
            "ids": item_id,
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "all_translations": all_translations,
            "wav_lens": wav_lens,
        }

################################################################################
# 2. collate_fn: 用於 batch 推理
################################################################################

class ml_superb_pseudo_collator:
    def __call__(self, features):
        ids, input_ids, labels, dec_input_ids, all_translations, wav_lens = [], [], [], [], [], []

        for f in features:
            ids.append(f["ids"])
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            all_translations.append(f["all_translations"])
            wav_lens.append(f["wav_lens"])

        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len = max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0) for audio, audio_len in zip(input_ids, audio_lengths)]

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)]

        # 建立 batch，根據是否有 translation_2 決定是否包含它
        batch = {
            "ids": ids,
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "all_translations": all_translations,
            "wav_lens": wav_lens,
        }

        # 只將數值類型的項目轉換為張量
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
    dataset = MLSuperbDataset(cfg,
                            split,
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
                    collate_fn=ml_superb_pseudo_collator(),
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
            
            teacher_feat = teacher.encoder(input_ids)
            teacher_out = teacher.decoder(dec_input_ids, teacher_feat, xt_list=all_xt)

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
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    csv_name = os.path.join(output_dir, f"pseudo_labels_{split}.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_name, index=False, encoding="utf-8")
    print(f"Done! Pseudo labels saved to: {csv_name}")

################################################################################
# 4. Main
################################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python generate_pseudo_labels_ml-superb.py config.yaml {split}")
        sys.exit(1)

    cfg_yaml = sys.argv[1]
    split = sys.argv[2]

    with open(cfg_yaml, 'r') as file:
        dct = yaml.safe_load(file)
        cfg = types.SimpleNamespace(**dct)

    generate_pseudo_labels(cfg, split=split)
