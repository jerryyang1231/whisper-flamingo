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

# my command
# python generate_pseudo_labels_librispeech_flamingo.py config/pseudo_labels/generate_pseudo_labels_librispeech_flamingo.yaml train.clean.100+train.clean.360+train.other.500
# python generate_pseudo_labels_librispeech_flamingo.py config/pseudo_labels/generate_pseudo_labels_librispeech_flamingo.yaml test.clean

################################################################################
# 1. Dataset 
################################################################################

class LibriSpeechTextDataset_Pseudo(Dataset):
    def __init__(self, cfg, hf_split, translation_base_dirs=None) -> None:
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
        self.sample_rate = 16000
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language='en', task='transcribe')
        self.model_name = cfg.model_name
        self.audio_max_length = cfg.audio_max_length

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
        for base_dir in self.translation_base_dirs:
            t = self.get_translation_text(file_id, base_dir)
            t = self.text_normalizer(t)  # 正規化
            all_translations.append(t)

        return {
            "ids": file_id,
            "wav_lens": wav_lens,
            "all_translations": all_translations,
            "input_ids": mel,
            "dec_input_ids": dec_input_ids,
            "labels": labels,
        }

################################################################################
# 2. collate_fn: 用於 batch 推理
################################################################################

class CollatorWhithPadding_librispeech_pseudo:
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
    dataset = LibriSpeechTextDataset_Pseudo(cfg,
                                            split,
                                            cfg.translation_base_dirs,
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
                    collate_fn=CollatorWhithPadding_librispeech_pseudo(),
                )

    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language='en', task='transcribe')
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
        print("Usage: python generate_pseudo_labels_librispeech.py config.yaml {split}")
        sys.exit(1)

    cfg_yaml = sys.argv[1]
    split = sys.argv[2]

    with open(cfg_yaml, 'r') as file:
        dct = yaml.safe_load(file)
        cfg = types.SimpleNamespace(**dct)

    generate_pseudo_labels(cfg, split=split)
