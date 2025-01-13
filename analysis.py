import os
import sys
import yaml
import types
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import whisper
from utils import wer_cer
from whisper.normalizers.basic import BasicTextNormalizer
from transformers import BertModel, BertTokenizer
from utils_batch_samplers import SortedBatchSampler
from generate_pseudo_labels import YTTDTaigiTRSDataset_Pseudo, WhisperTextCollatorWhithPadding_taigi_pseudo

# my command
# python analysis.py config/generate_pseudo_labels.yaml train
# python analysis.py config/generate_pseudo_labels.yaml val
# python analysis.py config/generate_pseudo_labels.yaml test

################################################################################
# 1. Dataset 定義 (模仿你的 YTTDTaigiTRSDataset)，但只保留 Teacher 前處理邏輯
################################################################################

valid_set_list = [
    '-d8TlAGYFmc', '3h8m__iwuJ4', '5mPJOkoIu3k', '87omMWX-DTw',
    'E0-HOPE7_QU', 'EhqcvfaaYu8', 'gDDbnFcvWcQ', 'iy1fPQQSA6c',
    'kGbjIuzvPR8', 'MrwSzSVGiRE', 'yht8d59dCpo'
]

class YTTDTaigiTRSDataset_Pseudo(Dataset):
    def __init__(self, cfg, split,  
                ) -> None:
        super().__init__()

        # 1) 載入資料集
        if split == 'train':
            dataset = load_dataset("formospeech/yttd_taigi_trs", name='train', split='train')
            self.dataset = dataset.filter(lambda sample: sample['id'][:11] not in valid_set_list)
            print(f"train set size: {len(self.dataset)}")
        elif split == 'val':
            dataset = load_dataset("formospeech/yttd_taigi_trs", name='train', split='train')
            self.dataset = dataset.filter(lambda sample: sample['id'][:11] in valid_set_list)
            print(f"valid set size: {len(self.dataset)}")
        else:  # 'test'
            self.dataset = load_dataset("formospeech/yttd_taigi_trs", name='test', split='train')
            print(f"test set size: {len(self.dataset)}")

        self.sample_rate = 16000
        
        self.cfg = cfg
        self.model_name = cfg.model_name
        self.max_length = cfg.audio_max_length
        
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language='zh', task='transcribe')
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        lang = cfg.lang
        item = self.dataset[id]
        
        wav_data = item['audio']['array'] 
        text = item['text']
        mandarin_text = item['text_mandarin']
        wav_lens = len(wav_data)
        
        text = self.text_normalizer(text)
        text = text.replace(" ", "")

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

        mandarin_text = mandarin_text.replace(" ", "")
        mandarin_text = self.text_normalizer(mandarin_text)
        
        return {
            "ids": item["id"],
            "input_ids": mel,  
            "labels": labels,
            "dec_input_ids": dec_input_ids, 
            "translations": mandarin_text, 
            "wav_lens": wav_lens
        }

################################################################################
# 2. collate_fn: 用於 batch 推理
################################################################################

class WhisperTextCollatorWhithPadding_taigi_pseudo:
    def __call__(self, features):
        # 收集各項
        ids, input_ids, labels, dec_input_ids, translations, wav_lens = [], [], [], [], [], []

        for f in features:
            ids.append(f["ids"])
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            translations.append(f["translations"])
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
            "translations": translations,
            "wav_lens": wav_lens,
        }

        # 3) 轉成 torch.Tensor
        # 只將數值類型的項目轉 tensor
        for key in ["input_ids", "labels", "dec_input_ids", "wav_lens"]:
            batch[key] = torch.tensor(np.array(batch[key]), requires_grad=False)

        return batch

# 記錄每筆樣本的 CER 並生成圖表
def plot_cer_curve(cer_list, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(sorted(cer_list), label="CER Curve", marker="o", linestyle="-")
    plt.title("CER Curve of Dataset")
    plt.xlabel("Samples (sorted by CER)")
    plt.ylabel("CER")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"CER curve saved to {output_path}")

def generate_pseudo_labels(cfg, split):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Teacher Model ...")
    teacher = whisper.load_model(cfg.model_name,
                                device="cpu",  # 先載到 CPU
                                download_root='/share/nas169/jerryyang/whisper-flamingo/models',
                                dropout_rate = cfg.dropout_rate,
                                add_gated_x_attn = cfg.add_gated_x_attn,
                                bert_encoder = cfg.bert_encoder,
                                mode = cfg.mode,
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

    # 載入 BERT (for xt_1)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    bert_model = BertModel.from_pretrained('bert-base-chinese').to(device)
    bert_model.eval()
    
    dataset = YTTDTaigiTRSDataset_Pseudo(cfg, split=split)
    batch_sampler = SortedBatchSampler(
        batch_size=cfg.batch_size,
        shapes=[(item['wav_lens']) for item in dataset],
        sort_in_batch='descending',
        sort_batch='descending',
        drop_last=False
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_worker,
        collate_fn=WhisperTextCollatorWhithPadding_taigi_pseudo()
    )

    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language='zh', task='transcribe')
    special_token_set = set(tokenizer.special_tokens.values())
    text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)

    results = []
    cer_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Generating pseudo-labels ({split})"):
            ids = batch["ids"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].long().to(device)
            dec_input_ids = batch["dec_input_ids"].long().to(device)
            translations = batch["translations"] 

            # 1) BERT embedding (batch)
            bert_inputs = bert_tokenizer(
                translations,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            bert_outputs = bert_model(**bert_inputs)
            translation_embeddings = bert_outputs.last_hidden_state  # [B, seq_bert, hidden]

            # 2) Teacher Forward
            teacher_feat = teacher.encoder(input_ids)  # [B, T', n_audio_state]
            teacher_out = teacher.decoder(
                dec_input_ids, teacher_feat, xt_1=translation_embeddings
            )  # => shape [B, dec_len_max, vocab_size]
            
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
                decoded_o = tokenizer.decode([t.item() for t in o if t.item() not in special_token_set])
                decoded_l = tokenizer.decode([t.item() for t in l if t.item() not in special_token_set])

                normalized_o = text_normalizer(decoded_o).replace(" ", "")
                normalized_l = text_normalizer(decoded_l).replace(" ", "")

                _, cer = wer_cer(hypo=[normalized_o], ref=[normalized_l])
                cer_list.append(cer)

                results.append({
                    "id": ids[i],
                    "pseudo_text": normalized_o,
                    "ground_truth": normalized_l,
                    "cer": cer
                })

    # 保存結果到 CSV
    csv_name = f"pseudo_labels_{split}_analysis.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_name, index=False, encoding="utf-8")
    print(f"Done! Pseudo labels saved to: {csv_name}")

    # 繪製 CER 曲線
    plot_cer_curve(cer_list, f"cer_curve_{split}_analysis.png")

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
