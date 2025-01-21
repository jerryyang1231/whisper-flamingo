#!/usr/bin/env python
# coding: utf-8

# my command
# python test.py config/test.yaml train

import os
import sys
import yaml
import types
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm
import whisper

from whisper.normalizers.basic import BasicTextNormalizer
from utils import wer_cer

# 指定 valid_set_list
valid_set_list = [
    '-d8TlAGYFmc', '3h8m__iwuJ4', '5mPJOkoIu3k', '87omMWX-DTw',
    'E0-HOPE7_QU', 'EhqcvfaaYu8', 'gDDbnFcvWcQ', 'iy1fPQQSA6c',
    'kGbjIuzvPR8', 'MrwSzSVGiRE', 'yht8d59dCpo'
]

class YTTDTaigiTRSDataset_TranscribeTest(Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        if split == 'train':
            ds = load_dataset("formospeech/yttd_taigi_trs", name='train', split='train')
            ds = ds.filter(lambda sample: sample['id'][:11] not in valid_set_list)
            print(f"Train set size: {len(ds)}")
        elif split == 'val':
            ds = load_dataset("formospeech/yttd_taigi_trs", name='train', split='train')
            ds = ds.filter(lambda sample: sample['id'][:11] in valid_set_list)
            print(f"Valid set size: {len(ds)}")
        else:  # 'test'
            ds = load_dataset("formospeech/yttd_taigi_trs", name='test', split='train')
            print(f"Test set size: {len(ds)}")

        self.dataset = ds
        self.sample_rate = 16000
        self.max_length = cfg.audio_max_length
        self.text_normalizer = BasicTextNormalizer(
            remove_diacritics=True, split_letters=False
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio_array = item['audio']['array']
        text = item['text']  # ground truth

        # 正規化 ground truth
        text = self.text_normalizer(text).replace(" ", "")

        # 若有需要，可裁切音訊
        if self.max_length is not None:
            audio_array = whisper.pad_or_trim(audio_array, length=self.max_length)

        return {
            "id": item["id"],
            "audio_array": audio_array.astype(np.float32),
            "ground_truth": text
        }

def main(cfg, split):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 載入模型 (teacher or student)
    print("Loading model ...")
    model = whisper.load_model(
        cfg.model_name,
        device="cpu",
        download_root="/share/nas169/jerryyang/whisper-flamingo/models",
        dropout_rate=cfg.dropout_rate,
        add_gated_x_attn=cfg.add_gated_x_attn,
        bert_encoder=cfg.bert_encoder,
        mode=cfg.mode,
    )

    # 如果指定了 checkpoint
    if hasattr(cfg, "teacher_ckpt") and cfg.teacher_ckpt != "":
        checkpoint_root = "/share/nas169/jerryyang/whisper-flamingo/models/checkpoints/"
        state_dict = torch.load(os.path.join(checkpoint_root, cfg.teacher_ckpt), map_location="cpu")
        # 這裡可能是 "state_dict" 裏頭的 key
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict_updated = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict_updated, strict=False)
        print(f"Loaded checkpoint from {cfg.teacher_ckpt}")

    model.to(device)
    model.eval()

    # 2) 讀 dataset
    dataset = YTTDTaigiTRSDataset_TranscribeTest(cfg, split)
    # 這邊簡單 for loop 處理
    print(f"Start inference on yttd_taigi_trs [{split}] ...")

    results = []
    text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)

    for i in tqdm(range(len(dataset)), desc=f"Transcribe({split})"):
        sample = dataset[i]
        audio_id = sample["id"]
        audio_array = sample["audio_array"] 
        ground_truth = sample["ground_truth"]
        
        # 3) 用 transcribe() 取得自由預測
        #   語言若是中文/台語，可設定 language="zh" ; 
        #   task="transcribe" => 不翻譯
        pred_result = model.transcribe(audio_array, language="zh", task="transcribe")
        pred_text = pred_result["text"]  # 模型輸出

        # 4) CER
        #   - 正規化 pred_text
        norm_pred = text_normalizer(pred_text).replace(" ", "")
        norm_ref = ground_truth

        _, cer = wer_cer(hypo=[norm_pred], ref=[norm_ref])

        results.append({
            "id": audio_id,
            "pred_text": norm_pred,
            "ref_text": norm_ref,
            "cer": cer
        })
    
    # 5) 存成 CSV
    df = pd.DataFrame(results)
    out_csv = f"transcribe_{split}_results.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Done! CSV saved: {out_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_transcribe.py config.yaml [train|val|test]")
        sys.exit(1)

    cfg_yaml = sys.argv[1]
    split = sys.argv[2]

    with open(cfg_yaml, 'r') as f:
        dct = yaml.safe_load(f)
        cfg = types.SimpleNamespace(**dct)

    main(cfg, split)
