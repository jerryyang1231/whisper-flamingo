import os
import logging
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

# 設定日誌格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 確認 GPU 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# 模型設定：這裡使用 facebook/seamless-m4t-v2-large 模型，來源語言設定為中文 (ISO 639-3: "zho")
MODEL_NAME = "facebook/seamless-m4t-v2-large"
# 目標語言設定，預設印尼語 (ISO 639-3: "ind")，若要翻譯成馬來語可改成 "zsm"
TARGET_LANG = "ind"

# 加載翻譯模型與標記器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.src_lang = "cmn_Hant"  
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

def translate_batch(texts, tgt_lang=TARGET_LANG):
    """
    將一個文本列表從中文翻譯到目標語言 (預設印尼語)
    """
    try:
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        # 指定目標語言參數
        generated_tokens = model.generate(**encoded, tgt_lang=tgt_lang)
        translated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return translated_texts
    except Exception as e:
        logging.error(f"Batch translation failed: {e}")
        return [""] * len(texts)

def process_dataset(dataset, batch_size=8):
    """
    針對 Hugging Face 下載的資料集，批次翻譯「chinese」欄位內容。
    生成的結果為 list of dict，包含：id、chinese 與 translation。
    """
    results = []
    batch_texts = []
    batch_ids = []
    for i, item in enumerate(dataset):
        # 若資料中無中文欄位則略過
        item_id = item.get("id")
        chinese_text = item.get("chinese", "").strip()
        if not chinese_text:
            continue

        # 這邊 id 可依據實際情況調整（此處以 index 作為 id）
        batch_ids.append(f"{item_id}")
        batch_texts.append(chinese_text)

        if len(batch_texts) == batch_size:
            translations = translate_batch(batch_texts)
            for bid, orig, trans in zip(batch_ids, batch_texts, translations):
                results.append({"id": bid, "chinese": orig, "translation": trans})
            batch_texts = []
            batch_ids = []

    # 處理剩餘不足 batch_size 的部分
    if batch_texts:
        translations = translate_batch(batch_texts)
        for bid, orig, trans in zip(batch_ids, batch_texts, translations):
            results.append({"id": bid, "chinese": orig, "translation": trans})

    return results

if __name__ == "__main__":
    # 設定 Hugging Face 資料集與 config
    # 請根據你要下載的族語資料集做調整，下面以 "formospeech/kloka_crawled_asr_train" 為例
    split = "train" # or eval
    # split = "eval" # or train
    
    dataset_name = f"formospeech/kloka_crawled_asr_{split}"
    
    # config_name = "太魯閣"
    # config_name = "賽德克_德固達雅"
    config_name = "阿美_秀姑巒"

    HF_TOKEN = "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"  # 如有需要請填入有效權杖

    try:
        ds = load_dataset(dataset_name, name=config_name, split="train", use_auth_token=HF_TOKEN)
        logging.info(f"Loaded dataset '{dataset_name}' config '{config_name}' with {len(ds)} examples.")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        exit(1)

    # 過濾掉中文欄位為空的例子
    ds = ds.filter(lambda x: x.get("chinese", "").strip() != "")

    # 執行翻譯（可調 batch_size 參數以符合記憶體狀況）
    translation_results = process_dataset(ds, batch_size=8)
    logging.info(f"Processed {len(translation_results)} translations.")

    # 存成 CSV，包含欄位：id, chinese, translation
    output_csv = f"{config_name}_translated_{TARGET_LANG}_{split}.csv"
    df = pd.DataFrame(translation_results)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    logging.info(f"Saved translation results to {output_csv}")
