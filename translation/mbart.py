import os
import logging
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import opencc
from concurrent.futures import ThreadPoolExecutor, as_completed

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 確認 GPU 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加載模型和標記器
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

converter = opencc.OpenCC('s2t')  # 簡體到繁體

def translate_batch(texts):
    try:
        # 設置源語言和目標語言
        tokenizer.src_lang = "en_XX"
        target_lang = "zh_CN"

        # 將多個文本編碼為批量輸入
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        # 批量生成翻譯文本
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang]
        )
        # 批量解碼
        simplified_translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return [converter.convert(text) for text in simplified_translations]
    except Exception as e:
        logging.error(f"Batch translation failed: {e}")
        return [""] * len(texts)

def process_file(trans_file_path, output_file_path, batch_size=8):
    try:
        with open(trans_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        with open(output_file_path, "w", encoding="utf-8") as f:
            text_batch = []
            id_batch = []
            for index, line in enumerate(lines):
                line = line.strip()
                if line:
                    try:
                        text_id, text = line.split(" ", 1)
                        text_batch.append(text)
                        id_batch.append(text_id)

                        # 當批量達到指定大小或是最後一行，開始翻譯
                        if len(text_batch) == batch_size or index == len(lines) - 1:
                            translated_texts = translate_batch(text_batch)
                            for text_id, translated_text in zip(id_batch, translated_texts):
                                f.write(f"{text_id} {translated_text}\n")
                                logging.info(f"Translated: {text_id} -> {translated_text}")

                            # 清空批量
                            text_batch = []
                            id_batch = []
                    except ValueError as ve:
                        logging.error(f"Skipping line due to parsing error: {line} - {ve}")
    except Exception as e:
        logging.error(f"Error processing file '{trans_file_path}': {e}")

def process_directory_multithreaded(root_dir, output_dir, max_workers=4, batch_size=8):
    os.makedirs(output_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".trans.txt"):
                    trans_file_path = os.path.join(subdir, file)
                    output_subdir = os.path.join(output_dir, os.path.relpath(subdir, root_dir))
                    os.makedirs(output_subdir, exist_ok=True)
                    output_file_path = os.path.join(output_subdir, file)

                    # 提交文件處理任務
                    futures.append(executor.submit(process_file, trans_file_path, output_file_path, batch_size))
        
        # 確保所有線程完成
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error in thread execution: {e}")

librispeech_path = "/share/nas169/jerryyang/corpus/LibriSpeech/LibriSpeech"  # 替換為實際路徑
output_path = "/share/nas169/jerryyang/corpus/mbart"  # 替換為實際輸出路徑

datasets = ["train-clean-100", "train-clean-360", "train-other-500"]
for dataset in datasets:
    process_directory_multithreaded(os.path.join(librispeech_path, dataset), os.path.join(output_path, dataset), max_workers=4, batch_size=4)
