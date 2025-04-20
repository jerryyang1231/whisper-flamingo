import os
import json
from datasets import load_dataset

# 設定 split，例如 'train' 或 'dev'
split = "train"

# 設定要篩選的語言（可以選擇多個）
selected_languages = {"eng"}

# 指定輸出資料夾（確保資料夾存在）
output_dir = "/share/nas169/jerryyang/corpus/ml-superb/eng"
os.makedirs(output_dir, exist_ok=True)

# 下載並載入數據集
dataset = load_dataset("espnet/ml_superb_hf", split=split)

# 過濾數據，只保留指定語言
filtered_dataset = dataset.filter(lambda example: example["language"] in selected_languages)
print("length of ds :", len(filtered_dataset))

# 定義只保留需要的欄位
def process_example(example):
    return {
        "id": example["id"],
        "language": example["language"],
        "text": example["text"].strip()
    }

# 轉換數據
filtered_data = [process_example(example) for example in filtered_dataset]

# **根據 selected_languages 和 split 來動態命名 JSON 檔案**
languages_str = "_".join(sorted(selected_languages))  # 例如 "cmn_eng"
json_filename = f"{languages_str}_{split}.json"

json_filepath = os.path.join(output_dir, json_filename)

with open(json_filepath , "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

print(f"✅ JSON 檔案已成功保存：{json_filepath }")
