import os
import re

# 設定資料夾路徑
directories = ["train-clean-100", "train-clean-360", "train-other-500"]
base_path = "/share/nas169/jerryyang/corpus/m2m100"  # 請替換為實際路徑

# 輸出檔案
output_file = "long_texts.txt"
total_texts = 0
long_texts = 0

def clean_text(line):
    # 將line根據第一個空格進行分割，分成index和text兩部分
    parts = line.split(" ", 1)
    
    # 如果沒有分割出兩部分，說明該行沒有text部分，直接返回None
    if len(parts) != 2:
        return None
    
    index, text = parts

    # 使用正則表達式去除標點符號
    cleaned_line = re.sub(r'[^\w\s]', '', text)  # 移除所有非字母數字和非空白字符
    
    return index, cleaned_line

with open(output_file, "w", encoding="utf-8") as out_f:
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            result = clean_text(line.strip())
                            if result is None:
                                continue  # 跳過沒有 text 部分的行

                            index, cleaned_line = result
                            total_texts += 1
                            char_count = len(cleaned_line)
                            if char_count > 448:
                                long_texts += 1
                                out_f.write(f"{index}: {cleaned_line}\n")

# 計算比例
if total_texts > 0:
    proportion = (long_texts / total_texts) * 100
else:
    proportion = 0

print(f"總共處理了 {total_texts} 條文本，其中有 {long_texts} 條文本超過448個字元，佔比 {proportion:.2f}%")

