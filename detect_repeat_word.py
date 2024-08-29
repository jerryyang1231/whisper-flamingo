import os
import string

def find_repeated_characters_in_line(line: str, threshold: int, ignore_set: set) -> dict:
    repeated_chars = {}
    
    for char in line:
        if char in ignore_set:
            continue  # 忽略指定的字元
        if char in repeated_chars:
            repeated_chars[char] += 1
        else:
            repeated_chars[char] = 1
    
    # 過濾出出現次數超過 threshold 的字元
    repeated_chars = {char: count for char, count in repeated_chars.items() if count > threshold}
    return repeated_chars

def process_file(file_path: str, threshold: int, ignore_set: set):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 建立輸出文件的路徑，並命名為 repeated_characters_<原始文件名>.txt
    output_file_path = f"{os.path.splitext(file_path)[0]}_repeated_characters.txt"
    
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in lines:
            if line.strip():  # 確保不是空行
                index, text = line.split(' ', 1)  # 假設index和text之間有一個空格分隔
                repeated_chars = find_repeated_characters_in_line(text.strip(), threshold, ignore_set)
                if repeated_chars:
                    output_file.write(f"Index: {index}, Text: {text.strip()}\n")
                    output_file.write(f"  Repeated characters: {repeated_chars}\n\n")
                    print(f"File: {file_path}")
                    print(f"Index: {index}, Text: {text.strip()}")
                    print(f"  Repeated characters: {repeated_chars}")

def process_all_files(root_dir: str, threshold: int, ignore_set: set):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name.endswith('trans.txt'):  # 只處理命名為 trans.txt 的文件
                file_path = os.path.join(root, file_name)
                process_file(file_path, threshold, ignore_set)

# 範例使用，假設你的根目錄為 'm2m100'
root_dir = '/share/nas169/jerryyang/corpus/m2m100'
threshold = 7

# 定義要忽略的字元集合，包括標點符號、空格和'的'
ignore_set = set(string.punctuation) | {' ', '的', '你', '妳', '我', '他', '她'}

process_all_files(root_dir, threshold, ignore_set)
