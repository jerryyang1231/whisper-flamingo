import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm

# 設定語言和資料分割
language = "en_us"  # 可以根據需求修改
splits = ['train', 'validation', 'test']  # 設定資料分割
durations_by_split = {split: [] for split in splits}

# 遍歷每個 split 並提取音檔時長，並檢查是否超過 30 秒
for split in splits:
    print(f"Processing {split}...")
    dataset = load_dataset('google/fleurs', language, split=split)
    
    # 提取時長並分類到對應組別
    for example in tqdm(dataset, desc=f"Processing {split}"):
        # 計算音檔時長
        duration = len(example['audio']['array']) / example['audio']['sampling_rate']
        durations_by_split[split].append(duration)
        
        # 檢查是否超過 30 秒，若超過則印出對應的 example
        if duration > 30:
            print(f"Audio file exceeding 30 seconds in {split}: ID = {example['id']}, Duration = {duration:.2f} seconds")

# 繪製每個 split 的音檔時長分布圖
for split, durations in durations_by_split.items():
    if durations:  # 確保有資料
        plt.figure(figsize=(10, 6))
        plt.hist(durations, bins=50, edgecolor='black')
        plt.title(f'Audio Length Distribution for {language} - {split}')
        plt.xlabel('Duration (sec)')
        plt.ylabel('Number of samples')
        plt.grid(True)
        plt.savefig(f'{language}_{split}_audio_length_distribution.pdf', dpi=800)
        plt.show()

print("所有分佈圖已保存為獨立的 PDF 文件")
