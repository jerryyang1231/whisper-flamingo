# import matplotlib.pyplot as plt
# from datasets import load_dataset
# from tqdm import tqdm

# # 要加載的子資料集名稱列表
# subsets = ['train.clean.100', 'train.clean.360', 'train.other.500', 'validation.clean', 'validation.other', 'test.clean', 'test.other']

# # 初始化列表來存儲每個子資料夾的時長數據
# durations_by_group = {
#     'train.clean.100': [],
#     'train.clean.360': [],
#     'train.other.500': [],
#     'validation.clean': [],
#     'validation.other': [],
#     'test.clean': [],
#     'test.other': []
# }

# # 遍歷每個子資料集並提取音檔時長
# for subset in subsets:
#     print(f"Processing {subset}...")
#     dataset = load_dataset('librispeech_asr', split=subset)
    
#     # 提取時長並分類到對應組別
#     for example in tqdm(dataset, desc=f"Processing {subset}"):
#         # 計算音檔時長
#         duration = len(example['audio']['array']) / example['audio']['sampling_rate']
#         durations_by_group[subset].append(duration)

# # 繪製每個資料夾的音檔時長分布圖
# for group, durations in durations_by_group.items():
#     if durations:  # 確保有資料
#         plt.figure(figsize=(10, 6))
#         plt.hist(durations, bins=50, edgecolor='black')
#         plt.title(f'Audio Length Distribution for {group}')
#         plt.xlabel('Duration (sec)')
#         plt.ylabel('Number of samples')
#         plt.grid(True)
#         plt.savefig(f'{group}_audio_length_distribution.pdf', dpi=800)
#         plt.show()

from datasets import load_dataset
from tqdm import tqdm

# 參數設置
SAMPLE_RATE = 16000  # LibriSpeech 的採樣率
MAX_AUDIO_LENGTH_SECONDS = 30  # 音檔時長限制

# 載入 Hugging Face 上的 LibriSpeech 資料集
dataset = load_dataset("librispeech_asr", split="train.clean.360")

# 統計超過30秒音檔的數量
long_audio_count = 0

# 使用 tqdm 顯示進度條
for item in tqdm(dataset, desc="Processing audio files"):
    audio_length_seconds = len(item['audio']['array']) / SAMPLE_RATE
    if audio_length_seconds > MAX_AUDIO_LENGTH_SECONDS:
        long_audio_count += 1

print(f"超過 30 秒的音檔數量: {long_audio_count}")
