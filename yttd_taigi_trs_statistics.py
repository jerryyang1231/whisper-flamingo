import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm

# 加載 Hugging Face 的訓練集資料集
train_dataset = load_dataset('formospeech/yttd_taigi_trs', name='train', split='train')

# 加載 Hugging Face 的測試集資料集
test_dataset = load_dataset('formospeech/yttd_taigi_trs', name='test', split='train')

# 初始化列表來存儲時長數據
train_durations = []
test_durations = []

# 遍歷訓練集，提取每個音檔的時長
print("Processing training set durations...")
for example in tqdm(train_dataset, desc="Processing train set"):
    train_durations.append(example['duration'])

# 遍歷測試集，提取每個音檔的時長
print("Processing test set durations...")
for example in tqdm(test_dataset, desc="Processing test set"):
    test_durations.append(example['duration'])

# 繪製訓練集音檔時長分布圖
plt.figure(figsize=(10, 6))
plt.hist(train_durations, bins=50, edgecolor='black')
plt.title('Training Set Audio Duration Distribution')
plt.xlabel('Duration (seconds)')
plt.ylabel('Number of Files')
plt.grid(True)
plt.savefig('yttd_taigi_trs_train_duration_distribution.pdf', dpi=800)
plt.show()

# 繪製測試集音檔時長分布圖
plt.figure(figsize=(10, 6))
plt.hist(test_durations, bins=50, edgecolor='black')
plt.title('Test Set Audio Duration Distribution')
plt.xlabel('Duration (seconds)')
plt.ylabel('Number of Files')
plt.grid(True)
plt.savefig('yttd_taigi_trs_test_duration_distribution.pdf', dpi=800)
plt.show()
