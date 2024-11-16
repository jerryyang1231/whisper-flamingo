import jieba
from pathlib import Path
import json
from datasets import load_dataset
from tqdm import tqdm

# 讀取 JSON 華台辭典
with open('mandarin2taibun.json', 'r', encoding='utf-8') as f:
    dictionary = json.load(f)

def get_keywords(mandarin_text, dictionary, separate=True):
    mandarin_text_list = mandarin_text.split()
    all_keywords = []

    for word in mandarin_text_list:
        if word in dictionary:
            all_keywords.extend(dictionary[word])
    
    return all_keywords if separate else "".join(all_keywords)

def update_jieba_dict_with_keywords(
        keywords: list,
        high_freq_words: list = [],
        high_freq_words_weight: int = 10
) -> None:
    # 去重並排序 keywords
    keywords = sorted(set(keywords))

    # 定義 Jieba 字典路徑
    jieba_dict_path = Path(jieba.__file__).parent / "dict.txt"
    jieba_dict_path.unlink(missing_ok=True)
    # Path("/tmp/jieba.cache").unlink(missing_ok=True)
    
    # 更改 Jieba 緩存文件的路徑
    user_cache_dir = Path.home() / ".cache" / "jieba"
    user_cache_dir.mkdir(parents=True, exist_ok=True)  # 確保目錄存在
    user_cache_path = user_cache_dir / "jieba.cache"
    
    # 將 keywords 寫入 Jieba 字典
    with jieba_dict_path.open("w") as file:
        for word in keywords:
            if word in high_freq_words:
                file.write(f"{word} {len(word) * high_freq_words_weight}\n")
            else:
                file.write(f"{word} {len(word)}\n")

    # 重置 Jieba 初始化狀態，讓其重新加載字典
    jieba.dt.initialized = False

def custom_cut(line: str) -> list:
    # 使用 Jieba 斷詞
    return list(jieba.cut(line, cut_all=False, HMM=False))

# 載入 yttd_taigi_trs 資料集
valid_set_list = ['-d8TlAGYFmc', '3h8m__iwuJ4', '5mPJOkoIu3k', '87omMWX-DTw', 
                'E0-HOPE7_QU', 'EhqcvfaaYu8', 'gDDbnFcvWcQ', 'iy1fPQQSA6c',
                'kGbjIuzvPR8', 'MrwSzSVGiRE', 'yht8d59dCpo']
# train set
# dataset = load_dataset("formospeech/yttd_taigi_trs", name='train', split='train')
# dataset = dataset.filter(lambda sample: sample['id'][:11] not in valid_set_list)
# print(f"train set size: {len(dataset)}")

# valid set
# dataset = load_dataset("formospeech/yttd_taigi_trs", name='train', split='train')
# dataset = dataset.filter(lambda sample: sample['id'][:11] in valid_set_list)
# print(f"valid set size: {len(dataset)}")

# test set
dataset = load_dataset("formospeech/yttd_taigi_trs", name='test', split='train')
print(f"test set size: {len(dataset)}")

# 用於記錄所有台文詞彙及缺少的詞彙
all_taiwanese_words = set()
missing_words = set()

# 使用 tqdm 來顯示進度條
# with open("train_texts.txt", "w", encoding="utf-8") as f:
# with open("valid_texts.txt", "w", encoding="utf-8") as f:
with open("test_texts.txt", "w", encoding="utf-8") as f:
    # for i, sample in enumerate(dataset):
        # if i >= 10:  # 只讀取前10個樣本
        #     break
    for sample in tqdm(dataset, desc="Processing samples"):
        
        # 從 mandarin_text 獲取 keywords
        keywords = get_keywords(sample['text_mandarin'], dictionary, separate=True)
        
        # 將 keywords 更新到 Jieba 字典
        update_jieba_dict_with_keywords(keywords)

        # 使用更新後的 Jieba 字典對 text 進行斷詞
        segmented_text = custom_cut(sample['text'].replace(' ', ''))
        
        # 累積本樣本的所有台文詞彙
        all_taiwanese_words.update(segmented_text)
                
        # 找出辭典中沒有的詞彙
        sample_missing_words = {text for text in segmented_text if text not in keywords}
        missing_words.update(sample_missing_words)
        
        # 將結果寫入文件
        f.write(f"ID: {sample['id']}\n")
        f.write(f"Text: {sample['text']}\n")
        f.write(f"Mandarin Text: {sample['text_mandarin']}\n")
        f.write(f"Keywords: {keywords}\n")
        f.write(f"Segmented Text: {' '.join(segmented_text)}\n")
        f.write(f"Sample Taiwanese Words: {set(segmented_text)}\n")
        # 如果 sample_missing_words 為空，寫出空白，不顯示 set()
        f.write(f"Sample Missing Words: {sample_missing_words if sample_missing_words else ''}\n")
        f.write("=" * 100 + "\n")

# 結果輸出
# print("所有台文詞彙:", all_taiwanese_words)
# print("辭典中缺少的台文詞彙:", missing_words)
print("所有台文詞彙數量:", len(all_taiwanese_words))
print("缺少台文詞彙的數量:", len(missing_words))
