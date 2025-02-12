from datasets import load_dataset

# 下載資料集
dataset = load_dataset(
    "formospeech/kloka_crawled_asr_train",
    name="太魯閣",
    split="train",
    use_auth_token="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"
)

# 原始數據筆數
original_count = len(dataset)
print(f"原始數據集總筆數: {original_count}")

# 過濾掉 chinese 欄位為空的資料
filtered_dataset = dataset.filter(lambda example: example.get("chinese", "").strip() != "")

# 過濾後的數據筆數
filtered_count = len(filtered_dataset)
print(f"過濾後剩餘的數據筆數: {filtered_count}")
