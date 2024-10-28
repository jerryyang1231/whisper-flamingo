from transformers import BertTokenizer

# 加載 BERT 分詞器
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# 定義測試句子
sequence = "Titan"

# 分詞後的結果（不含特殊標記）
tokenized_sequence = tokenizer.tokenize(sequence)
print("Tokenized sequence (without special tokens):", tokenized_sequence)

# 編碼後的結果（自動添加了特殊標記）
inputs = tokenizer(sequence, return_tensors="pt")
print("inputs :", inputs)
encoded_sequence = inputs["input_ids"][0]  # 取得 tensor 的第一個 batch
print("Encoded sequence (with special tokens):", encoded_sequence.tolist())

# 將編碼結果解碼回句子
decoded_sequence = tokenizer.decode(encoded_sequence)
print("Decoded sequence:", decoded_sequence)

# 獲取 [CLS] 和 [SEP] 的 token ID
cls_token_id = tokenizer.cls_token_id
sep_token_id = tokenizer.sep_token_id

print(f"[CLS] token ID: {cls_token_id}")  # 預期輸出 101
print(f"[SEP] token ID: {sep_token_id}")  # 預期輸出 102
