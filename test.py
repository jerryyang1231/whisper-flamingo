import whisper

# 載入 Whisper 的 tokenizer
tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

# 測試特殊 token ID
special_token_id = [50260, 50359, 50363]
decoded_token = tokenizer.decode(special_token_id)

# 輸出結果
print(f"Special token ID: {special_token_id}")
print(f"Decoded token: {decoded_token}")
