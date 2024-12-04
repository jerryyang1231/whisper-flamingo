from whisper.tokenizer import get_tokenizer

# 加載 Whisper 的多語言 tokenizer
tokenizer = get_tokenizer(multilingual=True)

# 要查詢的特殊 token ID
special_token_ids = [50359, 50260, 50359, 50363]

# 解碼特殊 token
decoded_tokens = [tokenizer.decode([token_id]) for token_id in special_token_ids]
print(decoded_tokens)
