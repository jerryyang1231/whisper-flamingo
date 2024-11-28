import whisper

# 加載 Whisper 模型
model = whisper.load_model("base")

# 獲取 tokenizer
tokenizer = whisper.tokenizer.get_tokenizer(is_multilingual=model.is_multilingual, language="en")

# 檢查是否有 vocab_size 屬性
vocab_size = tokenizer.tokenizer.vocab_size  # 注意這裡訪問內部 tokenizer 的 vocab_size
print(f"Vocabulary size: {vocab_size}")
