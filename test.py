# from transformers import MarianTokenizer, MarianMTModel

# # 使用正確的分詞器和模型類型
# tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
# model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

# # 定義輸入文本
# input_text = "The house is wonderful."

# # 將輸入文本編碼為模型可接受的格式
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# # 使用模型生成翻譯
# outputs = model.generate(input_ids)

# # 解碼生成的 token ID，得到翻譯文本
# translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(translated_text)

# from transformers import MarianTokenizer, MarianMTModel
# import torch

# # 加載分詞器和模型
# tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
# model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

# # 檢查是否有可用的 GPU，如果有，將模型轉移到 GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # 定義輸入文本
# input_text = "The house is wonderful."

# # 將輸入文本編碼並轉移到相同的設備
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids
# input_ids = input_ids.to(device)

# # 使用模型在 GPU 上生成翻譯
# outputs = model.generate(input_ids)

# # 將生成的 token ID 從 GPU 移回 CPU，並解碼得到翻譯文本
# translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(translated_text)

# from transformers import T5ForConditionalGeneration, T5Tokenizer

# # Set the device to 'cuda' if you are using GPU or 'cpu' if not
# device = 'cuda' #or 'cpu' for translate on cpu

# # Specify the model name
# model_name = 'utrobinmv/t5_translate_en_ru_zh_large_1024'
# model = T5ForConditionalGeneration.from_pretrained(model_name)
# model.to(device)  # Move model to the appropriate device
# tokenizer = T5Tokenizer.from_pretrained(model_name)

# # Set the translation direction with the appropriate prefix
# prefix = 'translate to zh: '  # Specifying Chinese as the target language
# src_text = prefix + "The quick brown fox jumps over the lazy dog."

# # Translate English to Chinese
# input_ids = tokenizer(src_text, return_tensors="pt", padding=True)

# # Generate the translation tokens
# generated_tokens = model.generate(**input_ids.to(device))

# # Decode the generated tokens to get the translation result
# result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# print(result)
# # This will output the translation of the provided English text into Chinese.

# from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# article_en = "An apple a day, keep doctor away."

# model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
# tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# # translate English to Chinese
# tokenizer.src_lang = "en_XX"
# encoded_en = tokenizer(article_en, return_tensors="pt")
# generated_tokens = model.generate(
#     **encoded_en,
#     forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
# )
# output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
# print("translated result :", output)

import whisper
import os

print(whisper.__file__)

