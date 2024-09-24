import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import whisper
from torch import nn

# 加載 Whisper 模型
model_name = "tiny"
model = whisper.load_model(model_name)  # 你可以選擇 base, small, medium, large 等模型

tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
token_embedding = model.decoder.token_embedding

# Whisper 支援的語言列表
languages = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
    "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
    "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
    "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
    "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
    "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
    "ba", "jw", "su", "yue"
]

# 提取每個語言的嵌入向量
language_embeddings = []

for lang in languages:
    # 使用 encode 來獲取正確的 token
    language_token = tokenizer.encode(f"<|{lang}|>", allowed_special={f"<|{lang}|>"} )[0]
    
    # 將 token 放入 tensor
    tokens = torch.tensor([[language_token]])
    # 將 tokens 放到與 token_embedding 相同的設備上
    tokens = tokens.to(token_embedding.weight.device)

    # 使用模型的嵌入層來獲取嵌入向量
    with torch.no_grad():
        embedding = token_embedding(tokens)
    language_embeddings.append(embedding)

# 修改這段程式碼，使每個 embedding 維度從 (1, 1, 384) 壓縮為 (384,)
language_embeddings = [embedding.squeeze().cpu().numpy() for embedding in language_embeddings]

# 將語言嵌入轉換為 numpy array
language_embeddings_np = np.array(language_embeddings)

# 使用 t-SNE 將嵌入降維到 2D 空間
tsne = TSNE(n_components=2, random_state=3407)
language_vectors_2d = tsne.fit_transform(language_embeddings_np)

# 可視化
plt.figure(figsize=(12, 8))
plt.scatter(language_vectors_2d[:, 0], language_vectors_2d[:, 1], marker='o')

# 給每個點添加語言標籤
for i, lang in enumerate(languages):
    plt.text(language_vectors_2d[i, 0], language_vectors_2d[i, 1], lang)

plt.title('t-SNE Visualization of Whisper Language Embeddings')

# 保存結果為 PDF 檔案
plt.savefig(f"whisper_{model_name}_languages_tsne.pdf", format="pdf")

plt.show()
