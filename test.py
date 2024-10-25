import json

# 假設的句子
mandarin_text = "你 為什麼 要 自作主張"
print("華文 :", mandarin_text)

# 句子斷詞
mandarin_text_list = mandarin_text.split()
print("華文列表:", mandarin_text_list)

# 讀取您的 JSON 華台辭典
with open('mandarin2taibun.json', 'r', encoding='utf-8') as f:
    dictionary = json.load(f)

# 初始化一個空的列表來存放所有查詢結果
all_keywords = []

# 查找每個詞彙是否有華台翻譯，並將所有翻譯加入 all_keywords
for word in mandarin_text_list:
    if word in dictionary:
        print(f"詞語: {word}, 華台翻譯: {dictionary[word]}")
        # all_keywords.append(word)  # 先加入原始詞
        all_keywords.extend(dictionary[word])  # 再加入所有翻譯

all_keywords = "".join(all_keywords)
# 輸出結果，列出所有原詞與其翻譯
print("所有關鍵詞與翻譯合併:", all_keywords)
