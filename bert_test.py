from transformers import BertModel, BertTokenizer

# 載入分詞器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# 示例使用
text = "這是一個示例文本。"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print("last_hidden_states' shape :", last_hidden_states.shape)