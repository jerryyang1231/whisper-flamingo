# # Load model directly
# from transformers import BertModel, BertTokenizer

# # 初始化 BERT 分詞器和模型
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# model = BertModel.from_pretrained('bert-base-chinese')

# # 獲取中文 BERT 的詞嵌入矩陣
# source_embedding = model.embeddings.word_embeddings.weight.detach()  # [vocab_size, bert_hidden_size]
# value_embedding = model.embeddings.word_embeddings.weight.detach()  # [vocab_size, bert_hidden_size]

# print("model.embeddings :", model.embeddings)
# print("model.embeddings.word_embeddings :", model.embeddings.word_embeddings)
# print("model.embeddings.word_embeddings.weight :", model.embeddings.word_embeddings.weight)
# print("source_embedding :", source_embedding)

def training_step(self, batch, batch_id):
    # ...（其他代碼保持不變）

    mandarin_words_list = batch["mandarin_words"]  # List[List[str]]

    # 為每個樣本的中文詞彙列表獲取嵌入
    source_embeddings_list = []
    for mandarin_words in mandarin_words_list:
        if mandarin_words:
            # 使用 BERT 分詞器對中文詞彙進行編碼
            bert_inputs = self.bert_tokenizer(
                mandarin_words,
                add_special_tokens=False,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(self.device)
            # 通過 BERT 模型獲取嵌入
            with torch.no_grad():
                bert_outputs = self.bert_model(**bert_inputs)
            # 獲取每個詞的嵌入，取最後一層隱狀態
            embeddings = bert_outputs.last_hidden_state  # [num_words, seq_len, hidden_size]
            # 對 seq_len 維度取平均（因為每個詞可能被分成多個子詞）
            embeddings = embeddings.mean(dim=1)  # [num_words, hidden_size]
            source_embeddings_list.append(embeddings)
        else:
            # 如果沒有中文詞彙，添加一個零向量
            embeddings = torch.zeros(1, self.bert_model.config.hidden_size).to(self.device)
            source_embeddings_list.append(embeddings)

    # 對每個樣本的 source_embeddings 進行填充，確保形狀一致
    max_num_words = max([emb.shape[0] for emb in source_embeddings_list])
    padded_source_embeddings = []
    for emb in source_embeddings_list:
        num_words = emb.shape[0]
        if num_words < max_num_words:
            pad_size = max_num_words - num_words
            pad_emb = torch.zeros(pad_size, emb.shape[1]).to(self.device)
            emb = torch.cat([emb, pad_emb], dim=0)
        padded_source_embeddings.append(emb)
    # 將列表轉換為張量，形狀為 [batch_size, S, hidden_size]
    source_embedding = torch.stack(padded_source_embeddings, dim=0)  # [batch_size, S, hidden_size]
    value_embedding = source_embedding  # 在這種情況下，value_embedding 與 source_embedding 相同

    # ...（後續代碼保持不變）

    # 在前向傳播中，傳遞 xt_1_token_ids、source_embedding、value_embedding
    out = self.model.decoder(
        dec_input_ids,
        audio_features,
        xt_1_token_ids=xt_1_token_ids,
        xt_2=translation_embeddings,
        source_embedding=source_embedding,
        value_embedding=value_embedding
    )

    # ...（後續代碼保持不變）
