def validation_step(self, batch, batch_idx, dataloader_idx):
    input_ids = batch["input_ids"]
    labels = batch["labels"].long()
    dec_input_ids = batch["dec_input_ids"].long()
    grouped_keywords_list = batch["grouped_keywords"]  # List[List[List[str]]]

    keyword_embeddings_list = []
    for grouped_keywords in grouped_keywords_list:
        # grouped_keywords 是一個列表，包含多個詞組，每個詞組是台文詞彙的列表
        phrase_representations = []
        for keywords in grouped_keywords:
            # keywords 是一個台文詞彙的列表
            embeddings_list = []
            for keyword in keywords:
                # 使用 Whisper 的 tokenizer 對台文詞彙進行編碼
                token_ids = self.tokenizer.encode(keyword)
                token_ids_tensor = torch.tensor(token_ids).to(self.device)
                token_embeddings = self.model.decoder.token_embedding(token_ids_tensor)  # [token_len, n_state]
                # 對詞元嵌入取平均，得到該詞彙的表示
                word_representation = token_embeddings.mean(dim=0)  # [n_state]
                embeddings_list.append(word_representation)
            if embeddings_list:
                # 將詞組中所有詞彙的表示取平均，得到該詞組的表示
                phrase_representation = torch.stack(embeddings_list, dim=0).mean(dim=0)  # [n_state]
            else:
                # 如果沒有詞彙，使用零向量
                phrase_representation = torch.zeros(self.model.decoder.token_embedding.embedding_dim).to(self.device)
            phrase_representations.append(phrase_representation)
        # 將該樣本的所有詞組表示堆疊起來
        keyword_embeddings = torch.stack(phrase_representations, dim=0)  # [num_phrases, n_state]
        keyword_embeddings_list.append(keyword_embeddings)

    # 找出最大詞組數量，進行填充
    max_num_phrases = max([emb.shape[0] for emb in keyword_embeddings_list])
    padded_keyword_embeddings = []
    for emb in keyword_embeddings_list:
        num_phrases = emb.shape[0]
        if num_phrases < max_num_phrases:
            pad_size = max_num_phrases - num_phrases
            pad_emb = torch.zeros(pad_size, emb.shape[1]).to(self.device)
            emb = torch.cat([emb, pad_emb], dim=0)
        padded_keyword_embeddings.append(emb)
    # 最終關鍵詞表示張量，形狀為 [batch_size, max_num_phrases, n_state]
    keyword_representations = torch.stack(padded_keyword_embeddings, dim=0)

    # 傳遞給編碼器
    audio_features = self.model.encoder(input_ids, keyword_representations=keyword_representations)

    # ...（後續代碼保持不變）
