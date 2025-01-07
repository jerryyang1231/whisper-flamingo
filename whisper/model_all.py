import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .resnet import ResEncoder
from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function

from transformers import BertModel, BertTokenizer
import math  # 添加必要的導入

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False, 
                 add_adapter: bool = False, adapter_dim: int = 256, add_gated_x_attn: int = 0,
                 sequential_gated_x_attn: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)
        
        self.add_gated_x_attn = add_gated_x_attn
        if self.add_gated_x_attn != 0:
            print("Adding gated x attn layers")
            self.gated_x_attn_1 = MultiHeadAttention(n_state, n_head)
            self.gated_x_attn_2 = MultiHeadAttention(n_state, n_head)

            self.gated_x_attn_ln_1 = LayerNorm(n_state)
            self.gated_x_attn_ln_2 = LayerNorm(n_state)

            self.attn_gate_1 = nn.Parameter(torch.tensor([0.]))
            self.attn_gate_2 = nn.Parameter(torch.tensor([0.]))

            self.ff_ln = LayerNorm(n_state)
            self.ff = nn.Sequential(
                Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
            )
            self.ff_gate = nn.Parameter(torch.tensor([0.]))  

        # 新增的參數，用於控制是否使用順序的門控交叉注意力
        self.sequential_gated_x_attn = sequential_gated_x_attn

    def apply_gated_x_attn(self, x, xt):
        x = x + self.gated_x_attn_1(self.gated_x_attn_ln_1(x), xt)[0] * self.attn_gate_1.tanh()
        x = x + self.ff(self.ff_ln(x)) * self.ff_gate.tanh()
        return x
    
    def apply_gated_x_attn_parallel(self, x, xt_1, xt_2):
        x_1 = self.gated_x_attn_1(self.gated_x_attn_ln_1(x), xt_1)[0] * self.attn_gate_1.tanh()
        x_2 = self.gated_x_attn_2(self.gated_x_attn_ln_2(x), xt_2)[0] * self.attn_gate_2.tanh()
        x = x + x_1 + x_2
        x = x + self.ff(self.ff_ln(x)) * self.ff_gate.tanh()
        return x
    
    def apply_gated_x_attn_sequential(self, x, xt_1, xt_2):
        x = x + self.gated_x_attn_1(self.gated_x_attn_ln_1(x), xt_1)[0] * self.attn_gate_1.tanh()
        x = x + self.gated_x_attn_2(self.gated_x_attn_ln_2(x), xt_2)[0] * self.attn_gate_2.tanh()
        x = x + self.ff(self.ff_ln(x)) * self.ff_gate.tanh()
        return x

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        xv: Optional[Tensor] = None,
        xt_1: Optional[Tensor] = None,
        xt_2: Optional[Tensor] = None
    ):
        if self.add_gated_x_attn != 0: 
            if xt_2 is not None:
                if self.sequential_gated_x_attn:
                    # 使用順序的門控交叉注意力
                    x = self.apply_gated_x_attn_sequential(x, xt_1, xt_2)
                else:
                    # 使用並行的門控交叉注意力
                    x = self.apply_gated_x_attn_parallel(x, xt_1, xt_2)
            else:
                x = self.apply_gated_x_attn(x, xt_1)
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))        
        return x

class ResNet1D(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ResNet1D, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            print(f"Add ResNet layers")
            self.layers.append(nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(input_dim)
            ))
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 形狀: (batch_size, seq_length, input_dim)
        x = x.permute(0, 2, 1)  # 轉換為 (batch_size, input_dim, seq_length)
        for layer in self.layers:
            identity = x
            out = layer(x)
            out += identity
            out = self.relu(out)
            x = out
        x = x.permute(0, 2, 1)  # 轉回 (batch_size, seq_length, input_dim)
        return x

class ReprogrammingLayer_m1(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer_m1, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_llm = d_llm or d_model  # 默認與 d_model 相同

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        # 投影查詢、鍵和值
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        # 計算注意力分數
        scale = 1.0 / math.sqrt(target_embedding.size(-1))
        scores = torch.einsum("blhd,shd->bhls", target_embedding, source_embedding) * scale

        # 計算注意力權重
        A = self.dropout(torch.softmax(scores, dim=-1))

        # 計算重新編程後的嵌入
        reprogrammed_embedding = torch.einsum("bhls,shd->blhd", A, value_embedding)
        reprogrammed_embedding = reprogrammed_embedding.reshape(B, L, -1)
        
        # 輸出
        out = self.out_projection(reprogrammed_embedding)
        return out
    
class ReprogrammingLayer_m2(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer_m2, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_llm = d_llm or d_model  # 默認與 d_model 相同

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        B_s, S, _ = source_embedding.shape  # B_s 應該等於 B
        H = self.n_heads

        # 投影查詢、鍵和值
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(B, S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(B, S, H, -1)

        # 計算注意力分數
        scale = 1.0 / math.sqrt(target_embedding.size(-1))
        scores = torch.einsum("blhd, bshd -> bhls", target_embedding, source_embedding) * scale

        # 計算注意力權重
        A = self.dropout(torch.softmax(scores, dim=-1))

        # 計算重新編程後的嵌入
        reprogrammed_embedding = torch.einsum("bhls, bshd -> blhd", A, value_embedding)
        reprogrammed_embedding = reprogrammed_embedding.reshape(B, L, -1)
        
        # 輸出
        out = self.out_projection(reprogrammed_embedding)
        return out

class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, 
              dropout_rate: float, video: bool, video_model_path: str, av_hubert_path: str,
              prob_av: float, prob_a: float, av_hubert_encoder: bool, av_fusion: str,
              add_adapter: bool, adapter_dim: int,
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, False,
                                    add_adapter, adapter_dim, add_gated_x_attn=0) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)
        self.dropout_rate = dropout_rate
        self.dropout = torch.nn.Dropout(dropout_rate)      
        
        # self.cross_attention_layers = [0, 1, 2, 3]  # 指定在哪些層加入交叉注意力
        # self.keyword_cross_attn = nn.ModuleList([
        #     MultiHeadAttention(n_state, n_head) for _ in self.cross_attention_layers
        # ])
        # self.keyword_cross_attn_ln = nn.ModuleList([
        #     LayerNorm(n_state) for _ in self.cross_attention_layers
        # ])

    def forward(self, x: Tensor, x_v=None, training=False, test_a=False, test_v=False, track_norm=False, 
                padding_mask=None):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """        
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        if track_norm:
            x_norm = torch.linalg.norm(x, dim=-1).mean()

        # NOTE: pos embedding has max length of 1500 (30s after conv downsample from 3000 mel frames)
        if x.shape[1] > 1500:
            x = x[ :, :1500, :]
    
        # NOTE: if max_len is 30s, then the cropping doesn't do anything.
        x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype) # trim pos embedding
        
        for layer, block in enumerate(self.blocks):
            x = block(x)
        
        # for idx, block in enumerate(self.blocks):
        #     # 印出目前的層索引
        #     # print(f"Passing through Encoder Block {idx}")
            
        #     x = block(x)
        #     if idx in self.cross_attention_layers and keyword_representations is not None:
        #         # 獲取對應的交叉注意力層
        #         layer_idx = self.cross_attention_layers.index(idx)
        #         cross_attn = self.keyword_cross_attn[layer_idx]
        #         cross_attn_ln = self.keyword_cross_attn_ln[layer_idx]
        #         # 印出交叉注意力層的資訊
        #         # print(f"Applying cross attention at Encoder Block {idx} (cross attention layer index {layer_idx})")
        #         # 準備關鍵詞表示
        #         kw = keyword_representations
        #         # 計算交叉注意力
        #         x = x + cross_attn(cross_attn_ln(x), kw)[0]
        #     # else:
        #     #     print(f"No cross attention applied at Encoder Block {idx}")

        x = self.ln_post(x)

        if track_norm:
            return x, x_norm
        return x

class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, dropout_rate: float, add_gated_x_attn: int,
        bert_encoder: bool, bert_hidden_size: int, add_resnet: bool, num_resnet_layer: int, mode: str, sequential_gated_x_attn: bool
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True, 
                                       add_gated_x_attn=add_gated_x_attn,
                                       sequential_gated_x_attn=sequential_gated_x_attn
                                       )
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        self.dropout_rate = dropout_rate
        self.dropout = torch.nn.Dropout(dropout_rate)     

        self.bert_encoder = bert_encoder
        if bert_encoder:
            # 初始化 BERT 模型
            self.bert_model = BertModel.from_pretrained('bert-base-chinese')
        
        if bert_hidden_size != n_state:
            self.xt_projection = nn.Linear(bert_hidden_size, n_state)
        else:
            self.xt_projection = nn.Identity()  # 如果維度一致，則不需要投影 
        
        self.add_resnet = add_resnet
        if add_resnet:
            self.resnet = ResNet1D(n_state, n_state, num_layers=num_resnet_layer)
        self.mode = mode
        
        if mode == "reprogram_m1":
            # 初始化 ReprogrammingLayer
            self.reprogramming_layer = ReprogrammingLayer_m1(
                d_model=n_state,  # 目標嵌入維度
                n_heads=n_head,   # 注意力頭數量
                d_llm=bert_hidden_size  # BERT 的嵌入維度
            )
        # elif mode == "reprogram_m2":
        elif mode in ["reprogram_m2", "keyword"]:
            # 初始化 ReprogrammingLayer
            self.reprogramming_layer = ReprogrammingLayer_m2(
                d_model=n_state,  # 目標嵌入維度
                n_heads=n_head,   # 注意力頭數量
                d_llm=bert_hidden_size  # BERT 的嵌入維度
            )

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None, 
                xt_1: Optional[Tensor] = None, xt_2: Optional[Tensor] = None,
                source_embedding: Optional[Tensor] = None, value_embedding: Optional[Tensor] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        # Process xt_1 and xt_2 based on the specified mode
        if self.mode == "mix":
            # old
            # # xt_1 without BERT and without positional embedding
            # if xt_1 is not None:
            #     xt_1 = self.token_embedding(xt_1)
            #     if self.add_resnet:
            #         xt_1 = self.resnet(xt_1)
            #     # No positional embedding for xt_1 in "mix" mode
            #     xt_1 = xt_1.to(xa.dtype)
            
            # new
            # xt_1 without BERT and without positional embedding
            if xt_1 is not None:
                if self.add_resnet:
                    xt_1 = self.resnet(xt_1)
                # No positional embedding for xt_1 in "mix" mode
                xt_1 = xt_1.to(xa.dtype)

            # xt_2 with BERT and positional embedding
            if xt_2 is not None:
                if xt_2.shape[-1] != x.shape[-1]:
                    xt_2 = self.xt_projection(xt_2)
                xt_2 = xt_2 + self.positional_embedding[offset: offset + xt_2.shape[1]]
                xt_2 = xt_2.to(xa.dtype)
        elif self.mode == "translation":
            # xt_1 with BERT and positional embedding
            if xt_1 is not None:
                if xt_1.shape[-1] != x.shape[-1]:
                    xt_1 = self.xt_projection(xt_1)
                xt_1 = xt_1 + self.positional_embedding[offset: offset + xt_1.shape[1]]
                xt_1 = xt_1.to(xa.dtype)   
        elif self.mode == "keyword":
            # xt_1 without BERT and without positional embedding
            if xt_1 is not None:
                if self.add_resnet:
                    xt_1 = self.resnet(xt_1)
                # No positional embedding for xt_1 in "keyword" mode
                xt_1 = xt_1.to(xa.dtype)
            
        #     # for reprogram
        #     if xt_1 is not None:
        #         # 獲取 target_embedding
        #         target_embedding = self.token_embedding(xt_1)  # [batch_size, seq_len, n_state]
                
        #         # 使用 ReprogrammingLayer
        #         reprogrammed_embedding = self.reprogramming_layer(
        #             target_embedding, source_embedding, value_embedding
        #         )  # [batch_size, seq_len, bert_hidden_size]
                
        #         # 通過 BERT 模型獲取輸出
        #         bert_outputs = self.bert_model(inputs_embeds=reprogrammed_embedding)
        #         xt_1 = bert_outputs.last_hidden_state  # [batch_size, seq_len, bert_hidden_size]
                
        #         # 投影到 n_state 維度
        #         if xt_1.shape[-1] != x.shape[-1]:
        #             xt_1 = self.xt_projection(xt_1)
        #         xt_1 = xt_1.to(xa.dtype)
        # elif self.mode == "cls":
        #     # xt_1 with BERT and without positional embedding
        #     if xt_1 is not None:
        #         if xt_1.shape[-1] != x.shape[-1]:
        #             xt_1 = self.xt_projection(xt_1)
        #         # No positional embedding for xt_1 in "cls" mode
        #         xt_1 = xt_1.to(xa.dtype)
                
        #     # xt_2 with BERT and positional embedding
        #     if xt_2 is not None:
        #         if xt_2.shape[-1] != x.shape[-1]:
        #             xt_2 = self.xt_projection(xt_2)
        #         xt_2 = xt_2 + self.positional_embedding[offset: offset + xt_2.shape[1]]
        #         xt_2 = xt_2.to(xa.dtype)
        # elif self.mode in ["reprogram_m1", "reprogram_m2"]:
        #     if xt_1 is not None:
        #         # 獲取 target_embedding
        #         target_embedding = self.token_embedding(xt_1)  # [batch_size, seq_len, n_state]
                
        #         # 使用 ReprogrammingLayer
        #         reprogrammed_embedding = self.reprogramming_layer(
        #             target_embedding, source_embedding, value_embedding
        #         )  # [batch_size, seq_len, bert_hidden_size]
                
        #         # 通過 BERT 模型獲取輸出
        #         bert_outputs = self.bert_model(inputs_embeds=reprogrammed_embedding)
        #         xt_1 = bert_outputs.last_hidden_state  # [batch_size, seq_len, bert_hidden_size]
                
        #         # 投影到 n_state 維度
        #         if xt_1.shape[-1] != x.shape[-1]:
        #             xt_1 = self.xt_projection(xt_1)
        #         xt_1 = xt_1.to(xa.dtype)
            
        #     # xt_2 with BERT and positional embedding
        #     if xt_2 is not None:
        #         if xt_2.shape[-1] != x.shape[-1]:
        #             xt_2 = self.xt_projection(xt_2)
        #         xt_2 = xt_2 + self.positional_embedding[offset: offset + xt_2.shape[1]]
        #         xt_2 = xt_2.to(xa.dtype)            
        
        # Pass through the layers
        for layer, block in enumerate(self.blocks):
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache, xt_1=xt_1, xt_2=xt_2)
        
        # Apply layer normalization
        x = self.ln(x)
        
        # Calculate logits
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        
        return logits

class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, dropout_rate: float, video: bool, 
                 video_model_path: str, av_hubert_path: str, prob_av: float, prob_a: float, av_hubert_encoder: bool,
                 av_fusion: str, add_adapter: bool, adapter_dim: int, add_gated_x_attn: int, 
                 bert_encoder: bool, bert_dim: int, add_resnet: bool, num_resnet_layer: int, mode: str, sequential_gated_x_attn: bool):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            dropout_rate,
            video,
            video_model_path,
            av_hubert_path,
            prob_av,
            prob_a,
            av_hubert_encoder,
            av_fusion,
            add_adapter,
            adapter_dim,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            dropout_rate,
            add_gated_x_attn,
            bert_encoder,
            bert_dim,
            add_resnet,
            num_resnet_layer,
            mode,
            sequential_gated_x_attn,
        )

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function