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
        # x_1 = self.gated_x_attn_1(self.gated_x_attn_ln_1(x), xt_1)[0]
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

class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, 
              dropout_rate: float, add_adapter: bool, adapter_dim: int,
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

    def forward(self, x: Tensor, track_norm=False, padding_mask=None):
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
        x = self.ln_post(x)

        if track_norm:
            return x, x_norm
        return x

class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, dropout_rate: float, add_gated_x_attn: int,
        bert_encoder: bool, bert_hidden_size: int, mode: str, sequential_gated_x_attn: bool
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
        
        if bert_hidden_size != n_state:
            self.xt_projection = nn.Linear(bert_hidden_size, n_state)
        else:
            self.xt_projection = nn.Identity()  # 如果維度一致，則不需要投影 
        
        self.mode = mode

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None, 
                xt_1: Optional[Tensor] = None, xt_2: Optional[Tensor] = None,
        ):
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
            # xt_1 without BERT and without positional embedding
            if xt_1 is not None:
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
                xt_1 = self.token_embedding(xt_1)
                # No positional embedding for xt_1 in "keyword" mode
                xt_1 = xt_1.to(xa.dtype)
            
            # # xt_1 without BERT and without positional embedding
            # if xt_1 is not None:
            #     # No positional embedding for xt_1 in "keyword" mode
            #     xt_1 = xt_1.to(xa.dtype)
        elif self.mode == "bilingual":
            # xt_1 with BERT and positional embedding
            if xt_1 is not None:
                if xt_1.shape[-1] != x.shape[-1]:
                    xt_1 = self.xt_projection(xt_1)
                xt_1 = xt_1 + self.positional_embedding[offset: offset + xt_1.shape[1]]
                xt_1 = xt_1.to(xa.dtype)

            # xt_2 with BERT and positional embedding
            if xt_2 is not None:
                if xt_2.shape[-1] != x.shape[-1]:
                    xt_2 = self.xt_projection(xt_2)
                xt_2 = xt_2 + self.positional_embedding[offset: offset + xt_2.shape[1]]
                xt_2 = xt_2.to(xa.dtype)
        
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

class AdaKWS_TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=4, batch_first=True)
        self.d_model = 768
        self.fc_mu = nn.Linear(hidden_dim, self.d_model)
        self.fc_sigma = nn.Linear(hidden_dim, self.d_model)
    
    def forward(self, tokenized_keyword):
        # tokenized_keyword: [B, K, L]
        B, K, L = tokenized_keyword.shape

        # 1. 展開到 [B*K, L]
        flattened = tokenized_keyword.view(B*K, L)

        # 2. Embedding: [B*K, L, embed_dim]
        emb = self.embedding(flattened)

        # 3. LSTM: 一次性 forward
        _, (h, _) = self.lstm(emb)  # h: [num_layers, B*K, hidden_dim]
        h_final = h[-1]            # [B*K, hidden_dim]

        # 4. 分別產生 mu_v, sigma_v
        mu_v = self.fc_mu(h_final)      # [B*K, d_model]
        sigma_v = self.fc_sigma(h_final)

        # 5. reshape 回 [B, K, d_model]
        mu_v = mu_v.view(B, K, -1)
        sigma_v = sigma_v.view(B, K, -1)
        return mu_v, sigma_v

class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, z, mu_v, sigma_v):
        # z: [B,T,D]
        # mu_v, sigma_v: [B,1,D] (已針對單一關鍵字)
        mu_z = z.mean(dim=1, keepdim=True)  # [B,1,D]
        sigma_z = z.var(dim=1, keepdim=True, unbiased=False).sqrt() + self.eps  # [B,1,D]
        z_norm = (z - mu_z) / sigma_z
        out = sigma_v * z_norm + mu_v  # 單一關鍵字，不需平均
        return out

class KeywordAdaptiveModule(nn.Module):
    def __init__(self, d_model=768, n_heads=8, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.adain1 = AdaIN()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.adain2 = AdaIN()
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mu_v, sigma_v):
        # x: [B,T,D], mu_v,sigma_v: [B,1,D]
        x_norm = self.adain1(x, mu_v, sigma_v)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)
        x_norm = self.adain2(x, mu_v, sigma_v)
        ff_out = self.fc2(F.relu(self.fc1(x_norm)))
        x = x + self.dropout2(ff_out)
        return x

class AdaKWS(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()  
        
        # Text encoder (φ)
        self.text_encoder = AdaKWS_TextEncoder(vocab_size=vocab_size)

        # Keyword-Adaptive modules (two sequential blocks)
        self.kw_module1 = KeywordAdaptiveModule(d_model=self.text_encoder.d_model)
        self.kw_module2 = KeywordAdaptiveModule(d_model=self.text_encoder.d_model)

        # Classifier head
        # 對每個 keyword 輸出二分類結果，因此維度是 d_model -> 2
        self.classifier = nn.Linear(768, 2)
    
    def forward(self, audio_features, keyword_tokens):

        # 1. Whisper encoder
        # audio_features = self.whisper.encoder(audio)  # [B, T, D=768]
        B, T, D = audio_features.shape
        
        # 2. Text encoder
        # mu_v, sigma_v: [B,K,D]
        mu_v, sigma_v = self.text_encoder(keyword_tokens)
        
        # 3. 平行處理，把[ B, T, D ]展開成[ B, K, T, D ]，再reshape成[ B*K, T, D ]
        #   同理mu_v, sigma_v reshape成[ B*K, D ] 
        K = mu_v.size(1)  # keyword數量
        audio_features = audio_features.unsqueeze(1).expand(B, K, T, D) # [B,K,T,D]
        audio_features = audio_features.contiguous().view(B*K, T, D)    # [B*K, T, D]
       
        mu_v = mu_v.view(B*K, D)       # [B*K, D]
        sigma_v = sigma_v.view(B*K, D) # [B*K, D]

        # 4. 通過 KeywordAdaptiveModule (兩層)
        z = self.kw_module1(audio_features, mu_v.unsqueeze(1), sigma_v.unsqueeze(1))  # shape [B*K,T,D]
        z = self.kw_module2(z, mu_v.unsqueeze(1), sigma_v.unsqueeze(1))               # shape [B*K,T,D]
        
        # 5. Pooling
        # z -> [B*K,D]
        z_pooled, _ = torch.max(z, dim=1)
        
        # 6. Classifier
        # logits -> [B*K,2]
        logits_flat = self.classifier(z_pooled)

        # 7. reshape回[B,K,2]
        logits = logits_flat.view(B, K, -1)  # [B,K,2]
        return logits

class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, dropout_rate: float, add_adapter: bool, adapter_dim: int,
                 add_gated_x_attn: int, bert_encoder: bool, bert_dim: int, mode: str,
                 sequential_gated_x_attn: bool, adakws_checkpoint: str):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            dropout_rate,
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
            mode,
            sequential_gated_x_attn,
        )
        self.keyword_spotter = AdaKWS(
            self.dims.n_vocab,
        )

        # 加載 AdaKWS 預訓練權重
        if adakws_checkpoint is not None:
            print(f"Loading AdaKWS checkpoint from {adakws_checkpoint}")
            checkpoint = torch.load(adakws_checkpoint, map_location="cpu")
            if 'state_dict' in checkpoint:
                print("Loading weights with strict=False")
                self.keyword_spotter.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.keyword_spotter.load_state_dict(checkpoint)


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