import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .resnet import ResEncoder
from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function

from transformers import BertModel, BertTokenizer
from wenet.transformer.ctc import CTC
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy,
                                reverse_pad_list)
from model.contextual_encoder import LSTMContextCoder
from wenet.transformer.attention import MultiHeadedAttention
from wenet.transformer.decoder_layer import DecoderLayer
from wenet.transformer.embedding import PositionalEncoding
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.utils.mask import (subsequent_mask, make_pad_mask)
from model.contextual_encoder import ContextualAttention, CrossAttention, SimpleAttention, DotAttention

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
        # self.subsampling_rate = 2

    # def forward(self, x: Tensor, x_lens: Tensor, track_norm=False, padding_mask=None):
    def forward(self, x: Tensor, track_norm=False, padding_mask=None):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        x_lens : torch.Tensor, shape = (batch_size,)
            the original lengths of each input sequence
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        # 計算 encoder_out_lens
        # encoder_out_lens = torch.ceil(x_lens / self.subsampling_rate).long()

        if track_norm:
            x_norm = torch.linalg.norm(x, dim=-1).mean()

        # Positional embedding
        if x.shape[1] > 1500:
            x = x[:, :1500, :]
        x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

        for layer, block in enumerate(self.blocks):
            x = block(x)
        x = self.ln_post(x)

        # if track_norm:
        #     return x, x_norm, encoder_out_lens
        # return x, encoder_out_lens
        return x

class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, dropout_rate: float,
        add_gated_x_attn: int, decoder_conf: dict,):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True, 
                                       add_gated_x_attn=add_gated_x_attn,
                                       )
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        self.dropout_rate = dropout_rate
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.contextual_att = decoder_conf["contextual_att"]
        self.conatt_type = decoder_conf["att_type"]
        self.add_copy_loss = decoder_conf["add_copy_loss"]
        self.cocoder_out_dim = decoder_conf["cocoder_out_dim"]
        
        if self.conatt_type == 'contextual':
            self.conatt = ContextualAttention(att_dim=n_state,        
                                        decoder_dim=n_state,
                                        cocoder_dim=self.cocoder_out_dim)
            self.output_layer = torch.nn.Linear(n_state+self.cocoder_out_dim, n_vocab)
        elif self.conatt_type == 'crossatt':
            self.conatt = CrossAttention(n_state, 4)
            self.output_layer = torch.nn.Linear(n_state+n_state, n_vocab)
        elif self.conatt_type == 'simpleatt':
            self.conatt = SimpleAttention(att_dim=n_state,        
                                        decoder_dim=n_state,
                                        cocoder_dim=self.cocoder_out_dim)
            self.output_layer = torch.nn.Linear(n_state+self.cocoder_out_dim, n_vocab)
        else:
            raise ValueError(f'No this att type: {conatt_type}!')

    def forward(self, x: Tensor, xa: Tensor, context: Tensor, need_att_mask: Tensor, kv_cache: Optional[dict] = None, 
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

        # Pass through the layers
        for layer, block in enumerate(self.blocks):
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
        
        # Apply layer normalization
        x = self.ln(x)
        
        # add attention module here, attention with vocabulary
        if self.contextual_att :
            context_repr, p, score = self.conatt(x, context, need_att_mask)
            x = torch.cat((x, context_repr), dim=-1)  # shape = (B, seq_len, n_state + context_dim)
            logits = self.output_layer(x)
        else:
            logits = (
                x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
            ).float()
        
        return logits  
        # if not self.add_copy_loss:
        #     return logits 
        # else:
        #     return logits, p

class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, dropout_rate: float, add_adapter: bool, adapter_dim: int, 
                add_gated_x_attn: int, tokenizer: object,
                ctc_weight: float, lsm_weight: float, length_normalized_loss: bool, add_context_att: bool,
                add_null_context: bool, add_copy_loss: bool, concoder_cofig: dict, decoder_conf:dict,
                ):
        super().__init__()

        self.tokenizer = tokenizer
        self.sot = self.tokenizer.sot
        self.eot = self.tokenizer.eot
        self.ctc_weight = ctc_weight
        self.lsm_weight = lsm_weight
        self.length_normalized_loss = length_normalized_loss
        self.add_context_att = add_context_att
        self.add_null_context = add_null_context
        self.add_copy_loss = add_copy_loss
        self.ignore_id = -100

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
            decoder_conf,
        )
        
        if self.add_context_att:
            self.concoder = LSTMContextCoder(self.dims.n_vocab, self.eot, add_null_context=add_null_context, **concoder_cofig)

        # self.ctc = CTC(self.dims.n_vocab, self.dims.n_audio_state)

        # self.criterion_att = LabelSmoothingLoss(
        #     size=self.dims.n_vocab,
        #     padding_idx=self.ignore_id,
        #     smoothing=self.lsm_weight,
        #     normalize_length=self.length_normalized_loss,
        # )

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)
    
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        dec_input_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        labels: torch.Tensor,
        context=None,
        need_att_mask=None,
        att_tgt=None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        
        # 1. Encoder       
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)

        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            if not self.add_copy_loss:
                loss_att, acc_att, decoder_out = self._calc_att_loss(encoder_out, 
                                                    dec_input_ids, text_lengths, labels, 
                                                    context, need_att_mask)
            else:
                # TODO
                loss_att, loss_copy, acc_att, decoder_out = self._calc_att_loss(encoder_out, 
                                                    dec_input_ids, text_lengths, labels, 
                                                    context, need_att_mask, att_tgt)
        else:
            loss_att = None
        text = dec_input_ids[:, 5:]
        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, 
                                text, text_lengths)
        else:
            loss_ctc = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 -
                                                 self.ctc_weight) * loss_att
        
        if not self.add_copy_loss:
            return loss, loss_att, loss_ctc, decoder_out
        else:
            loss = loss + 0.5 * loss_copy
            return loss, loss_att, loss_ctc, loss_copy, decoder_out
    
    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        dec_input_ids: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        labels: torch.Tensor,
        context=None,
        need_att_mask=None,
        att_tgt=None,
    ) -> Tuple[torch.Tensor, float]:

        ys_in_pad, ys_out_pad = dec_input_ids, labels
        # print("ys_in_pad.shape :", ys_in_pad.shape)
        # print("ys_out_pad.shape :", ys_out_pad.shape)
        # input("null")
        # ys_in_lens = ys_pad_lens + 4

        if not self.add_context_att:
            decoder_out = self.decoder(ys_in_pad, encoder_out,
                                    context, need_att_mask,
                                    )
        else:
            # print("ys_in_pad :", ys_in_pad)
            # print("ys_in_pad.shape :", ys_in_pad.shape)
            need_att_mask = ys_in_pad.ne(self.eot)
            # print("need_att_mask :", need_att_mask)
            # print("need_att_mask.shape :", need_att_mask.shape)
            # input("null")
            # need_att_mask[:, 0] = True
            if not self.add_copy_loss:
                decoder_out = self.decoder(ys_in_pad, encoder_out, 
                                            context, need_att_mask,
                                            )
            else:
                decoder_out, att_p = self.decoder(ys_in_pad, encoder_out, 
                                                context, need_att_mask,
                                                )
                # Compute copy attention loss
                # att_p: [batch_size, max_len, n], att_tgt: [batch_size, max_len]
                # print("att_p :", att_p)
                # print("att_p.shape :", att_p.shape)
                # print("need_att_mask :", need_att_mask)
                # print("need_att_mask.shape :", need_att_mask.shape, )
                # 更新 mask
                need_att_mask[:, :5] = False
                # print("new need_att_mask :", need_att_mask)
                # print("new need_att_mask.shape :", need_att_mask.shape, )
                # print("att_tgt :", att_tgt)
                # print("att_tgt.shape :", att_tgt.shape)
                # input("null")
    
                loss_copy = -torch.gather(att_p[need_att_mask], -1, att_tgt[need_att_mask].unsqueeze(-1)).log().sum() / need_att_mask.shape[0]
                # print("loss_copy :", loss_copy)
                # input("null")
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.dims.n_vocab,),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        if not self.add_copy_loss:
            return loss_att, acc_att, decoder_out
        else:
            return loss_att, loss_copy, acc_att, decoder_out
    
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