import os
import random
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as at
import numpy as np
import editdistance
from scipy.io import wavfile
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
from operator import itemgetter
from typing import Iterator, Optional
from torch.utils.data import Dataset, DistributedSampler
from torch.utils.data.sampler import Sampler
import json
from supar.utils.fn import pad
import re

def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform


def select_noise(noise_wavs):
    rand_indexes = np.random.randint(0, len(noise_wavs), size=1)
    noise_wav = []
    for x in rand_indexes:
        noise_wav.append(wavfile.read(noise_wavs[x])[1].astype(np.float32))
    return noise_wav[0]

def add_noise(clean_wav, noise_wavs, noise_snr=0):
    clean_wav = clean_wav.astype(np.float32)
    noise_wav = select_noise(noise_wavs)
    if type(noise_snr) == int or type(noise_snr) == float:
        snr = noise_snr
    elif type(noise_snr) == tuple:
        snr = np.random.randint(noise_snr[0], noise_snr[1]+1)
    clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
    if len(clean_wav) > len(noise_wav):
        ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
        noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
    if len(clean_wav) < len(noise_wav):
        start = 0
        noise_wav = noise_wav[start: start + len(clean_wav)]
    noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1))
    adjusted_noise_rms = clean_rms / (10**(snr/20))
    adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
    mixed = clean_wav + adjusted_noise_wav

    #Avoid clipping noise
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
        if mixed.max(axis=0) >= abs(mixed.min(axis=0)): 
            reduction_rate = max_int16 / mixed.max(axis=0)
        else :
            reduction_rate = min_int16 / mixed.min(axis=0)
        mixed = mixed * (reduction_rate)
    mixed = mixed.astype(np.int16)
    return mixed

class WhisperDataCollatorWhithPadding:
    def __call__(self, features):
        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len =  max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0) for audio, audio_len in zip(input_ids, audio_lengths)]

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length) # seems redundant

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}

        return batch

class WhisperDataCollatorWhithPadding_librispeech:
    def __call__(self, features):
        input_ids, labels, dec_input_ids, wav_lens, audio = [], [], [], [], []

        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            wav_lens.append(f["wav_lens"])
            audio.append(f["audio"])

        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len = max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0)
                    for audio, audio_len in zip(input_ids, audio_lengths)]

        # Pad audio (apply the same padding logic)
        audio_lengths = [a.shape[0] for a in audio]  # Assuming audio is a 1D array of raw waveform
        max_audio_len = max(audio_lengths)
        audio = [np.pad(a, (0, max_audio_len - a_len), 'constant', constant_values=0) 
                for a, a_len in zip(audio, audio_lengths)]
        
        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) 
                    for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) 
                        for e, e_len in zip(dec_input_ids, dec_input_ids_length)]  # 50257 is eot token id

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "wav_lens": wav_lens,  # Add wav_lens to the batch
            "audio": audio # Add the padded audio to the batch
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}

        return batch

class WhisperDataCollatorWhithPadding_taigi:
    def __call__(self, features):
        input_ids, labels, dec_input_ids, wav_lens = [], [], [], []

        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            wav_lens.append(f["wav_lens"])

        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len = max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0)
                    for audio, audio_len in zip(input_ids, audio_lengths)]
        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) 
                    for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) 
                        for e, e_len in zip(dec_input_ids, dec_input_ids_length)]  # 50257 is eot token id

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "wav_lens": wav_lens, 
        }

        for key in ["input_ids", "labels", "dec_input_ids", "wav_lens"]:
            batch[key] = torch.tensor(np.array(batch[key]), requires_grad=False)

        return batch

class keyword_prompt_collator:
    def __call__(self, features):
        input_ids, labels, dec_input_ids, wav_lens, prompt_lens = [], [], [], [], []

        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            wav_lens.append(f["wav_lens"])
            prompt_lens.append(f["prompt_lens"])

        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len = max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0)
                    for audio, audio_len in zip(input_ids, audio_lengths)]
        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) 
                    for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) 
                        for e, e_len in zip(dec_input_ids, dec_input_ids_length)]  # 50257 is eot token id

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "wav_lens": wav_lens,
            "prompt_lens": prompt_lens,
        }

        for key in ["input_ids", "labels", "dec_input_ids", "wav_lens", "prompt_lens"]:
            batch[key] = torch.tensor(np.array(batch[key]), requires_grad=False)

        return batch
    
class keyword_prompt_translation_collator:
    def __call__(self, features):
        input_ids, labels, dec_input_ids, wav_lens, prompt_lens, translations = [], [], [], [], [], []

        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            wav_lens.append(f["wav_lens"])
            prompt_lens.append(f["prompt_lens"])
            translations.append(f["translations"])

        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len = max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0)
                    for audio, audio_len in zip(input_ids, audio_lengths)]
        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) 
                    for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) 
                        for e, e_len in zip(dec_input_ids, dec_input_ids_length)]  # 50257 is eot token id

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "wav_lens": wav_lens,
            "prompt_lens": prompt_lens,
            "translations": translations,
        }

        for key in ["input_ids", "labels", "dec_input_ids", "wav_lens", "prompt_lens"]:
            batch[key] = torch.tensor(np.array(batch[key]), requires_grad=False)

        return batch
    
class copyne_collate_fn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        input_ids, labels, dec_input_ids, wav_lens, ne_set, raw_sentence_lst, raw_audio_length, raw_text_length = [], [], [], [], [], [], [], []

        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            wav_lens.append(f["wav_lens"])
            ne_set.append(f["ne_set"])
            raw_sentence_lst.append(f["sentence"])
            raw_audio_length.append(f["raw_audio_length"])
            raw_text_length.append(f["raw_text_length"])

        # padding 過的音檔長度
        audio_lengths = [audio.shape[1] for audio in input_ids]

        max_audio_len = max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0)
                    for audio, audio_len in zip(input_ids, audio_lengths)]

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) 
                    for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) 
                        for e, e_len in zip(dec_input_ids, dec_input_ids_length)]  # 50257 is eot token id
        
        batch_ne_set = set()
        for instance_ne_set in ne_set:
            if len(instance_ne_set) > 0:
                batch_ne_set.update(instance_ne_set)
    
        batch_ne_lst = sorted(list(batch_ne_set), key=lambda x: len(x), reverse=True)
        context_tensor = build_ne_vocab_tensor_with_tokenizer(batch_ne_lst, self.tokenizer)
        att_tgt = pad([build_copy_tgt(sent, batch_ne_lst) for sent in raw_sentence_lst], padding_value=-100)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "wav_lens": wav_lens, 
            "batch_ne_lst": batch_ne_lst,
            "context_tensor": context_tensor,
            "att_tgt": att_tgt,
            "raw_audio_length": raw_audio_length,
            "raw_text_length": raw_text_length,
        }

        for key in ["input_ids", "labels", "dec_input_ids", "wav_lens", "raw_audio_length", "raw_text_length"]:
            batch[key] = torch.tensor(np.array(batch[key]), requires_grad=False)

        return batch
    
class KG_WhisperDataCollatorWhithPadding_taigi:
    def __call__(self, features):
        input_ids, labels, dec_input_ids, wav_lens, keyword_tokens, = [], [], [], [], []

        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            wav_lens.append(f["wav_lens"])
            keyword_tokens.append(f["keyword_tokens"]) # List of lists: each keyword has its own token list

        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len = max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0)
                    for audio, audio_len in zip(input_ids, audio_lengths)]
        
        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) 
                    for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) 
                        for e, e_len in zip(dec_input_ids, dec_input_ids_length)]  # 50257 is eot token id

        # 處理 keyword_tokens
        # (a) 取得該 batch 中每個樣本最多的關鍵字數量
        max_keywords_per_sample = max(len(sample) for sample in keyword_tokens)
        # (b) 取得整個 batch 中所有關鍵字中最長 token 長度
        max_keyword_token_len = max(
            max(len(kt) for kt in sample) 
            for sample in keyword_tokens
        )
        padded_keyword_tokens = []
        for sample in keyword_tokens:
            padded_keywords = [
                np.pad(kt, (0, max_keyword_token_len - len(kt)), 
                       'constant', constant_values=50257) 
                for kt in sample
            ]
            # 若該 sample 的關鍵字數量少於 max_keywords_per_sample，進行填充
            while len(padded_keywords) < max_keywords_per_sample:
                padded_keywords.append(
                    np.full((max_keyword_token_len,), 50257)
                )
            padded_keyword_tokens.append(padded_keywords) # shape: [batch_size, max_keywords_per_sample, max_keyword_token_len]

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "wav_lens": wav_lens, 
            "keyword_tokens": padded_keyword_tokens,
        }

        # 只將數值類型的項目轉換為張量
        for key in ["input_ids", "labels", "dec_input_ids", "wav_lens", "keyword_tokens"]:
            batch[key] = torch.tensor(np.array(batch[key]), requires_grad=False)

        return batch

class KG_WhisperDataCollatorWhithPadding_taigi_translation:
    def __call__(self, features):
        input_ids, labels, dec_input_ids, wav_lens, keyword_tokens, translations = [], [], [], [], [], []

        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            wav_lens.append(f["wav_lens"])
            keyword_tokens.append(f["keyword_tokens"]) # List of lists: each keyword has its own token list
            translations.append(f["translations"])

        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len = max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0)
                    for audio, audio_len in zip(input_ids, audio_lengths)]
        
        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) 
                    for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) 
                        for e, e_len in zip(dec_input_ids, dec_input_ids_length)]  # 50257 is eot token id

        # 處理 keyword_tokens
        # (a) 取得該 batch 中每個樣本最多的關鍵字數量
        max_keywords_per_sample = max(len(sample) for sample in keyword_tokens)
        # (b) 取得整個 batch 中所有關鍵字中最長 token 長度
        max_keyword_token_len = max(
            max(len(kt) for kt in sample) 
            for sample in keyword_tokens
        )
        padded_keyword_tokens = []
        for sample in keyword_tokens:
            padded_keywords = [
                np.pad(kt, (0, max_keyword_token_len - len(kt)), 
                       'constant', constant_values=50257) 
                for kt in sample
            ]
            # 若該 sample 的關鍵字數量少於 max_keywords_per_sample，進行填充
            while len(padded_keywords) < max_keywords_per_sample:
                padded_keywords.append(
                    np.full((max_keyword_token_len,), 50257)
                )
            padded_keyword_tokens.append(padded_keywords) # shape: [batch_size, max_keywords_per_sample, max_keyword_token_len]

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "wav_lens": wav_lens, 
            "keyword_tokens": padded_keyword_tokens,
            "translations": translations,
        }

        # 只將數值類型的項目轉換為張量
        for key in ["input_ids", "labels", "dec_input_ids", "wav_lens", "keyword_tokens"]:
            batch[key] = torch.tensor(np.array(batch[key]), requires_grad=False)

        return batch
    
class AdaKWSDataCollatorWhithPadding:
    def __call__(self, features):
        input_ids, keyword_tokens, labels, wav_lens, all_keywords = [], [], [], [], []

        for f in features:
            input_ids.append(f["input_ids"])
            keyword_tokens.append(f["keyword_tokens"]) # List of lists: each keyword has its own token list
            labels.append(f["labels"])
            wav_lens.append(f["wav_lens"])
            all_keywords.append(f["all_keywords"])

        # 1. 音訊數據填充
        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len = max(audio_lengths)
        padded_input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 
                                   'constant', constant_values=0)
                            for audio, audio_len in zip(input_ids, audio_lengths)]
        
        # 2. 處理 keyword_tokens
        # (a) 取得該 batch 中每個樣本最多的關鍵字數量
        max_keywords_per_sample = max(len(sample) for sample in keyword_tokens)
        # (b) 取得整個 batch 中所有關鍵字中最長 token 長度
        max_keyword_token_len = max(
            max(len(kt) for kt in sample) 
            for sample in keyword_tokens
        )
        padded_keyword_tokens = []
        for sample in keyword_tokens:
            padded_keywords = [
                np.pad(kt, (0, max_keyword_token_len - len(kt)), 
                       'constant', constant_values=50257) 
                for kt in sample
            ]
            # 若該 sample 的關鍵字數量少於 max_keywords_per_sample，進行填充
            while len(padded_keywords) < max_keywords_per_sample:
                padded_keywords.append(
                    np.full((max_keyword_token_len,), 50257)
                )
            padded_keyword_tokens.append(padded_keywords) # shape: [batch_size, max_keywords_per_sample, max_keyword_token_len]

        # 3. 處理標籤
        padded_labels = [label + [False]*(max_keywords_per_sample - len(label)) for label in labels] # shape: [batch_size, max_keywords_per_sample]

        
        batch = {
            "input_ids": padded_input_ids,
            "keyword_tokens": padded_keyword_tokens,
            "labels": padded_labels,
            "wav_lens": wav_lens, 
            "all_keywords": all_keywords,
        }

        # 4. 只將數值類型的項目轉換為張量
        for key in ["input_ids", "keyword_tokens", "labels", "wav_lens"]:
            batch[key] = torch.tensor(np.array(batch[key]), requires_grad=False)

        return batch

class WhisperTextCollatorWhithPadding_taigi_without_bert:
    def __call__(self, features):
        input_ids, labels, dec_input_ids, translated_texts, wav_lens = [], [], [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            translated_texts.append(f["translated_text"])
            wav_lens.append(f["wav_lens"])

        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len = max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0) for audio, audio_len in zip(input_ids, audio_lengths)]

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)]

        # 為 translated_texts 添加 padding，如有需要
        translated_text_lengths = [len(t) for t in translated_texts]
        max_translated_text_len = max(translated_text_lengths)
        translated_texts = [np.pad(t, (0, max_translated_text_len - t_len), 'constant', constant_values=50257) 
                              for t, t_len in zip(translated_texts, translated_text_lengths)]
        
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "translated_text": translated_texts,
            "wav_lens": wav_lens  # Add wav_lens to the batch
        }
        
        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}

        return batch

class WhisperTextCollatorWhithPadding_taigi_with_bert:
    def __call__(self, features):
        input_ids, labels, dec_input_ids, translations, wav_lens = [], [], [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            translations.append(f["translations"])
            wav_lens.append(f["wav_lens"])

        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len = max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0) for audio, audio_len in zip(input_ids, audio_lengths)]

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)]

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "translations": translations,
            "wav_lens": wav_lens,
        }
        
        # 只將數值類型的項目轉換為張量
        for key in ["input_ids", "labels", "dec_input_ids", "wav_lens"]:
            batch[key] = torch.tensor(np.array(batch[key]), requires_grad=False)

        return batch

class WhisperTextCollatorWhithPadding_taigi_mix:
    def __call__(self, features):
        # input_ids, labels, dec_input_ids, translations, keywords, wav_lens = [], [], [], [], [], []
        input_ids, labels, dec_input_ids, keywords, wav_lens = [], [], [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            # translations.append(f["translation"])
            keywords.append(f["keywords"])
            wav_lens.append(f["wav_lens"])

        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len = max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0) for audio, audio_len in zip(input_ids, audio_lengths)]

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)
        
        keywords_lengths = [len(t) for t in keywords]

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) 
                  for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) 
                         for e, e_len in zip(dec_input_ids, dec_input_ids_length)]

        # 為 keywords 添加 padding，如有需要
        max_keyword_len = max(keywords_lengths)
        keywords = [np.pad(t, (0, max_keyword_len - t_len), 'constant', constant_values=50257) 
                    for t, t_len in zip(keywords, keywords_lengths)]
        
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "keywords": keywords,
            # "translations": translations,
            "wav_lens": wav_lens 
        }
        
        # 只將數值類型的項目轉換為張量
        for key in ["input_ids", "labels", "dec_input_ids", "keywords", "wav_lens"]:
            batch[key] = torch.tensor(np.array(batch[key]), requires_grad=False)

        return batch
    
# class WhisperTextCollatorWhithPadding_taigi_cls:
#     def __call__(self, features):
#         input_ids, labels, dec_input_ids, translations, grouped_keywords, wav_lens= [], [], [], [], [], []
#         mandarin_words = []  # 初始化 mandarin_words 為空列表
        
#         # 檢查 features 中是否包含 "mandarin_words" 欄位
#         has_mandarin_words = "mandarin_words" in features[0]
        
#         for f in features:
#             input_ids.append(f["input_ids"])
#             labels.append(f["labels"])
#             dec_input_ids.append(f["dec_input_ids"])
#             translations.append(f["translations"])
#             grouped_keywords.append(f["grouped_keywords"])
#             wav_lens.append(f["wav_lens"])
            
#             # 如果有 "mandarin_words"，則加入列表
#             if has_mandarin_words:
#                 mandarin_words.append(f["mandarin_words"])        
        
#         audio_lengths = [audio.shape[1] for audio in input_ids]
#         max_audio_len = max(audio_lengths)
#         input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0) for audio, audio_len in zip(input_ids, audio_lengths)]

#         label_lengths = [len(lab) for lab in labels]
#         dec_input_ids_length = [len(e) for e in dec_input_ids]
#         max_label_len = max(label_lengths + dec_input_ids_length)

#         # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
#         labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) 
#                   for lab, lab_len in zip(labels, label_lengths)]
#         dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) 
#                          for e, e_len in zip(dec_input_ids, dec_input_ids_length)]
        
#         batch = {
#             "input_ids": input_ids,
#             "labels": labels,
#             "dec_input_ids": dec_input_ids,
#             "grouped_keywords": grouped_keywords,
#             "translations": translations,
#             "wav_lens": wav_lens,  # Add wav_lens to the batch
#         }
        
#         # 只有在 features 包含 "mandarin_words" 時才加入 batch
#         if has_mandarin_words:
#             batch["mandarin_words"] = mandarin_words
        
#         # 只將數值類型的項目轉換為張量
#         for key in ["input_ids", "labels", "dec_input_ids", "wav_lens"]:
#             batch[key] = torch.tensor(np.array(batch[key]), requires_grad=False)

#         return batch

class WhisperTextCollatorWhithPadding_librispeech_without_bert:
    def __call__(self, features):
        input_ids, labels, dec_input_ids, translated_texts_1, translated_texts_2, wav_lens, audio = [], [], [], [], [], [], []
        
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            translated_texts_1.append(f["translated_text_1"])
            if f.get("translated_text_2") is not None:  # 檢查 translated_text_2 是否為 None
                translated_texts_2.append(f["translated_text_2"])
            wav_lens.append(f["wav_lens"])
            audio.append(f["audio"])
        
        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len = max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0) 
                     for audio, audio_len in zip(input_ids, audio_lengths)]
        
        # Pad audio (apply the same padding logic)
        audio_lengths = [a.shape[0] for a in audio]  # Assuming audio is a 1D array of raw waveform
        max_audio_len = max(audio_lengths)
        audio = [np.pad(a, (0, max_audio_len - a_len), 'constant', constant_values=0) 
                for a, a_len in zip(audio, audio_lengths)]

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) 
                  for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) 
                         for e, e_len in zip(dec_input_ids, dec_input_ids_length)]
        
        # 為 translated_texts_1 添加 padding，如有需要
        translated_text_lengths_1 = [len(t) for t in translated_texts_1]
        max_translated_text_len_1 = max(translated_text_lengths_1)
        translated_texts_1 = [np.pad(t, (0, max_translated_text_len_1 - t_len), 'constant', constant_values=50257) 
                              for t, t_len in zip(translated_texts_1, translated_text_lengths_1)]
        
        # 為 translated_texts_2 添加 padding，如有需要
        if translated_texts_2:  # 只有當 translated_text_2 存在時才進行處理
            translated_text_lengths_2 = [len(t) for t in translated_texts_2]
            max_translated_text_len_2 = max(translated_text_lengths_2)
            translated_texts_2 = [np.pad(t, (0, max_translated_text_len_2 - t_len), 'constant', constant_values=50257) 
                                  for t, t_len in zip(translated_texts_2, translated_text_lengths_2)]

       # 建立 batch，根據是否有 translated_text_2 決定是否包含它
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "translated_text_1": translated_texts_1,
            "wav_lens": wav_lens,  # Add wav_lens to the batch
            "audio": audio  # Add the padded audio to the batch
        }

        if translated_texts_2:  # 如果 translated_text_2 存在，將其添加到 batch
            batch["translated_text_2"] = translated_texts_2

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        
        return batch

class WhisperTextCollatorWhithPadding_librispeech_with_bert:
    def __call__(self, features):
        input_ids, labels, dec_input_ids, translation_1, translation_2, wav_lens, audio = [], [], [], [], [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            translation_1.append(f["translation_1"])
            if f.get("translation_2") is not None:  # 檢查 translation_2 是否為 None
                translation_2.append(f["translation_2"])
            wav_lens.append(f["wav_lens"])
            audio.append(f["audio"])

        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len = max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0) for audio, audio_len in zip(input_ids, audio_lengths)]
        
        # Pad audio (apply the same padding logic)
        audio_lengths = [a.shape[0] for a in audio]  # Assuming audio is a 1D array of raw waveform
        max_audio_len = max(audio_lengths)
        audio = [np.pad(a, (0, max_audio_len - a_len), 'constant', constant_values=0) 
                for a, a_len in zip(audio, audio_lengths)]

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)]
        
        # 建立 batch，根據是否有 translation_2 決定是否包含它
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "translation_1": translation_1,
            "wav_lens": wav_lens,  # Add wav_lens to the batch
            "audio": audio  # Add the padded audio to the batch
        }
        
        if translation_2:  # 如果 translation_2 存在，將其添加到 batch
            batch["translation_2"] = translation_2

        
        # 只將數值類型的項目轉換為張量
        for key in ["input_ids", "labels", "dec_input_ids", "wav_lens", "audio"]:
            batch[key] = torch.tensor(np.array(batch[key]), requires_grad=False)
        
        return batch

def create_padding_mask(T, padding_amounts):
    """
    Creates a padding mask for a batch of B x T tensors, given padding amounts.

    Args:
        padding_amounts: A list or tensor of integers, where each element
                         specifies the amount of padding for the corresponding
                         sequence in the batch.

    Returns:
        A PyTorch tensor of shape (B, T) containing 1s for padded elements and 0s
        for non-padded elements.
    """

    padded_lens = T - torch.tensor(padding_amounts, dtype=torch.long)[:, None]  # Add a dimension for broadcasting
    mask = padded_lens <= torch.arange(T, dtype=torch.long)[None, :]  # Add a dimension for broadcasting
    return mask

def whisper_optimizer(model, cfg, t_total):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                        lr=cfg.learning_rate,
                        eps=cfg.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps,
        num_training_steps=t_total
    )
    return optimizer, scheduler

def AdaKWS_optimizer(model, cfg, t_total):
    # 定義哪些參數不做 weight_decay
    no_decay = ["bias", "LayerNorm.weight"]

    # phi_params: text_encoder 的參數 (φ)
    phi_params = list(model.text_encoder.named_parameters())

    # theta_params: kw_module1, kw_module2, classifier 的參數 (θ)
    theta_params = list(model.kw_module1.named_parameters()) + \
                   list(model.kw_module2.named_parameters()) + \
                   list(model.classifier.named_parameters())

    # 對 phi_params 分 decay/no_decay
    phi_decay = [p for n, p in phi_params if not any(nd in n for nd in no_decay)]
    phi_no_decay = [p for n, p in phi_params if any(nd in n for nd in no_decay)]

    # 對 theta_params 分 decay/no_decay
    theta_decay = [p for n, p in theta_params if not any(nd in n for nd in no_decay)]
    theta_no_decay = [p for n, p in theta_params if any(nd in n for nd in no_decay)]

    # 建立 optimizer group
    # 這裡根據論文或需求，phi_params 用 cfg.lr_text，theta_params 用 cfg.lr_classifier
    optimizer_grouped_parameters = [
        {
            "params": theta_decay,
            "lr": cfg.lr_classifier,
            "weight_decay": cfg.weight_decay
        },
        {
            "params": theta_no_decay,
            "lr": cfg.lr_classifier,
            "weight_decay": 0.0
        },
        {
            "params": phi_decay,
            "lr": cfg.lr_text,
            "weight_decay": cfg.weight_decay
        },
        {
            "params": phi_no_decay,
            "lr": cfg.lr_text,
            "weight_decay": 0.0
        }
    ]

    # 建立 AdamW 優化器
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=cfg.lr_classifier,  # 主 LR 可視為 theta 的 default
                      eps=cfg.adam_epsilon)

    # 建立 linear scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=t_total
    )

    return optimizer, scheduler

def whisper_flamingo_optimizer(model, cfg, t_total):
    x_attn = ["gated_x_attn", "attn_gate", "ff", "keyword_cross_attn"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in x_attn )],
            "lr": cfg.learning_rate,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                        lr=cfg.learning_rate,
                        eps=cfg.adam_epsilon,
                        weight_decay=cfg.weight_decay)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps,
        num_training_steps=t_total
    )
    return optimizer, scheduler

def whisper_copyne_optimizer(model, cfg, t_total):
    x_attn = ["gated_x_attn", "attn_gate", "ff", "keyword_cross_attn"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in x_attn )],
            "lr": cfg.learning_rate,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                        lr=cfg.learning_rate,
                        eps=cfg.adam_epsilon,
                        weight_decay=cfg.weight_decay)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps,
        num_training_steps=t_total
    )
    return optimizer, scheduler

def setup_logging_and_checkpoint(log_output_dir, check_output_dir, train_name, train_id, monitor):
    Path(log_output_dir).mkdir(exist_ok=True)
    Path(check_output_dir).mkdir(exist_ok=True)

    tflogger = TensorBoardLogger(
        save_dir=log_output_dir,
        name=train_name,
        version=train_id
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{check_output_dir}/{train_id}",
        filename="step-{step:05d}-wer={val/wer:.4f}-acc={val/acc:.4f}",
        monitor=monitor,
        mode='max',
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    # monitor = monitor.replace('test', 'val') if 'test' in monitor else monitor.replace('val', 'test')
    val_checkpoint = ModelCheckpoint(
        dirpath=f"{check_output_dir}/{train_id}",
        # filename="step-{step:05d}-cer={dev-other/cer:.4f}-wer={dev-other/wer:.4f}",
        filename="step-{step:05d}-cer={val/cer_at:.4f}-wer={val/wer_at:.4f}",
        monitor=monitor,
        mode='min',
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    latest_checkpoint = ModelCheckpoint(
        dirpath=f"{check_output_dir}/{train_id}",
        filename="step-{step:05d}-wer={val/wer:.4f}-acc={val/acc:.4f}",
        monitor="step",
        mode='max',
        every_n_train_steps=5000,
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    # callback_list = [checkpoint_callback,
    #                  val_checkpoint,
    #                  latest_checkpoint, 
    #                  LearningRateMonitor(logging_interval="step")]
    callback_list = [val_checkpoint,
                     LearningRateMonitor(logging_interval="step")]
    return tflogger, checkpoint_callback, callback_list

def setup_logging_and_checkpoint_taigi(log_output_dir, check_output_dir, train_name, train_id, monitor, filename):
    Path(log_output_dir).mkdir(exist_ok=True)
    Path(check_output_dir).mkdir(exist_ok=True)

    tflogger = TensorBoardLogger(
        save_dir=log_output_dir,
        name=train_name,
        version=train_id
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{check_output_dir}/{train_id}",
        filename="step-{step:05d}-wer={val/wer:.4f}-acc={val/acc:.4f}",
        monitor=monitor,
        mode='max',
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    
    val_checkpoint = ModelCheckpoint(
        dirpath=f"{check_output_dir}/{train_id}",
        filename=filename,
        monitor=monitor,
        mode='min',
        save_top_k=3,
        auto_insert_metric_name=False,
    )

    callback_list = [val_checkpoint,
                     LearningRateMonitor(logging_interval="step")]
    return tflogger, checkpoint_callback, callback_list

def setup_checkpoint_kws(log_output_dir, check_output_dir, train_name, train_id, monitor, filename):
    Path(log_output_dir).mkdir(exist_ok=True)
    Path(check_output_dir).mkdir(exist_ok=True)
   
    val_checkpoint = ModelCheckpoint(
        dirpath=f"{check_output_dir}/{train_id}",
        filename=filename,
        monitor=monitor,
        mode='max',
        save_top_k=3,
        auto_insert_metric_name=False,
    )

    callback_list = [val_checkpoint,
                     LearningRateMonitor(logging_interval="step")]
    return callback_list

def setup_logging_and_checkpoint_librispeech(log_output_dir, check_output_dir, train_name, train_id, monitor, filename):
    Path(log_output_dir).mkdir(exist_ok=True)
    Path(check_output_dir).mkdir(exist_ok=True)

    tflogger = TensorBoardLogger(
        save_dir=log_output_dir,
        name=train_name,
        version=train_id
    )

    val_checkpoint = ModelCheckpoint(
        dirpath=f"{check_output_dir}/{train_id}",
        filename=filename,
        monitor=monitor,
        mode='min',
        save_top_k=3,
        auto_insert_metric_name=False,
    )

    callback_list = [val_checkpoint,
                     LearningRateMonitor(logging_interval="step")]
    return tflogger, callback_list

def wer_cer(hypo, ref):
    c_err, c_len, w_err, w_len = 0, 0, 0, 0
    for h, r in zip(hypo, ref):
        pred_words = h.split()
        pred_units = h.replace(' ', '|').replace('', ' ').split() # chars-space separated
        
        gt_words = r.split()
        gt_units = r.replace(' ', '|').replace('', ' ').split() # chars-space separated\
        c_err += editdistance.eval(pred_units, gt_units)
        c_len += len(gt_units)

        w_err += editdistance.eval(pred_words, gt_words)
        w_len += len(gt_words)
    return w_err/w_len, c_err/c_len

# https://github.com/mpc001/auto_avsr/blob/main/datamodule/samplers.py
class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()

        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.sampler.set_epoch(epoch)

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

def get_all_keywords(mandarin_text, dictionary):
    # 句子斷詞
    mandarin_text_list = mandarin_text.split()

    # 初始化一個空的列表來存放所有查詢結果
    all_keywords = []

    # 查找每個詞彙是否有華台翻譯，並將所有翻譯加入 all_keywords
    for word in mandarin_text_list:
        if word in dictionary:
            all_keywords.extend(dictionary[word])  # 再加入所有翻譯
   
    return all_keywords

def get_grouped_keywords(mandarin_text, dictionary, separate=False):
    # 句子斷詞
    mandarin_text_list = mandarin_text.split()
    # print("mandarin_text_list :", mandarin_text_list)

    # 初始化一個空的列表來存放所有查詢結果
    grouped_keywords = []    
    for word in mandarin_text_list:
        # print("word :", word)
        # 查詢華台辭典，獲取該中文詞對應的台文詞彙列表
        keywords = dictionary.get(word, [])
        # print("keywords :", keywords)
        if keywords:
            # 將台文詞彙列表作為一個子列表存入 grouped_keywords
            grouped_keywords.append(keywords)
        else:
            # 如果沒有對應的台文詞彙，則添加一個空列表
            grouped_keywords.append([])
        # print("grouped_keywords :", grouped_keywords)

    # 如果選擇分開顯示
    if separate:
        return grouped_keywords
    
    # 將所有子列表展開並合併為一個單一的列表
    flattened_keywords = [keyword for sublist in grouped_keywords for keyword in sublist]
    return flattened_keywords

def build_ne_vocab_tensor_with_tokenizer(ne_lst, tokenizer):
    """
    使用 Whisper.tokenizer 將命名實體列表轉換為張量
    Args:
        ne_lst (list[str]): 命名實體的字符串列表
        tokenizer (Tokenizer): Whisper 模型的 tokenizer

    Returns:
        torch.Tensor: 填充後的張量，形狀為 (num_ne, max_len)
    """
    # 1. 將命名實體列表轉換為 token 索引
    res = []
    for ne in ne_lst:
        token_ids = tokenizer.encode(ne)
        res.append(torch.tensor(token_ids, dtype=torch.long))
    # 2. 使用 pad 函數填充
    return pad(res, padding_value=50257)

def build_copy_tgt(sentence, ne_lst):
    res = [len(ne_lst)] * (len(sentence)+1)
    for idx, ne in enumerate(ne_lst):
        lst = [(item.span()[0], item.span()[1]-1) for item in re.finditer(ne, sentence)]
        for st, ed in lst:
            if res[st] != len(ne_lst):
                continue
            res[st] = idx
    res = [len(ne_lst)] * 5 + res
    return torch.tensor(res, dtype=torch.int64)
