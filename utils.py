import os
import cv2
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

def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform

def load_video_feats(video_path, train=False, image_crop_size=88, 
               image_mean=0.421, image_std=0.165):
    feats = load_video_av_hubert(video_path)
    if train:
        transform = Compose([
            Normalize( 0.0,255.0 ),
            RandomCrop((image_crop_size, image_crop_size)),
            HorizontalFlip(0.5),
            Normalize(image_mean, image_std)])
    else:
        transform = Compose([
            Normalize( 0.0,255.0 ),
            CenterCrop((image_crop_size, image_crop_size)),
            Normalize(image_mean, image_std)])
    feats = transform(feats)
    feats = np.expand_dims(feats, axis=-1) # T, H, W, C
    return feats

def load_video_av_hubert(path):
    for i in range(3):
        try:
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(frame)
                else:
                    break
            frames = np.stack(frames)
            return frames
        except Exception:
            print(f"failed loading {path} ({i} / 3)")
            if i == 2:
                raise ValueError(f"Unable to load {path}")


class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)

class CenterCrop(object):
    """Crop the given image at the center
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw))/2.)
        delta_h = int(round((h - th))/2.)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames


class RandomCrop(object):
    """Crop the given image at the center
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w-tw)
        delta_h = random.randint(0, h-th)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class HorizontalFlip(object):
    """Flip image horizontally.
    """

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames

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

def load_data(AUDIO_MAX_LENGTH, TEXT_MAX_LENGTH, langs=['en', 'ar', 'de', 'el', 'es', 'fr', 'it', 'pt', 'ru'],
              muavic_root='/data/sls/scratch/roudi/datasets/muavic/', reduce_val=None, include_audio_lens=False,
              AUDIO_MAX_LENGTH_VAL=480000, translate=False):
    # reduce_val: If not None, keep this number of samples from the validation set
    audio_transcript_pair_list = {'train':[], 'valid':[], 'test':[]}
    for lang in langs:
        for split in audio_transcript_pair_list:
            if not translate or lang == 'en':
                tsv_fn = os.path.join(muavic_root, 'muavic', lang, '{}.tsv'.format(split))
                txt_fn = os.path.join(muavic_root, 'muavic', lang, '{}.{}'.format(split, lang))
            else: # translate, only support en -> X for now
                tsv_fn = os.path.join(muavic_root, 'muavic', 'en', lang, '{}.tsv'.format(split))
                txt_fn = os.path.join(muavic_root, 'muavic', 'en', lang, '{}.{}'.format(split, lang))
            with open(tsv_fn) as tsv:
                with open(txt_fn) as txt:
                    audio_lns = tsv.readlines()[1:]
                    txt_lns = txt.readlines()
                    # audio path, audio length, text, text length, video_length
                    wav_fns = [(audio.strip().split('\t')[2],  int(audio.strip().split('\t')[-1]), txt.strip(), 
                                len(txt.strip()), int(audio.strip().split('\t')[-2])) for audio, txt in zip(audio_lns, txt_lns)]
                    pre_video_check = len(wav_fns)
                    wav_fns =  list(filter(lambda x: x[4] > 0, wav_fns))
                    post_video_check = len(wav_fns)
                    print("Removed {} samples with missing video (before filtering lengths)".format(pre_video_check - post_video_check))
                    len_before = len(wav_fns)
                    if split == 'train': 
                        wav_fns =  list(filter(lambda x: x[1] <= AUDIO_MAX_LENGTH, wav_fns))
                        wav_fns =  list(filter(lambda x: x[3] <= TEXT_MAX_LENGTH, wav_fns))
                    elif split == 'valid': # whisper pos. embedding only up to 30s long, don't filter test
                        wav_fns =  list(filter(lambda x: x[1] <= AUDIO_MAX_LENGTH_VAL, wav_fns))
                    print("Total hours {} : {}".format(split, sum([int(x[1]) for x in wav_fns]) / 16000 / 3600))
                    if not include_audio_lens:
                        lang_filtered = [(lang, i[0], i[2]) for i in wav_fns]
                    else: 
                        lang_filtered = [(lang, i[0], i[2], i[1]) for i in wav_fns]
                    if split == 'valid' and reduce_val is not None:
                        lang_filtered = lang_filtered[:reduce_val]
                    len_after = len(lang_filtered)
                    audio_transcript_pair_list[split] += lang_filtered
            print(lang, split, len_before, len_after)
    print("Total data lengths")
    print(len(audio_transcript_pair_list['train']))
    print(len(audio_transcript_pair_list['valid']))
    print(len(audio_transcript_pair_list['test']))
    return audio_transcript_pair_list

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
        # input_ids, labels, dec_input_ids, wav_lens, prompt_lens, translations = [], [], [], [], [], []
        # input_ids, labels, dec_input_ids, wav_lens, prompt_lens, = [], [], [], [], []
        input_ids, labels, dec_input_ids, wav_lens, keyword_tokens, = [], [], [], [], []

        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            wav_lens.append(f["wav_lens"])
            keyword_tokens.append(f["keyword_tokens"]) # List of lists: each keyword has its own token list
            # prompt_lens.append(f["prompt_lens"])
            # translations.append(f["translations"])

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
            # "prompt_lens": prompt_lens,
            # "translations": translations,
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

class WhisperDataCollatorWhithPadding_fleurs:
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
            "wav_lens": wav_lens,  # Add wav_lens to the batch
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}

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
            "wav_lens": wav_lens  # Add wav_lens to the batch
        }
        
        # 只將數值類型的項目轉換為張量
        for key in ["input_ids", "labels", "dec_input_ids", "wav_lens"]:
            batch[key] = torch.tensor(np.array(batch[key]), requires_grad=False)

        return batch

class WhisperTextCollatorWhithPadding_taigi_mix:
    def __call__(self, features):
        input_ids, labels, dec_input_ids, translations, keywords, wav_lens = [], [], [], [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            translations.append(f["translation"])
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
            "translations": translations,
            "wav_lens": wav_lens  # Add wav_lens to the batch
        }
        
        # 只將數值類型的項目轉換為張量
        for key in ["input_ids", "labels", "dec_input_ids", "keywords", "wav_lens"]:
            batch[key] = torch.tensor(np.array(batch[key]), requires_grad=False)

        return batch
    
class WhisperTextCollatorWhithPadding_taigi_cls:
    def __call__(self, features):
        input_ids, labels, dec_input_ids, translations, grouped_keywords, wav_lens= [], [], [], [], [], []
        mandarin_words = []  # 初始化 mandarin_words 為空列表
        
        # 檢查 features 中是否包含 "mandarin_words" 欄位
        has_mandarin_words = "mandarin_words" in features[0]
        
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            translations.append(f["translations"])
            grouped_keywords.append(f["grouped_keywords"])
            wav_lens.append(f["wav_lens"])
            
            # 如果有 "mandarin_words"，則加入列表
            if has_mandarin_words:
                mandarin_words.append(f["mandarin_words"])        
        
        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len = max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0) for audio, audio_len in zip(input_ids, audio_lengths)]

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) 
                  for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) 
                         for e, e_len in zip(dec_input_ids, dec_input_ids_length)]
        
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "grouped_keywords": grouped_keywords,
            "translations": translations,
            "wav_lens": wav_lens,  # Add wav_lens to the batch
        }
        
        # 只有在 features 包含 "mandarin_words" 時才加入 batch
        if has_mandarin_words:
            batch["mandarin_words"] = mandarin_words
        
        # 只將數值類型的項目轉換為張量
        for key in ["input_ids", "labels", "dec_input_ids", "wav_lens"]:
            batch[key] = torch.tensor(np.array(batch[key]), requires_grad=False)

        return batch

class WhisperTextCollatorWhithPadding_taigi_biasing:
    def __call__(self, features):
        input_ids, biasing_list, wav_lens, targets =[], [], [], []
        
        for f in features:
            input_ids.append(f["input_ids"])
            biasing_list.append(f["biasing_list"])
            wav_lens.append(f["wav_lens"])
            targets.append(f["targets"])         
        
        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len = max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant',
                            constant_values=0) for audio, audio_len in zip(input_ids, audio_lengths)]

        target_lengths = [len(t) for t in targets]
        max_target_len = max(target_lengths)

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot 
        targets = [np.pad(t, (0, max_target_len - t_len), 'constant', constant_values=-100) 
           for t, t_len in zip(targets, target_lengths)]
   

        batch = {
            "input_ids": input_ids,
            "biasing_list": biasing_list,
            "wav_lens": wav_lens,
            "targets": targets,
        }
        
        # 只將數值類型的項目轉換為張量
        for key in ["input_ids", "wav_lens", "targets"]:
            batch[key] = torch.tensor(np.array(batch[key]), requires_grad=False)

        return batch
    
class WhisperDataCollatorWhithPadding_taigi_no_biasing:
    def __call__(self, features):
        input_ids, targets, wav_lens,  =[], [], [], 
        
        for f in features:
            input_ids.append(f["input_ids"])
            wav_lens.append(f["wav_lens"])
            targets.append(f["targets"])         
        
        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len = max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant',
                            constant_values=0) for audio, audio_len in zip(input_ids, audio_lengths)]

        target_lengths = [len(t) for t in targets]
        max_target_len = max(target_lengths)

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot     
        targets = [np.pad(t, (0, max_target_len - t_len), 'constant', constant_values=-100) 
           for t, t_len in zip(targets, target_lengths)]
   

        batch = {
            "input_ids": input_ids,
            "wav_lens": wav_lens,
            "targets": targets,
        }
        
        # 只將數值類型的項目轉換為張量
        for key in ["input_ids", "wav_lens", "targets"]:
            batch[key] = torch.tensor(np.array(batch[key]), requires_grad=False)

        return batch

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

class WhisperVideoCollatorWithPadding:
    def __call__(self, features):
        input_ids, labels, dec_input_ids, video = [], [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            video.append(f["video"])

        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len =  max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0) for audio, audio_len in zip(input_ids, audio_lengths)]

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length) # seems redundant

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id

        # 0 pad the videos
        video_lengths = [len(vid) for vid in video]
        max_video_len = max(video_lengths)
        video = [np.pad(vid, ((0, max_video_len - vid_len), (0, 0), (0, 0), (0, 0)), 'constant', constant_values=0) for vid, vid_len in zip(video, video_lengths)]
        padding_mask = create_padding_mask(max_video_len, [max_video_len - vid_len for vid_len in video_lengths])
        
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "video": video,
            "padding_mask": padding_mask,
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch['video'] = batch['video'].permute((0, 4, 1, 2, 3)).contiguous() # [B, T, H, W, C] -> [B, C, T, H, W]

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

def whisper_optimizer(model, cfg, t_total, video=True):
    no_decay = ["bias", "LayerNorm.weight"]
    projection = ["video_projection"] # linear layer and scalar
    if video and cfg.video_projection_separate_lr != '': # ft video projection separate lr
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                            if not any(nd in n for nd in projection)],
                "lr": cfg.learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters()
                            if any(nd in n for nd in projection)],
                "lr": cfg.video_projection_separate_lr,
            },
        ]
    else:
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

def whisper_video_projection_optimizer(model, cfg, t_total):
    if cfg.video_projection_linear_scale != 1.0:
        print("Scaling video projection scaler by {}".format(cfg.video_projection_linear_scale))
        print(model.encoder.video_projection_scalar)
        with torch.no_grad():
            model.encoder.video_projection_scalar *= cfg.video_projection_linear_scale
        print(model.encoder.video_projection_scalar)

    optimizer_grouped_parameters = [
        {
            "params": [*model.encoder.video_projection.parameters(),
                       model.encoder.video_projection_scalar],
            "lr" : cfg.video_projection_lr, 
            "weight_decay": cfg.weight_decay,
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

def whisper_flamingo_projection_optimizer(model, cfg, t_total):
    x_attn = ["gated_x_attn", "attn_gate", "ff"]
    video_projection = ["video_projection"]
    # x_attn = ["gated_x_attn", "attn_gate", "ff"] if cfg.freeze_video_model else ["video_model", "video_blocks" "gated_x_attn", "attn_gate", "ff"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in x_attn + video_projection)],
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

def whisper_biasing_optimizer(model, cfg, t_total):
    biasing_keywords = ["pointer_gate", "Qproj", "Kproj", "ooKBemb"]
    
    optimizer_grouped_parameters = [
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if any(keyword in name for keyword in biasing_keywords)
            ],
            "lr": cfg.learning_rate,
        },
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=cfg.learning_rate,
        eps=cfg.adam_epsilon,
        weight_decay=cfg.weight_decay,
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=t_total,
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
        save_top_k=1,
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
