import json
import os
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import whisper
from transformers import WhisperTokenizer

def make_lexical_tree(word_dict, subword_dict, word_unk):
    # node [dict(subword_id -> node), word_id, word_set[start-1, end]]
    root = [{}, -1, None]
    for w, wid in word_dict.items():
        if wid > 0 and wid != word_unk:  # skip <blank> and <unk>
            if True in [c not in subword_dict for c in w]:  # skip unknown subword
                continue
            succ = root[0]  # get successors from root node
            for i, c in enumerate(w):
                cid = subword_dict[c]
                if cid not in succ:  # if next node does not exist, make a new node
                    succ[cid] = [{}, -1, (wid - 1, wid)]
                else:
                    prev = succ[cid][2]
                    succ[cid][2] = (min(prev[0], wid - 1), max(prev[1], wid))
                if i == len(w) - 1:  # if word end, set word id
                    succ[cid][1] = wid
                succ = succ[cid][0]  # move to the child successors
    return root

class BiasingProcessor_taigi(object):
    def __init__(self, tokenizer, vocab_size):
        # 這邊 WhisperBiasing 的 code 不確定為什麼 vocab_size 印出來是 50257
        self.chardict = {idx:idx for idx in range(vocab_size)}

    def construct_tree(self, uttblist):
        worddict = {word: i+1 for i, word in enumerate(uttblist)}
        lextree = make_lexical_tree(worddict, self.chardict, -1)
        return lextree

    def get_lextree(self, uttblist):
        uttblist = [tuple(bword) for bword in uttblist]
        lextree = self.construct_tree(uttblist)
        return lextree