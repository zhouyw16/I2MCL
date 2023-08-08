from __future__ import print_function
import os
import json
import pickle
import numpy as np
import utils
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import CLIPTokenizer


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('.', '').replace('!', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, sentence, label):
    entry = {
        'image'       : img,
        'sentence'    : sentence,
        'label'       : label}
    return entry


def _load_dataset(dataroot, name, img_id2val):
    """Load entries
    """

    entries = []
    path = os.path.join(dataroot, 'snli-ve', 've_%s.json' % name)
    with open(path) as file:
        for line in json.load(file):
            image, sentence, label = line['image'], line['sentence'], line['label']
            img_id = '%s.jpg' % (image)
            entries.append(_create_entry(img_id2val[img_id], sentence, label))
    return entries


class VEImageTextDataset(Dataset):
    def __init__(self, name, dictionary, images, img_id2idx, transform, dataroot='data'):
        super(VEImageTextDataset, self).__init__()
        assert name in ['train', 'dev', 'test']

        self.ans2label = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
        self.label2ans = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.images = images
        self.img_id2idx = img_id2idx
        self.transform = transform

        self.entries = _load_dataset(dataroot, name, self.img_id2idx)
        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=14):
        """Tokenizes the sentences.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        tokenizer.padding_side = 'left'
        for entry in self.entries:
            clip_tokens = tokenizer.encode(entry['sentence'], padding='max_length', truncation=True, max_length=max_length)
            tokens = self.dictionary.tokenize(entry['sentence'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(clip_tokens), max_length)
            utils.assert_eq(len(tokens), max_length)
            entry['clip_token'] = clip_tokens
            entry['q_token'] = tokens

    def tensorize(self):
        for entry in self.entries:
            clip_sentence = torch.from_numpy(np.array(entry['clip_token']))
            entry['clip_token'] = clip_sentence
            sentence = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = sentence
            entry['label'] = self.ans2label[entry['label']]

    def __getitem__(self, index):
        entry = self.entries[index]
        image = self.transform(Image.fromarray(self.images[entry['image']].astype('uint8')).convert('RGB'))
        sentence = entry['q_token']
        clip_sentence = entry['clip_token']
        label = entry['label']

        return image, sentence, clip_sentence, label

    def __len__(self):
        return len(self.entries)
