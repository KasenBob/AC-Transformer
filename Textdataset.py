import argparse
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import sys
import time
import neptune.new as neptune
import os
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import linecache
from torchtext import data
from torchtext.data import Iterator, BucketIterator
from torch.nn import init


def get_max_len(p = None):
    with open(p.train_data_path) as f:
        train_max_len = 0
        '''
        for line in f.readlines():
            if len(line.split()) > train_max_len:
                train_max_len = len(line.split())
        '''
        for line in f.readlines():
            if len(line) > train_max_len:
                train_max_len = len(line)

    with open(p.test_data_path) as f:
        test_max_len = 0
        '''
        for line in f.readlines():
            if len(line.split()) > test_max_len:
                test_max_len = len(line.split())
        '''
        for line in f.readlines():
            if len(line) > test_max_len:
                test_max_len = len(line)
    
    p.train_max_len = train_max_len
    p.test_max_len = test_max_len

    return train_max_len if train_max_len > test_max_len else test_max_len


def get_train_dataset(q_filed, a_filed, p = None):
    xy = open(p.train_data_path, "r")
    fields = [("input", q_filed), ("output", a_filed)]
    examples = []
    for line in tqdm(xy.readlines()):
        src, tgt = line.replace('\n', '').split("\t")
        question = str(src + ' <cls> ')
        answer = str(tgt + ' <eos>')
        examples.append(data.Example.fromlist([question, answer], fields))
    return examples, fields


def get_test_dataset(q_filed, a_filed, p = None):
    xy = open(p.test_data_path, "r")
    fields = [("question", q_filed), ("answer", a_filed)]
    examples = []
    for line in tqdm(xy.readlines()):
        src, tgt = line.replace('\n', '').split("\t")
        question = str(src + ' <cls> ')
        answer = str(' <eos>')
        examples.append(data.Example.fromlist([question, answer], fields))
    return examples, fields


def TextDataset(p = None):

    max_len = get_max_len(p=p)
    
    split_chars = lambda x: list(x)
    q_field = data.Field(sequential = True, batch_first=True, tokenize=split_chars, fix_length=60)
    a_field = data.Field(sequential = True, batch_first=True, tokenize=split_chars, fix_length=60)
    #q_field = data.Field(sequential = True, batch_first=True, tokenize=str.split, fix_length=max_len + 10)
    #a_field = data.Field(sequential = True, batch_first=True, tokenize=str.split, fix_length=max_len + 10)
    # get train dataset
    train_examples, train_fields = get_train_dataset(q_field, a_field, p=p)
    train = data.Dataset(train_examples, train_fields)
    # get test dataset
    test_examples, test_fields = get_test_dataset(q_field, a_field, p=p)
    test = data.Dataset(test_examples, test_fields)
    # build vocab
    #q_field.build_vocab(train, vectors='glove.6B.100d')
    q_field.build_vocab(train)
    #q_field.vocab.vectors.unk_init = init.xavier_uniform
    #a_field.build_vocab(train, vectors='glove.6B.100d')
    a_field.build_vocab(train)
    #a_field.vocab.vectors.unk_init = init.xavier_uniform
    # build iter
    train_iter = data.BucketIterator(train, batch_size=p.batch_size, sort_key=lambda x: len(x.text), shuffle=True)
    test_iter = data.Iterator(test, batch_size=p.batch_size, train=False, sort=False)

    p.q_vocab = q_field.vocab
    p.a_vocab = a_field.vocab
    p.q_vocab_size = len(q_field.vocab)
    p.a_vocab_size = len(a_field.vocab)

    p.UNK = a_field.vocab.stoi['<unk>']
    p.PAD = a_field.vocab.stoi['<pad>']
    p.CLS = a_field.vocab.stoi['<cls>']
    p.EOS = a_field.vocab.stoi['<eos>']

    return train_iter, test_iter
