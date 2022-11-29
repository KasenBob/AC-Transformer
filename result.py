import argparse
import math
from re import A
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
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from transformer import *
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]

def result(model, test_iter = None, p = None):
    score = 0
    word_score = 0
    count = 0
    word_count = 0

    criterion = nn.CrossEntropyLoss(ignore_index=1)
    test_loss = []

    for batch in tqdm(test_iter):
        inputs, outputs = batch.question.to(p.device), batch.answer.to(p.device)
        for i in range(len(inputs)):
            # greedy_dec_input = greedy_decoder(model, question[i].view(1, -1), start_symbol=p.CLS, final_symbol=p.EOS, device=p.device)

            predict = model(inputs[i].view(1, -1))

            loss = criterion(predict, outputs[i].reshape(-1))
            # 可视化
            tmp = loss
            with torch.no_grad():
                test_loss.append(tmp.cpu().numpy())

            predict = predict.data.max(1, keepdim=True)[1]
            #print(predict.squeeze())
            predict_list = []
            predict = predict.squeeze().cpu().numpy().tolist()
            if isinstance (predict, int):
                predict_list.append(predict)
            else:
                predict_list = predict
            #print(enc_inputs[i], '->', [p.a_vocab.itos[n] for n in predict_list])
            answer = outputs[i].squeeze().cpu().numpy().tolist()

            final = answer.index(p.EOS)
            answer = answer[: final]
            predict_list = predict_list[: len(answer)]

            if answer == predict_list:
                score += 1
                #print(score)
            count += 1

            for a, b in zip(answer, predict_list) :
                if a == b:
                    word_score += 1
                word_count += 1

    print("The acc_score is " + str(score))
    print("The acc is " + str(score / count))
    print("The word_score is " + str(word_score))
    print("The word_acc is " + str(word_score / word_count))

    return test_loss

