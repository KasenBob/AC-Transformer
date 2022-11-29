import argparse
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import sys
import time
import os
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import linecache
from torchtext import data
from torchtext.data import Iterator, BucketIterator
from torch.nn import init
from Textdataset import *
from testdataset import *
from transformer import *
from result import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device_ids = [0, 1]

parser = argparse.ArgumentParser(description='This is a shit code.')
# model
parser.add_argument('--seed', default=147, type=int, help='seed')
parser.add_argument('--batch_size', default=256, type=int, help='train batch size')
parser.add_argument('--num_workers', default=12, type=int, help='num_workers')
parser.add_argument('--epochs', default=300, type=int, help='epochs')
parser.add_argument('--debug', default=False, type=bool, help='debug or not')
parser.add_argument('--re_train', default=False, type=bool, help='re_train or not')
parser.add_argument('--work_mode', default='train', type=str, help='train or test')

parser.add_argument('--d_model', default=512, type=int, help='seed')
parser.add_argument('--d_ff', default=2048, type=int, help='seed')
parser.add_argument('--d_k', default=64, type=int, help='seed')
parser.add_argument('--d_v', default=64, type=int, help='seed')
parser.add_argument('--n_layers', default=6, type=int, help='seed')
parser.add_argument('--n_heads', default=8, type=int, help='seed')

parser.add_argument('--save_model_path', default="weight/", type=str, help='the path to save the model')
parser.add_argument('--train_data_path', default='data/train.xy', type=str, help='the path of the data')
parser.add_argument('--test_data_path', default='data/test.xy', type=str, help='the path of the data')

parser.add_argument('--flooding', default=1e-2, type=float, help='flooding')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout_rate')
parser.add_argument('--lr', default=0.001, type=float, help='learning_rate')
parser.add_argument('--optimizer', default="Adam", type=str, help='optimizer')

p = parser.parse_args()
if torch.cuda.is_available():
    p.device = 'cuda'
else:
    print("ERROR: No CUDA device available.")
    sys.exit()

def main():
    
    train_iter, test_iter = TextDataset(p = p)

    model = Transformer(p = p).to(p.device)
    total_params = sum(p.numel() for p in model.parameters())
    print('total ' + str(total_params) + ' parameters.')

    if p.work_mode == 'test':
        model.load_state_dict(torch.load('weight/model.G'))
        print("The model was loaded!")
        result(model, test_iter, p)
    elif p.work_mode == 'check':
        print(model)
    else:
        if p.re_train == True:
            model.load_state_dict(torch.load(p.save_model_path + 'model.G'))
            print("The model was loaded!")

        criterion = nn.CrossEntropyLoss(ignore_index=1)

        if p.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=p.lr, betas=(0.9, 0.98), eps=1e-09)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        elif p.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(),lr=p.lr, momentum=0.99)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        # train
        flooding_count = 0
        train_loss = []
        test_loss = []

        for epoch in range(p.epochs):
            print("The epoch " + str(epoch) + " is starting at " + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            # train
            for batch in tqdm(train_iter):
                dec_input, dec_output = batch.input.to(p.device), batch.output.to(p.device)

                outputs = model(dec_input)
                loss = criterion(outputs, dec_output.reshape(-1))
                if flooding_count % 2 == 0:
                    loss = loss - (0.5) * loss 
                else:
                    loss = loss + (0.4) * loss
                flooding_count += 1

                if p.debug == True:
                    print('Epoch:','%04d' % (epoch+1), 'loss =','{:.6f}'.format(loss))

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

            scheduler.step()
            #save model 
            torch.save(model.state_dict(), p.save_model_path + 'model.G')
            if epoch % 5 == 0:
                torch.save(model.state_dict(), p.save_model_path + 'model' + str(epoch) +'.G')

            print("The epoch " + str(epoch) + " was ended at " + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

            # Test
            if p.debug == False:
                tmp = result(model, test_iter, p)
                test_loss.extend(tmp)

        if p.debug == False:
            result(model, test_iter, p)


if __name__ == '__main__':
    main()