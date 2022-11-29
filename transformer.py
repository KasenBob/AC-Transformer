import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F


from utils import *
from convolution import *


class EncoderLayer(nn.Module):
    def __init__(self, p):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(p = p)
        self.conv = ConformerConvModule(p = p)
        self.pos_ffn = PoswiseFeedForwardNet(p = p)

        self.resweight = nn.Parameter(torch.Tensor([0]))
        self.dropout1 = nn.Dropout(p.dropout)
        self.dropout2 = nn.Dropout(p.dropout)
        self.dropout3 = nn.Dropout(p.dropout)

        self.flag = 0

    def forward(self,enc_inputs, enc_self_attn_mask, flag):
        '''
        :param enc_inputs: [batch_size, src_len, d_model]
        :param enc_self_attn_mask: [batch_size, src_len, src_len]
        :return:
        '''
        self.flag += 1

        # Self attention layer
        enc_outputs = enc_inputs
        enc_outputs = self.enc_self_attn(enc_outputs, enc_outputs, enc_outputs, enc_self_attn_mask, self.flag)
        enc_outputs = enc_outputs * self.resweight
        enc_inputs = enc_inputs + self.dropout1(enc_outputs)

        
        enc_outputs = enc_inputs
        enc_outputs = self.conv(enc_outputs)
        enc_outputs = enc_outputs * self.resweight
        enc_inputs = enc_inputs + self.dropout2(enc_outputs)

        # Pointiwse FF Layer
        enc_outputs = enc_inputs
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = enc_outputs * self.resweight
        enc_inputs = enc_inputs + self.dropout3(enc_outputs)

        return enc_inputs


class Encoder(nn.Module):
    def __init__(self, p):
        super(Encoder, self).__init__()
        self.p = p 
        self.src_emb = nn.Embedding(p.q_vocab_size, p.d_model)
        self.pos_emb = PositionalEncoding(p.d_model)
        self.layers = nn.ModuleList([EncoderLayer(p = p) for _ in range(p.n_layers)])

    def forward(self, enc_inputs):
        '''
        :param enc_inputs: [batch_size, src_len]
        :return:
        '''
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1) # [batch_size, src_len, src_len]
        #enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attn_pad_mask = get_attn_pad_mask(enc_inputs, enc_inputs, self.p).to(self.p.device) # [batch_size, tgt_len, tgt_len]
        enc_self_attn_subsequence_mask = get_attn_subsequence_mask(enc_inputs).to(self.p.device) #[batch_size, tgt_len, tgt_len]
        enc_self_attn_mask = torch.gt((enc_self_attn_pad_mask + enc_self_attn_subsequence_mask),0).to(self.p.device)

        flag = 0
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, enc_self_attn_mask, flag)
        return enc_outputs


class Transformer(nn.Module):
    def __init__(self, p):
        super(Transformer,self).__init__()
        self.encoder = Encoder(p = p).to(p.device)
        #self.decoder = Decoder(p = p).to(p.device)
        self.projection = nn.Linear(p.d_model, p.a_vocab_size, bias=False).to(p.device)

    def forward(self,enc_inputs):
        '''
        :param enc_inputs: [batch_size, src_len]
        :return:
        '''
        enc_outputs = self.encoder(enc_inputs)
        #dec_outputs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        logits = self.projection(enc_outputs)
        #logits = torch.nn.functional.softmax(logits, dim=1)
        return logits.view(-1, logits.size(-1))