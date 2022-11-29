import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len,1], pos向量
        # div_term [d_model/2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 10000^{2i/d_model}
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位赋值 [max_len,d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)  # 技术位赋值 [max_Len,d_model/2]
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len,1,d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: [seq_len, batch_size, d_model]
        :return:
        '''
        x = x + self.pe[:x.size(0), :] # 直接将pos_embedding 和 vocab_embedding相加
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k, p):
    '''
    :param seq_q: [batch_size, seq_len]
    :param seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    :return:
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    #eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(p.PAD).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, p):
        super(ScaledDotProductAttention, self).__init__()
        self.p = p
    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.p.d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        '''
        print("context:")
        print(torch.sum(context, dim=3))
        print(torch.sum(context, dim=3).size())
        '''

        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, p):
        super(MultiHeadAttention, self).__init__()
        self.p = p
        self.W_Q = nn.Linear(p.d_model, p.d_k * p.n_heads, bias=False)
        self.W_K = nn.Linear(p.d_model, p.d_k * p.n_heads, bias=False)
        self.W_V = nn.Linear(p.d_model, p.d_v * p.n_heads, bias=False)
        self.fc = nn.Linear(p.n_heads * p.d_v, p.d_model, bias=False)

    def forward(self,input_Q, input_K, input_V, attn_mask, flag):
        '''
        :param input_Q: [batch_size, len_q, d_model]
        :param input_K: [batch_size, len_k, d_model]
        :param input_V: [batch_size, len_v(=len_k), d_model]
        :param attn_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size,-1, self.p.n_heads, self.p.d_k).transpose(1,2) # Q:[batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size,-1, self.p.n_heads, self.p.d_k).transpose(1,2) # K:[batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size,-1, self.p.n_heads, self.p.d_v).transpose(1,2) # V:[batch_size, n_heads, len_v(=len_k, d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.p.n_heads, 1, 1)

        context = ScaledDotProductAttention(p = self.p)(Q, K, V, attn_mask)

        context = context.transpose(1,2).reshape(batch_size, -1, self.p.n_heads * self.p.d_v)
        #
        #print(context)
        #print(context.size())
        output = self.fc(context)

        #m = nn.Softmax(dim=2)
        #print(m(output))
        #print(m(output).size())

        #return nn.LayerNorm(self.p.d_model).to(self.p.device)(output + residual) # Layer Normalization
        return output


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, p):
        super(PoswiseFeedForwardNet, self).__init__()
        self.p = p
        self.fc = nn.Sequential(
            nn.Linear(p.d_model, p.d_ff,bias=False),
            nn.ReLU(),
            nn.Linear(p.d_ff, p.d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        :param inputs: [batch_size, seq_len, d_model]
        :return:
        '''
        #residual = inputs
        output = self.fc(inputs)
        #return nn.LayerNorm(self.p.d_model).to(self.p.device)(output+residual) #[batch_size, seq_len, d_model]
        return output


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)


class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, inputs):
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    """
    The gating mechanism is called Gated Linear Units (GLU), which was first introduced for natural language processing
    in the paper “Language Modeling with Gated Convolutional Networks”
    """
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class print_size(nn.Module):
    def __init__(self, flag:int) -> None:
        super(print_size, self).__init__()
        self.flag = flag
    def forward(self, inputs):
        print(str(self.flag) + ':')
        print(inputs.size())
        return inputs
