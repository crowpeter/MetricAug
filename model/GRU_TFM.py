#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:35:21 2022

@author: crowpeter
"""

import torch
from torch import nn

class TFM(nn.Module):
    def __init__(self, layers, hidden_size, head):
        super(TFM, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.head = head
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, dim_feedforward=self.hidden_size, nhead=self.head, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.layers, norm=nn.LayerNorm(self.hidden_size))
    def forward(self, encoder_outputs, pad_mask=None):
        if pad_mask is None:
            out = self.transformer_encoder.forward(encoder_outputs)
        else:
            out = self.transformer_encoder.forward(encoder_outputs, src_key_padding_mask=pad_mask)
        return out

class Bi_GRU_dynamic(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(Bi_GRU_dynamic, self).__init__()
        self.feat_dim = feat_dim
        self.gru_hidden_dim = hidden_dim
        # self.hidden_layers_num = hidden_layers_num
        # self.max_length = max_length
        
        self.gru_model = nn.GRU(self.feat_dim, self.gru_hidden_dim, 1, bidirectional=True)
    
    def forward(self, input_seqs, input_lens, hidden=None):
        input_seqs = input_seqs.transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_seqs, input_lens)
        outputs, hidden = self.gru_model(packed)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)  #[T,B,E]
        outputs = outputs[:, :, :self.gru_hidden_dim] \
                + outputs[:, :, self.gru_hidden_dim:]   
        outputs = outputs.transpose(0, 1)
        hidden = hidden.transpose(0, 1)
        
        return outputs, hidden, input_lens

class GRU_dynamic(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(GRU_dynamic, self).__init__()
        self.feat_dim = feat_dim
        self.gru_hidden_dim = hidden_dim
        
        self.gru_model = nn.GRU(self.feat_dim, self.gru_hidden_dim, 1, bidirectional=False)
    
    def forward(self, input_seqs, input_lens, hidden=None):
        input_seqs = input_seqs.transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_seqs, input_lens)
        outputs, hidden = self.gru_model(packed)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)  #[T,B,E]      
        outputs = outputs.transpose(0, 1)
        hidden = hidden.transpose(0, 1)
        
        return outputs, hidden, input_lens
        

        
class GRU_TFM_reDim_clf(nn.Module):
    def __init__(self, feat_dim, hidden_dim, hidden_layers_num, cl_num, tfm_head, max_length, dropout_r=0.0):
        super(GRU_TFM_reDim_clf, self).__init__()
        
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers_num = hidden_layers_num
        self.cl_num = cl_num
        self.tfm_head = tfm_head
        self.max_length = max_length

        # input fc
        network = [nn.Linear(self.feat_dim, self.hidden_dim)]
        self.input_layer = nn.Sequential(*network)
        
        # bi_GRU_encoder
        self.gru = GRU_dynamic(self.hidden_dim, self.hidden_dim)
        
        # tfm_encoder
        self.tfm = TFM(self.hidden_layers_num, self.hidden_dim, self.tfm_head)
        
        # drop_out
        self.drop_out = nn.Dropout(p=dropout_r)
        
        # output fc
        network = [nn.Linear(self.hidden_dim, self.cl_num), nn.LogSoftmax(dim=1)]
        self.clf = nn.Sequential(*network)
        
    def forward(self, input_seqs, input_lens):

        # input fc
        input_fc_op = torch.zeros((len(input_seqs), input_lens.max() ,self.hidden_dim)).to(input_seqs.device)
        input_mask = torch.ones((len(input_seqs), input_lens.max()), dtype = torch.bool).to(input_seqs.device)
        
        for i in range(len(input_lens)):
            input_fc_op[i, :input_lens[i], :] = self.input_layer.forward(input_seqs[i,:input_lens[i],:])
            input_mask[i, :input_lens[i]] = False
        
        # gru encode
        gru_op, _, _ = self.gru.forward(input_fc_op, input_lens)
        
        gru_op = gru_op.transpose(0, 1)
        # tfn_emcoder
        tfm_output = self.tfm.forward(gru_op, input_mask)
        # # output fc
        tfm_output = tfm_output.transpose(0, 1)
        clf_output = []
        for i, data in enumerate(tfm_output):
            lens = input_lens[i]
            clf_output.append(self.clf.forward(torch.mean(data[:lens,:],dim=0,keepdim=True)))
            
        return tfm_output, clf_output
    
#%%
if __name__ == '__main__':
    import numpy as np
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    gru_tfm = GRU_TFM_reDim_clf(feat_dim=512, hidden_dim=16, hidden_layers_num=2, cl_num=4, tfm_head=2, max_length=None, dropout_r=0.3).to(device)
    feat_seqs = []
    for i in range(32):
        m = np.random.randint(300,1098)
        temp = np.random.normal(size=(m,512))
        feat_seqs.append(temp)
    true_y = torch.randint(4, (len(feat_seqs),))
    seq_lengths = torch.LongTensor([len(seq) for seq in feat_seqs])
    batch_x = torch.rand((len(feat_seqs),  seq_lengths.max(), 512))
    sort_index = torch.argsort(-seq_lengths)
    batch_x = batch_x[sort_index].to(device)
    true_y = true_y[sort_index].to(device)
    seq_lengths = seq_lengths[sort_index]    
    _, outputs = gru_tfm.forward(batch_x, seq_lengths)
