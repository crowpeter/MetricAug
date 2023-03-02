#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 21:54:47 2022

@author: crowpeter
"""

import torch
import numpy as np
from loaders.data_loader import MELD_vqwav2vec_noisy_0_5_10_Dataset, seq_collate_pad_zeros_with_intv
from torch.utils.data import DataLoader
import joblib
import argparse
import os
from sklearn.metrics import f1_score

#%% load data and label
parser = argparse.ArgumentParser(description='Testing SNR 0db 5db 10db case')
parser.add_argument('--test_meta_file', default='meta/MELD_emo_train_clean.csv',
                    help='meta file of feature')
parser.add_argument('--dataset', default='MELD',
                    help='dataset name')
parser.add_argument('--exp_name', default='exp/original/MELD_stoi_gmm/',
                    help='exp folder for save')
parser.add_argument('--gpu_idx', default='0',
                    help='select gpu')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
#%% parameter
BATCH_SIZE = 32
CLF_NUM = 4
#%% some list for record loss, UAR, AC, etc.
RE_BOX = {'epoch_loss':[],\
            'batch_loss':[],\
            'va_pred':[],\
            'va_true':[],\
            'ts_pred':[],\
            'ts_true':[],\
            'pred_final_metric_intv':{i:[] for i in range(-1, 5)},\
            'true_final_metric_intv':{i:[] for i in range(-1, 5)}
            }

#%% test phase
count=0
RE_BOX['ts_snr_level'] = []
RE_BOX['ts_snr_level_result'] = {i:0 for i in range(0, 15, 5)}

TFM = torch.load(args.exp_name+'/best_va_result.pt')
TFM.eval()
ts_dataset = MELD_vqwav2vec_noisy_0_5_10_Dataset(args.test_meta_file.replace('train', 'test'), emo_num=CLF_NUM)
ts_loader = DataLoader(ts_dataset, batch_size=BATCH_SIZE*4, shuffle=False, collate_fn=seq_collate_pad_zeros_with_intv)
RE_BOX['ts_pred'] = []
RE_BOX['ts_true'] = []
with torch.no_grad():
    for step, batch in enumerate(ts_loader):
        # feature, label and length for model input
        batch_X_ts = batch[0]
        batch_Y_ts = batch[1]
        seq_lengths = batch[2]
        snr_level = batch[3]

        # sort for gru
        sorted_index = torch.argsort(-seq_lengths)
        batch_X_ts = batch_X_ts[sorted_index].to(device)
        batch_Y_ts = batch_Y_ts[sorted_index]
        snr_level = snr_level[sorted_index]
        seq_lengths = seq_lengths[sorted_index]
        
        # throw them through your LSTM (remember to give batch_first=True here if you packed with it)
        _, outputs = TFM.forward(batch_X_ts, seq_lengths)
        pred = torch.cat(outputs)
        
        # result record           
        RE_BOX['ts_pred'].extend(pred.max(1)[1].data.cpu().numpy())
        RE_BOX['ts_true'].extend(batch_Y_ts.data.numpy())
        RE_BOX['ts_snr_level'].extend(snr_level.data.cpu().numpy())
    # metric compute
ts_F1 = f1_score(RE_BOX['ts_true'], RE_BOX['ts_pred'], average='weighted')
    

RE_BOX['wf1'] = ts_F1
print('exp:', args.exp_name)
print(
      ' ts_F1: '+str(round(ts_F1,4))
      )
#%% compute clean
pred = np.array(RE_BOX['ts_pred'])[np.where(np.array(RE_BOX['ts_snr_level'])==-1)]
true = np.array(RE_BOX['ts_true'])[np.where(np.array(RE_BOX['ts_snr_level'])==-1)]
ts_WF1 = f1_score(true, pred, average='weighted')
print('ts clean WF1: '+str(round(ts_WF1,4)))
RE_BOX['ts_clean_wf1'] = ts_WF1

for intv in range(0 ,15, 5):
    # compute wf1 in each interval
    pred = np.array(RE_BOX['ts_pred'])[np.where(np.array(RE_BOX['ts_snr_level'])==intv)]
    true = np.array(RE_BOX['ts_true'])[np.where(np.array(RE_BOX['ts_snr_level'])==intv)]
    ts_WF1 = f1_score(true, pred, average='weighted')
    print('ts rank: '+str(intv)+' WF1: '+str(round(ts_WF1,4)))
    RE_BOX['ts_snr_level_result'][intv] = ts_WF1
    
joblib.dump(RE_BOX, args.exp_name+'/info_0_5_10.pkl')