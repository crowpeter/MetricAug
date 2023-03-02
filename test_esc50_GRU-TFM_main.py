#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 21:54:47 2022

@author: crowpeter
"""

import torch
from loaders.data_loader import MELD_vqwav2vec_clean_Dataset, seq_collate_pad_zeros
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import joblib
import argparse
import os


#%% load data and label
parser = argparse.ArgumentParser(description='Testing unseen and ransom case')
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
TEST_NOISE_NAME = args.test_meta_file.split('/')[-1].replace('MELD_test_meta_wav_','').replace('.csv','')
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
          'true_final_metric_intv':{i:[] for i in range(-1, 5)},\
          'va_best_f1':0,\
         }

#%% test phase
count=0
TFM = torch.load(args.exp_name+'/best_va_result.pt')
TFM.eval()
ts_dataset = MELD_vqwav2vec_clean_Dataset(args.test_meta_file)
ts_loader = DataLoader(ts_dataset, batch_size=BATCH_SIZE*4, shuffle=False, collate_fn=seq_collate_pad_zeros)
RE_BOX['ts_pred'] = []
RE_BOX['ts_true'] = []
with torch.no_grad():
    for step, batch in enumerate(ts_loader):
        # feature, label and length for model input
        batch_X_ts = batch[0]
        batch_Y_ts = batch[1]
        seq_lengths = batch[2]

        # sort for gru
        sorted_index = torch.argsort(-seq_lengths)
        batch_X_ts = batch_X_ts[sorted_index].to(device)
        batch_Y_ts = batch_Y_ts[sorted_index]
        seq_lengths = seq_lengths[sorted_index]
        
        # model inference
        _, outputs = TFM.forward(batch_X_ts, seq_lengths)
        pred = torch.cat(outputs)
        
        # result record           
        RE_BOX['ts_pred'].extend(pred.max(1)[1].data.cpu().numpy())
        RE_BOX['ts_true'].extend(batch_Y_ts.data.numpy())
    # metric compute
ts_F1 = f1_score(RE_BOX['ts_true'], RE_BOX['ts_pred'], average='weighted')
RE_BOX['wf1'] = ts_F1

joblib.dump(RE_BOX, args.exp_name+'/info_'+TEST_NOISE_NAME+'.pkl')
print(' exp: '+args.exp_name+'\n'+\
      ' noise_name: '+ str(TEST_NOISE_NAME)+\
      ' ts_F1: '+str(round(ts_F1,3))
      )