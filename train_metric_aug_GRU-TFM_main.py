#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 21:03:55 2022

@author: crowpeter
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.GRU_TFM import GRU_TFM_reDim_clf
from loaders.data_loader import MELD_vqwav2vec_noisy_aug_Dataset, seq_collate_pad_zeros_with_intv
from tqdm import tqdm
from sklearn.metrics import recall_score, accuracy_score,f1_score
from torch.utils.data import DataLoader
from pathlib import Path
import joblib
import argparse
import os
from data_sample_weight import sampling_w_compute

#%% load data and label
parser = argparse.ArgumentParser(description='TFM att model training')
parser.add_argument('--train_fea_meta_file', default='meta/MELD_emo_train_clean.csv',
                    help='meta file of feature & label')
parser.add_argument('--dataset', default='MELD',
                    help='dataset name')
parser.add_argument('--data_metric', default='stoi',
                    help='select metric')
parser.add_argument('--exp_name', default='exp/new_exp',
                    help='exp folder for save')
parser.add_argument('--data_weighted', default=True,
                    help='data augmentation by ranked metric with weighted adjustment')
parser.add_argument('--gpu_idx', default='0',
                    help='gpu select')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
#%% model & training parameter
# feat_dim, hidden_dim, hidden_layers_num, cl_num, tfm_head, dropout_r
FEAT_DIM = 512
HIDDEN_DIM = 16
HIDDEN_LAYERS_NUM = 1
CLF_NUM = 4
DROPOUT_R = 0.0
TFM_HEAD = 2
BATCH_NORM = False
LR=1e-3
EPOCH = 100
BATCH_SIZE = 32
WEIGHTED_LABEL = False
PATIENCE = 10
assign_metric = args.data_metric
DATA_WEIGHTED = args.data_weighted
TOTAL_W = 1
MIN_RM = 0.05
GMM_QTZ = True
#%% some list for record loss, UAR, AC, etc.
RE_BOX = {'epoch_loss':[],\
            'batch_loss':[],\
            'va_pred':[],\
            'va_true':[],\
            'ts_pred':[],\
            'ts_true':[],\
            'pred_final_metric_intv':{i:[] for i in range(-1, 5)},\
            'true_final_metric_intv':{i:[] for i in range(-1, 5)},\
            'record_weight_box':{},\
            'va_best_UAR':0,\
            'va_best_f1':0,\
            }

#%% make exp path for save
path = Path(args.exp_name+'/')
path.mkdir(parents=True, exist_ok=True)
#%% start training
count=0

# model optimizer and CLF_NUM
TFM = GRU_TFM_reDim_clf(FEAT_DIM, HIDDEN_DIM, HIDDEN_LAYERS_NUM, CLF_NUM, TFM_HEAD, max_length=None, dropout_r=DROPOUT_R).to(device)
criterion = nn.NLLLoss().cuda()
optimizer = optim.Adam(TFM.parameters(), lr = LR)
metric_init_weight = {0:TOTAL_W/5, 1:TOTAL_W/5, 2:TOTAL_W/5, 3:TOTAL_W/5, 4:TOTAL_W/5}
metric_weight = metric_init_weight.copy()

# loss uar init
RE_BOX['epoch_loss'] = []
RE_BOX['va_best_noisy_f1'] = 0
RE_BOX['va_best_clean_f1'] = 0
early_stop = 0
#%%
for epoch in tqdm(range(EPOCH)):
    print('epoch:', epoch, 'metric_weight:', metric_weight)
    # Data loadertraining data and testing data
    tr_dataset = MELD_vqwav2vec_noisy_aug_Dataset(args.train_fea_meta_file, emo_num=CLF_NUM, assign_metric=assign_metric, metric_weight = metric_weight, gmm_qtz=GMM_QTZ)
    va_dataset = MELD_vqwav2vec_noisy_aug_Dataset(args.train_fea_meta_file.replace('train', 'validation'), emo_num=CLF_NUM, assign_metric=assign_metric, metric_weight = metric_weight, load_all=True, gmm_qtz=GMM_QTZ)
        
    tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=seq_collate_pad_zeros_with_intv)
    va_loader = DataLoader(va_dataset, batch_size=BATCH_SIZE*2, shuffle=False, collate_fn=seq_collate_pad_zeros_with_intv)
     
    RE_BOX['record_weight_box'] = []
#%% training phase
    TFM.train()
    for step, batch in enumerate(tr_loader):
        # break
        # feature, label and length for model input
        batch_X_tr = batch[0]
        batch_Y_tr = batch[1]
        seq_lengths = batch[2]
        
        # sort for gru
        sorted_index = torch.argsort(-seq_lengths)
        batch_X_tr = batch_X_tr[sorted_index].to(device)
        batch_Y_tr = batch_Y_tr[sorted_index]
        seq_lengths = seq_lengths[sorted_index]
        
        TFM.zero_grad()

        # model
        _, outputs = TFM.forward(batch_X_tr, seq_lengths)
        pred = torch.cat(outputs)
        
        # metric compute
        batch_Y_tr = batch_Y_tr.to(device)
        loss = criterion(pred, batch_Y_tr)
        loss.backward()
        optimizer.step()
        RE_BOX['batch_loss'].append(loss.data.cpu().numpy())
        if (step+1) % 50 ==0:
            print('batch loss:'+str(np.mean(RE_BOX['batch_loss'])))
        # torch.cuda.empty_cache()
    RE_BOX['epoch_loss'].append(np.mean(RE_BOX['batch_loss']))
    
#%% validation phase      
    RE_BOX['va_pred'] = []
    RE_BOX['va_true'] = []
    RE_BOX['va_metric_intv'] = []
    RE_BOX['va_glob_WF1_by_metric_intv'] = {i:0 for i in range(-1, 5)}      
    TFM.eval()
    with torch.no_grad():
        for step, batch in enumerate(va_loader):
            # step
            # feature, label and length for model input
            batch_X_va = batch[0]
            batch_Y_va = batch[1]
            seq_lengths = batch[2]
            metric_intv = batch[3]
    
            # sort for gru
            sorted_index = torch.argsort(-seq_lengths)
            batch_X_va = batch_X_va[sorted_index].to(device)
            batch_Y_va = batch_Y_va[sorted_index]
            metric_intv = metric_intv[sorted_index]
            seq_lengths = seq_lengths[sorted_index]
            
            # model
            _, outputs = TFM.forward(batch_X_va, seq_lengths)
            pred = torch.cat(outputs)
            
            # result record           
            RE_BOX['va_pred'].extend(pred.max(1)[1].data.cpu().numpy())
            RE_BOX['va_true'].extend(batch_Y_va.data.numpy())
            RE_BOX['va_metric_intv'].extend(metric_intv.data.numpy())
    # Find best UAR
    va_UAR = recall_score(RE_BOX['va_true'], RE_BOX['va_pred'], average='macro')
    va_AC = accuracy_score(RE_BOX['va_true'], RE_BOX['va_pred'])
    va_F1 = f1_score(RE_BOX['va_true'], RE_BOX['va_pred'], average='weighted')
    
        
    print(
          ' epoch: '+str(epoch)+\
          ' tr_loss: '+str(round(np.mean(RE_BOX['epoch_loss']), 3))+\
          ' va_F1: '+str(round(va_F1,3))
          )
    
#%% sampling weight compute 
    if DATA_WEIGHTED:
        total_gap = 0
        total_weight = TOTAL_W
        for intv in range(-1,5):
            # compute wf1 in each interval
            pred = np.array(RE_BOX['va_pred'])[np.where(np.array(RE_BOX['va_metric_intv'])==intv)]
            true = np.array(RE_BOX['va_true'])[np.where(np.array(RE_BOX['va_metric_intv'])==intv)]
            va_WF1 = f1_score(true, pred, average='weighted')
            
            print(' rank: '+str(intv)+' WF1: '+str(round(va_WF1,3)))
            
            RE_BOX['va_glob_WF1_by_metric_intv'][intv] = va_WF1
        metric_weight = sampling_w_compute(TOTAL_W, MIN_RM, RE_BOX['va_glob_WF1_by_metric_intv'])

        print('New data aug weight:', metric_weight)
#%% early stopping 
    if va_F1 > RE_BOX['va_best_f1']:
        RE_BOX['va_best_f1'] = va_F1
        torch.save(TFM,args.exp_name+'/best_va_result.pt')
        joblib.dump(RE_BOX, args.exp_name+'/info_'+str(epoch)+'_ckpt.pkl')
        early_stop = 0
        print('best model save')
        print('F1: '+str(RE_BOX['va_best_f1']))

    elif early_stop < PATIENCE:
        early_stop += 1
    elif early_stop == PATIENCE:
        break
    print(
          ' epoch: '+str(epoch)+\
          ' tr_loss: '+str(round(np.mean(RE_BOX['epoch_loss']), 3))+\
          ' va_F1: '+str(round(va_F1,3))
          )
    RE_BOX['va_pred'] = []
    RE_BOX['va_true'] = []
#%% test phase
RE_BOX['ts_true_metric_intv'] = {i:[] for i in range(-1,5)}
RE_BOX['ts_pred_metric_intv'] = {i:[] for i in range(-1,5)}
RE_BOX['ts_metric_intv'] = []
RE_BOX['ts_glob_WF1_by_metric_intv'] = {i:0 for i in range(-1, 5)}

TFM = torch.load(args.exp_name+'/best_va_result.pt')
TFM.eval()
ts_dataset = MELD_vqwav2vec_noisy_aug_Dataset(args.train_fea_meta_file.replace('train', 'test'), emo_num=CLF_NUM, assign_metric=assign_metric, metric_weight = metric_weight, load_all=True, gmm_qtz=GMM_QTZ)
ts_loader = DataLoader(ts_dataset, batch_size=BATCH_SIZE*2, shuffle=False, collate_fn=seq_collate_pad_zeros_with_intv)
RE_BOX['ts_pred'] = []
RE_BOX['ts_true'] = []
with torch.no_grad():
    for step, batch in enumerate(ts_loader):
        # step
        # feature, label and length for model input
        batch_X_ts = batch[0]
        batch_Y_ts = batch[1]
        seq_lengths = batch[2]
        metric_intv = batch[3]

        # sort for gru
        sorted_index = torch.argsort(-seq_lengths)
        batch_X_ts = batch_X_ts[sorted_index].to(device)
        batch_Y_ts = batch_Y_ts[sorted_index]
        metric_intv = metric_intv[sorted_index]
        seq_lengths = seq_lengths[sorted_index]
        
        # model
        _, outputs = TFM.forward(batch_X_ts, seq_lengths)
        pred = torch.cat(outputs)
        
        # result record           
        RE_BOX['ts_pred'].extend(pred.max(1)[1].data.cpu().numpy())
        RE_BOX['ts_true'].extend(batch_Y_ts.data.numpy())
        RE_BOX['ts_metric_intv'].extend(metric_intv.data.cpu().numpy())

    # metric compute
ts_F1 = f1_score(RE_BOX['ts_true'], RE_BOX['ts_pred'], average='weighted')
RE_BOX['wf1'] = ts_F1
print(
      ' ts_F1: '+str(round(ts_F1,3))
      )

for intv in range(-1,5):
    # compute wf1 in each interval
    pred = np.array(RE_BOX['ts_pred'])[np.where(np.array(RE_BOX['ts_metric_intv'])==intv)]
    true = np.array(RE_BOX['ts_true'])[np.where(np.array(RE_BOX['ts_metric_intv'])==intv)]
    ts_WF1 = f1_score(true, pred, average='weighted')
    print('ts rank: '+str(intv)+' WF1: '+str(round(ts_WF1,3)))
    RE_BOX['ts_glob_WF1_by_metric_intv'][intv] = ts_WF1
    
joblib.dump(RE_BOX, args.exp_name+'/info'+str(count)+'.pkl')