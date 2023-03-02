#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:59:35 2021

@author: crowpeter
"""

import joblib
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd
    
class MELD_vqwav2vec_clean_Dataset(Dataset):
    def __init__(self, fea_meta_file):
        fea_meta_file = pd.read_csv(fea_meta_file)
        self._Y = []
        self._X = []
        
        print('Start loading wav2vec feature')
        for i in range(len(fea_meta_file)):
            fea_path = fea_meta_file.iloc[i]['fea_path']
            target_emo = fea_meta_file.iloc[i]['emo']
            if target_emo != 4:
                self._X.append(fea_path)
                self._Y.append(target_emo)
            if (i+1)%10000 == 0:
                print('Load '+str(i+1)+' data...')
                  
        assert len(self._X) == len(self._Y)
        print('Finish loading wav2vec feature')
        print('Length:'+str(len(self._Y)))
        
    def __len__(self):
        return len(self._X) 
    
    def __getitem__(self, idx):
        x_data = joblib.load(self._X[idx])
        y_data = self._Y[idx]
        
        return x_data, y_data, len(x_data)

class MELD_vqwav2vec_noisy_aug_Dataset(Dataset):
    def __init__(self, fea_meta_file, emo_num, assign_metric, metric_weight, load_all=False, gmm_qtz=False):
        fea_meta = pd.read_csv(fea_meta_file)
        self._Y = []
        self._X = []
        self.metric_weight = metric_weight
        print('Start loading wav2vec feature')
        for i in range(len(fea_meta)):
            fea_path = fea_meta.iloc[i]['fea_path']
            target_emo = fea_meta.iloc[i]['emo']
            if target_emo != 4:
                self._X.append(fea_path)
                self._Y.append(target_emo)
            if (i+1)%10000 == 0:
                print('Load '+str(i+1)+' data...')
        assert len(self._X) == len(self._Y)
        print('Finish loading wav2vec clean feature', 'emo class=', emo_num)
        print('Length:'+str(len(self._Y)))
        
        clean_length = len(self._X)
        
        split_set = fea_meta_file.split('/')[-1].split('_')[2]
        if gmm_qtz:
            aug_data_meta = pd.read_csv('meta/MELD_'+split_set+'_all_noisy_aug_data.csv')
            rank_level = 'gmm_rank_'+assign_metric
        else:
            aug_data_meta = pd.read_csv('meta/MELD_'+split_set+'_all_noisy_aug_data.csv')
            rank_level = 'rank_'+assign_metric
        if emo_num == 4:
            aug_data_meta = aug_data_meta.loc[aug_data_meta['emo'] != 4]
        self.metric_intv = [-1]*len(self._X)
        for intv_idx in range(5):
            temp_df = aug_data_meta.loc[aug_data_meta[rank_level] == intv_idx]
            if not load_all:
                temp_df = temp_df.sample(n=int(clean_length*self.metric_weight[intv_idx]))
            self._X += list(temp_df['fea_path'])
            self._Y += list(temp_df['emo'])
            self.metric_intv += list(temp_df[rank_level])
            assert len(self._Y) == len(self._X)
            assert len(self._Y) == len(self.metric_intv)
        print('Total Length: '+str(len(self._Y)))
    def __len__(self):
        return len(self._X) 
    def __getitem__(self, idx):
        intv = self.metric_intv[idx]
        x_data = joblib.load(self._X[idx])
        y_data = self._Y[idx]
        return x_data, y_data, len(x_data), intv

def seq_collate_pad_zeros_with_intv(batch):
    """
    Returns:
        [B, F, T (Longest)]
    """
    X_list = [torch.tensor(item[0], dtype=torch.float) for item in batch]
    Y_list = [item[1] for item in batch]
    seq_lengths = torch.LongTensor([item[2] for item in batch])
    intv = torch.tensor([item[3] for item in batch])
    Y_list = torch.LongTensor(np.array(Y_list))
    X_list_pad = pad_sequence(X_list ,batch_first=True)
    return X_list_pad, Y_list, seq_lengths, intv

class MELD_vqwav2vec_noisy_0_5_10_Dataset(Dataset):
    def __init__(self, fea_meta_file, emo_num):
        fea_meta = pd.read_csv(fea_meta_file)
        self._Y = []
        self._X = []
        self._SNR = []
        print('Start loading wav2vec feature')
        for i in range(len(fea_meta)):
            fea_path = fea_meta.iloc[i]['fea_path']
            target_emo = fea_meta.iloc[i]['emo']
            if emo_num == 4:
                if target_emo != 4:
                    self._X.append(fea_path)
                    self._Y.append(target_emo)
            else:
                self._X.append(fea_path)
                self._Y.append(target_emo)
            if (i+1)%10000 == 0:
                print('Load '+str(i+1)+' data...')
        assert len(self._X) == len(self._Y)
        print('Finish loading wav2vec clean feature', 'emo class=', emo_num)
        print('Length:'+str(len(self._Y)))
        self._SNR = [-1]*len(self._X)
        
        split_set = fea_meta_file.split('/')[-1].split('_')[2]
        aug_data_meta = pd.read_csv('meta/MELD_emo_'+split_set+'_0_5_10.csv')
        if emo_num == 4:
            aug_data_meta = aug_data_meta.loc[aug_data_meta['emo'] != 4]
        self._X += list(aug_data_meta['fea_path'])
        self._Y += list(aug_data_meta['emo'])
        self._SNR += list(aug_data_meta['snr_level'])
        assert len(self._Y) == len(self._X)
        assert len(self._Y) == len(self._SNR)
            
        print('Total Length: '+str(len(self._Y)))
    def __len__(self):
        return len(self._X) 
    
    def __getitem__(self, idx):
        snr = self._SNR[idx]
        x_data = joblib.load(self._X[idx])
        y_data = self._Y[idx]
        return x_data, y_data, len(x_data), snr
    
#%%