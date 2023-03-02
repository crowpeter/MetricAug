#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:26:02 2022

@author: crowpeter
"""

import joblib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch
import sys
import pandas as pd
import glob
from collections import Counter
import os
# from torch.utils.data import DataLoader
class IEMOCAP_wav2vec_clean_Dataset(Dataset):
    def __init__(self, fea_meta_file, feature_type, label_csv_file, test_sess, label_type, mode, label_weighted=None):
        fea_meta_file = pd.read_csv(fea_meta_file)
        self._x_fea_dict = {path.split('/')[-1].replace('.pkl',''):path for path in fea_meta_file[feature_type]}
        self._y_csv = pd.read_csv(label_csv_file)
        self.test_sess = test_sess
        if test_sess[-1] == 'F':
            self.validation_sess = test_sess[:5]+'M'
        elif test_sess[-1] == 'M':
            self.validation_sess = test_sess[:5]+'F'
        self._Y = []
        self._X = []
        print('Start loading wav2vec '+str(test_sess)+' '+str(mode)+' feature')
        for i in range(len(self._y_csv)):
            
            if self._y_csv.iloc[i]['Name_String'][:5] != test_sess[:5] and mode == 'train':
                if label_type == 'Emotion':
                    if self._y_csv.iloc[i][label_type]=='sad':
                        self._Y.append(0)
                        # self._X.append(joblib.load(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')]).squeeze(dim=0).to('cpu'))
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='neu':
                        self._Y.append(1)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='hap' or self._y_csv.iloc[i][label_type]=='exc':
                        self._Y.append(2)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='ang':
                        self._Y.append(3)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                        
            elif self._y_csv.iloc[i]['Name_String'][:6] == self.validation_sess and mode == 'validation':
                if label_type == 'Emotion':
                    if self._y_csv.iloc[i][label_type]=='sad':
                        self._Y.append(0)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='neu':
                        self._Y.append(1)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='hap' or self._y_csv.iloc[i][label_type]=='exc':
                        self._Y.append(2)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='ang':
                        self._Y.append(3)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                        
            elif self._y_csv.iloc[i]['Name_String'][:6] == test_sess and mode == 'test':
                if label_type == 'Emotion':
                    if self._y_csv.iloc[i][label_type]=='sad':
                        self._Y.append(0)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='neu':
                        self._Y.append(1)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='hap' or self._y_csv.iloc[i][label_type]=='exc':
                        self._Y.append(2)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='ang':
                        self._Y.append(3)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    
        assert len(self._Y) == len(self._X)
        if label_weighted:
            self.label_weight = [item[1]/len(self._Y) for item in sorted(Counter(self._Y).items())]
            print(sorted(Counter(self._Y).items()))
        print('Finish loading wav2vec '+str(test_sess)+' '+str(mode)+' feature')
        print('Length:'+str(len(self._Y)))
    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        x_data = joblib.load(self._X[idx])
        y_data = self._Y[idx]
        return dict(X=x_data,Y=y_data)
        # return x_data, y_data
    
class IEMOCAP_wav2vec_noisy_Dataset(Dataset):
    def __init__(self, fea_meta_file, label_csv_file, test_sess, label_type, mode, label_weighted=None, noise_level=None, noise_type=None):
        fea_meta_file = pd.read_csv(fea_meta_file)
        _x_fea_dict = {path.split('/')[-1].replace('.pkl',''):path for path in fea_meta_file[noise_level]}
        
        self._x_fea_dict={}
        for key in _x_fea_dict.keys():
            if noise_type in _x_fea_dict[key]:
                new_key = '_'.join(key.split('_')[:-1])
                self._x_fea_dict[new_key] = _x_fea_dict[key]
        del _x_fea_dict
                
        self._y_csv = pd.read_csv(label_csv_file)
        self.test_sess = test_sess
        if test_sess[-1] == 'F':
            self.validation_sess = test_sess[:5]+'M'
        elif test_sess[-1] == 'M':
            self.validation_sess = test_sess[:5]+'F'
        self._Y = []
        self._X = []
        print('Start loading wav2vec '+noise_level+' '+noise_type+' '+str(test_sess)+' '+str(mode)+' feature')
        for i in range(len(self._y_csv)):
            
            # if self._y_csv.iloc[i]['Name_String'][:5] != test_sess[:5] and mode == 'train':
            #     if label_type == 'Emotion':
            #         if self._y_csv.iloc[i][label_type]=='sad':
            #             self._Y.append(0)
            #             self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
            #         elif self._y_csv.iloc[i][label_type]=='neu':
            #             self._Y.append(1)
            #             self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
            #         elif self._y_csv.iloc[i][label_type]=='hap' or self._y_csv.iloc[i][label_type]=='exc':
            #             self._Y.append(2)
            #             self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
            #         elif self._y_csv.iloc[i][label_type]=='ang':
            #             self._Y.append(3)
            #             self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                        
            # elif self._y_csv.iloc[i]['Name_String'][:6] == self.validation_sess and mode == 'validation':
            #     if label_type == 'Emotion':
            #         if self._y_csv.iloc[i][label_type]=='sad':
            #             self._Y.append(0)
            #             self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
            #         elif self._y_csv.iloc[i][label_type]=='neu':
            #             self._Y.append(1)
            #             self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
            #         elif self._y_csv.iloc[i][label_type]=='hap' or self._y_csv.iloc[i][label_type]=='exc':
            #             self._Y.append(2)
            #             self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
            #         elif self._y_csv.iloc[i][label_type]=='ang':
            #             self._Y.append(3)
            #             self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                        
            if self._y_csv.iloc[i]['Name_String'][:6] == test_sess and mode == 'test':
                if label_type == 'Emotion':
                    if self._y_csv.iloc[i][label_type]=='sad':
                        self._Y.append(0)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='neu':
                        self._Y.append(1)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='hap' or self._y_csv.iloc[i][label_type]=='exc':
                        self._Y.append(2)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='ang':
                        self._Y.append(3)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    
        assert len(self._Y) == len(self._X)
        if label_weighted:
            self.label_weight = [item[1]/len(self._Y) for item in sorted(Counter(self._Y).items())]
            print(sorted(Counter(self._Y).items()))
        # print('Finish loading wav2vec '+str(test_sess)+' '+str(mode)+' feature')
        print('Finish loading wav2vec '+noise_level+' '+noise_type+' '+str(test_sess)+' '+str(mode)+' feature')
        print('Length:'+str(len(self._Y)))
    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        x_data = joblib.load(self._X[idx])
        y_data = self._Y[idx]
        return dict(X=x_data,Y=y_data)

class MSP_wav2vec_clean_Dataset(Dataset):
    def __init__(self, fea_meta_file):
        target_emo_dict = {'A':0, 'H':1, 'S':2, 'N':3, 'D':4}
        all_annotation_csv = pd.read_csv('/mnt/sdd/crowpeter/MSP_PODCAST1.8/labels_concensus.csv')
        all_annotation_df = all_annotation_csv.set_index('FileName').T.to_dict('list')
        fea_meta_file = pd.read_csv(fea_meta_file)
        self._Y = []
        self._X = []
        # self._Metric = []
        # self.st_list = []
        # self.ed_list = []
        
        print('Start loading wav2vec feature')
        for i in range(len(fea_meta_file)):
            fea_path = fea_meta_file.iloc[i]['fea_path']
            fea_name = fea_path.split('/')[-1].replace('.pkl','.wav')
            
            emo_str_label = all_annotation_df[fea_name][0]
            assert emo_str_label in list(target_emo_dict.keys())
            
            self._X.append(fea_path)
            self._Y.append(target_emo_dict[emo_str_label])
            # self._Metric.append(fea_meta_file.iloc[i][metric])
            # self.st_list.append(fea_meta_file.iloc[i]['st'])
            # self.ed_list.append(fea_meta_file.iloc[i]['ed'])
            if (i+1)%10000 == 0:
                print('Load '+str(i+1)+' data...')
                  
        assert len(self._X) == len(self._Y)
        print('Finish loading wav2vec feature')
        print('Length:'+str(len(self._Y)))
    def __len__(self):
        return len(self._X) 
    
    def __getitem__(self, idx):
        pad_size = 1098
        # print(st,ed)
        
        x_data = joblib.load(self._X[idx])
        if len(x_data) >= pad_size:
            X = x_data[:pad_size, :]
            length = pad_size
        else:
            length = len(x_data)
            pad_tensor = np.zeros((pad_size-length, x_data.shape[1]))
            X = np.concatenate((x_data, pad_tensor),axis=0).astype(np.float32)
            
        y_data = self._Y[idx]
        return X, y_data, length

class MSP_1_4_wav2vec_clean_Dataset(Dataset):
    def __init__(self, fea_meta_file):
        target_emo_dict = {'A':0, 'H':1, 'S':2, 'N':3, 'D':4}
        # all_annotation_csv = pd.read_csv('/mnt/sdd/crowpeter/MSP_PODCAST1.8/labels_concensus.csv')
        # all_annotation_df = all_annotation_csv.set_index('FileName').T.to_dict('list')
        all_annotation_df = {} 
        with open('/mnt/sdd/crowpeter/MSP_PODCAST1.4/labels.txt') as f:
            temp = 1
            while temp:
                temp = f.readline()
                if temp[:3] == 'MSP':
                    wav_name = temp.split('; ')[0]
                    emo_label = temp.split('; ')[1]
                    if emo_label in list(target_emo_dict.keys()):
                        all_annotation_df[wav_name] = emo_label
        fea_meta_file = pd.read_csv(fea_meta_file)
        self._Y = []
        self._X = []
        # self._Metric = []
        # self.st_list = []
        # self.ed_list = []
        
        print('Start loading wav2vec feature')
        for i in range(len(fea_meta_file)):
            fea_path = fea_meta_file.iloc[i]['fea_path']
            fea_name = fea_path.split('/')[-1].replace('.pkl','.wav')
            
            emo_str_label = all_annotation_df[fea_name][0]
            assert emo_str_label in list(target_emo_dict.keys())
            
            self._X.append(fea_path)
            self._Y.append(target_emo_dict[emo_str_label])
            # self._Metric.append(fea_meta_file.iloc[i][metric])
            # self.st_list.append(fea_meta_file.iloc[i]['st'])
            # self.ed_list.append(fea_meta_file.iloc[i]['ed'])
            if (i+1)%10000 == 0:
                print('Load '+str(i+1)+' data...')
                  
        assert len(self._X) == len(self._Y)
        print('Finish loading wav2vec feature')
        print('Length:'+str(len(self._Y)))
    def __len__(self):
        return len(self._X) 
    
    def __getitem__(self, idx):
        pad_size = 1098
        # print(st,ed)
        
        x_data = joblib.load(self._X[idx])
        if len(x_data) >= pad_size:
            X = x_data[:pad_size, :]
            length = pad_size
        else:
            length = len(x_data)
            pad_tensor = np.zeros((pad_size-length, x_data.shape[1]))
            X = np.concatenate((x_data, pad_tensor),axis=0).astype(np.float32)
            
        y_data = self._Y[idx]
        return X, y_data, length
    
class MSP_1_4_wav2vec_dst_Dataset(Dataset):
    def __init__(self, meta_file_root, dst_type):
        target_emo_dict = {'A':0, 'H':1, 'S':2, 'N':3, 'D':4}
        # all_annotation_csv = pd.read_csv('/mnt/sdd/crowpeter/MSP_PODCAST1.8/labels_concensus.csv')
        # all_annotation_df = all_annotation_csv.set_index('FileName').T.to_dict('list')
        all_annotation_df = {} 
        with open('/mnt/sdd/crowpeter/MSP_PODCAST1.4/labels.txt') as f:
            temp = 1
            while temp:
                temp = f.readline()
                if temp[:3] == 'MSP':
                    wav_name = temp.split('; ')[0]
                    emo_label = temp.split('; ')[1]
                    if emo_label in list(target_emo_dict.keys()):
                        all_annotation_df[wav_name] = emo_label
        dst_type_box = dst_type.split(',')
        self._Y = []
        self._X = []
        print('Start loading wav2vec feature')
        snr_box = ['0_5','5_10','10_15']
        for dst_type in dst_type_box:
            print('Loading '+dst_type+'...')
            meta_file_path = meta_file_root+'/MSP_1_4_train_meta_wav_'+dst_type+'.csv'
            if dst_type == 'noisy' or dst_type == 'music':
                fea_meta_file = pd.read_csv(meta_file_path)
                temp = []
                for snr in snr_box:
                    temp.append(fea_meta_file[['clean_wav',snr]].rename(columns={snr:'wav_path'}))
                fea_meta_file = pd.concat(temp)
            elif dst_type == 'clean':
                meta_file_path = meta_file_root+'/MSP_1_4_train_clean_fea.csv'
                fea_meta_file = pd.read_csv(meta_file_path)
            elif dst_type == 'fake_rir':
                fea_meta_file = pd.read_csv(meta_file_path)[['clean_wav','fake_rir']].rename(columns={'fake_rir':'wav_path'})
            elif dst_type == 'mit_rir':
                fea_meta_file = pd.read_csv(meta_file_path)[['clean_wav','real_rir']].rename(columns={'real_rir':'wav_path'})
            for i in range(len(fea_meta_file)):
                if dst_type == 'clean':
                    fea_path = fea_meta_file.iloc[i]['fea_path']
                    fea_name = fea_path.split('/')[-1].replace('.pkl','.wav')
                else:
                    fea_path = fea_meta_file.iloc[i]['wav_path'].replace('.wav','.pkl').replace('wav','feature')
                    fea_name = fea_meta_file.iloc[i]['clean_wav'].split('/')[-1]
                
                emo_str_label = all_annotation_df[fea_name][0]
                assert emo_str_label in list(target_emo_dict.keys())
                
                self._X.append(fea_path)
                self._Y.append(target_emo_dict[emo_str_label])
                # self._Metric.append(fea_meta_file.iloc[i][metric])
                # self.st_list.append(fea_meta_file.iloc[i]['st'])
                # self.ed_list.append(fea_meta_file.iloc[i]['ed'])
                if len(self._X)%10000 == 0:
                    print('Load '+str(len(self._X))+' data...')
                      
            assert len(self._X) == len(self._Y)
        print('Finish loading wav2vec feature')
        print('Length:'+str(len(self._Y)))
    def __len__(self):
        return len(self._X) 
    
    def __getitem__(self, idx):
        pad_size = 1098
        # print(st,ed)
        
        x_data = joblib.load(self._X[idx])
        if len(x_data) >= pad_size:
            X = x_data[:pad_size, :]
            length = pad_size
        else:
            length = len(x_data)
            pad_tensor = np.zeros((pad_size-length, x_data.shape[1]))
            X = np.concatenate((x_data, pad_tensor),axis=0).astype(np.float32)
            
        y_data = self._Y[idx]
        return X, y_data, length

def seq_collate_pad(batch):
    """
    Returns:
        [B, F, T (Longest)]
    """
    X_list = [torch.tensor(item['X'], dtype=torch.float) for item in batch]
    Y_list = [item['Y'] for item in batch]
    seq_lengths = torch.LongTensor([len(seq) for seq in X_list])
    Y_list = torch.LongTensor(np.array(Y_list))
    X_list_pad = pad_sequence(X_list ,batch_first=True)

    return X_list_pad, Y_list, seq_lengths

class MSP_1_4_wav2vec_fake_Dataset(Dataset):
    def __init__(self, meta_file_root, dst_type):
        target_emo_dict = {'A':0, 'H':1, 'S':2, 'N':3, 'D':4}
        # all_annotation_csv = pd.read_csv('/mnt/sdd/crowpeter/MSP_PODCAST1.8/labels_concensus.csv')
        # all_annotation_df = all_annotation_csv.set_index('FileName').T.to_dict('list')
        all_annotation_df = {} 
        with open('/mnt/sdd/crowpeter/MSP_PODCAST1.4/labels.txt') as f:
            temp = 1
            while temp:
                temp = f.readline()
                if temp[:3] == 'MSP':
                    wav_name = temp.split('; ')[0]
                    emo_label = temp.split('; ')[1]
                    if emo_label in list(target_emo_dict.keys()):
                        all_annotation_df[wav_name] = emo_label
        # dst_type_box = dst_type.split(',')
        self._Y = []
        self._X = []
        print('Start loading wav2vec feature')
        # snr_box = ['0_5','5_10','10_15']
        dir_box = [path for path in glob.glob(meta_file_root+'/'+dst_type+'*')]
        dir_box.append(meta_file_root+'/'+'clean')
        for fea_dir in dir_box:
            print('Loading from '+fea_dir+' ...')
            fea_type = fea_dir.split('/')[-1]
            for fea_path in glob.glob(fea_dir+'/*'):
                if fea_type == 'clean':
                    fea_name = fea_path.split('/')[-1].replace('.pkl','.wav')
                else:
                    # print(fea_path)
                    fea_name = '_'.join(fea_path.split('/')[-1].split('_')[:-1])+'.wav'
                    # print(fea_name)
            # meta_file_path = meta_file_root+'/MSP_1_4_train_meta_wav_'+dst_type+'.csv'
            # if dst_type == 'noisy' or dst_type == 'music':
            #     fea_meta_file = pd.read_csv(meta_file_path)
            #     temp = []
            #     for snr in snr_box:
            #         temp.append(fea_meta_file[['clean_wav',snr]].rename(columns={snr:'wav_path'}))
            #     fea_meta_file = pd.concat(temp)
            # elif dst_type == 'clean':
            #     meta_file_path = meta_file_root+'/MSP_1_4_train_clean_fea.csv'
            #     fea_meta_file = pd.read_csv(meta_file_path)
            # elif dst_type == 'fake_rir':
            #     fea_meta_file = pd.read_csv(meta_file_path)[['clean_wav','fake_rir']].rename(columns={'fake_rir':'wav_path'})
            # elif dst_type == 'mit_rir':
            #     fea_meta_file = pd.read_csv(meta_file_path)[['clean_wav','real_rir']].rename(columns={'real_rir':'wav_path'})
            # for i in range(len(fea_meta_file)):
                # if dst_type == 'clean':
                #     fea_path = fea_meta_file.iloc[i]['fea_path']
                #     fea_name = fea_path.split('/')[-1].replace('.pkl','.wav')
                # else:
                #     fea_path = fea_meta_file.iloc[i]['wav_path'].replace('.wav','.pkl').replace('wav','feature')
                #     fea_name = fea_meta_file.iloc[i]['clean_wav'].split('/')[-1]
                
                emo_str_label = all_annotation_df[fea_name][0]
                assert emo_str_label in list(target_emo_dict.keys())
                
                self._X.append(fea_path)
                self._Y.append(target_emo_dict[emo_str_label])
                # self._Metric.append(fea_meta_file.iloc[i][metric])
                # self.st_list.append(fea_meta_file.iloc[i]['st'])
                # self.ed_list.append(fea_meta_file.iloc[i]['ed'])
                if len(self._X)%10000 == 0:
                    print('Load '+str(len(self._X))+' data...')
                      
            assert len(self._X) == len(self._Y)
        print('Finish loading wav2vec feature')
        print('Length:'+str(len(self._Y)))
    def __len__(self):
        return len(self._X) 
    
    def __getitem__(self, idx):
        pad_size = 1098
        # print(st,ed)
        
        x_data = joblib.load(self._X[idx])
        if len(x_data) >= pad_size:
            X = x_data[:pad_size, :]
            length = pad_size
        else:
            length = len(x_data)
            pad_tensor = np.zeros((pad_size-length, x_data.shape[1]))
            X = np.concatenate((x_data, pad_tensor),axis=0).astype(np.float32)
            
        y_data = self._Y[idx]
        return X, y_data, length

class IEMOCAP_wav2vec2_clean_Dataset(Dataset):
    def __init__(self, fea_meta_file,  label_csv_file, test_sess, label_type, mode, max_len, label_weighted=None):
        fea_meta_file = pd.read_csv(fea_meta_file)
        self._x_fea_dict = {path.split('/')[-1].replace('.pkl',''):path for path in fea_meta_file['fea_path']}
        self._y_csv = pd.read_csv(label_csv_file)
        self.test_sess = test_sess
        self.max_len = max_len
        self.mode = mode
        self._Y = []
        self._X = []
        print('Start loading wav2vec '+str(test_sess)+' '+str(mode)+' feature')
        for i in range(len(self._y_csv)):
            
            if self._y_csv.iloc[i]['Name_String'][:5] != self.test_sess[:5] and mode == 'train':
                if label_type == 'Emotion':
                    if self._y_csv.iloc[i][label_type]=='sad':
                        self._Y.append(0)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='neu':
                        self._Y.append(1)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='hap' or self._y_csv.iloc[i][label_type]=='exc':
                        self._Y.append(2)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='ang':
                        self._Y.append(3)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                        
            elif self._y_csv.iloc[i]['Name_String'][:6] == self.test_sess+self.mode and mode == 'F':
                if label_type == 'Emotion':
                    if self._y_csv.iloc[i][label_type]=='sad':
                        self._Y.append(0)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='neu':
                        self._Y.append(1)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='hap' or self._y_csv.iloc[i][label_type]=='exc':
                        self._Y.append(2)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='ang':
                        self._Y.append(3)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                        
            elif self._y_csv.iloc[i]['Name_String'][:6] == self.test_sess+self.mode and mode == 'M':
                if label_type == 'Emotion':
                    if self._y_csv.iloc[i][label_type]=='sad':
                        self._Y.append(0)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='neu':
                        self._Y.append(1)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='hap' or self._y_csv.iloc[i][label_type]=='exc':
                        self._Y.append(2)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='ang':
                        self._Y.append(3)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    
        assert len(self._Y) == len(self._X)
        if label_weighted:
            self.label_weight = [item[1]/len(self._Y) for item in sorted(Counter(self._Y).items())]
            print(sorted(Counter(self._Y).items()))
        print('Finish loading wav2vec '+str(test_sess)+' '+str(mode)+' feature')
        print('Length:'+str(len(self._Y)))
        
    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        x_data = joblib.load(self._X[idx])
        if len(x_data) > self.max_len:
            x_data = x_data[:self.max_len,:]
        y_data = self._Y[idx]
        return dict(X=x_data,Y=y_data)
     
class IEMOCAP_wav2vec2_noisy_Dataset(Dataset):
    def __init__(self, fea_meta_file,  label_csv_file, test_sess, snr_range, label_type, mode, max_len, label_weighted=None):
        fea_meta_file = pd.read_csv(fea_meta_file)
        # self._x_fea_dict = {path.split('/')[-1].replace('.pkl',''):path for path in fea_meta_file['fea_path']}
        self._x_fea_dict = {}
        for path in fea_meta_file['fea_path']:
            name = '_'.join(path.split('/')[-1].replace('.pkl','').split('_')[:-1])
            if snr_range in os.path.dirname(path):
                self._x_fea_dict[name] = path
        self._y_csv = pd.read_csv(label_csv_file)
        self.test_sess = test_sess
        self.max_len = max_len
        self.mode = mode
        self._Y = []
        self._X = []
        print('Start loading wav2vec '+str(test_sess)+' '+str(mode)+' feature')
        for i in range(len(self._y_csv)):
            
            if self._y_csv.iloc[i]['Name_String'][:5] != self.test_sess[:5] and mode == 'train':
                if label_type == 'Emotion':
                    if self._y_csv.iloc[i][label_type]=='sad':
                        self._Y.append(0)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='neu':
                        self._Y.append(1)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='hap' or self._y_csv.iloc[i][label_type]=='exc':
                        self._Y.append(2)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='ang':
                        self._Y.append(3)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                        
            elif self._y_csv.iloc[i]['Name_String'][:6] == self.test_sess+self.mode and mode == 'F':
                if label_type == 'Emotion':
                    if self._y_csv.iloc[i][label_type]=='sad':
                        self._Y.append(0)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='neu':
                        self._Y.append(1)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='hap' or self._y_csv.iloc[i][label_type]=='exc':
                        self._Y.append(2)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='ang':
                        self._Y.append(3)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                        
            elif self._y_csv.iloc[i]['Name_String'][:6] == self.test_sess+self.mode and mode == 'M':
                if label_type == 'Emotion':
                    if self._y_csv.iloc[i][label_type]=='sad':
                        self._Y.append(0)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='neu':
                        self._Y.append(1)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='hap' or self._y_csv.iloc[i][label_type]=='exc':
                        self._Y.append(2)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    elif self._y_csv.iloc[i][label_type]=='ang':
                        self._Y.append(3)
                        self._X.append(self._x_fea_dict[self._y_csv.iloc[i]['Name_String'].replace('.txt','')])
                    
        assert len(self._Y) == len(self._X)
        if label_weighted:
            self.label_weight = [item[1]/len(self._Y) for item in sorted(Counter(self._Y).items())]
            print(sorted(Counter(self._Y).items()))
        print('Finish loading wav2vec '+str(test_sess)+' '+str(mode)+' feature')
        print('Length:'+str(len(self._Y)))
        
    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        x_data = joblib.load(self._X[idx])
        if len(x_data) > self.max_len:
            x_data = x_data[:self.max_len,:]
        y_data = self._Y[idx]
        return dict(X=x_data,Y=y_data)
    
class MSP_1_4_wav2vec_noisy_Dataset(Dataset):
    def __init__(self, fea_meta_file, snr_range):
        target_emo_dict = {'A':0, 'H':1, 'S':2, 'N':3, 'D':4}
        # all_annotation_csv = pd.read_csv('/mnt/sdd/crowpeter/MSP_PODCAST1.8/labels_concensus.csv')
        # all_annotation_df = all_annotation_csv.set_index('FileName').T.to_dict('list')
        all_annotation_df = {} 
        with open('/mnt/sdd/crowpeter/MSP_PODCAST1.4/labels.txt') as f:
            temp = 1
            while temp:
                temp = f.readline()
                if temp[:3] == 'MSP':
                    wav_name = temp.split('; ')[0]
                    emo_label = temp.split('; ')[1]
                    if emo_label in list(target_emo_dict.keys()):
                        all_annotation_df[wav_name] = emo_label
        fea_meta_file = pd.read_csv(fea_meta_file)
        self._Y = []
        self._X = []
        # self._Metric = []
        # self.st_list = []
        # self.ed_list = []
        
        print('Start loading wav2vec feature')
        for i in range(len(fea_meta_file)):
            fea_path = fea_meta_file.iloc[i]['fea_path']
            # fea_name = fea_path.split('/')[-1].replace('.pkl','.wav')
            fea_name = '_'.join(fea_path.split('/')[-1].split('_')[:-1])+'.wav'
            emo_str_label = all_annotation_df[fea_name][0]
            assert emo_str_label in list(target_emo_dict.keys())
            if snr_range in os.path.dirname(fea_path):
                self._X.append(fea_path)
                self._Y.append(target_emo_dict[emo_str_label])
            # self._Metric.append(fea_meta_file.iloc[i][metric])
            # self.st_list.append(fea_meta_file.iloc[i]['st'])
            # self.ed_list.append(fea_meta_file.iloc[i]['ed'])
            if (i+1)%10000 == 0:
                print('Load '+str(i+1)+' data...')
                  
        assert len(self._X) == len(self._Y)
        print('Finish loading wav2vec feature')
        print('Length:'+str(len(self._Y)))
    def __len__(self):
        return len(self._X) 
    
    def __getitem__(self, idx):
        pad_size = 1098
        # print(st,ed)
        
        x_data = joblib.load(self._X[idx])
        if len(x_data) >= pad_size:
            X = x_data[:pad_size, :]
            length = pad_size
        else:
            length = len(x_data)
            pad_tensor = np.zeros((pad_size-length, x_data.shape[1]))
            X = np.concatenate((x_data, pad_tensor),axis=0).astype(np.float32)
            
        y_data = self._Y[idx]
        return X, y_data, length
#%%
# fea_meta_file = '/homes/ssd0/crowpeter/SE_SER/data_meta_file/IEMOCAP_ESC50_fea_meta.csv'
# label_csv_file = '/homes/ssd0/crowpeter/SE_SER/IEMOCAP/IEMOCAP_Org_Emo.csv'
# test_sess = 'Ses01F' 
# label_type = 'Emotion'
# # noise_level = '20'
# # noise_type = 'Animals'
# # mode='test'
# ts_dataset = IEMOCAP_wav2vec_noisy_Dataset(fea_meta_file, label_csv_file, test_sess, label_type, mode='test', noise_level='20' , noise_type='Animals')
# # tr_loader = DataLoader(tr_dataset, batch_size=16, shuffle=True, collate_fn=seq_collate_pad)
# # # tr_loader = DataLoader(tr_dataset, batch_size=2, shuffle=True)
# # c=0
# for step, batch in enumerate(ts_dataset):
#     aa=batch
#     break
#%%
# meta_file_root = '/homes/ssd0/crowpeter/SE_SER/robust_meta_file'
# dst_type = 'clean,noisy,music,fake_rir,mit_rir'
# # test_sess = 'Ses01F' 
# # label_type = 'Emotion'
# # noise_level = '20'
# # noise_type = 'Animals'
# # mode='test'
# ts_dataset = MSP_1_4_wav2vec_dst_Dataset(meta_file_root, dst_type)
# # tr_loader = DataLoader(tr_dataset, batch_size=16, shuffle=True, collate_fn=seq_collate_pad)
# # # tr_loader = DataLoader(tr_dataset, batch_size=2, shuffle=True)
# # c=0
# for step, batch in enumerate(ts_dataset):
#     aa=batch
#     break
#%%
# tr_dataset = MSP_1_4_wav2vec_fake_Dataset('/mnt/sdd/crowpeter/MSP_1_4_emo_train/feature','fake_v4_pesq')
