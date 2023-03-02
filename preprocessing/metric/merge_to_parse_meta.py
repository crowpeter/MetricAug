#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:07:10 2022

@author: crowpeter
"""

import pandas as pd
import glob
import os
DATA_FORMAT = 'superset'
#%% for noisy aug
if DATA_FORMAT == 'superset':
    split_sets = ['train', 'test', 'validation']
    for split_set in split_sets:
        noisy = pd.read_csv('MELD_'+split_set+'_meta_wav_noisy_0_30_with_metric.csv')
        music = pd.read_csv('MELD_'+split_set+'_meta_wav_music_0_30_with_metric.csv')
        all_meta = pd.concat([noisy, music])
        all_meta.to_csv('MELD_data_process/parser_input/MELD_'+split_set+'_all_noisy_aug_data.csv')
#%% for 0 5 10
elif DATA_FORMAT == '0_5_10':
    clean_meta_root = 'MELD_data_process/'
    split_set = 'validation'
    ROOT = 'MELD_emo_'+split_set+'/wav/'
    OUT_ROOT = 'MELD_data_process/parser_input/'+'MELD_emo_'+split_set+'_0_5_10.csv'
    clean_meta = pd.read_csv(clean_meta_root+'/MELD_emo_'+split_set+'_clean.csv')
    typees = ['noisy', 'music']
    dBs = ['0', '5', '10']
    
    wav_emo = {}
    for i in range(len(clean_meta)):
        wav_emo[clean_meta.iloc[i]['wav_path']]=clean_meta.iloc[i]['emo']
    
    final_df = {}
    wav_list = []
    clean_wav_list = []
    emo_list = []
    fea_path_list = []
    snr_list = []
    for typee in typees:
        for db in dBs:
            wav_root = ROOT+'/'+typee+'/'+db
            clean_root = 'MELD_emo_'+split_set+'/wav/clean/'
            for wav_path in glob.glob(wav_root+'/*.wav'):
                # break
                clean_wav_name = '_'.join(wav_path.split('/')[-1].split('_')[0:2])+'.wav'
                feature_path = wav_path.replace('.wav', '.pkl').replace('wav', 'feature')
                if os.path.exists(feature_path):
                    clean_wav_list.append(clean_root+clean_wav_name)
                    emo_list.append(wav_emo[clean_root+clean_wav_name])
                    fea_path_list.append(feature_path)
                    wav_list.append(wav_path)
                    snr_list.append(int(db))
    final_df['wav'] = wav_list
    final_df['clean_wav'] = clean_wav_list
    final_df['emo'] = emo_list
    final_df['fea_path'] = fea_path_list
    final_df['snr_level'] = snr_list
    final_df = pd.DataFrame(final_df)
    final_df.to_csv(OUT_ROOT)