#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 21:55:13 2022

@author: crowpeter
"""

import soundfile as sf
import argparse
import numpy as np
import pandas as pd
from pesq import pesq
from scipy.io import wavfile
from utils import cal_metric_single
import datetime
import time
parser = argparse.ArgumentParser(description='add noise config')
parser.add_argument('--wav_meta_csv', default='MELD_data_process/MELD_test_meta_wav_noisy_0_30.csv',
                    help='wav meta dir')
parser.add_argument('--out_folder', default='MELD_data_process/with_se_metric/',
                    help='noise dir')
parser.add_argument('--gpu_idx', default='0',
                    help='training or testing')
args = parser.parse_args()
#%%
meta_csv = pd.read_csv(args.wav_meta_csv)
out_meta_name = args.wav_meta_csv.split('/')[-1].replace('.csv','')+'_with_metric.csv'
total_step = len(meta_csv)
metrics_list = ['fwSNRseg', 'stoi']
total_step = int(total_step)
op_se_metric_df = {'clean_wav':[],'wav':[], 'stoi':[], 'pesq':[], 'fwSNRseg':[], 'snr':[], 'fea_path':[], 'emo':[]}
start_time = time.time()
c=0
pesq_error = []

#%%
for idx in range(len(meta_csv)):
    wav = meta_csv.iloc[idx]['clean_wav']
    out_path = meta_csv.iloc[idx]['wav']
    snr = meta_csv.iloc[idx]['snr']
    wav_name = wav.split('/')[-1].replace('.wav','')
    dis_name = out_path.split('/')[-1]
    
    fea_path = meta_csv.iloc[idx]['fea_path']
    emo = meta_csv.iloc[idx]['emo']
    # print(wav)
    if (c+1) % 10 == 0:
        et = time.time() - start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "Elapsed [{}], Iteration [{}/{}]".format(et, c+1, total_step)
        print(log)
        op_se_metric_df_ckpt = pd.DataFrame.from_dict(op_se_metric_df)
        op_se_metric_df_ckpt.to_csv(args.out_folder+'/'+out_meta_name)
    #%% metric compute
    ref, rate = sf.read(wav)
    deg, rate = sf.read(out_path)
    
    # 'fwSNRseg', 'stoi'
    metrics = cal_metric_single(ref.astype(np.float32), deg.astype(np.float32), dis_name, fs=16000, metric=metrics_list)
    
    # pesq compute
    rate, ref = wavfile.read(wav)
    rate, deg = wavfile.read(out_path)
    try:
        pesq_value = pesq(rate, ref, deg, 'wb')
    except:
        pesq_value = -1
        print(out_path+'\t'+'Wrong!!!')

    op_se_metric_df['stoi'].append(metrics['stoi'].values[0])
    op_se_metric_df['pesq'].append(pesq_value)
    op_se_metric_df['fwSNRseg'].append(metrics['fwSNRseg'].values[0])
    op_se_metric_df['snr'].append(snr)
    op_se_metric_df['wav'].append(out_path)
    op_se_metric_df['clean_wav'].append(wav)
    op_se_metric_df['fea_path'].append(fea_path)
    op_se_metric_df['emo'].append(emo)
    c+=1
op_se_metric_df_ckpt = pd.DataFrame.from_dict(op_se_metric_df)
op_se_metric_df_ckpt.to_csv(args.out_folder+'/'+out_meta_name)
