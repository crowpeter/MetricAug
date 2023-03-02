#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:51:08 2022

@author: crowpeter
"""

import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
import numpy as np
from collections import Counter
IN_PATH = 'parser_input_gmm/MELD_train_all_noisy_aug_data.csv'
OUT_PATH = 'parser_input_gmm/MELD_train_all_noisy_aug_data.csv'
final_df = pd.read_csv(IN_PATH)
del final_df['Unnamed: 0']
#%%
metrics_list = ['fwSNRseg', 'stoi', 'pesq']
# metrics_list = ['stoi']
k = 5
for metric in metrics_list:
    if metric == 'pesq':
        final_df = final_df.loc[final_df['pesq'] != -1]
    elif metric == 'stoi':
        final_df = final_df.loc[final_df['stoi'] != 1e-5]
        
    final_df = final_df.sort_values(by=[metric],ascending=False).copy()
    sess_list = []
    rank_idx = []

    gm = BayesianGaussianMixture(n_components=k, random_state=2022, max_iter=100).fit(np.reshape(list(final_df[metric]),(len(list(final_df[metric])),1)))
    rank_dict = {}
    original_rank_mean = gm.means_
    new_rank = np.argsort(-original_rank_mean[:,0])
    for i in range(k):
        # rank_dict[i] = new_rank[i]
        rank_dict[new_rank[i]] = i
    for i in range(len(final_df)):
        metric_value = np.array(final_df.iloc[i][metric]).reshape(1,-1)
        rank = gm.predict(metric_value)[0]
        rank_idx.append(rank_dict[rank])

    assert len(rank_idx) == len(final_df)
    final_df['gmm_rank_'+metric] = rank_idx
    cc = Counter(rank_idx)
    print('train', metric)
    print(gm.means_)
    print(cc)
    final_df.to_csv(OUT_PATH)
    
    # read validation set
    va_path = IN_PATH.replace('train','validation')
    va_out_path = OUT_PATH.replace('train','validation')
    va_df = pd.read_csv(va_path)
    
    rank_idx = []
    for i in range(len(va_df)):
        metric_value = np.array(va_df.iloc[i][metric]).reshape(1,-1)
        rank = gm.predict(metric_value)[0]
        rank_idx.append(rank_dict[rank])

    assert len(rank_idx) == len(va_df)
    va_df['gmm_rank_'+metric] = rank_idx
    cc = Counter(rank_idx)
    print('validation', metric)
    print(cc)    
    va_df.to_csv(va_out_path)
    
    # read_testing set
    ts_path = IN_PATH.replace('train','test')
    ts_out_path = OUT_PATH.replace('train','test')
    ts_df = pd.read_csv(ts_path)
    
    rank_idx = []
    for i in range(len(ts_df)):
        metric_value = np.array(ts_df.iloc[i][metric]).reshape(1,-1)
        rank = gm.predict(metric_value)[0]
        rank_idx.append(rank_dict[rank])

    assert len(rank_idx) == len(ts_df)
    ts_df['gmm_rank_'+metric] = rank_idx
    cc = Counter(rank_idx)
    print('test', metric)
    print(cc)
    ts_df.to_csv(ts_out_path)
    
