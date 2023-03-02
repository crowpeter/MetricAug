#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 20:12:26 2022

@author: crowpeter
"""


import pandas as pd

IN_PATH = 'MELD_test_all_noisy_aug_data.csv'
OUT_PATH = 'MELD_test_all_noisy_aug_data.csv'
final_df = pd.read_csv(IN_PATH)
del final_df['Unnamed: 0']
#%%
metrics_list = ['fwSNRseg', 'stoi', 'pesq']
# metrics_list = ['stoi']
k = 5
for metric in metrics_list:
    rank_idx = []
    rank_range = [[i,(i+len(final_df)//k)] for i in range(0, len(final_df), len(final_df)//k)]
    final_df = final_df.sort_values(by=[metric],ascending=False).copy()
    for i in range(len(final_df)):
        for r, (low,high) in enumerate(rank_range):
            if i < high and i >= low:
                rank_idx.append(r)
                break

    assert len(rank_idx) == len(final_df)
    

    final_df['rank_'+metric] = rank_idx
    final_df.to_csv(OUT_PATH)
