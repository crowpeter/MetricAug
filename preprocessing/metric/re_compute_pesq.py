#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 21:55:27 2022

@author: crowpeter
"""

from pesq import pesq
import pandas as pd
from scipy.io import wavfile
import numpy as np
import glob
import soundfile as sf
int16_max = (2 ** 15) - 1
def normalize_volume(wav, target_dBFS, increase_only=True, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))
#%%

INPATH = 'MELD_data_process/with_se_metric/MELD_train_meta_wav_music_0_30_with_metric.csv'
OUTPATH = 'MELD_data_process/pesq_mdf/'
file_name = INPATH.split('/')[-1]
meta_file = pd.read_csv(INPATH)
err_box = [path for path in glob.glob('MELD_pesq_err/train/*.wav')]+\
        [path for path in glob.glob('MELD_pesq_err/validation/*.wav')]+\
        [path for path in glob.glob('MELD_pesq_err/test/*.wav')]
#%%
c = 0
still_err_box = []
for i in range(len(meta_file)):
    row = meta_file.iloc[i]
    pesq_check = row['pesq']
    clean_wav_name = row['clean_wav'].split('/')[-1]
    noisy_wav_path = row['wav']
    if pesq_check == -1:
        # really speical case
        if clean_wav_name in err_box:
            try:
                # pesq compute
                wav = row['clean_wav']
                out_path = row['wav']
                ref, rate = sf.read(wav)
                deg, rate = sf.read(out_path)

                ref = normalize_volume(ref, 50)
                ref = ref*np.power(2, 15)
                ref = ref.astype(np.int16)

                deg = normalize_volume(deg, 50)
                deg = deg*np.power(2, 15)
                deg = deg.astype(np.int16)

                ref_q_temp = np.concatenate((np.zeros((rate*20, )), ref)).astype(np.int16)
                deg_q_temp = np.concatenate((np.zeros((rate*20, )), deg)).astype(np.int16)
                pesq_value = pesq(rate, ref_q_temp, deg_q_temp, 'wb')
                meta_file.at[i,'pesq'] = pesq_value
            except:
                pesq_value = -1
                still_err_box.append(out_path)
                print(out_path+'\t'+'still Wrong!!!')
                c += 1
        else:
        # break
            wav = row['clean_wav']
            out_path = row['wav']
            
            # pesq compute
            rate, ref = wavfile.read(wav)
            rate, deg = wavfile.read(out_path)
            
            try:
                ref_q_temp = np.concatenate((np.zeros(ref.shape), ref)).astype(np.int16)
                deg_q_temp = np.concatenate((np.zeros(deg.shape), deg)).astype(np.int16)
                pesq_value = pesq(rate, ref_q_temp, deg_q_temp, 'wb')
                meta_file.at[i,'pesq'] = pesq_value
                print(meta_file.iloc[i]['pesq'])
            except:
                pesq_value = -1
                still_err_box.append(out_path)
                print(out_path+'\t'+'still Wrong!!!')
                c += 1

# if c == 0:
meta_file.to_csv(OUTPATH+'/'+file_name)
print(OUTPATH+file_name+' save!!')