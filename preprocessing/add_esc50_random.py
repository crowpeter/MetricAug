#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 21:56:54 2022

@author: crowpeter
"""

import soundfile as sf
import glob
import argparse
import os
import numpy as np
from pathlib import Path
import random
import librosa
import pandas as pd

parser = argparse.ArgumentParser(description='add noise config')
parser.add_argument('--speech_dataset', default='MELD',
                    help='speech dir')
parser.add_argument('--noise_dataset', default='ESC50',
                    help='noise dir')
parser.add_argument('--data_set', default='test',
                    help='training or testing')

args = parser.parse_args()
def snr_noise_mix(s ,ns, snr):
    snr = 10**(snr/10.0)
    speech_power = np.sum(np.power(s,2))/len(s)
    noise_power = np.sum(np.power(ns,2))/len(ns)
    noise_update = ns / np.sqrt(snr * noise_power/speech_power)
    return noise_update + s

def s_n_p_align(s, ns):
    if len(s) > len(ns):
        new_ns = []
        count = len(s)//len(ns)+1
        for i in range(count):
            new_ns.append(ns)
        new_ns = np.hstack(new_ns)[:len(s)]
    elif len(s) <= len(ns):
        new_ns = ns[:len(s)]
    return s, new_ns

#%%
    
speech_dict = {}
                
if args.speech_dataset == 'MSP':
    wav_root_path = 'MSP_PODCAST1.8/Audios/'
    use_list = []
    target_emo_label = ['A', 'H', 'S', 'N', 'D']
    wav_list = [path for path in glob.glob('MSP_emo_'+args.data_set+'/wav/clean/*')]
    speech_dict[args.data_set] = wav_list
    if args.data_set == 'train':
        sesses = ['Train']
    elif args.data_set == 'test1':
        sesses = ['Test1']
    elif args.data_set == 'test2':
        sesses = ['Test2']
    elif args.data_set == 'validation':
        sesses = ['Validation']
    y_csv = pd.read_csv('MSP_PODCAST1.8/labels_concensus.csv')
    
    for wav_idx in range(len(y_csv)):
        if y_csv.iloc[wav_idx]['EmoClass'] in target_emo_label:
            if y_csv.iloc[wav_idx]['Split_Set'] == sesses[0]:
                use_list.append(y_csv.iloc[wav_idx]['FileName'])

    for sess in sesses:
        sess_idx = sess.split('/')[-1]
        speech_dict[sess_idx] = []
        for wav_path in glob.glob(wav_root_path+'/*.wav'):
            wav_name = wav_path.split('/')[-1]
            if wav_name in use_list:
                speech_dict[sess_idx].append(wav_path)

elif args.speech_dataset == 'MELD':
    target_emo_label = ['anger', 'joy', 'sadness', 'neutral']
    wav_list = []
    if args.data_set == 'train':
        wav_root_path = 'MELD_emo_train/wav/clean/'
        meta_csv = pd.read_csv('MELD/train_sent_emo.csv')
    elif args.data_set == 'validation':
        wav_root_path = 'MELD_emo_validation/wav/clean/'
        meta_csv = pd.read_csv('MELD/dev_sent_emo.csv')        
    elif args.data_set == 'test':
        wav_root_path = 'MELD_emo_test/wav/clean/'
        meta_csv = pd.read_csv('MELD/test_sent_emo.csv') 
    meta_csv = meta_csv
    for i in range(len(meta_csv)):
        if meta_csv.iloc[i]['Emotion'] in target_emo_label:
            dia = str(meta_csv.iloc[i]['Dialogue_ID'])
            utt = str(meta_csv.iloc[i]['Utterance_ID'])
            if os.path.exists(wav_root_path+'dia'+dia+'_utt'+utt+'.wav'):
                wav_list.append(wav_root_path+'dia'+dia+'_utt'+utt+'.wav')
    speech_dict[args.data_set] = wav_list
#%%
noise_dict = {}

if args.noise_dataset == 'ESC50':
    n_root_path = 'ESC-50-master/audio/'
    meta_csv = pd.read_csv('ESC-50-master/meta/esc50.csv')
    for noise_idx in range(len(meta_csv)):
        wav_path = n_root_path+meta_csv.iloc[noise_idx]['filename']
        cate = int(wav_path.split('/')[-1].split('-')[-1].replace('.wav',''))
        
        if cate//10 not in noise_dict.keys():
            noise_dict[cate//10] = []
        noise_dict[cate//10].append(wav_path)
    noise_index_type = {0:'Animals',\
                        1:'Natural_water',\
                        2:'Human_non_speech',\
                        3:'Interior_domestic',\
                        4:'Exterior_urban_noises'}
        
elif args.noise_dataset == 'noisy':
    noise_dict = {path.split('/')[-1].replace('.wav',''):path for path in glob.glob('musan/noise/*/*.wav', recursive=True)}

elif args.noise_dataset == 'music':
    noise_dict = {path.split('/')[-1].replace('.wav',''):path for path in glob.glob('musan/music/*/*.wav', recursive=True)}
#%%        
snr_ratio_box = ['0_30']
op_se_metric_df = {'clean_wav':[], 'wav':[], 'snr':[]}
    
pesq_error = []
for ses in speech_dict.keys():
    for wav in speech_dict[ses]:
        wav_name = wav.split('/')[-1].replace('.wav','')
        s, sr = sf.read(wav)
        if sr != 16000:
            s, sr = librosa.load(wav, sr = 16000)
        for snr_range in snr_ratio_box:

            noisy_select = np.random.randint(len(noise_dict))
            magic_number = np.random.randint(len(noise_dict[noisy_select]))
            n_path = noise_dict[noisy_select][magic_number]
            n_name = n_path.split('/')[-1].replace('.wav','')
            ns, nsr = librosa.load(n_path,sr = 16000)
            s, ns = s_n_p_align(s, ns)
            
            snr_low, snr_up = snr_range.split('_')
            snr = random.uniform(int(snr_low), int(snr_up))
            path = Path('MELD_emo_'+args.data_set+'/wav/random/'+snr_range+'/')
            path.mkdir(parents=True, exist_ok=True)
            op = snr_noise_mix(s, ns, snr)
            out_path = str(path.as_posix())+'/'+wav_name+'_'+n_name+'.wav'
            sf.write(out_path, op, 16000)
    
op_se_metric_df = pd.DataFrame.from_dict(op_se_metric_df)
op_se_metric_df.to_csv('MELD_data_process'+'/MELD_'+args.data_set+'_meta_wav_random_noisy.csv')