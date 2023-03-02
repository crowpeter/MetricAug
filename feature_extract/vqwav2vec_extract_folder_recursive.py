#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 18:43:24 2022

@author: crowpeter
"""

import os
import argparse
from pathlib import Path
import joblib
import torch
import pandas as pd
from vqwav2vec_extract import AudioDataPreprocessing
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='MELD_emo_train/wav/music/',\
                    help='the root folder you want to extract wav recursively')
parser.add_argument("--dataset_name", type=str, default='MELD',\
                    help='dataset name for special case')
parser.add_argument("--save_meta_dir", type=str, default="MELD_feature_meta/",\
                    help='the root folder you want to create a folder "feature" to save feature')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
#%%
# call wav2vec model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W2V_NAME = 'vqw2v'
w2v_path = './vq-wav2vec_kmeans.pt'
bert_filename = './bert_kmeans.pt'
processor = AudioDataPreprocessing(W2V_NAME, w2v_path, bert_filename, device)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#%%
# load data
data_name =  args.data_dir.split('/')[-2]
op_df = {'wav_path':[], 'fea_path':[]}
wav_list = [str(path.as_posix()) for path in Path(args.data_dir).rglob('*.wav')]
out_list = [path.replace('.wav','.pkl').replace('wav','feature') for path in wav_list]
#%%
for c, wav_path in enumerate(wav_list):
    print(wav_path)

    audio_features = processor.preprocess_audio_file(wav_path)
    audio_z = processor.get_audio_feature(audio_features)
    
    out_path = '/'.join(out_list[c].split('/')[:-1])
    path = Path(out_path)
    
    if not os.path.exists(str(path.as_posix())):
        path.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(audio_z, out_list[c])
    print(out_list[c]+'\t'+'saved!')
    op_df['wav_path'].append(wav_path)
    op_df['fea_path'].append(out_list[c])
    if c % 1000 == 0:
        torch.cuda.empty_cache()
        
op_df = pd.DataFrame.from_dict(op_df)
typee = args.data_dir.split('/')[-3]
op_df.to_csv(args.save_meta_dir+'/'+typee+'_'+data_name+'_fea.csv')