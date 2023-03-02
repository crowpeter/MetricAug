#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:22:44 2021

@author: johnli
"""

import os
import glob
import torch
import soundfile as sf
from fairseq.models.wav2vec import Wav2VecModel, Wav2Vec2Model


class AudioDataPreprocessing():
    def __init__(self, name, w2v_path, bert_filename, devices):

        cp = torch.load(w2v_path)
        if name == "vqw2v":
            self.model = Wav2VecModel.build_model(cp['args'], task=None)
        else:
            self.model = Wav2Vec2Model.build_model(cp['args'], task=None)
        self.model.load_state_dict(cp['model'])
        self.device = devices
        self.model.to(self.device)
        self.model.eval()

    def indices_to_string(self,idxs):
        # based on fairseq/examples/wav2vec/vq-wav2vec_featurize.py
        return "<s>"+" " +" ".join("-".join(map(str, a.tolist()))
                                   for a in idxs.squeeze(0)
                                   )
    def preprocess_audio_file(self, filename):
        feats_audio, sr = sf.read(filename)

        assert feats_audio.ndim == 1, feats_audio.ndim
        feats_audio = torch.FloatTensor(feats_audio).reshape((1, -1))
        print("Audio: ",feats_audio.size())

        return feats_audio

    def preprocess_data(self, data_path, base_path='*.wav', token=False):

        audio_files = sorted(glob.glob(os.path.join(data_path, base_path)))
        print(len(audio_files), " audio_files found")
        w2v_dict = {}
        for audio_file in audio_files:
            print(audio_file)
            key = os.path.basename(audio_file).strip('.wav')
            audio_features = self.preprocess_audio_file(audio_file)
            with torch.no_grad():
                audio_z = self.get_audio_feature(audio_features)
            if token:
                audio_z = audio_z[1]
            elif type(audio_z) == tuple:
                audio_z = audio_z[0]

            w2v_dict[key] = audio_z
            print('encoded vec', audio_z.shape)
        return w2v_dict


    def get_audio_feature(self, feature):
        with torch.no_grad():
            feature = feature.to(self.device)
            z = self.model.feature_extractor(feature)
            z = z.cpu().detach().numpy().squeeze().T
        return z



#%%
if __name__ == "__main__":

    W2V_NAME = 'vqw2v'
    w2v_path = './vq-wav2vec_kmeans.pt'
    bert_filename = './bert_kmeans.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AudioDataPreprocessing(W2V_NAME, w2v_path, bert_filename, device)
    wav_path = 'MSP_PODCAST1.8/Audios/MSP-PODCAST_0001_0008.wav'

    audio_feat = processor.preprocess_audio_file(wav_path)
    audio_z = processor.get_audio_feature(audio_feat)

