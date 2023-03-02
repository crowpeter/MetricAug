#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 17:16:51 2020

@author: chadyang
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pysepm
from multiprocessing import Pool
from functools import partial
import soundfile as sf

import warnings
warnings.filterwarnings("ignore")

# import librosa
#%%
def sisdr(reference, estimation, dtype=np.float64):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    From:
        https://github.com/fgnt/pb_bss/blob/master/pb_bss/evaluation/module_si_sdr.py#L40
        
    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]
        
    Returns:
        SI-SDR
    
    [1] SDR– Half- Baked or Well Done?
    http://www.merl.com/publications/docs/TR2019-013.pdf
    
    Example:
        >>> np.random.seed(0)
        >>> reference = np.random.randn(100)
        >>> si_sdr(reference, reference)
        inf
        >>> si_sdr(reference, reference * 2)
        inf
        >>> si_sdr(reference, np.flip(reference))
        -25.127672346460717
        >>> si_sdr(reference, reference + np.flip(reference))
        0.481070445785553
        >>> si_sdr(reference, reference + 0.5)
        6.3704606032577304
        >>> si_sdr(reference, reference * 2 + 1)
        6.3704606032577304
        >>> si_sdr([1., 0], [0., 0])  # never predict only zeros
        nan
        >>> si_sdr([reference, reference], [reference * 2 + 1, reference * 1 + 0.5])
        array([6.3704606, 6.3704606])
    """
    estimation, reference = np.broadcast_arrays(estimation, reference)

    assert reference.dtype == dtype, reference.dtype
    assert estimation.dtype == dtype, estimation.dtype

    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    # This is $\alpha$ after Equation (3) in [1].
    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) \
        / reference_energy

    # This is $e_{\text{target}}$ in Equation (4) in [1].
    projection = optimal_scaling * reference

    # This is $e_{\text{res}}$ in Equation (4) in [1].
    noise = estimation - projection

    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
    return 10 * np.log10(ratio)
#%% core function calculate metric
def cal_metric_single(sig_ref:np.array, sig_test:np.array, wavName:str, fs:int=16000, metric:list=['pesq','srmr','sisdr']) -> pd.DataFrame:
    '''
    Parameters
    ----------
    sig_ref : np.array
        1-D reference( clean) speech array with shape (L,)
    sig_test : np.array
        1-D test (reverbed / noisy / enhanced) speech array with shape (L,)
    wavName : str
        wav file name
    fs : int, optional
        sample frequency. The default is 16000.
    metric : list, optional
        the metric to calculate. The default is ['pesq','srmr','sisdr'].

    Returns
    -------
    metrics_return : pd.DataFrame
        calculated metrics with wavName as index
    '''
    
    metrics_return = {}

    # =============================================================================
    # --- Speech Quality Measures ---
    # =============================================================================
    # Segmental Signal-to-Noise Ratio (SNRseg)
    if 'SNRseg' in metric:
        metrics_return['SNRseg'] = pysepm.SNRseg(sig_ref, sig_test, fs)

    # Frequency-weighted Segmental SNR (fwSNRseg)
    if 'fwSNRseg' in metric:
        metrics_return['fwSNRseg'] = pysepm.fwSNRseg(sig_ref, sig_test, fs)
    # Log-likelihood Ratio (LLR)
    if 'llr' in metric:
        metrics_return['llr'] = pysepm.llr(sig_ref, sig_test, fs)
    
    # Weighted Spectral Slope (WSS)
    if 'wss' in metric:
        metrics_return['wss']  = pysepm.wss(sig_ref, sig_test, fs)

    # Perceptual Evaluation of Speech Quality (PESQ)
    if 'pesq' in metric:
        metrics_return['pesq'] = pysepm.pesq(sig_ref, sig_test, fs)[1]

    # Composite Objective Speech Quality (composite)
    if 'comp' in metric:
        for cIdx, (cName, c) in enumerate(zip(['sig','bak','ovl'], pysepm.composite(sig_ref, sig_test, fs))):
            metrics_return[f'comp-{cName}'] = c
            
    # Cepstrum Distance Objective Speech Quality Measure (CD)
    if 'cd' in metric:
        metrics_return['cd']  = pysepm.cepstrum_distance(sig_ref, sig_test, fs)


    # =============================================================================
    # --- Speech Intelligibility Measures ---
    # =============================================================================
    # Short-time objective intelligibility (STOI)
    if 'stoi' in metric:
        metrics_return['stoi'] = pysepm.stoi(sig_ref, sig_test, fs)

    # Coherence and speech intelligibility index (CSII)
    if 'csii' in metric:
        for cIdx, (cName, c) in enumerate(zip(['h','m','l'], pysepm.csii(sig_ref, sig_test, fs))):
            metrics_return[f'csii-{cName}'] = c

    # Normalized-covariance measure (NCM)
    if 'ncm' in metric:
        metrics_return['ncm'] = pysepm.ncm(sig_ref, sig_test, fs)



    # =============================================================================
    #  --- Dereverberation Measures ---
    # =============================================================================
    # basd
    if 'bsd' in metric: # remember to modify pysepm/reverberationMeasures.py line 111 into np.nanmean
        metrics_return['bsd'] = pysepm.bsd(sig_ref, sig_test, fs)

    # srmr
    if 'srmr' in metric:
        metrics_return['srmr'] = pysepm.srmr(sig_test, fs) # default use fast
        
    # sisdr
    if 'sisdr' in metric:
        metrics_return['sisdr'] = sisdr(sig_ref, sig_test, dtype=np.float32)
    
    metrics_return = pd.DataFrame(metrics_return, index=[wavName])
    return metrics_return

def cal_metric_array_parallel(sig_ref_array:np.array, sig_test_array:np.array, fs:int, wavName_list:list, metric:list=['pesq','srmr','sisdr'], n_proc=8, print_pbar=False) -> pd.DataFrame:
    metric_all = pd.DataFrame()
    with Pool(processes=n_proc) as pool:
        for metrics_return in tqdm(pool.map(partial(cal_metric_single, fs=fs, metric=metric), zip(sig_ref_array, sig_test_array, wavName_list)), total=len(wavName_list), disable=not(print_pbar)):
            metric_all = metric_all.append(metrics_return)
    return metric_all

def cal_metric_single_parallel(sig_ref_array:np.array, sig_test_array:np.array, fs:int, wavName_list:list, metric:list=['pesq','srmr','sisdr'], n_proc=8, print_pbar=False) -> pd.DataFrame:
    metric_all = pd.DataFrame()
    with Pool(processes=n_proc) as pool:
        for metrics_return in pool.map(partial(cal_metric_single, fs=fs, metric=metric), zip(sig_ref_array, sig_test_array), wavName_list):
            metric_all = metric_all.append(metrics_return)
    return metric_all
    
    
#%%
def cal_metric_single_wav(wav_path_ref:str, wav_path_test:str, metric:list=['pesq','srmr','sisdr']) -> pd.DataFrame:
    wavName = '.'.join(os.path.basename(wav_path_ref).split('.')[:-1])
    # fs_ref, sig_ref = wavfile.read(wav_path_ref)
    # fs_test, sig_test = wavfile.read(wav_path_test)
    sig_ref, fs_ref  = sf.load(wav_path_ref,sr=16000)
    sig_test, fs_test = sf.load(wav_path_test,sr=16000)
    if len(sig_ref) == len(sig_test):
        pass
    elif len(sig_ref) > len(sig_test):
        sig_ref = sig_ref[:len(sig_test)]
    elif len(sig_test) > len(sig_ref):
        sig_test = sig_test[:len(sig_ref)]
    assert fs_ref==fs_test, 'Error: the sampling rate is not the same !!!'
    fs = fs_ref
    return cal_metric_single(sig_ref, sig_test, wavName, fs, metric=metric)

#%%