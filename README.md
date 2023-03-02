# MetricAug: A Distortion Metric-Lead Augmentation Strategy for Training Noise-Robust Speech Emotion Recognizer official implementation

This repository is official implementation of MetricAug: A Distortion Metric-Lead Augmentation Strategy for Training Noise-Robust Speech Emotion Recognizer.

## 1. Install requirement
```
create env -n metricaug python==3.8
pip install scikit-learn  
pip install joblib  
pip install pandas  
pip install tqdm  
pip lnstall librosa  
pip install soundfile  
pip install fairseq  
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113  
pip install https://github.com/schmiph2/pysepm/archive/master.zip  
```
## 2. Download dataset: MSP-Podcast, MELD, MUSAN and ESC-50
Please download the dataset which you want to implement on it.  

**MSP-Pocast**: <https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html>  
**MELD**: <https://affective-meld.github.io/>  
**MUSAN**:<https://www.openslr.org/17/>  
**ESC-50**:<https://github.com/karolpiczak/ESC-50>  

If you used these dataset, please reference the corresponding paper from the original author.
## 3. Data preprocessing 
 
#### Converting .mp4 to .wav (optional)
If you used **MELD**, please install **ffmpeg** and run the following command to extract feature:
```
ffmpeg -i input.mp4 output.wav
```
#### Superset generate
Follow the scripts in these code to generate superset:
```
preprocessing/add_musan.py
preprocessing/add_esc50_random.py
```

## 4. Feature extraction

```
feature_extract/vqwav2vec_extract_folder_recursive.py
```
Follow this code to complete the feature extraction.

## 5. Distortion metric computation & clustering

#### Step 1. Speech distortion metric computation
Follow these scripts to compute the speech distortion metrics (fwSNRseg, stoi and pesq):
```
preprocessing/metric/compute_se_metric.py
```
If you have problem in computing pesq, using ``preprocessing/metric/re_compute_pesq.py`` to fix it.

#### Step 2. Merge to one csv meta

```
preprocessing/metric/merge_to_parse_meta.py
```
Once you are done, run this code to merge all noisy data.

#### Step 3. Speech distortion metric clustering
```
se_metric_statistical_by_gmm_metric.py
se_metric_statistical_by_rank_metric.py
```
Using these two scripts to complete the clustering for speech distortion metric, the default level is 5.
If you have any questions for code I/O, we made examples in ``example_meta``, please check the format and file path.


## 6. Training
```
train_metric_aug_GRU-TFM_main.py
```
The training code is here. ``data_sample_weight.py`` shows the algorithm 1 in our paper.

## 7. Inference on different testing set

```
test_musan_0_5_10_aug_GRU-TFM_main.py
test_esc50_GRU-TFM_main.py
```
They are the code for inference our model, we also provide the best performance in our paper, which are in the folder ``exp/original_exp/MELD_stoi_gmm`` and ``exp/original_exp/MSP_stoi_gmm``.


#### We also provide the Superset in our paper, contact me with an e-mail if you need it.

TO DO LIST:
 * Optimized the code .  
 * Write a shell bash to make a pipeline.
 * Detail the code I/O .