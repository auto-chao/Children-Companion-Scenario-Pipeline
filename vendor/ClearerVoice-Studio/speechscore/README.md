# SpeechScore : A toolkit for intrusive and non-intrusive speech quality assessment

## 👉🏻[HuggingFace Space Demo](https://huggingface.co/spaces/alibabasglab/SpeechScore)👈🏻

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Usage](#2-usage)
- [3. Acknowledgements](#3-acknowledgements)

## 1. Introduction

SpeechScore is a wrapper designed for assessing speech quality. It includes a collection of commonly used speech quality metrics, as listed below:
| Index | Metrics | Non-Intrusive | Short Description | Externel Link |
|-------|---------|---------------|-------------------|---------------|
|1.| BSSEval {ISR, SAR, SDR} | No |ISR (Source Image-to-Spatial distortion Ratio) measures preservation/distortion of target source. SDR (Source-to-Distortion Ratio) measures global quality. SAR (Source-to-Artefact Ratio) measures the presence of additional artificial noise|(See <a href="https://github.com/sigsep/sigsep-mus-eval">the official museval page</a>)|
|2.| {CBAK, COVL, CSIG} | No | CSIG predicts the signal distortion mean opinion score (MOS), CBAK measures background intrusiveness, and COVL measures speech quality. CSIG, CBAK, and COVL are ranged from 1 to 5| See paper: <a href="https://ecs.utdallas.edu/loizou/speech/obj_paper_jan08.pdf">Evaluation of Objective Quality Measures for Speech Enhancement</a>|
|3.| DNSMOS {BAK, OVRL, SIG, P808_MOS} |Yes|DNSMOS (Deep Noise Suppression Mean Opinion Score) measures the overall quality of the audio clip based on the ITU-T Rec. P.808 subjective evaluation. It outputs 4 scores: i) speech quality (SIG), ii) background noise quality (BAK), iii) the overall quality (OVRL), and iv) the P808_MOS of the audio.  DNSMOS does not require clean references. | See paper: <a href="https://arxiv.org/pdf/2010.15258.pdf">Dnsmos: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors</a> and <a href="https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS">github page</a>|
|4.| FWSEGSNR | No | FWSEGSNR (Frequency-Weighted SEGmental SNR) is commonly used for evaluating dereverberation performance |See paper: <a href="https://ecs.utdallas.edu/loizou/speech/obj_paper_jan08.pdf">Evaluation of Objective Quality Measures for Speech Enhancement</a> | 
|5.| LLR | No | LLR (Log Likelihood Ratio) measures how well an estimated speech signal matches the target (clean) signal in terms of their short-term spectral characteristics. |See paper: <a href="https://ecs.utdallas.edu/loizou/speech/obj_paper_jan08.pdf">Evaluation of Objective Quality Measures for Speech Enhancement</a> |
|6.| LSD | No |LSD (Log-Spectral Distance) measures the spectral differences between a clean reference signal and a processed speech signal.| See <a href="https://github.com/haoheliu/ssr_eval"> github page </a>|
|7.| MCD | No |MCD (Mel-Cepstral Distortion) measures the difference between the mel-cepstral coefficients (MCCs) of an estimated speech signal and the target (clean) speech signal. |See <a href="https://github.com/chenqi008/pymcd"> github page </a> |
|8.| NB_PESQ | No| NB-PESQ (NarrowBand Perceptual Evaluation of Speech Quality) meaures speech quality that reflects human auditory perception. It is defined in the ITU-T Recommendation P.862 and is developed for assessing narrowband speech codecs and enhancement algorithms. | See <a href="https://github.com/ludlows/PESQ"> github page </a> |
|9.| PESQ | No | PESQ (Perceptual Evaluation of Speech Quality) assesses the quality of speech signals to mimic human perception. It is standardized by the International Telecommunication Union (ITU-T P.862) and is widely used in evaluating telecommunication systems and speech enhancement algorithms. |See <a href="https://github.com/ludlows/PESQ"> github page </a> |
|10.| SISDR | No | SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) quantifies the ratio between the power of the target signal component and the residual distortion. It measures how well an estimated speech signal matches the target (clean) speech signal, while being invariant to differences in scale. |See paper: <a href="https://arxiv.org/abs/1811.02508">SDR - half-baked or well done?<a/> |
|11.| SNR | No | SNR (Signal-to-Noise Ratio) is a fundamental metric used in speech quality measurement to evaluate the relative level of the desired speech signal compared to unwanted noise. It quantifies the clarity and intelligibility of speech in decibels (dB).| See paper: <a href="https://www.isca-archive.org/icslp_1998/hansen98_icslp.pdf">An effective quality evaluation protocol for speech enhancement algorithms<a/>|
|12.| SRMR | Yes | SRMR (Speech-to-Reverberation Modulation Energy Ratio) evaluates the ratio of speech-dominant modulation energy to reverberation-dominant modulation energy. It quantifies the impact of reverberation on the quality and intelligibility of speech signals. SRMR does not require clean references. | See <a href="https://github.com/jfsantos/SRMRpy">SRMRpy<a/> and <a href="https://github.com/MuSAELab/SRMRToolbox">SRMR Toolbox<a/>|
|13.| SSNR | No | SSNR (Segmental Signal-to-Noise Ratio) is an extension of SNR (Signal-to-Noise Ratio) and for evaluating the quality of speech signals in shorter segments or frames. It is calculated by dividing the power of the clean speech signal by the power of the noise signal, computed over small segments of the speech signal. | See paper: <a href="https://www.isca-archive.org/icslp_1998/hansen98_icslp.pdf">An effective quality evaluation protocol for speech enhancement algorithms<a/>|
|14.| STOI| No | STOI (Short-Time Objective Intelligibility Index) measures speech quality and intelligibility by operateing on short-time segments of the speech signal and computes a score between 0 and 1. | See <a href="https://github.com/mpariente/pystoi">github page <a/> |
|15.| NISQA| Yes |NISQA is a deep learning model for predicting speech quality in communication systems, providing overall quality scores and detailed insights into Noisiness, Coloration, Discontinuity, and Loudness.|see <a href="https://github.com/gabrielmittag/NISQA">gtihub page<a/>|
|16.| DISTILL_MOS |Yes | Distill-MOS is a compact and efficient speech quality assessment model learned from a larger speech quality assessment model based on wav2vec2.0 XLS-R embeddings.|The work is described in the paper: <a href="https://arxiv.org/pdf/2502.05356">Distillation and Pruning for Scalable Self-Supervised Representation-Based Speech Quality Assessment"<a/> ICASSP 2025|

## 2. Usage

### Step-by-Step Guide

If you haven't created a Conda environment for ClearerVoice-Studio yet, follow steps 1 and 2. Otherwise, skip directly to step 3.

1. **Clone the Repository**

``` sh
git clone https://github.com/modelscope/ClearerVoice-Studio.git
```

2. **Create Conda Environment**

``` sh
cd ClearerVoice-Studio
conda create -n ClearerVoice-Studio python=3.8
conda activate ClearerVoice-Studio
pip install -r requirements.txt
```

3. Run demo script

``` sh
cd speechscore
python demo.py
```
or use the following script:
``` python
# Import pprint for pretty-printing the results in a more readable format
import pprint
# Import the SpeechScore class to evaluate speech quality metrics
from speechscore import SpeechScore 

# Main block to ensure the code runs only when executed directly
if __name__ == '__main__':
    # Initialize a SpeechScore object with a list of score metrics to be evaluated
    # Supports any subsets of the list
    
    # Non-intrusive tests ['NISQA', 'DNSMOS', 'DISTILL_MOS', SRMR'] : No reference audio is required
    
    mySpeechScore = SpeechScore([
        'SRMR', 'PESQ', 'NB_PESQ', 'STOI', 'SISDR', 
        'FWSEGSNR', 'LSD', 'BSSEval', 'DNSMOS', 
        'SNR', 'SSNR', 'LLR', 'CSIG', 'CBAK', 
        'COVL', 'MCD', 'NISQA', 'DISTILL_MOS'
    ])
    
    # Call the SpeechScore object to evaluate the speech metrics between 'noisy' and 'clean' audio
    # Arguments:
    # - {test_path, reference_path} supports audio directories or audio paths (.wav or .flac)
    # - window (float): seconds, set None to specify no windowing (process the full audio)
    # - score_rate (int): specifies the sampling rate at which the metrics should be computed
    # - return_mean (bool): set True to specify that the mean score for each metric should be returned

    
    print('score for a signle wav file')
    scores = mySpeechScore(test_path='audios/noisy.wav', reference_path='audios/clean.wav', window=None, score_rate=16000, return_mean=False)
    #scores = mySpeechScore(test_path='audios/noisy.wav', reference_path=None) # for Non-instrusive tests
    # Pretty-print the resulting scores in a readable format
    pprint.pprint(scores)

    print('score for wav directories')
    scores = mySpeechScore(test_path='audios/noisy/', reference_path='audios/clean/', window=None, score_rate=16000, return_mean=True)

    # Pretty-print the resulting scores in a readable format
    pprint.pprint(scores)

    # Print only the resulting mean scores in a readable format
    #pprint.pprint(scores['Mean_Score'])
```
The results should be looking like below:

```sh
score for a signle wav file
{'BSSEval': {'ISR': 22.74466768594795,
             'SAR': -0.1921607960486268,
             'SDR': -0.2392167019930811},
 'CBAK': 1.5908301020179343,
 'COVL': 1.5702211032246858,
 'CSIG': 2.3259380853008342,
 'DISTILL_MOS': 2.7404244,
 'DNSMOS': {'BAK': 1.3532921035219425,
            'OVRL': 1.371476750718355,
            'P808_MOS': 2.354834,
            'SIG': 1.8698046623352382},
 'FWSEGSNR': 6.414399025759908,
 'LLR': 0.8532984,
 'LSD': 2.136734818644344,
 'MCD': 11.013451521306235,
 'NB_PESQ': 1.2447538375854492,
 'NISQA': {'col_pred': 2.1640655994415283,
           'dis_pred': 3.664355993270874,
           'loud_pred': 2.142110824584961,
           'mos_pred': 1.5944541692733765,
           'noi_pred': 1.601306676864624},
 'PESQ': 1.0545592308044434,
 'SISDR': -0.2370745117626482,
 'SNR': -0.9504614142497446,
 'SRMR': 6.202590182397147,
 'SSNR': -0.6363067113236048,
 'STOI': 0.8003376411051097}
score for wav directories
{'Mean_Score': {'BSSEval': {'ISR': 23.728811184377903,
                            'SAR': 4.839625092004949,
                            'SDR': 4.9270216975279135},
                'CBAK': 1.9391528046230797,
                'COVL': 1.5400282284547386,
                'CSIG': 2.128618074752965,
                'DISTILL_MOS': 2.9505499601364136,
                'DNSMOS': {'BAK': 1.9004393746158654,
                           'OVRL': 1.8606212162135691,
                           'P808_MOS': 2.5821499824523926,
                           'SIG': 2.6799127209039266},
                'FWSEGSNR': 9.079539440199575,
                'LLR': 1.1992592215538025,
                'LSD': 2.00452909961033,
                'MCD': 8.916492705343465,
                'NB_PESQ': 1.431145429611206,
                'NISQA': {'col_pred': 3.115872025489807,
                          'dis_pred': 4.023841500282288,
                          'loud_pred': 2.707561731338501,
                          'mos_pred': 2.045940101146698,
                          'noi_pred': 1.6330870985984802},
                'PESQ': 1.141619324684143,
                'SISDR': 4.778657656271212,
                'SNR': 4.571920494312266,
                'SRMR': 9.221118316293257,
                'SSNR': 2.9965604574762796,
                'STOI': 0.8585249663711918},
 'audio_1.wav': {'BSSEval': {'ISR': 22.74466768594795,
                             'SAR': -0.1921607960486268,
                             'SDR': -0.2392167019930811},
                 'CBAK': 1.5908301020179345,
                 'COVL': 1.570221103224686,
                 'CSIG': 2.3259380853008342,
                 'DISTILL_MOS': 2.7404244,
                 'DNSMOS': {'BAK': 1.3532921035219425,
                            'OVRL': 1.371476750718355,
                            'P808_MOS': 2.354834,
                            'SIG': 1.8698046623352382},
                 'FWSEGSNR': 6.414399025759908,
                 'LLR': 0.8532984,
                 'LSD': 2.136734818644344,
                 'MCD': 11.013451521306235,
                 'NB_PESQ': 1.2447538375854492,
                 'NISQA': {'col_pred': 2.1640655994415283,
                           'dis_pred': 3.664355993270874,
                           'loud_pred': 2.142110824584961,
                           'mos_pred': 1.5944541692733765,
                           'noi_pred': 1.601306676864624},
                 'PESQ': 1.0545592308044434,
                 'SISDR': -0.2370745117626482,
                 'SNR': -0.9504614142497446,
                 'SRMR': 6.202590182397147,
                 'SSNR': -0.6363067113236048,
                 'STOI': 0.8003376411051097},
 'audio_2.wav': {'BSSEval': {'ISR': 24.71295468280786,
                             'SAR': 9.871410980058526,
                             'SDR': 10.093260097048908},
                 'CBAK': 2.287475507228225,
                 'COVL': 1.5098353536847915,
                 'CSIG': 1.9312980642050954,
                 'DISTILL_MOS': 3.1606755,
                 'DNSMOS': {'BAK': 2.447586645709788,
                            'OVRL': 2.3497656817087833,
                            'P808_MOS': 2.809466,
                            'SIG': 3.4900207794726152},
                 'FWSEGSNR': 11.744679854639243,
                 'LLR': 1.54522,
                 'LSD': 1.8723233805763162,
                 'MCD': 6.819533889380694,
                 'NB_PESQ': 1.617537021636963,
                 'NISQA': {'col_pred': 4.067678451538086,
                           'dis_pred': 4.383327007293701,
                           'loud_pred': 3.273012638092041,
                           'mos_pred': 2.4974260330200195,
                           'noi_pred': 1.6648675203323364},
                 'PESQ': 1.2286794185638428,
                 'SISDR': 9.794389824305073,
                 'SNR': 10.094302402874277,
                 'SRMR': 12.239646450189369,
                 'SSNR': 6.629427626276164,
                 'STOI': 0.9167122916372739}}
```
Any subset of the full score list is supported, specify your score list using the following objective:

```
mySpeechScore = SpeechScore(['.'])
```

## 3. Acknowledgements
We referred to <a href="https://github.com/aliutkus/speechmetrics">speechmetrics<a/>, <a href="https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS">DNSMOS <a/>, <a href="https://github.com/sigsep/bsseval/tree/master">BSSEval<a/>, <a href="https://github.com/chenqi008/pymcd/blob/main/pymcd/mcd.py">pymcd<a/>, <a href="https://github.com/mpariente/pystoi">pystoi<a/>, <a href="https://github.com/ludlows/PESQ">PESQ<a/>, and <a href="https://github.com/santi-pdp/segan_pytorch/tree/master">segan_pytorch<a/> for implementing this repository.
