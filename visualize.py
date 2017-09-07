# -*- coding: utf-8 -*-
### 參考 keras/example 裡的 neural_style_transfer.py
### 詳細可到這裡觀看他們的原始碼：
### https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py
from __future__ import print_function
import os
import os.path
import sys
import math
import numpy as np
import scipy
import argparse
import scipy.io.wavfile
import librosa
import matplotlib
matplotlib.use('Agg') ## headless
import matplotlib.pyplot as plt
from keras.models import Model
import conv_net_sound

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

parser = argparse.ArgumentParser(description='Neural style transfer layer visualizer.')
parser.add_argument('wav_path', metavar='wavfile', type=str,
                    help='Path to the wav file.')
parser.add_argument('prefix', metavar='prefix', type=str,
                    help='Prefix of extracted feature maps.')
parser.add_argument('layer', metavar='layer', type=str,
                    help='Layer name in this model.')
parser.add_argument('--ar', type=int, default=11025, required=False,
                    help='Sample rate.')
parser.add_argument('--offset', type=int, default=5, required=False,
                    help='Time offset.')

args = parser.parse_args()
wav_path = args.wav_path
rate = args.ar
offset = args.offset
prefix = args.prefix
layer  = args.layer

segment_n = 64
FFT_n = 2048
FFT_t = FFT_n//8
total_samp = segment_n*FFT_n
img_nrows, img_ncols = 505, 1025

def spectrogram(x):
    stfted = librosa.stft(x, FFT_n, FFT_t)
    stfted = np.transpose(stfted, (1, 0)) ## (freq, frame) -> (frame, freq)
    y = np.log1p(np.abs(stfted[:img_nrows, :img_ncols]))
    y = np.reshape(y, (1, img_nrows, img_ncols, 1))
    return y

def invert_spectrogram(result, iterations=800): ## shape: 1, img_nrows, img_ncols, 1
    ## Formula: x_{n+1}=istft(S∗exp(1i∗angle(stft(x_{n}))))
    ## More detail: https://dsp.stackexchange.com/questions/9877/reconstruction-of-audio-signal-from-spectrogram
    result = np.reshape(result, (img_nrows, img_ncols)).T
    result = np.exp(result) - 1.
    signal = np.zeros((FFT_n/2+1, img_nrows), dtype=np.float64)
    signal[:img_ncols, :img_nrows] = result[:,:]
    p = 2 * np.pi * np.random.random_sample(signal.shape) - np.pi
    for i in xrange(iterations):
        S = signal * np.exp(1j*p)
        x = librosa.istft(S, FFT_t)
        p = np.angle(librosa.stft(x, FFT_n, FFT_t))
    return x

def preprocess_wav(wavfile, offset, sample_n):
    input_rate, wavData = scipy.io.wavfile.read(str(wavfile))
    assert input_rate == rate
    offset *= rate
    if wavData.ndim>1:
        wavData = wavData[...,0]
    wavData = wavData[offset:offset+sample_n] ## skip sample
    wavData = wavData.astype(np.float32) / 32768. ## pcm_s16le -> pcm_f32le
    return spectrogram(wavData)

wav_data = preprocess_wav(wav_path, offset, total_samp)

model = conv_net_sound.conv_net(input_tensor = None,
                          input_shape = [img_nrows, img_ncols, 1],
                          weight_path = './conv_net.h5'
                         )
model.summary()

eprint('Model loaded.')

#feature_layers = [
        #'block1_conv1_gpu1',
        #'block1_conv1_gpu2',
        #'block2_conv1_gpu1',
        #'block2_conv1_gpu2',
        #'block3_conv1_gpu1',
        #'block3_conv1_gpu2',
        #'block3_conv2_gpu1',
        #'block3_conv2_gpu2'
        #]

def plot_spectrogram(x, fname):
    plt.ylabel('Frequency [HZ]')
    plt.xlabel('Time [not scaled]')
    plt.imshow(x)
    plt.savefig(prefix+str(fname))

def layer_extract(x, layer_name):
    feature_extractor = Model(input=model.input, output=model.get_layer(layer_name).output)
    h = feature_extractor.predict(x, verbose=0, batch_size=1)[0] ## shape: (frame_n, freq_n, channel_n)
    h = np.transpose(h, [2, 1, 0]) ## shape: (channel_n, freq_n, frame_n)
    for i, feature_map in enumerate(h):
        plot_spectrogram(feature_map, layer_name+'_'+str(i))

plot_spectrogram(np.reshape(wav_data, (img_nrows, img_ncols)).T, 'spectrogram.png')

layer_extract(wav_data, layer)

