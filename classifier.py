# -*- coding: utf-8 -*-
### 參考 keras/example 裡的 neural_style_transfer.py
### 詳細可到這裡觀看他們的原始碼：
### https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py
from __future__ import print_function
import os
import os.path
import sys
import math
import wave
import random
import numpy as np
import keras
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, Flatten, Lambda, RepeatVector, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras import losses
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.engine.topology import Layer
from keras import regularizers
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
import scipy
import scipy.io.wavfile
import librosa
import matplotlib
matplotlib.use('Agg') ## headless
import matplotlib.pyplot as plt
from conv_net_sound import conv_net

parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('wav_path', metavar='wav', type=str,
                    help='Path to the wav file.')
parser.add_argument('--offset', type=int, default=5, required=False,
                    help='Time offset.')
parser.add_argument('--ffmpeg', action='store_true', default=False,
                    help='use FFmpeg')

args = parser.parse_args()
wav_path = args.wav_path
rate = 11025
offset = args.offset
use_ffmpeg = args.ffmpeg

def readFromFFmpeg(filepath):
    import subprocess as sp
    command = ['ffmpeg',
               '-i', filepath,
               '-acodec', 'pcm_s16le',
               '-f', 's16le',
               '-ac', '1', '-ar', str(rate)]
    command.append('-')
    p = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, _ = p.communicate()
    return np.fromstring(stdout, dtype="int16")

segment_n = 64
FFT_n = 2048
FFT_t = FFT_n//8
total_samp = segment_n*FFT_n
img_nrows, img_ncols = 505, 1025
class_list = ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Jazz', 'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken']

def spectrogram(x):
    stfted = librosa.stft(x, FFT_n, FFT_t)
    stfted = np.transpose(stfted, (1, 0)) ## (freq, frame) -> (frame, freq)
    y = np.log1p(np.abs(stfted[:img_nrows, :img_ncols]))
    y = np.reshape(y, (1, img_nrows, img_ncols, 1))
    return y

def preprocess_wav(wavfile, offset, sample_n):
    wavData = None
    if use_ffmpeg:
        wavData = readFromFFmpeg(str(wavfile))
    else:
        input_rate, wavData = scipy.io.wavfile.read(str(wavfile))
        assert input_rate == rate
    offset *= rate
    if wavData.ndim>1:
        wavData = wavData[...,0]
    wavData = wavData[offset:offset+sample_n] ## skip sample
    wavData = wavData.astype(np.float32) / 32768. ## pcm_s16le -> pcm_f32le
    return spectrogram(wavData)

wav_data = preprocess_wav(wav_path, offset, total_samp)

model = conv_net(input_tensor = None,
                          input_shape = [img_nrows, img_ncols, 1],
                          class_n = len(class_list),
                          weight_path = './top_weight.h5'
                         )
pred_class = model.predict(wav_data, batch_size=1, verbose=0)[0]
od = np.argsort(-pred_class)
for i in od:
    print('%s, confidence: %.2f' % (class_list[i], pred_class[i]))

