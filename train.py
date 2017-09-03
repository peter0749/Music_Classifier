# -*- coding: utf-8 -*-
### 參考 keras/example 裡的 neural_style_transfer.py
### 詳細可到這裡觀看他們的原始碼：
### https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py
from __future__ import print_function
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
from keras.optimizers import RMSprop
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from scipy.optimize import fmin_l_bfgs_b
import argparse
import scipy
import scipy.io.wavfile
import vgg19_sound
from utils import preprocess_wav
from utils import deprocess_wav

parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('base_wav_path', metavar='base', type=str,
                    help='Path to the wav to transform.')
parser.add_argument('style_reference_wav_path', metavar='ref', type=str,
                    help='Path to the style reference wav.')
parser.add_argument('--iter', type=int, default=10, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--ar', type=int, default=11025, required=False,
                    help='Sample rate.')
parser.add_argument('--lr', type=float, default=0.01, required=False,
                    help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=4, required=False,
                    help='Batch size.')

args = parser.parse_args()
base_wav_path = args.base_wav_path
style_reference_wav_path = args.style_reference_wav_path
iterations = args.iter
rate = args.ar
lr   = args.lr
batch_size = args.batch_size

segment_n = 64
FFT_n = 2048
FFT_t = FFT_n/8
total_samp = segment_n*FFT_n
img_nrows, img_ncols = 505, 768

def normalize_wav(wavData):
    wavData = wavData.astype(np.float64) ## convert int32 -> float64
    minAmp = wavData[np.argmin(np.abs(wavData))] ## find min amptitude
    wavData[:] -= minAmp ## shift wave to center (a.k.a. 0)
    maxAmp = max(1e-7, np.max(np.abs(wavData))) ## find max amptitude
    wavData[:] /= maxAmp ## normalize to (-1, 1)
    return wavData

def generator():
    base_rate, base_wav_data = scipy.io.wavfile.read(base_wav_path)
    ref_rate , ref_wav_data  = scipy.io.wavfile.read(style_reference_wav_path)
    assert base_rate==ref_rate and base_rate==rate
    if base_wav_data.ndim>1:
        base_wav_data = base_wav_data[...,0]
    if ref_wav_data.ndim>1:
        ref_wav_data = ref_wav_data[...,0]
    base_wav_data = normalize_wav(base_wav_data)
    ref_wav_data  = normalize_wav(ref_wav_data)
    base_wav_data = base_wav_data.astype(np.float32)
    ref_wav_data  = ref_wav_data.astype(np.float32)
    data_shape = (batch_size, total_samp, 1)
    label_shape= (batch_size, 2)
    fetch_size = batch_size*total_samp
    y_base = np.zeros(label_shape, dtype=np.bool)
    y_ref  = np.zeros(label_shape, dtype=np.bool)
    y_base[:,0] = 1
    y_ref[:,1]  = 1
    while True:
        for i in xrange(0, len(base_wav_data)-fetch_size):
            yield np.reshape(base_wav_data[i:i+fetch_size], data_shape), y_base
        for i in xrange(0, len(ref_wav_data)-fetch_size):
            yield np.reshape(ref_wav_data[i:i+fetch_size], data_shape), y_ref

model = vgg19_sound.vgg19(input_tensor = None,
                          segment_n = segment_n,
                          FFT_n = FFT_n,
                          FFT_t = FFT_t,
                          img_nrows = img_nrows, img_ncols = img_ncols,
                          class_n = 2
                         )
model.summary()
optimizer = RMSprop(lr=lr)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.fit_generator(generator(), steps_per_epoch=1, epochs=iterations)
to_save = vgg19_sound.vgg19(input_tensor = None,
                          segment_n = segment_n,
                          FFT_n = FFT_n,
                          FFT_t = FFT_t,
                          img_nrows = img_nrows, img_ncols = img_ncols,
                          class_n = None
                         )
for l in model.layers:
    for r in to_save.layers:
        try:
            r.set_weights(l.get_weights())
        except:
            continue

to_save.save_weights('./vgg19.h5')
