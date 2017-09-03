# -*- coding: utf-8 -*-
### 參考 keras/example 裡的 neural_style_transfer.py
### 詳細可到這裡觀看他們的原始碼：
### https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py
from __future__ import print_function
import os
import os.path
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
from keras.optimizers import SGD
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import ModelCheckpoint
from scipy.optimize import fmin_l_bfgs_b
import argparse
import scipy
import scipy.io.wavfile
import conv_net_sound
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
parser.add_argument('--lr', type=float, default=0.0001, required=False,
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
    x = np.zeros(data_shape, dtype=np.float32)
    y = np.zeros(label_shape, dtype=np.bool)
    baseIdx = 0
    refIdx  = 0
    cap     = 0
    od      = 0
    while True:
        if od % 2 == 0:
            x[cap, :, 0] = base_wav_data[baseIdx:baseIdx+total_samp]
            y[cap, 0] = 1
            y[cap, 1] = 0
            baseIdx = (baseIdx+1) % (len(base_wav_data)-total_samp)
        else:
            x[cap, :, 0] = ref_wav_data[refIdx:refIdx+total_samp]
            y[cap, 0] = 0
            y[cap, 1] = 1
            refIdx = (refIdx+1) % (len(ref_wav_data)-total_samp)
        cap = (cap+1) % batch_size
        od  = (od+1) % 2
        if cap==0: ## full
            yield x, y

model = conv_net_sound.conv_net(input_tensor = None,
                          segment_n = segment_n,
                          FFT_n = FFT_n,
                          FFT_t = FFT_t,
                          img_nrows = img_nrows, img_ncols = img_ncols,
                          class_n = 2
                         )
if (os.path.isfile('./top_weight.h5')):
    model.load_weights('./top_weight.h5')
model.summary()
optimizer = SGD(lr=lr, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
checkPoint = ModelCheckpoint(filepath="./top_weight.h5", verbose=1, save_best_only=True, monitor='loss', mode='min', save_weights_only=True, period=50)
model.fit_generator(generator(), steps_per_epoch=1, epochs=iterations, callbacks=[checkPoint])
to_save = conv_net_sound.conv_net(input_tensor = None,
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

to_save.save_weights('./conv_net.h5')
