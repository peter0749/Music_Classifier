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
parser.add_argument('--offset_base', type=int, default=5, required=False,
                    help='Time offset of base wav.')
parser.add_argument('--offset_ref', type=int, default=5, required=False,
                    help='Time offset of reference wav.')

args = parser.parse_args()
base_wav_path = args.base_wav_path
style_reference_wav_path = args.style_reference_wav_path
iterations = args.iter
rate = args.ar
lr   = args.lr
offset_base = args.offset_base
offset_ref  = args.offset_ref

segment_n = 64
FFT_n = 2048
FFT_t = FFT_n/8
total_samp = segment_n*FFT_n
img_nrows, img_ncols = 505, 768

base_wav_data = preprocess_wav(base_wav_path, offset_base, total_samp, True)
style_reference_wav_data = preprocess_wav(style_reference_wav_path, offset_ref, total_samp, True)

data = np.concatenate((base_wav_data, style_reference_wav_data), axis=0)
label= np.zeros((2, 2), dtype=np.bool)
label[0,0] = True
label[1,1] = True

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
model.fit(data, label, batch_size=2, shuffle=False, epochs=iterations)

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
