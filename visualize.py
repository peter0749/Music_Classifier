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
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('wav_path', metavar='base', type=str,
                    help='Path to the wav file.')
parser.add_argument('--ar', type=int, default=11025, required=False,
                    help='Sample rate.')

args = parser.parse_args()
wav_path = args.wav_path
rate = args.ar

segment_n = 64
FFT_n = 2048
FFT_t = FFT_n/8
total_samp = segment_n*FFT_n
img_nrows, img_ncols = 505, 768

def vis(x, base_model, layer_name):
    model = Model(base_model.input, output=base_model.get_layer(str(layer_name)).output)
    y = model.predict(x)
    kernel_n = np.shape(y)[-1]
    for i in range(kernel_n):
        plt.title(layer_name+'_'+str(i))
        plt.imshow(y[0, :, :, i])
        plt.show()


data = preprocess_wav(wav_path, 0, total_samp, True)

model = conv_net_sound.conv_net(input_tensor = None,
                          segment_n = segment_n,
                          FFT_n = FFT_n,
                          FFT_t = FFT_t,
                          img_nrows = img_nrows, img_ncols = img_ncols,
                          class_n = None,
                          weight_path = './conv_net.h5'
                         )
model.summary()

vis_layers = ['block1_conv1', 'block1_conv2',
              'block2_conv1', 'block2_conv2',
              'block3_conv1', 'block3_conv2',
              'block3_conv3', 'block3_conv4'
             ]

for lay in vis_layers:
    vis(data, model, lay)
