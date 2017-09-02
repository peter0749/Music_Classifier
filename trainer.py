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
import matplotlib
matplotlib.use('Agg') ## headless
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('base_wav_path', metavar='base', type=str,
                    help='Path to the wav to transform.')
parser.add_argument('style_reference_wav_path', metavar='ref', type=str,
                    help='Path to the style reference wav.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')
parser.add_argument('--iter', type=int, default=10, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--content_weight', type=float, default=0.025, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
                    help='Total Variation weight.')
parser.add_argument('--ar', type=int, default=11025, required=False,
                    help='Sample rate')
parser.add_argument('--offset_base', type=int, default=5, required=False,
                    help='Time offset of base wav')
parser.add_argument('--offset_ref', type=int, default=5, required=False,
                    help='Time offset of reference wav')

args = parser.parse_args()
base_wav_path = args.base_wav_path
style_reference_wav_path = args.style_reference_wav_path
result_prefix = args.result_prefix
iterations = args.iter
rate = args.ar
offset_base = args.offset_base
offset_ref  = args.offset_ref

# these are the weights of the different loss components
total_variation_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight

segment_n = 64
FFT_n = 2048
FFT_t = FFT_n/8
total_samp = segment_n*FFT_n
img_nrows, img_ncols = 505, 505

def preprocess_wav(wavfile, offset, sample_n, normalize=False):
    input_rate, wavData = scipy.io.wavfile.read(str(wavfile))
    assert input_rate == rate
    offset *= rate
    if wavData.ndim>1.:
        wavData = wavData[...,0]
    wavData = wavData[offset:offset+sample_n] ## skip sample
    if normalize:
        wavData = wavData.astype(np.float64) ## convert int32 -> float64
        minAmp = wavData[np.argmin(np.abs(wavData))] ## find min amptitude
        wavData[:] -= minAmp ## shift wave to center (a.k.a. 0)
        maxAmp = max(1e-7, np.max(np.abs(wavData))) ## find max amptitude
        wavData[:] /= maxAmp ## normalize to (-1, 1)
    wavData = wavData.astype(np.float32) ## downcast to float32 ( float64 -> float32)
    wavData = np.reshape(wavData, (1, sample_n, 1))
    return wavData

def deprocess_wav(x):
    return x.flatten().clip(-1., 1.)

base_wav_data = preprocess_wav(base_wav_path, offset_base, total_samp, True)
style_reference_wav_data = preprocess_wav(style_reference_wav_path, offset_ref, total_samp, True)

base_wav = K.variable(base_wav_data)
style_reference_wav = K.variable(style_reference_wav_data)
combination_wav = K.placeholder((1, total_samp, 1))

input_tensor = K.concatenate([base_wav,
                              style_reference_wav,
                              combination_wav], axis=0)
input_layer = Input(tensor=input_tensor, shape=(total_samp, 1))

def custom_STFT_layer(x):
    ## input_shape: (batch_size, timestep, channel)
    ## output_shape:(batch_size, sample, freq_range, channel)
    stft = tf.contrib.signal.stft(x[...,0], FFT_n, FFT_t)
    dense = tf.abs(stft)
    spec  = tf.log1p(dense)
    ##  end of spectrogram
    y = tf.expand_dims(spec, -1)
    return y[:, :img_nrows, :img_ncols, :]

stfted = Lambda(custom_STFT_layer, name='STFT')(input_layer)
## VGG19 Net:
# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_initializer='random_normal')(stfted)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_initializer='random_normal')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_initializer='random_normal')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_initializer='random_normal')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_initializer='random_normal')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_initializer='random_normal')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_initializer='random_normal')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4', kernel_initializer='random_normal')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_initializer='random_normal')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_initializer='random_normal')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_initializer='random_normal')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4', kernel_initializer='random_normal')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_initializer='random_normal')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_initializer='random_normal')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_initializer='random_normal')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4', kernel_initializer='random_normal')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

model = Model(input_layer, x) ## model: wav -> features
model.summary()

stft_input = Input(shape=(total_samp, 1))
stft_output = Lambda(custom_STFT_layer)(stft_input)
stft_model = Model(input=stft_input, output=stft_output)

print('Model loaded.')

# get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# compute the neural style loss
# first we need to define 4 util functions

# the gram matrix of an image tensor (feature-wise outer product)


def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1))) ## Supports Tensorflow only. "channel last" mode by default
    gram = K.dot(features, K.transpose(features))
    return gram

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image


def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 1
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image


def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent


def total_variation_loss(orix):
    x = custom_STFT_layer(orix)
    assert K.ndim(x) == 4
    a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# combine these loss functions into a single scalar
loss = K.variable(0.)
layer_features = outputs_dict['block5_conv2']
base_wav_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(base_wav_features,
                                      combination_features)

feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl
loss += total_variation_weight * total_variation_loss(combination_wav)

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_wav)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_wav], outputs)


def eval_loss_and_grads(x):
    x = x.reshape((1, total_samp, 1))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

def plot_spectrogram(x, fname):
    x = np.reshape(x, (1, total_samp, 1))
    x = stft_model.predict(x)
    x = np.reshape(x, (img_nrows, img_ncols))
    x = np.transpose(x, (1, 0))
    plt.ylabel('Frequency [HZ]')
    plt.xlabel('Time [not scaled]')
    plt.gca().invert_yaxis()
    plt.imshow(x)
    plt.savefig(str(fname))


# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
x = np.random.randn(1, total_samp, 1) * 1e-3

plot_spectrogram(base_wav_data, 'base.png')
plot_spectrogram(style_reference_wav_data, 'style.png')

for i in xrange(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    wav = deprocess_wav(x.copy())
    fname = result_prefix + '_at_iteration_%d.wav' % i
    plot_spectrogram(wav, fname+'.png')
    scipy.io.wavfile.write(fname, rate, wav)
    end_time = time.time()
    print('wav saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))


