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
import conv_net_sound

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

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
parser.add_argument('--ar', type=int, default=11025, required=False,
                    help='Sample rate.')
parser.add_argument('--offset_base', type=int, default=5, required=False,
                    help='Time offset of base wav.')
parser.add_argument('--offset_ref', type=int, default=5, required=False,
                    help='Time offset of reference wav.')
parser.add_argument('--init', type=str, default='noise', required=False,
                    help='Initial state of output wav file. [noise/base] (default=noise)')
parser.add_argument('--result_only', type=int, default=1, required=False,
                    help='Just generate final result. (default:1)')

args = parser.parse_args()
base_wav_path = args.base_wav_path
style_reference_wav_path = args.style_reference_wav_path
result_prefix = args.result_prefix
iterations = args.iter
rate = args.ar
offset_base = args.offset_base
offset_ref  = args.offset_ref
init_mode = args.init
result_only = True if args.result_only!=0 else False

# these are the weights of the different loss components
style_weight = args.style_weight
content_weight = args.content_weight

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

def deprocess_wav(x):
    data = invert_spectrogram(x).flatten()
    data *= 32768.
    return data.clip(-32768, 32767).astype(np.int16)

base_wav_data = preprocess_wav(base_wav_path, offset_base, total_samp)
style_reference_wav_data = preprocess_wav(style_reference_wav_path, offset_ref, total_samp)

base_wav = K.variable(base_wav_data.copy())
style_reference_wav = K.variable(style_reference_wav_data.copy())
combination_wav = K.placeholder((1, img_nrows, img_ncols, 1))

input_tensor = K.concatenate([base_wav,
                              style_reference_wav,
                              combination_wav], axis=0)
model = conv_net_sound.conv_net(input_tensor = input_tensor,
                          input_shape = [img_nrows, img_ncols, 1],
                          weight_path = './conv_net.h5'
                         )
model.summary()

eprint('Model loaded.')

# get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# compute the neural style loss
# first we need to define 4 util functions

# the gram matrix of an image tensor (feature-wise outer product)


def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1))) ## Supports Tensorflow only. "channel last" mode by default
    gram = K.dot(features, K.transpose(features))
    return gram / img_nrows ## gram / sample_n

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
    return K.sum(K.square(S - C)) ## mse / feature size

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image


def content_loss(base, combination): ## mse
    return K.sum(K.square(combination - base))

# combine these loss functions into a single scalar
loss = K.variable(0.)

feature_layers = [
        'block3_conv4_gpu1', ## extract feature reprentation of the deepest layer
        'block3_conv4_gpu2'
        ]
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    base_wav_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    cl = content_loss(base_wav_features, combination_features)
    loss += (content_weight / len(feature_layers)) * cl

feature_layers = [
        'block1_conv1_gpu1',
        'block1_conv1_gpu2',
        'block2_conv1_gpu1',
        'block2_conv1_gpu2',
        'block3_conv1_gpu1',
        'block3_conv1_gpu2'
        ]
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_wav)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_wav], outputs)


def eval_loss_and_grads(x):
    x = x.reshape((1, img_nrows, img_ncols, 1))
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
    x = np.reshape(x, (img_nrows, img_ncols))
    x = np.transpose(x, (1, 0))
    plt.ylabel('Frequency [HZ]')
    plt.xlabel('Time [not scaled]')
    plt.imshow(x)
    plt.savefig(str(fname))


# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss

if init_mode == 'noise':
    x = np.random.randn(1, img_nrows, img_ncols, 1) * 1e-6
else:
    x = preprocess_wav(base_wav_path, offset_base, total_samp)

plot_spectrogram(base_wav_data.copy(), 'base.png')
plot_spectrogram(style_reference_wav_data.copy(), 'style.png')

wav = None

for i in xrange(iterations):
    eprint('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=32)
    eprint('Current loss value:', min_val)
    # save current generated image
    if not result_only:
        wav = deprocess_wav(x.copy())
        fname = result_prefix + '_at_iteration_%d.wav' % i
        plot_spectrogram(x.copy(), fname+'.png')
        scipy.io.wavfile.write(fname, rate, wav)
        eprint('wav saved as', fname)
    end_time = time.time()
    eprint('Iteration %d completed in %ds' % (i, end_time - start_time))

wav = deprocess_wav(x.copy())
fname = result_prefix + 'result.wav'
plot_spectrogram(x.copy(), fname+'.png')
scipy.io.wavfile.write(fname, rate, wav)
