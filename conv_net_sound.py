# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, Flatten, Lambda, RepeatVector, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras import losses
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.engine.topology import Layer
from keras import regularizers

def conv_net(input_tensor = None,
               input_shape = None,
               weight_path = None
               ):
    dev1 = '/gpu:0'
    dev2 = '/gpu:1'
    if input_tensor is None:
        input_layer = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            input_layer = Input(tensor=input_tensor, shape=input_shape)
        else:
            input_layer = input_tensor

    ## VGG19 Net:
    # Block 1
    with tf.device(dev1):
        x1_0 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1_gpu1', kernel_initializer='random_normal')(input_layer)
        x1_0 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool1_gpu1')(x1_0)
    with tf.device(dev2):
        x2_0 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1_gpu2', kernel_initializer='random_normal')(input_layer)
        x2_0 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool1_gpu2')(x2_0)
    with tf.device(dev1):
        x1_1 = concatenate([x1_0, x2_0], axis=-1)
        x1_1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2_gpu1', kernel_initializer='random_normal')(x1_1)
        x1_1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool2_gpu1')(x1_1)
    with tf.device(dev2):
        x2_1 = concatenate([x1_0, x2_0], axis=-1)
        x2_1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2_gpu2', kernel_initializer='random_normal')(x2_1)
        x2_1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool2_gpu2')(x2_1)

    # Block 2
    with tf.device(dev1):
        x1_2 = concatenate([x1_1, x2_1], axis=-1)
        x1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1_gpu1', kernel_initializer='random_normal')(x1_2)
    with tf.device(dev2):
        x2_2 = concatenate([x1_1, x2_1], axis=-1)
        x2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1_gpu2', kernel_initializer='random_normal')(x2_2)
    with tf.device(dev1):
        x1_3 = concatenate([x1_2, x2_2], axis=-1)
        x1_3 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2_gpu1', kernel_initializer='random_normal')(x1_3)
        x1_3 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool_gpu1')(x1_3)
    with tf.device(dev2):
        x2_3 = concatenate([x1_2, x2_2], axis=-1)
        x2_3 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2_gpu2', kernel_initializer='random_normal')(x2_3)
        x2_3 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool_gpu2')(x2_3)

    # Block 3
    with tf.device(dev1):
        x1_4 = concatenate([x1_3, x2_3], axis=-1)
        x1_4 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1_gpu1', kernel_initializer='random_normal')(x1_4)
    with tf.device(dev2):
        x2_4 = concatenate([x1_3, x2_3], axis=-1)
        x2_4 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1_gpu2', kernel_initializer='random_normal')(x2_4)
    with tf.device(dev1):
        x1_5 = concatenate([x1_4, x2_4], axis=-1)
        x1_5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2_gpu1', kernel_initializer='random_normal')(x1_5)
        x1_5 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool1_gpu1')(x1_5)
    with tf.device(dev2):
        x2_5 = concatenate([x1_4, x2_4], axis=-1)
        x2_5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2_gpu2', kernel_initializer='random_normal')(x2_5)
        x2_5 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool1_gpu2')(x2_5)
    with tf.device(dev1):
        x1_6 = concatenate([x1_5, x2_5], axis=-1)
        x1_6 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv3_gpu1', kernel_initializer='random_normal')(x1_6)
    with tf.device(dev2):
        x2_6 = concatenate([x1_5, x2_5], axis=-1)
        x2_6 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv3_gpu2', kernel_initializer='random_normal')(x2_6)
    with tf.device(dev1):
        x1_7 = concatenate([x1_6, x2_6], axis=-1)
        x1_7 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv4_gpu1', kernel_initializer='random_normal')(x1_7)
        x1_7 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool2_gpu1')(x1_7)
    with tf.device(dev2):
        x2_7 = concatenate([x1_6, x2_6], axis=-1)
        x2_7 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv4_gpu2', kernel_initializer='random_normal')(x2_7)
        x2_7 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool2_gpu2')(x2_7)
    with tf.device(dev1):
        x = concatenate([x1_7, x2_7], axis=-1)
    model = Model(input_layer, x) ## model: wav -> features
    if weight_path is not None:
        model.load_weights(str(weight_path))
    return model

