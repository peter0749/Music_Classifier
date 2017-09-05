# -*- coding: utf-8 -*-
import numpy as np
import scipy
import scipy.io.wavfile

def preprocess_wav(wavfile, offset, sample_n):
    rate, wavData = scipy.io.wavfile.read(str(wavfile))
    offset *= rate
    if wavData.ndim>1:
        wavData = wavData[...,0]
    wavData = wavData[offset:offset+sample_n] ## skip sample
    wavData = wavData.astype(np.float32)
    wavData = np.reshape(wavData, (1, sample_n))
    return wavData

def deprocess_wav(x):
    return x.flatten().clip(-32768, 32767).astype(np.int16)

