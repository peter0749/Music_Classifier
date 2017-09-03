# -*- coding: utf-8 -*-
import numpy as np
import scipy
import scipy.io.wavfile

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

