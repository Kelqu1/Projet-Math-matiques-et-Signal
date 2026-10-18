# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:16:26 2023

@author: JDION
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.io.wavfile as wavfile 
import sounddevice as sd  

from my_signal_td3 import Signal 
        
"""
Exercice 3
"""

fe = 44100

"""
Q1
"""

def read_test(wav_file):
    return None


"""
Q2
"""

#def play(wav_data):
    #sd.play(wav_data, fe)
    #sd.wait()
    

"""
Q3
"""

def plot(data):
    return None

"""
Q4
"""

def sinus(freq, t):
    return None


"""
Q5
"""

def chord(frequency_list, t):
    return None


"""
Q6
"""

DTMF_TABLE = {
    '1': [1209, 697],
    '2': [1336, 697],
    '3': [1477, 697],
    'A': [1633, 697],

    '4': [1209, 770],
    '5': [1336, 770],
    '6': [1477, 770],
    'B': [1633, 770],

    '7': [1209, 852],
    '8': [1336, 852],
    '9': [1477, 852],
    'C': [1633, 852],

    '*': [1209, 941],
    '0': [1336, 941],
    '#': [1477, 941],
    'D': [1633, 941],
} 

def DTMF(t):
    return None

    