# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import mne
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt
rawEEG = mne.io.read_raw_edf('EDF_EEG.edf', preload=True)
rawEEG.plot(block=False, duration=10.0, title='raw EEG data')
rawEEG.info
print(rawEEG.info['sfreq'])
print(rawEEG.info['nchan'])
print(rawEEG.info['ch_names'])

"""Filtering"""
rawEEG.filter(0.1, 30)
rawEEG.plot(block=False, duration=10.0, title='filtered EEG data')

EEGfiltered = rawEEG.copy().filter(0.1, 40)
EEGfiltered.save('EEGfiltered.fif')

"""Creating a standard EEG montage"""
montage = mne.channels.make_standard_montage('standard_1020')
rawEEG, montage(montage)
rawEEG.plot()
