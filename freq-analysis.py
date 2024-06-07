# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 19:16:18 2024

@author: varsh
"""

# importing one channel EEG in text format
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('One_Ch_EEG.txt')
print(data.shape)
plt.plot(data)

#Changing to seconds frompoints
srate = 256
# creating a time vector based on length of data and sampling rate
time  = np.arange(len(data))/srate
plt.plot(time, data)
plt.xlabel('time')
plt.ylabel('amplitude')

#calculating power spectrum using FFT
powerSpec = np.fft.fft(data)
freqs = np.fft.fftfreq(data.size, d=1/256)
#ploting the result
fig, ax = plt.subplots(1, 2)
ax[0].plot(time, data)
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Amplitude (uV)')

#Creating a mask to retain only positive frequencies
masks = freqs > 0
freqsPos = freqs[masks]
specPos = np.abs(powerSpec)[masks]
ax[1].plot(freqsPos, specPos)
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Amplitude (uV)')
ax[1].set_title('Amplitude spectrum')
ax[1].set_xlim([1, 40])

#Frequency analysis
import mne
from mne.preprocessing import ica
import matplotlib.pyplot as plt
%matplotlib qt    

#Importing raw EEG data and importing into memory
EEG = mne.io.read_raw_fif('EEGFiltered.fif', preload=True)

#Filtering the data
EEG.filter(0.1, 30)

#Creating the standard montage
montage = mne.channels.make_standard_montage('standard_1020')
#Setting monatage to the EEG data
EEG.set_montage(montage)

#Generating power spectral density 
psd, freqs = mne.time_frequency.psd_array_multitaper(EEG.get_data(), EEG.info['sfreq'], fmin=0.5, fmax=30)
psd.shape

#PSD for 19 channels
#Creating subplots for visulaizing PSD for all channels
fig, ax = plt.subplots(figsize = (20, 10))
for i in range(len(EEG.ch_names)):
    ax.plot(freqs, psd[i], label=EEG.ch_names[i])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD for 19  channels')
plt.legend()
plt.show()
EEG.ch_names

#PSD for desired channel
plt.plot(freqs, psd[18], label='02')
plt.legend()

#Printing indices and names for all channels
for i in range (len(EEG.ch_names)):
    print(i, EEG.ch_names[i])
plt.plot(freqs, psd[8], label='C3')
plt.plot(freqs, psd[10], label='C4')
plt.legend()
plt.show()

#Custom topomaps
im, _ = mne.viz.plot_topomap(psd.mean(axis=-1), pos=EEG.info, cmap='RdBu_r', names=EEG.info['ch_names'])
plt.colorbar()

#Topomaps for alpha,beta and theta ranges
#Defining g=frquency bands of interest
freqBands = [(3,7), (8, 12), (13, 30)]
#Creating subplots for each frequency band
fig, axes = plt.subplots(1, 3, figsize=(15,5))
#Looping for band
for i, (fmin, fmax) in enumerate(freqBands):
    ax = axes[i]
    #Computing PSD
    psd, freqs = mne.time_frequency.psd_array_multitaper(EEG.get_data(), sfreq=EEG.info['sfreq'], fmin=fmin, fmax=fmax)
    #Plotting topomap for average PSD across channels
    mne.viz.plot_topomap(psd.mean(axis=-1), pos=EEG.info, cmap='RdBu_r', axes=ax, names=EEG.info['ch_names'])
    #Setting title of subplot
    ax.set(title='PSD({:.1f}-{:.1f})Hz'.format(fmin, fmax))