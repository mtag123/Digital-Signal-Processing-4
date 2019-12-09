# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:13:05 2019

@author: Martin Greer (2138689g) and Andrew Ritchie (2253409R)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

#Import audio data
fs, data = wavfile.read('./original.wav')

#Create a time axis in seconds
time = np.zeros(len(data))
for i in range(len(data)):
    time[i] = i/fs
    

#Plot the audio signal
plt.figure(figsize=(14,8))      
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')  
plt.title('Raw Audio Data')
plt.plot(time,data)
plt.show()
plt.savefig('RawAudio.svg',format='svg')

#Perform fft and remove mirror
fdata = np.fft.fft(data)/len(data)
half = np.split(fdata,2)

#Create freqeuncy axes in Hz with and without mirror
faxis = np.linspace(0,fs, len(fdata))
faxis_without_mirror = np.linspace(0,fs, len(half[0]))


#Plot fourier spectrum on log scales
db_values = 20*np.log10(abs(half[0]))
plt.figure(figsize=(14,8))     
plt.xscale("log")
plt.xlabel("Frequency")
plt.ylabel("dB")
plt.title('Spectrum')
plt.plot(faxis_without_mirror,db_values)
plt.show()
plt.savefig('Spectrum.svg',format='svg')

#Alters spectrum between two frequencies linearly between two decibel amounts
def alter(flow,fhigh,dbstart,dbend):
    number_elements = faxis[(faxis>flow) & (faxis<fhigh)]
    step = np.linspace(dbstart,dbend,len(number_elements))
    x = 0
    for i in range(len(faxis)):
        if flow<faxis[i]<fhigh:
            fdata[i] *= (10**(step[x]/20))
            fdata[-i] *= (10**(step[x]/20))
            x += 1
            
#Set values in frequency range equal to 0
def zero_values(flow, fhigh):
    for i in range(len(faxis)):
        if flow<faxis[i]<fhigh:
            fdata[i] = 0
            fdata[-i] = 0
            
#Alter frequency spectrum by values based on literature and our own tests
zero_values(0,50) # HPF cut low frequencies
alter(325, 350, -6, -6)# cut to make voice more clear
alter(200,600, 5, 5) # boost to enhance bass
alter(2500, 4000, -5, -5) # cut to make voice less harsh

# cut any sibilance
alter(5000,55000, -5, -5) 
alter(5500,6000, -5, -5)
alter(6000,6500, -10, -10)
alter(6500,8000, -10, -10)

# boost to enhance the voice
alter(8000, 10000,5, 5)
alter(200, 250,10, 10)

alter(10000, fs/2, 0, -15) #roll off high frequencies



#Create new decibel values for improved spectrum
half = np.split(fdata,2)       
improved_db_values = np.zeros(len(half[0]))

#If we set any dB values to 0, just make 
#them null for the plot to avoid flat lines
for i in range(len(half[0])):
    if fdata[i] != 0:
       improved_db_values[i] = (20*np.log10(abs(fdata[i])))
    else:
        improved_db_values[i] = None

#Plot improved fourier spectrum on log scales
plt.figure(figsize=(14,8))     
plt.xscale("log")
plt.xlabel("Frequency")
plt.ylabel("dB")
plt.title('Improved Spectrum')
plt.plot(faxis_without_mirror,improved_db_values)
plt.show()
plt.savefig('ImprovedSpectrum.svg',format='svg')



#Perform inverse forier transform
improved_data = np.fft.ifft(fdata).real

#Increase amplitude of audio data
improved_data*=3

#Create new time axis for improved audio
time2 = np.zeros(len(improved_data))
for i in range(len(improved_data)):
    time2[i] = i/fs
    
#Plot new audio signal
plt.figure(figsize=(14,8))      
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')  
plt.title('Improved Audio Data')
plt.plot(time2,improved_data)
plt.show()
plt.savefig('ImprovedAudio.svg',format='svg')

#Write new audio file
wavfile.write("improved.wav",fs,improved_data)
