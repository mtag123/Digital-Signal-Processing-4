#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 10:14:41 2019

@author: labuser
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class IIR2Filter:
    def __init__(self,coeffs):
        self.b0 = coeffs[0]
        self.b1 = coeffs[1]
        self.b2 = coeffs[2]
        self.a0 = coeffs[3]
        self.a1 = coeffs[4]
        self.a2 = coeffs[5]
        
        self.buffer1 = 0
        self.buffer2 = 0
        
        self.input_acc = 0
        self.output_acc = 0
    
    def filter(self,x):
        self.input_acc = 0
        self.output_acc = 0
        
        self.input_acc = x-(self.a1*self.buffer1)-(self.a2*self.buffer2)
        self.output_acc = (self.input_acc*self.b0)+(self.b1*self.buffer1)+(self.b2*self.buffer2)
        
        self.buffer2 = self.buffer1
        self.buffer1 = self.input_acc
        
        return self.output_acc

class IIRFilter:
    def __init__(self,_sos):
        self.sos = _sos
        self.nIIR2 = len(self.sos)
        
        self.longfilter = []
        
        for coeffs in self.sos:
            self.longfilter.append(IIR2Filter(coeffs))
       
        self.longfilter
    
    
    def filter(self,x):
        for i in range(self.nIIR2):
            x = self.longfilter[i].filter(x)
            
        return x
    
samplingRate = 100

sos1 = signal.butter(6, 0.5, btype='highpass',output='sos')
iirfilter1 = IIRFilter(sos1)
sos2 = signal.butter(6, 0.01, btype='highpass',output='sos')
iirfilter2 = IIRFilter(sos2)
time = np.linspace(0,1,10000)
faxis = np.linspace(0,samplingRate,len(time))
inputdata = np.zeros(len(time))
inputdata[5] = 1
outputdata1 = np.zeros(len(time))
outputdata2 = np.zeros(len(time))
for t in range(len(time)):
    outputdata1[t] = iirfilter1.filter(inputdata[t])
    
for t in range(len(time)):
    outputdata2[t] = iirfilter2.filter(inputdata[t])
    
plt.figure(figsize=(14,8))
plt.title("Filter Input")
plt.xlabel("Time")
plt.ylabel("Amplitude")    
plt.plot(time,inputdata)
plt.savefig("FilterInput.svg", format='svg')

plt.figure(figsize=(14,8))
plt.title("Impulse Response, Cutoff = 25Hz")
plt.xlabel("Time")
plt.ylabel("Amplitude")    
plt.plot(time,outputdata1)
plt.xlim(0,0.01) 
plt.savefig("Impulseresponse05.svg", format='svg')

plt.figure(figsize=(14,8))
plt.title("Impulse Response, Cutoff = 0.5Hz")
plt.xlabel("Time")
plt.ylabel("Amplitude") 
plt.xlim(0,0.01) 
plt.plot(time,outputdata2)
plt.savefig("Impulseresponse001.svg", format='svg')


iXF = np.fft.fft(inputdata)
plt.figure(figsize=(14,8))
plt.title("Frequency Input")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")    
plt.plot(faxis,abs(iXF.real))
plt.savefig("freqeuncyinput.svg", format='svg')


XF1 = np.fft.fft(outputdata1)
plt.figure(figsize=(14,8))
plt.title("Amplitude Response , Cutoff = 25Hz")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")    
plt.plot(faxis,abs(XF1.real))
plt.savefig("amplituderesponse05.svg", format='svg')

XF2 = np.fft.fft(outputdata2)
plt.figure(figsize=(14,8))
plt.title("Amplitude Response , Cutoff = 0.5Hz")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")    
plt.plot(faxis,abs(XF2.real))
plt.savefig("amplituderesponse001.svg", format='svg')
