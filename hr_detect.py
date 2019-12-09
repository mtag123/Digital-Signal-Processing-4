# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

#FIR FILTERING
class FIR_filter:
    def __init__(self,_coefficients):
        self.nTaps = len(_coefficients)
        self.coefficients = _coefficients
        self.filter_states = np.zeros(len(_coefficients))
        self.offset = 0
    
    def dofilter(self,v):
        if self.offset >= self.nTaps:
            self.offset = 0
        self.filter_states[self.offset] = v
        result = 0
        for i in range(len(self.coefficients)):
            
            result+=self.filter_states[(i+self.offset)%self.nTaps]*self.coefficients[len(self.coefficients)-i-1]
        
        self.offset+=1
        
        return result

class matched_filter(FIR_filter):
    def detect(self,v):
        if self.offset >= self.nTaps:
            self.offset = 0
        self.filter_states[self.offset] = v
        result = 0
        for i in range(len(self.coefficients)):
            
            result+=self.filter_states[(i+self.offset)%self.nTaps]*self.coefficients[len(self.coefficients)-i-1]
        
        self.offset+=1
        
        return result**2


def create_filter_coefficients():
    #Number of taps
    M = 200

    #For removing 50Hz
    k1 = int(30/fs * M)
    k2 = int(500/fs * M)

    #For removing DC
    k3 = int(0/fs * M)
    k4 = int(0/fs * M)

    #Create coefficient array
    X = np.ones(M)

    #Set unwatned frequences to 0
    X[k1:k2+1] = 0
    X[M-k2:M-k1+1] = 0

    X[k3:k4+1] = 0 #0,1
    X[M-k4:M-k3+1] = 0 #m-1,m-0+1

    #Perform inverse Fourier transform to get filter coefficients
    x = np.fft.ifft(X)
    x = np.real(x)
    
    
    #Swap t nd -t valeus to correct places
    h = np.zeros(M)
    h[0:int(M/2)] = x[int(M/2):M]
    h[int(M/2):M] = x[0:int(M/2)]
    
    return h, X



def convert_to_mV(input_array):
    voltage_graph = np.zeros(len(input_array))
    
    #It's really 24 bits but the ADC runs from -1.325V to +1.325V and
    #all of our values are positive, so we can pretend its 23 bits
    adc_bits = 23
    gain = 500
    max_voltage = 1.325
    
    voltage_graph = input_array*((max_voltage/(2**adc_bits-1))/gain)*1000
    
    return voltage_graph



#GRAPH PLOTTING
def plot_ecg2_graphs(time,ecgin,template,output):
    
    #Plot unfiltered signal
    plt.figure(figsize=(14,8))
    plt.title('Unfiltered ECG')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    plt.plot(time/1000,ecgin)
    plt.savefig('ecg2in.svg')
    
    #Plot matched filter impulse reponse
    plt.figure(figsize=(14,8))
    plt.plot(template)
    plt.title('Mexican Hat Template')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.savefig('Matched_filter_template.svg')
    
    #Blot heartbeat detection
    plt.figure(figsize=(14,8))
    plt.title('Beat Detection')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.plot(time/1000,output)
    plt.savefig('beat_detection.svg')


#TEMPLATE CREATION
def create_mexican_hat():
    time = np.linspace(-5,5, 200)
    output = np.zeros(len(time))
    
    for i, t in enumerate(time):
        output[i] = ((2/np.sqrt(3))*np.pi**(-1/4))*(1-t**2)*np.exp(-(t**2)/2)
    
    return output


def create_gaussian():
    time = np.linspace(-5,5, 200)
    output = np.zeros(len(time))
    a = 1/(np.sqrt(0.2)*np.sqrt(2*np.pi))
    b = 0
    c = np.sqrt(0.2)
    
    for i, t in enumerate(time):
        output[i] = a*np.exp(-(((t-b)**2))/(2*(c**2)))
    
    return output 


#HEARTBEAT DETECTION
class BPM_detector:
    def __init__(self):
        self.previous_beat = 0
        self.current_beat = 0
        self.BPM = 0
        
    def momentary_heart_rate(self,matched_signal,time):
        if matched_signal > 1e14 and matched_signal < 4e14:
            self.current_beat = time
        if self.current_beat - self.previous_beat > 405:
            self.temp_BPM = 60000/(self.current_beat-self.previous_beat)

            if time<=500:
                self.BPM = 60000/(self.current_beat-self.previous_beat)
                self.previous_BPM = self.BPM

                
            #ignore BPM changes greater than 8
            if time>500 and abs(self.temp_BPM - self.previous_BPM) < 8:
                self.BPM = 60000/(self.current_beat-self.previous_beat)
                self.previous_BPM = self.BPM

            
            self.previous_beat = self.current_beat
    
        return self.BPM
            

       
               

#ACTUAL PROGRAM


#Read in ECG data
data = np.loadtxt('ecg2.dat')

fs = 1000

#Get time and values from the channel we want
time = data[:,0]
channel = data[:,1]

ecgin_voltage = convert_to_mV(channel)

mexican_hat = create_mexican_hat()
gaussian = create_gaussian()
               
#Get FIR filter coefficients
h, spectrum = create_filter_coefficients()

#Create FIR filter
myfilter = FIR_filter(h)

#Create matchd filter
mydetector = matched_filter(mexican_hat)

beat_detector = BPM_detector()

#Perform filtering and save in output array
output = np.zeros(len(channel))
detection = np.zeros(len(output))
BPM = np.zeros(len(detection))

#Output of FIR filter feeds straight into matched filter for causal processing
for i in range(len(channel)):
    output[i] = (myfilter.dofilter(channel[i]))
    detection[i] = (mydetector.detect(output[i]))
    
    #Calculate momentary heartrate by time difference between the peaks
    BPM[i] = beat_detector.momentary_heart_rate(detection[i],i)
        

detection_voltage = convert_to_mV(detection)
plot_ecg2_graphs(time,ecgin_voltage,mexican_hat,detection)



plt.figure(figsize=(14,8))
plt.title('Beats per minute')
plt.xlabel('Time (s)')
plt.ylabel('Beats per minute')
plt.ylim(100,200)
plt.plot(np.linspace(0,max(time),len(BPM))/1000,BPM)   

plt.show()