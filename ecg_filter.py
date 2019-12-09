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



def create_filter_coefficients():
    #Number of taps
    M = 200

    #For removing 50Hz
    k1 = int(45/fs * M)
    k2 = int(100/fs * M)

    #For removing DC
    k3 = int(0/fs * M)
    k4 = int(0/fs * M)
    
     #For removing DC
    k5 = int(100/fs * M)
    k6 = int(500/fs * M)


    #Create coefficient array
    X = np.ones(M)

    #Set unwatned frequences to 0
    X[k1:k2+1] = 0
    X[M-k2:M-k1+1] = 0

    X[k3:k4+1] = 0 #0,1
    X[M-k4:M-k3+1] = 0 #m-1,m-0+1
    
    X[k5:k6+1] = 0 #0,1
    X[M-k6:M-k5+1] = 0 #m-1,m-0+1

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
    
   #It's really 24 bits but the ADC runs from -1.325V to +1.325V so we can
    #pretend its 23 bits either side of 0
    adc_bits = 23
    gain = 500
    max_voltage = 1.325
    
    voltage_graph = input_array*((max_voltage/(2**adc_bits-1))/gain)*1000
    
    return voltage_graph

#GRAPH PLOTTING
def plot_ecg1_graphs(time,faxis,ecgin,spectrum,h,faxis_ecg,ecgout):
    
    #Plot unfiltered signal
    plt.figure(figsize=(14,8))
    plt.title('Unfiltered ECG')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    plt.plot(time/1000,ecgin)
    plt.savefig('ecgunfiltered.svg',format='svg')
    
    plt.figure(figsize=(14,8))
    plt.title('Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.plot(faxis,spectrum)
    plt.savefig('filter_spectrum.svg',format='svg')
             
    plt.figure(figsize=(14,8))
    plt.title('Impulse Response')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.plot(h)
    plt.savefig('FIR_filter.svg',format='svg')  
    
    plt.figure(figsize=(14,8))
    plt.title('Filtered ECG')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    plt.plot(time/1000,ecgout)
    plt.savefig('ecgfiltered.svg',format='svg')
    
    single_heartbeat = ecgout_voltage[1400:2100]
    plt.figure(figsize=(14,8))
    plt.title('Single Beat')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.plot(single_heartbeat)
    plt.savefig('Single_beat.svg',format='svg')
    
    unfiltered_spectrum = np.fft.fft(ecgin_voltage)
    plt.figure(figsize=(14,8))
    plt.title('Unfiltered Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.ylim(0,600)
    plt.plot(faxis_ecg,abs(unfiltered_spectrum))
    plt.savefig('unfiltered_spectrum.svg',format='svg')
    
    filtered_spectrum = np.fft.fft(ecgout_voltage)
    plt.figure(figsize=(14,8))
    plt.title('Filtered Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.plot(faxis_ecg,abs(filtered_spectrum))
    plt.savefig('filtered_spectrum.svg',format='svg')
    


#TEMPLATE CREATION
def create_mexican_hat():
    time = np.linspace(-5,5, 100)
    output = np.zeros(len(time))
    
    for i, t in enumerate(time):
        output[i] = ((2/np.sqrt(3))*np.pi**(-1/4))*(1-t**2)*np.exp(-(t**2)/2)
    
    return output

def create_gaussian():
    time = np.linspace(-5,5, 100)
    output = np.zeros(len(time))
    a = 1/(np.sqrt(0.2)*np.sqrt(2*np.pi))
    b = 0
    c = np.sqrt(0.2)
    
    for i, t in enumerate(time):
        output[i] = a*np.exp(-(((t-b)**2))/(2*(c**2)))
    
    return output 


#HEARTBEAT DETECTION
def detect_beat(matched_signal):
    BPM = []
    previous_beat = 0
    current_beat = 0
    for i in range(len(matched_signal)):
        if matched_signal[i] > 2*(10**15):
            current_beat = i
        if current_beat - previous_beat > 300:
           BPM.append(60000/(current_beat-previous_beat))
           previous_beat = current_beat
    return BPM  
           

#ACTUAL PROGRAM


#Read in ECG data
data = np.loadtxt('ecg1.dat')

fs = 1000

#Get time and values from the channel we want
time = data[:,0]
channel = data[:,2]
ecgin_voltage = convert_to_mV(channel)


    
#Get FIR filter coefficients
h, spectrum = create_filter_coefficients()

faxis = np.linspace(0,fs, len(spectrum))
faxis_ecg = np.linspace(0,fs, len(channel))

#Create FIR filter
myfilter = FIR_filter(h)

#Perform filtering and save in output arra
output = np.zeros(len(channel))
for i in range(len(channel)):
    output[i] = (myfilter.dofilter(channel[i]))
        
ecgout_voltage =convert_to_mV(output)

#Plot the necessary graphs 
plot_ecg1_graphs(time,faxis,ecgin_voltage,spectrum,h,faxis_ecg,ecgout_voltage)

plt.show()  






