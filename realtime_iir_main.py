
#!/usr/bin/python3
"""
Plots channels zero and one in two different windows. Requires pyqtgraph.
"""

import sys

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from scipy import signal
from pyfirmata2 import Arduino


PORT = Arduino.AUTODETECT

# create a global QT application object
app = QtGui.QApplication(sys.argv)

# signals to all threads in endless loops that we'd like to run these
running = True

class QtPanningPlot:

    def __init__(self,title):
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle(title)
        self.plt = self.win.addPlot()
        self.plt.setYRange(-1,1)
        self.plt.setXRange(0,500)
        self.curve = self.plt.plot()
        self.data = []
        # any additional initalisation code goes here (filters etc)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(100)
        self.layout = QtGui.QGridLayout()
        self.win.setLayout(self.layout)
        self.win.show()
        
    def update(self):
        self.data=self.data[-500:]
        if self.data:
            self.curve.setData(np.hstack(self.data))

    def addData(self,d):
        self.data.append(d)
        
        #FIR FILTERING
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


class Speed:
    def __init__(self):
        self.rpm = 0
        self.count = 0
        self.flag = None
        self.rpm_history = [0]
        self.x = 0

   

    def speed_of_motor(self,signal):
        self.count += 1
        if self.x >=100:
            self.x = 0
        else:
            self.x += 1
        
        if signal < -0.05:
            self.flag = True
            
        if signal >0.05 and self.flag == True:
            
            self.rpm = 60 / (self.count*0.01)
            if len(self.rpm_history) < 100:
                self.rpm_history.append(self.rpm)
            else:
                self.rpm_history[self.x] = self.rpm
                
            self.count = 0
            
        if signal > 0.05:
           self.flag = False
        total = 0
        for i in self.rpm_history:
            total += i
        
        return round(total/len(self.rpm_history))
            
        

# Let's create two instances of plot windows
qtPanningPlot1 = QtPanningPlot("Sensor Input")
qtPanningPlot2 = QtPanningPlot("Filtered signal")

# sampling rate: 100Hz
samplingRate = 100

sos = signal.butter(6, 0.01, btype='highpass',output='sos')
iirfilter = IIRFilter(sos)
speedDetector = Speed()

# called for every new sample at channel 0 which has arrived from the Arduino
# "data" contains the new sample
def callBack(data):
    # filter your channel 0 samples here:
    # data = self.filter_of_channel0.dofilter(data)
    # send the sample to the plotwindow
    qtPanningPlot1.addData(data)
    ch1 = iirfilter.filter(data)
    # 1st sample of 2nd channel might arrive later so need to check
    if ch1:
        # filter your channel 1 samples here:
        # ch1 = self.filter_of_channel1.dofilter(ch1)
        qtPanningPlot2.addData(ch1)
        rpm = speedDetector.speed_of_motor(ch1)
        print(rpm)

# Get the Ardunio board.
board = Arduino(PORT)

# Set the sampling rate in the Arduino
board.samplingOn(1000 / samplingRate)

# Register the callback which adds the data to the animated plot
# The function "callback" (see above) is called when data has
# arrived on channel 0.
board.analog[0].register_callback(callBack)

# Enable the callback
board.analog[0].enable_reporting()


# showing all the windows
app.exec_()

# needs to be called to close the serial port
board.exit()

print("Finished")

