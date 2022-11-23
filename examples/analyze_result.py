

import numpy as np
import numpy.fft as fft
import sys
from matplotlib import pyplot as plt
import math

def blackman_harris(npoints):
    x=np.arange(0,npoints)
    y= 0.35875 - 0.48829*np.cos(2*np.pi*x/npoints) + 0.14128*np.cos(4*np.pi*x/npoints) - 0.01168*np.cos(6*np.pi*x/npoints)
    return y

def plot_spect(indata, window=True):
    
    for wf, fs in indata:
        print(sum(wf))
        npoints = len(wf)
        divfact = npoints/2
        if window:
            wind = blackman_harris(npoints)
            wf = wf*wind*wind
            divfact = sum(wind)/2
        print(npoints)
        t = np.linspace(0, npoints/fs, npoints, endpoint=False) 
        f = np.linspace(0, fs/2.0, math.floor(npoints/2))
        valfft = fft.fft(wf)
        cut = valfft[0:math.floor(npoints/2)]
        ampl = 20*np.log10(np.abs(cut)/divfact)
        phase = 180/np.pi*np.angle(cut)
        #plt.subplot(2,1,1)
        plt.figure(1)
        plt.plot(f, ampl)
        #plt.subplot(2,1,2)
        #plt.semilogx(f, phase)

        #plt.gca().set(xlim=(10, srate/2.0))
        #plt.subplot(2,1,2)
        plt.figure(2)
        plt.plot(wf)
        plt.figure(3)
        plt.plot(np.diff(wf,5))
    plt.show()

file_in = sys.argv[1]
channels = int(sys.argv[2])
bits = int(sys.argv[4])
srate = int(sys.argv[3])

if bits == 64:
    values = np.fromfile(file_in, dtype=float)
elif bits == 32:
    values = np.fromfile(file_in, dtype=np.float32)


# Let's look at the first channel only..
values = values.reshape((-1,channels))
values = values[:,0]
plot_spect([(values, srate)], window=True)



