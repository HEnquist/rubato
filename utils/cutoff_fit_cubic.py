# Script for fitting the relative cutoff as function of sinc length, for the various window functions.
# Needs numpy, scipy and matplotlib

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize, curve_fit
from matplotlib import pyplot as plt
import numpy.fft as fft
import math

def blackman_harris(npoints):
    x=np.arange(0,npoints)
    y= 0.35875 - 0.48829*np.cos(2*np.pi*x/npoints) + 0.14128*np.cos(4*np.pi*x/npoints) - 0.01168*np.cos(6*np.pi*x/npoints)
    return y

def blackman(npoints):
    x=np.arange(0,npoints)
    y= 0.42 - 0.5*np.cos(2*np.pi*x/npoints) + 0.08*np.cos(4*np.pi*x/npoints)
    return y

def sine(npoints):
    x=np.arange(0,npoints)
    y= np.sin(np.pi*x/npoints)
    return y

def raised_cosine(npoints, a0):
    x=np.arange(0,npoints)
    y= a0 - (1-a0)*np.cos(2*np.pi*x/npoints)
    return y

def hann(npoints):
    a0=0.5
    return raised_cosine(npoints, a0)

def rect(npoints):
    a0=1.0
    return raised_cosine(npoints, a0)

def hamming(npoints):
    a0=0.53836
    return raised_cosine(npoints, a0)

def make_sinc(npoints, cutoff, factor, power, window):
    totpoints = npoints*factor
    x=np.arange(-totpoints/2, totpoints/2)
    y=np.sinc(x*cutoff/factor)
    for n in range(power):
        y=y*window
    return y

def pad_vec(vec, npoints):
    ynew = np.zeros(npoints)
    ynew[0:len(vec)] = vec
    return ynew

FACTOR = 10
SINCLENGTHS = [32, 48, 64, 96, 128, 160, 256, 320, 512, 768, 1024, 1536, 2048]
FS=20000

windows_bh = []
windows_hann = []
windows_blackman = []
labels = []
windows = {"BlackmanHarris": [], "Blackman": [], "Hann": []}


for sinclen in SINCLENGTHS:
    wind_bh = blackman_harris(sinclen*FACTOR)
    wind_blackman = blackman(sinclen*FACTOR)
    wind_hann = hann(sinclen*FACTOR)
    windows["BlackmanHarris"].append(wind_bh)
    windows["Blackman"].append(wind_blackman)
    windows["Hann"].append(wind_hann)


waves = []
mins = {}

def get_first_min(sinc):
    npoints = len(sinc)
    divfact = npoints/2
    f = np.linspace(0, FACTOR*FS/2.0, math.floor(npoints/2))
    valfft = fft.fft(sinc)
    cut = valfft[0:math.floor(npoints/2)]
    ampl = 20*np.log10(np.abs(cut)/divfact)
    minima, _ = find_peaks(-ampl, height=100)
    #print(minima[0])
    #plt.figure(10)
    #plt.plot(f, ampl, f[minima[0]], ampl[minima[0]], '*')
    #print("min at", f[minima[0]])
    return f[minima[0]]

def plot_sinc_fft(cutoff, window, power):
    sinc_len = len(window)/FACTOR
    sinc = pad_vec(make_sinc(sinc_len, cutoff, FACTOR, power, window), 2**16)
    divfact = sinc_len/2
    f = np.linspace(0, FACTOR*FS/2.0, math.floor(len(sinc)/2))
    valfft = fft.fft(sinc)
    cut = valfft[0:math.floor(len(sinc)/2)]
    ampl = 20*np.log10(np.abs(cut)/divfact)
    minima, _ = find_peaks(-ampl, height=100)
    plt.figure()
    plt.plot(f, ampl)
    plt.axvline(x = f[minima[0]])


if __name__ == "__main__":
    cutoffs = {"BlackmanHarris": [[], []], "Blackman": [[], []], "Hann": [[], []]}

    # Fit the cutoff frequency to place the first minimum at the desired frequency.
    for name, winds in windows.items():
        for power in range(2):
            for wind in winds:
                def get_offset(cutoff):
                    sinc_len = len(wind)/FACTOR
                    sinc = pad_vec(make_sinc(sinc_len, cutoff, FACTOR, power+1, wind), 2**16)
                    diff = get_first_min(sinc) - FS/2
                    return abs(diff)
                res = minimize(get_offset, [1.0], method='Nelder-Mead', tol=1e-7)
                cutoffs[name][power].append(res.x[0]) 
                #plot_sinc_fft(res.x[0], wind, power+1)

    def func(x, a, b, c):
        return 1/(a/x + b/x**2 +c/x**3 + 1)

    fignbr = 100
    constants = {"BlackmanHarris": [], "Blackman": [], "Hann": []}
    for name, powers in cutoffs.items():
        for power, values in enumerate(powers):
            popt, pcov = curve_fit(func, SINCLENGTHS, values)
            constants[name].append(popt[0:3])
            fitted = [func(l, popt[0], popt[1], popt[2]) for l in SINCLENGTHS]
            residuals = [f - d for f, d in zip(fitted, values)]
            fig, axs = plt.subplots(2, num=fignbr)
            axs[0].plot(SINCLENGTHS, values, '*', SINCLENGTHS, fitted, '-')
            axs[0].set_title(f"{name}, power {power+1}")
            axs[0].set_xlabel("sinc length")
            axs[0].set_ylabel("relative cutoff")
            axs[1].plot(SINCLENGTHS, residuals)
            axs[1].set_title(f"{name}, power {power+1}")
            axs[1].set_xlabel("sinc length")
            axs[1].set_ylabel("fit residual")
            fignbr += 1

    print("\nFitting results:")
    for name, values in constants.items():
        print(name)
        for power in range(2):
            print(f"{power+1}: {values[power]}")

    print("\nCopy to check script:")
    print("consts = [")
    for name, values in constants.items():
        for power in range(2):
            vals = [str(v) for v in values[power]]
            print(f"    ({', '.join(vals)}),")
    print("]")
    print("")

    print("\nCopy to windows.rs:")
    for name, values in constants.items():
        for power in range(2):
            print(f"    WindowFunction::{name}{power + 1 if power>0 else ''} => (")
            for val in values[power]:
                print(f"        T::coerce({val}),")
            print("    ),")

    plt.show()
