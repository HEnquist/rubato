# Script for viewing the results from the fitting script.
# Needs numpy and matplotlib

import numpy as np
from matplotlib import pyplot as plt
import numpy.fft as fft
import math
from cutoff_fit_cubic import blackman_harris, blackman, hann, pad_vec, make_sinc, FACTOR, SINCLENGTHS, FS

windows_bh = []
windows_hann = []
windows_blackman = []

for sinclen in SINCLENGTHS:
    wind_bh = blackman_harris(sinclen*FACTOR)
    wind_blackman = blackman(sinclen*FACTOR)
    wind_hann = hann(sinclen*FACTOR)
    windows_bh.append(wind_bh)
    windows_blackman.append(wind_blackman)
    windows_hann.append(wind_hann)

waves = []
mins_bh = []
mins_bh2 = []
mins_blackman = []
mins_blackman2 = []
mins_hann = []
mins_hann2 = []

consts = [
    (8.041443677716476, 55.9506779343387, 898.0287985384213),
    (13.745202940783823, 121.73532586374934, 5964.163279612051),
    (6.159598046201173, 18.926415097606878, 653.4247430458968),
    (9.506235102129398, 79.13120634953742, 1502.2316160588925),
    (3.3481080887677166, 10.106519434875038, 78.96345249024414),
    (5.38751148378734, 29.69451915489501, 184.82117462266237),
]

def calc_cutoff(length, idx):
    return 1.0 / (consts[idx][0]/length + consts[idx][1]/length**2 + consts[idx][2]/length**3 + 1)

def plot_sinc_fft(sinc, fignbr, title):
    sinc = 2**15*sinc/np.sum(sinc)
    npoints = len(sinc)
    divfact = npoints/2
    f = np.linspace(0, FACTOR*FS/2.0, math.floor(npoints/2))
    valfft = fft.fft(sinc)
    cut = valfft[0:math.floor(npoints/2)]
    ampl = 20*np.log10(np.abs(cut)/divfact)
    plt.figure(fignbr)
    plt.plot(f, ampl)
    plt.title(title)
    plt.legend(SINCLENGTHS)

for w_bh, w_bm, w_h in zip(windows_bh, windows_blackman, windows_hann):
    sinc_len = len(w_bh)/FACTOR
    sinc = pad_vec(make_sinc(sinc_len, calc_cutoff(sinc_len, 0), FACTOR, 1, w_bh), 2**16)
    plot_sinc_fft(sinc, 1, "BlackmanHarris")
    sinc = pad_vec(make_sinc(sinc_len, calc_cutoff(sinc_len, 1), FACTOR, 2, w_bh), 2**16)
    plot_sinc_fft(sinc, 2, "BlackmanHarris2")
    sinc = pad_vec(make_sinc(sinc_len, calc_cutoff(sinc_len, 2), FACTOR, 1, w_bm), 2**16)
    plot_sinc_fft(sinc, 3, "Blackman")
    sinc = pad_vec(make_sinc(sinc_len, calc_cutoff(sinc_len, 3), FACTOR, 2, w_bm), 2**16)
    plot_sinc_fft(sinc, 4, "Blackman2")
    sinc = pad_vec(make_sinc(sinc_len, calc_cutoff(sinc_len, 4), FACTOR, 1, w_h), 2**16)
    plot_sinc_fft(sinc, 5, "Hann")
    sinc = pad_vec(make_sinc(sinc_len, calc_cutoff(sinc_len, 5), FACTOR, 2, w_h), 2**16)
    plot_sinc_fft(sinc, 6, "Hann2")

plt.show()

