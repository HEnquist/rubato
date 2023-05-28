# Make a signal for testing purposes
import numpy as np

t = np.linspace(0, 1.0, num=int(1.0*44100), endpoint=False)

#freqs = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
#ampls = [500000/(f*f) for f in freqs]

freqs = list(range(1000, 21000, 1000)) 
ampls = [1.0/len(freqs)]*len(freqs)

wave = np.zeros(len(t))
for f, a in zip(freqs, ampls):
    wave = wave + a * np.sin(f*2*np.pi*t)
wave= np.reshape(wave,(-1,1))
wave = np.concatenate((wave, wave), axis=1)

wave64 = wave.astype('float64')
wave32 = wave.astype('float32')


#print(wave64)
wave64.tofile("multi_44.1_f64_2ch_1.0s.raw")
wave32.tofile("multi_44.1_f32_2ch_1.0s.raw")


