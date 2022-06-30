# Make a simple spike for testing purposes
import numpy as np


#wave64 = np.zeros((2,44100), dtype="float64")
#wave32 = np.zeros((2,44100), dtype="float32")
t = np.linspace(0, 1.0, num=int(1.0*44100), endpoint=False)
wave = np.sin(10000*2*np.pi*t)
wave= np.reshape(wave,(-1,1))
wave = np.concatenate((wave, wave), axis=1)

wave64 = wave.astype('float64')
wave32 = wave.astype('float32')


#print(wave64)
wave64.tofile("sine_44.1_10000_f64_2ch_1.0s.raw")
wave32.tofile("sine_44.1_10000_f32_2ch_1.0s.raw")


