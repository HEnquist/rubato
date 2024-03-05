# Make a sine for testing purposes
# Store as interleaved stereo
import numpy as np

t = np.linspace(0, 10.0, num=int(10.0*44100), endpoint=False)
wave = np.sin(1000*2*np.pi*t)
wave= np.reshape(wave,(-1,1))
wave = np.concatenate((wave, wave), axis=1)

wave64 = wave.astype('float64')
wave32 = wave.astype('float32')


#print(wave64)
wave64.tofile("sine_44.1_1000_f64_2ch_10.0s.raw")
wave32.tofile("sine_44.1_1000_f32_2ch_10.0s.raw")


