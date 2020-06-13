# Make a simple spike for testing purposes
import numpy as np


spike = np.zeros(2048, dtype="float64")
spike[0] = 1.0
spike[512]=1.0

spike.tofile("spike_1024.raw")