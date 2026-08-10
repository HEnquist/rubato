# Make a multitone signal for testing purposes.
# This is a comb of equally spaced tones of equal amplitude, for looking at aliasing
# and intermodulation across the whole band at once, which a single sine cannot show.
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("samplerate", type=int,
                    help="sample rate of file")
parser.add_argument("channels", type=int,
                    help="number of channels")
parser.add_argument("length", type=float,
                    help="length in seconds")
parser.add_argument("--first", type=float, default=1000.0,
                    help="frequency of the lowest tone (default: %(default)s)")
parser.add_argument("--last", type=float, default=20000.0,
                    help="highest frequency to place a tone at (default: %(default)s)")
parser.add_argument("--spacing", type=float, default=1000.0,
                    help="spacing between the tones (default: %(default)s)")

args = parser.parse_args()

# Add half a step to the end, so that the last tone is included despite rounding.
freqs = np.arange(args.first, args.last + args.spacing / 2, args.spacing)

# Tones at or above Nyquist would alias in the generated signal itself,
# which would defeat the purpose of the test.
nyquist = args.samplerate / 2
if np.any(freqs >= nyquist):
    dropped = np.count_nonzero(freqs >= nyquist)
    print(f"Skipping {dropped} tone(s) at or above the Nyquist frequency of {nyquist:.0f} Hz")
    freqs = freqs[freqs < nyquist]
if len(freqs) == 0:
    raise SystemExit("No tones below the Nyquist frequency, nothing to generate")

# Equal amplitudes, scaled so that the sum of all tones cannot clip.
ampls = [1.0 / len(freqs)] * len(freqs)

t = np.linspace(0, args.length, num=int(args.length*args.samplerate), endpoint=False)
wave = np.zeros(len(t))
for f, a in zip(freqs, ampls):
    wave = wave + a * np.sin(f*2*np.pi*t)
wave= np.reshape(wave,(-1,1))
wave = np.concatenate((wave,)*args.channels, axis=1)

srkhz = args.samplerate/1000

fname = (f"multi_{srkhz:.1f}kHz_{freqs[0]/1000:.1f}-{freqs[-1]/1000:.1f}kHz_"
         f"{args.channels}ch_{args.length:.1f}s_f64.raw")

print(f"{len(freqs)} tones from {freqs[0]:.0f} to {freqs[-1]:.0f} Hz")
print("Saving:", fname)
wave.astype('float64').tofile(fname)
