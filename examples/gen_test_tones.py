#!/usr/bin/env python3
"""Generate a "song" of stepped pure sine tones as raw little-endian f64.

The output format matches what the `adjust_ratio_f64` example reads and writes:
interleaved 64-bit floats, little-endian, no header, two channels.

The pitch steps through a major scale from `--root`, four octaves up, one note
every `--seconds`. Sustained pure tones are the worst case for hearing a
resampler's artefacts, and stepping them as a scale is easier on the ears than
octave leaps. A short raised-cosine fade at every note boundary keeps the pitch
changes click-free, so the only discontinuities in the file are the ones the
resampler introduces.

The Slip resampler makes one single-frame correction each time the accumulated
drift reaches a whole frame, i.e.

    corrections_per_second = (offset_ppm / 1e6) * fs

so this script also prints the ppm offset to feed the example for a chosen,
easily audible correction rate (default 2 Hz).

Example:
    python examples/gen_test_tones.py
    cargo run --release --example adjust_ratio_f64 \\
        SlipFixedOutput test_tones_f64_2ch.raw out.raw 2 45
"""
import argparse
import math
import struct
import sys

CHANNELS = 2      # interleaved stereo, both channels identical
OCTAVES = 4       # the scale ascends this many octaves from the root
FADE_MS = 15.0    # raised-cosine fade at each note edge, for click-free boundaries


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", default="test_tones_f64_2ch.raw",
                   help="output raw f64 file (default: %(default)s)")
    p.add_argument("--fs", type=float, default=44100.0,
                   help="sample rate in Hz (default: %(default)s)")
    p.add_argument("--seconds", type=float, default=2.0,
                   help="duration of each note in seconds (default: %(default)s)")
    p.add_argument("--amp", type=float, default=0.5,
                   help="tone amplitude, 0..1 (default: %(default)s, ~-6 dBFS)")
    p.add_argument("--root", type=float, default=130.813,
                   help="scale root frequency in Hz (default: %(default)s, ~C3)")
    p.add_argument("--slip-hz", type=float, default=2.0,
                   help="target Slip correction rate to compute the ppm for (default: %(default)s)")
    args = p.parse_args()

    fs = args.fs
    n_per_tone = int(round(args.seconds * fs))
    fade = min(int(round(FADE_MS * 1e-3 * fs)), n_per_tone // 2)

    # A major scale ascending over OCTAVES, ending on the top root. Semitone
    # offsets within an octave: do re mi fa sol la ti.
    major = [0, 2, 4, 5, 7, 9, 11]
    semitones = [o * 12 + s for o in range(OCTAVES) for s in major]
    semitones.append(OCTAVES * 12)  # finish on the octave
    freqs = [args.root * 2.0 ** (st / 12.0) for st in semitones]

    # Build one channel: each note is a sine faded to silence at both edges.
    mono = []
    for f in freqs:
        w = 2.0 * math.pi * f / fs
        for i in range(n_per_tone):
            s = args.amp * math.sin(w * i)
            if fade > 0:
                if i < fade:
                    s *= 0.5 - 0.5 * math.cos(math.pi * i / fade)
                elif i >= n_per_tone - fade:
                    j = n_per_tone - 1 - i
                    s *= 0.5 - 0.5 * math.cos(math.pi * j / fade)
            mono.append(s)

    # Interleave identical channels and pack as little-endian f64.
    packer = struct.Struct("<" + "d" * CHANNELS)
    with open(args.out, "wb") as fh:
        buf = bytearray()
        for s in mono:
            buf += packer.pack(*([s] * CHANNELS))
            if len(buf) >= 1 << 20:
                fh.write(buf)
                buf = bytearray()
        fh.write(buf)

    total_frames = len(mono)
    ppm = args.slip_hz * 1e6 / fs
    print(f"Wrote {args.out}")
    print(f"  {len(freqs)} notes x {args.seconds:g} s = {total_frames / fs:.1f} s, "
          f"{CHANNELS} ch, fs = {fs:g} Hz")
    print(f"  frequencies (Hz): {', '.join(f'{f:.1f}' for f in freqs)}")
    print()
    print(f"For a {args.slip_hz:g} Hz Slip correction rate use offset = {ppm:.1f} ppm:")
    print(f"  cargo run --release --example adjust_ratio_f64 "
          f"SlipFixedOutput {args.out} out.raw {CHANNELS} {round(ppm)}")


if __name__ == "__main__":
    sys.exit(main())
