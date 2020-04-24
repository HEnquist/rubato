# camillaresampler

An audio sample rate conversion library for Rust.

This library provides resamplers to process audio in chunks.
The ratio between input and output sample rates is completely free.
Implementations are available that accept a fixed length input
while returning a variable length output, and vice versa.
The resampling is based on band-limited interpolation using sinc
interpolation filters. The sinc interpolation upsamples by an adjustable factor,
and then the new sample points are calculated by interpolating between these points.

### Example
Resample a single chunk of a dummy audio file from 44100 to 48000 Hz.
See also the "fixedin64" example that can be used to process a file from disk.
```rust
use camillaresampler::{Resampler, SincFixedIn, Interpolation};
let mut resampler = SincFixedIn::<f64>::new(
    48000 as f32 / 44100 as f32,
    256,
    0.95,
    160,
    Interpolation::Nearest,
    1024,
    2,
);

let waves_in = vec![vec![0.0f64; 1024];2];
let waves_out = resampler.process(&waves_in).unwrap();
```

### Compatibility

The `camillaresampler` crate only depends on the `num` crate and should work with any rustc version that crate supports.

License: MIT
