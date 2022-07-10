# rubato

An audio sample rate conversion library for Rust.

This library provides resamplers to process audio in chunks.

The ratio between input and output sample rates is completely free.
Implementations are available that accept a fixed length input
while returning a variable length output, and vice versa.

Rubato can be used in realtime applications without any allocation during
processing by preallocating a [Resampler] and using its
[input_buffer_allocate](Resampler::input_buffer_allocate) and
[output_buffer_allocate](Resampler::output_buffer_allocate) methods before
beginning processing. The [log feature](#log-enable-logging) feature should be disabled
for realtime use (it is disabled by default).

#### Input and output data format

Input and output data is stored non-interleaved.

The output data is stored in a vector of vectors, `Vec<Vec<f32>>` or `Vec<Vec<f64>>`.
The inner vectors (`Vec<f32>` or `Vec<f64>`) hold the sample values for one channel each.

The input data is similar, except that it allows the inner vectors to be `AsRef<[f32]>` or `AsRef<[f64]>`.
Normal vectors can be used since `Vec` implements the `AsRef` trait.

#### Asynchronous resampling

The asynchronous resamplers are available with and without anti-aliasing filters.

Resampling with anti-aliasing is based on band-limited interpolation using sinc
interpolation filters. The sinc interpolation upsamples by an adjustable factor,
and then the new sample points are calculated by interpolating between these points.
The resampling ratio can be updated at any time.

Resampling without anti-aliasing omits the cpu-heavy sinc interpolation.
This runs much faster but produces a lower quality result.

#### Synchronous resampling

Synchronous resampling is implemented via FFT. The data is FFT:ed, the spectrum modified,
and then inverse FFT:ed to get the resampled data.
This type of resampler is considerably faster but doesn't support changing the resampling ratio.

#### SIMD acceleration

##### Asynchronous resampling with anti-aliasing

The asynchronous resampler supports SIMD on x86_64 and on aarch64.
The SIMD capabilities of the CPU are determined at runtime.
If no supported SIMD instruction set is available, it falls back to a scalar implementation.

On x86_64 it will try to use AVX. If AVX isn't available, it will instead try SSE3.

On aarch64 (64-bit Arm) it will use Neon if available.

##### Synchronous resampling

The synchronous resamplers benefit from the SIMD support of the RustFFT library.

#### Cargo features

###### `log`: Enable logging

This feature enables logging via the `log` crate. This is intended for debugging purposes.
Note that outputting logs allocates a [std::string::String] and most logging implementations involve various other system calls.
These calls may take some (unpredictable) time to return, during which the application is blocked.
This means that logging should be avoided if using this library in a realtime application.

### Example

Resample a single chunk of a dummy audio file from 44100 to 48000 Hz.
See also the "fixedin64" example that can be used to process a file from disk.
```rust
use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction};
let params = SincInterpolationParameters {
    sinc_len: 256,
    f_cutoff: 0.95,
    interpolation: SincInterpolationType::Linear,
    oversampling_factor: 256,
    window: WindowFunction::BlackmanHarris2,
};
let mut resampler = SincFixedIn::<f64>::new(
    48000 as f64 / 44100 as f64,
    2.0,
    params,
    1024,
    2,
).unwrap();

let waves_in = vec![vec![0.0f64; 1024];2];
let waves_out = resampler.process(&waves_in, None).unwrap();
```

### Compatibility

The `rubato` crate requires rustc version 1.61 or newer.

### Changelog

- v0.13.0
  - Add faster (lower quality) asynchronous resamplers.
  - Optional smooth ramping of ratio changes to avoid audible steps.
  - Refactoring for a more logical structure.
- v0.12.0
  - Always enable all simd acceleration (and remove the simd Cargo features).
- v0.11.0
  - New api to allow use in realtime applications.
  - Configurable adjust range of asynchronous resamplers.
- v0.10.1
  - Fix compiling with neon feature after changes in latest nightly.
- v0.10.0
  - Add an object-safe wrapper trait for Resampler.
- v0.9.0
  - Accept any AsRef<[T]> as input.

License: MIT
