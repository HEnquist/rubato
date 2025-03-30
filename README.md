# Rubato

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

## Input and output data format

Input and output data are stored in a non-interleaved format.

Input and output data are stored as slices of references, `&[AsRef<[f32]>]` or `&[AsRef<[f64]>]`.
The inner references (`AsRef<[f32]>` or `AsRef<[f64]>`) hold the sample values for one channel each.

Since normal vectors implement the `AsRef` trait,
`Vec<Vec<f32>>` and `Vec<Vec<f64>>` can be used for both input and output.

## Asynchronous resampling

The asynchronous resamplers are available with and without anti-aliasing filters.

Resampling with anti-aliasing is based on band-limited interpolation using sinc
interpolation filters. The sinc interpolation upsamples by an adjustable factor,
and then the new sample points are calculated by interpolating between these points.
The resampling ratio can be updated at any time.

Resampling without anti-aliasing omits the cpu-heavy sinc interpolation.
This runs much faster but produces a lower quality result.

## Synchronous resampling

Synchronous resampling is implemented via FFT. The data is FFT:ed, the spectrum modified,
and then inverse FFT:ed to get the resampled data.
This type of resampler is considerably faster but doesn't support changing the resampling ratio.

## Usage
The resamplers provided by this library are intended to process audio in chunks.
The optimal chunk size is determined by the application,
but will likely end up somwhere between a few hundred to a few thousand frames.
This gives a good compromize between efficiency and memory usage.

### Real time considerations
Rubato is suitable for real-time applications when using the `Resampler::process_into_buffer()` method.
This stores the output in a pre-allocated output buffer, and performs no allocations or other
operations that may block the thread.

### Resampling a given audio clip
A suggested simple process for resampling an audio clip of known length to a new sample rate is as follows.
Here it is assumed that the source data is stored in a vec,
or some other structure that supports reading arbitrary number of frames at a time.
For simplicity, the output is stored in a temporary buffer during resampling,
and copied to the destination afterwards.

Preparations:
1. Create a resampler of suitable type, for example FFTFixedIn which is quite fast and gives good quality.
   Since neither input or output has any restrictions for the number of frames that can be read or written at a time,
   the chunk size can be chosen arbitrarily. Start with a chunk size of for example 1024.
2. Create an input buffer.
3. Create a temporary buffer for collecting the resampled output data.
4. Call `Resampler::output_delay()` to know how many frames of delay the resampler gives.
   Store the number as `delay`.
5. Calculate the new clip length as `new_length = original_length * new_rate / original_rate`.

Now it's time to process the bulk of the clip by repeated procesing calls. Loop:
1. Call `Resampler::input_frames_next()` to learn how many frames the resampler needs.
2. Check the number of available frames in the source. If it is less than the needed input size, break the loop.
3. Read the required number of frames from the source, convert the sample values to float, and copy them to the input buffer.
4. Call `Resampler::process()` or `Resampler::process_into_buffer()`.
5. Append the output frames to the temporary output buffer.

The next step is to process the last remaining frames.
1. Read the available frames fom the source, convert the sample values to float, and copy them to the input buffer.
2. Call `Resampler::process_partial()` or `Resampler::process_partial_into_buffer()`.
3. Append the output frames to the temporary buffer.

At this point, all frames have been sent to the resampler,
but because of the delay through the resampler,
it may still have some frames in its internal buffers.
When all wanted frames have been generated, the length of the temporary
output buffer should be at least `new_length + delay`.
If this is not the case, call `Resampler::process_partial()`
or `Resampler::process_partial_into_buffer()` with `None` as input,
and append the output to the temporary output buffer.
If needed, repeat until the length is sufficient.

Finally, copy the data from the temporary output buffer to the desired destination.
Skip the first `delay` frames, and copy `new_length` frames.

If there is more than one clip to resample from and to the same sample rates,
the same resampler should be reused.
Creating a new resampler is an expensive task and should be avoided if possible.
Start the procedire from the start, but instead of creating a new resampler,
call `Resampler::reset()` on the existing one to prepare it for a new job.

### Resampling a stream
When resamping a stream, the process is normally performed in real time,
and either the input of output is some API that provides or consumes frames at a given rate.

#### Example, record to file from an audio API
Audio APIs such as [CoreAudio](https://crates.io/crates/coreaudio-rs) on MacOS,
or the cross platform [cpal](https://crates.io/crates/cpal) crate,
often use callback functions for data exchange.

A complete

When capturing audio from these, the application passes a function to the audio API.
The API then calls this function periodically, with a pointer to a data buffer containing new audio frames.
The data buffer size is usually the same on every call, but that varies between APIs.
It is important that the function does not block,
since this would block some internal loop of the API and cause loss of some audio data.
It is recommended to keep the callback function light.
Ideally it should read the provided audio data from the buffer provided by the API,
and optionally perform some light processing such as sample format conversion.
No heavy processing such as resampling should be performed here.
It should then store the audio data to a shared buffer.
The buffer may be a `Arc<Mutex<VecDeque<T>>>`,
or something more advanced such as [ringbuf](https://crates.io/crates/ringbuf).

A separate loop, running either in the main or a separate thread,
should then read from that buffer, resample, and save to file.
If the Audio API provides a fixed buffer size,
then this number of frames is a good choice for the resampler chunk size.
If the size varies, the shared buffer can be used to adapt the chunk sizes of the audio API and the resampler.
A good starting point for the resampler chunk size is to use an "easy" value
near the average chunk size of the audio API.
Make sure that the shared buffer is large enough to not get full
in case for the loop gets blocked waiting for example for disk access.

The loop should follow a process similar to [resampling a clip](#resampling-a-given-audio-clip),
but the input is now the shared buffer.
The loop needs to wait for the needed number of frames to become available in the buffer,
before reading and passing them to the resampler.

It would also be appropriate to omit the temporary output buffer,
and write the output directly to the destination.
The [hound](https://crates.io/crates/hound) crate is a popular choice
for reading and writing uncompressed audio formats.


## SIMD acceleration

### Asynchronous resampling with anti-aliasing

The asynchronous resampler supports SIMD on x86_64 and on aarch64.
The SIMD capabilities of the CPU are determined at runtime.
If no supported SIMD instruction set is available, it falls back to a scalar implementation.

On x86_64, it will try to use AVX. If AVX isn't available, it will instead try SSE3.

On aarch64 (64-bit Arm), it will use Neon if available.

### Synchronous resampling

The synchronous resamplers benefit from the SIMD support of the RustFFT library.

## Cargo features

### `fft_resampler`: Enable the FFT based synchronous resamplers

This feature is enabled by default. Disable it if the FFT resamplers are not needed,
to save compile time and reduce the resulting binary size.

### `log`: Enable logging

This feature enables logging via the `log` crate. This is intended for debugging purposes.
Note that outputting logs allocates a [std::string::String] and most logging implementations involve various other system calls.
These calls may take some (unpredictable) time to return, during which the application is blocked.
This means that logging should be avoided if using this library in a realtime application.

The `log` feature can be enabled when running tests, which can be very useful when debugging.
The logging level can be set via the `RUST_LOG` environment variable.

Example:
```sh
RUST_LOG=trace cargo test --features log
```

## Example

Resample a single chunk of a dummy audio file from 44100 to 48000 Hz.
See also the "process_f64" example that can be used to process a file from disk.
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

## Included examples

The `examples` directory contains a few sample applications for testing the resamplers.
There are also Python scripts for generating simple test signals as well as analyzing the resampled results.

The examples read and write raw audio data in 64-bit float format.
They can be used to process .wav files if the files are first converted to the right format.
Use `sox` to convert a .wav to raw samples:
```sh
sox some_file.wav -e floating-point -b 64 some_file_f64.raw
```
After processing, the result can be converted back to new .wav. This examples converts to 16-bits at 44.1 kHz:
```sh
sox -e floating-point -b 64 -r 44100 -c 2 resampler_output.raw -e signed-integer -b 16 some_file_resampled.wav
```

Many audio editors, for example Audacity, are also able to directly import and export the raw samples.

## Compatibility

The `rubato` crate requires rustc version 1.61 or newer.

## Changelog
- v0.16.2
  - Fix issues when using on 32-bit systems.
- v0.16.1
  - Fix issue in test suite when building without FFT resamplers.
- v0.16.0
  - Add support for changing the fixed input or output size of the asynchronous resamplers.
- v0.15.0
  - Make FFT resamplers optional via `fft_resampler` feature.
  - Fix calculation of input and output sizes when creating FftFixedInOut resampler.
  - Fix panic when using very small chunksizes (less than 5).
- v0.14.1
  - More bugfixes for buffer allocation and max output length calculation.
  - Fix building with `log` feature.
- v0.14.0
  - Add argument to let `input/output_buffer_allocate()` optionally pre-fill buffers with zeros.
  - Add convenience methods for managing buffers.
  - Bugfixes for buffer allocation and max output length calculation.
- v0.13.0
  - Switch to slices of references for input and output data.
  - Add faster (lower quality) asynchronous resamplers.
  - Add a macro to help implement custom object safe resamplers.
  - Optional smooth ramping of ratio changes to avoid audible steps.
  - Add convenience methods for handling last frames in a stream.
  - Add resampler reset method.
  - Refactoring for a more logical structure.
  - Add helper function for calculating cutoff frequency.
  - Add quadratic interpolation for sinc resampler.
  - Add method to get the delay through a resampler as a number of output frames.
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
  - Accept any AsRef<\[T\]> as input.


License: MIT
