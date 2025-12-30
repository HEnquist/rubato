# Rubato

Rubato is a flexible audio sample rate conversion library for Rust,
providing a choice of resamplers that can be optimized for either high quality or high speed.
It processes audio in chunks, making it suitable for everything
from real-time audio streams to offline batch processing.

The library allows for completely free selection of resampling ratios,
which can even be updated on the fly.
It features several resampler implementations,
including high-quality asynchronous sinc resamplers
and fast synchronous FFT-based resamplers.

Rubato is designed with real-time safety in mind,
avoiding allocations during processing to ensure smooth and predictable performance.
See [Real-time considerations](#real-time-considerations) for more details.

## Input and output data format

Input and output data is handled via
[`Adapter`](https://docs.rs/audioadapter/2.0.0/audioadapter/trait.Adapter.html)
and [`AdapterMut`](https://docs.rs/audioadapter/2.0.0/audioadapter/trait.AdapterMut.html)
objects from the [audioadapter](https://crates.io/crates/audioadapter) crate.
By using a suitable adapter, any sample layout and format can be used.

The [audioadapter-buffers](https://crates.io/crates/audioadapter-buffers) crate
provides a selection of adapters for common data structures,
and the `audioadapter` traits are kept simple in order to make it easy to implement them
for new structures if needed.

For projects migrating from a previous version of `rubato`, the
[`SequentialSliceOfVecs`](https://docs.rs/audioadapter-buffers/2.0.0/audioadapter_buffers/direct/struct.SequentialSliceOfVecs.html)
adapter is a good starting point, since it wraps the vector of vectors
commonly used with `rubato` v0.16 and earlier.


## Asynchronous resampling

Asynchronous resampling is when the input and output sample rates are not locked,
and the ratio may vary slightly over time.
This is common in real-time audio streams where the input and output devices
have different clocks that may drift relative to each other.
It allows for changing the resampling ratio at any time to compensate for this drift.

The asynchronous resamplers are available with and without anti-aliasing filters.

Resampling with anti-aliasing is based on band-limited interpolation using sinc
interpolation filters. The sinc interpolation upsamples by an adjustable factor,
and then the new sample points are calculated by interpolating between these points.
The resampling ratio can be updated at any time.

Resampling without anti-aliasing omits the CPU-heavy sinc interpolation.
This runs much faster but produces a lower quality result.

## Synchronous resampling

Synchronous resampling is the case when the input and output sample rates
are fixed and locked to each other.
For example, converting a file from 44.1 kHz to 48 kHz.
The ratio, 48 kHz / 44.1 kHz (equivalent to 160 / 147),
is fixed and constant throughout the process.

Synchronous resampling is implemented via FFT. The data is FFT:ed, the spectrum modified,
and then inverse FFT:ed to get the resampled data.
This type of resampler is considerably faster but doesn't support changing the resampling ratio.

## Usage
The resamplers provided by this library are intended to support processing streams of audio.
To enable this, they process audio in chunks.
The optimal chunk size is determined by the application,
but will likely end up somewhere between a few hundred to a few thousand frames.
This gives a good compromise between efficiency and memory usage.

### Chunk size and fixed size options

Rubato processes audio in chunks.
The size of these chunks is determined by the chunk size parameter given to the resampler constructor.
Depending on the configuration, this parameter determines
the number of frames in the input or output chunk, or both.

The resamplers allow specifying which side should have a fixed size.

*   **Fixed input**: The input chunk size is fixed to the given value.
    The output chunk size will vary depending on how many samples can be calculated using the available input data.
    This is convenient to use for resampling data from a source that delivers data in fixed size chunks.
*   **Fixed output**: The output chunk size is fixed to the given value.
    The input chunk size will vary depending on how many new samples the resampler needs to calculate the output.
    This is meant to be used for resampling data that will be sent to some target that requires fixed size chunks.
*   **Both input and output fixed**: Both input and output chunk sizes are fixed.
    This is only available for the synchronous resampler.
    In this mode, the chunk size parameter is used as a hint,
    and the actual chunk sizes are calculated to fit the resampling ratio exactly.
    For example, a 44.1 kHz to 48 kHz resampler must use an input chunk size that is a multiple of 147,
    and an output chunk size that is a multiple of 160, in order to maintain the correct resampling ratio.
    For asynchronous resamplers, fixing both input and output chunk sizes is not possible
    since the resampling ratio can change, requiring at least one side to be variable.

### Resampling quality
The synchronous resampler has no quality settings, it always delivers the best quality.

When using cubic sinc interpolation, the quality of the asynchronous resampler
is equivalent to the synchronous resampler.
This mode is however computationally heavy, and therefore there are some settings
that can be used when a different balance between speed and quality is required.

When using sinc interpolation, the length of the sinc function can be reduced.
Each halving of the sinc function length nearly doubles the speed,
at the cost of increased roll-off at high frequencies.
It is also posible to lower the polynomial degree to quadratic or linear,
which also increases the speed while producing higher amounts of distortion.

The fastest option is to use plain polynomial interpolation.
This is significantly faster as it avoids the expensive sinc interpolation,
but does not provide any anti-alias filtering.
The effect of this is often subtle, and many applications can use this mode
to save a significant amount of CPU time with little or no percieved quality loss.

### Real-time considerations
Rubato is suitable for real-time applications when using the `Resampler::process_into_buffer()` method.
This stores the output in a pre-allocated output buffer, and performs no allocations or other
operations that may block the thread.
Ensure that the resampler instance and any needed input and output buffers are created
before entering time-sensitive parts of the application.

The [log feature](#log-enable-logging) feature is disabled by default,
and should not be enabled for real-time use.

### Resampling a given audio clip
The resampler trait provides the `Resampler::process_all_into_buffer()` method
for resampling a full audio clip of arbitrary length.
To use this, create a resampler of suitable type, for example `Fft` which is fast and gives good quality.
The chunk size can be chosen arbitrarily. Start with a chunk size of for example 1024.
Then call `Resampler::process_all_needed_output_len()` to find out the length of the result.
Create an output buffer, and call `Resampler::process_all_into_buffer()`.

If there is more than one clip to resample from and to the same sample rates,
the same resampler should be reused.
Creating a new resampler is an expensive task and should be avoided when possible.
Start the procedure from the start, but instead of creating a new resampler,
call `Resampler::reset()` on the existing one to prepare it for a new job.

### Resampling a stream
When resampling a stream, the process is normally performed in real time,
and either the input or output is some API that provides or consumes frames at a given rate.

#### Use case example, record from an audio API and save to a file
Audio APIs such as [CoreAudio](https://crates.io/crates/coreaudio-rs) on MacOS,
or the cross platform [cpal](https://crates.io/crates/cpal) crate,
often use callback functions for data exchange.

##### Callback function
When capturing audio from these, the application passes a function to the audio API.
The API then calls this function periodically,
with a pointer to a data buffer containing new audio frames.
The data buffer size is usually the same on every call, but that varies between APIs.
It is important that the function does not block,
since this would block some internal loop of the API and cause loss of some audio data.
It is also recommended to keep the callback function light.
**No heavy processing such as resampling should be performed here.**
Ideally it should read the provided audio data from the buffer provided by the API,
and optionally perform some light processing such as sample format conversion.
It should then store the audio data to a shared buffer.
The buffer may be a `Arc<Mutex<VecDeque<T>>>`,
or something more advanced such as [ringbuf](https://crates.io/crates/ringbuf).

##### Processing loop
A separate loop, running either in the main or a separate thread,
should then read from that buffer, resample, and save to file.
The resampler loop needs to wait for the needed number
of frames to become available in the buffer,
before reading and passing them to the resampler.

If the Audio API provides a fixed buffer size,
then this number of frames is a good choice for the resampler input chunk size.
If the size varies, the shared buffer can be used to adapt
the chunk sizes of the audio API and the resampler.
A good starting point for the resampler chunk size is to use an "easy" value,
for example a power of two, near the average chunk size of the audio API.
Make sure that the shared buffer is large enough to not get full
in case for the loop gets blocked waiting for example for disk access.

The output of the resampler is then written to a file.
The [hound](https://crates.io/crates/hound) crate is a popular choice
for reading and writing uncompressed audio formats.

## SIMD acceleration

### Asynchronous resampling with anti-aliasing

The asynchronous sinc resampler supports SIMD on x86_64 and on aarch64 (64-bit Arm).
The SIMD capabilities of the CPU are determined at runtime.
If no supported SIMD instruction set is available, it falls back to a scalar implementation.

On x86_64, it will try to use AVX. If AVX isn't available, it will instead try SSE3.

On aarch64, it will use Neon if available.

### Synchronous resampling

The synchronous FFT resampler benefits from the SIMD support of the RustFFT library.

## Cargo features

### `fft_resampler`: Enable the FFT based synchronous resampler

This feature is enabled by default. Disable it if the FFT resampler is not needed,
to save compile time and reduce the resulting binary size.

### `log`: Enable logging

This feature enables logging via the `log` crate.
This is intended for debugging purposes, when working on `rubato` itself or investigating issues.
Applications using this library should normally keep this feature disabled to avoid
cluttering logs with unnecessary messages.

Note that outputting a log message allocates a [std::string::String],
and most logging implementations involve various other system calls.
These calls may take some (unpredictable) time to return, during which the application is blocked.
This means that logging should be avoided if using this library in a realtime application.

The `log` feature can be enabled when running tests, which can be very useful when debugging.
The logging level can be set via the `RUST_LOG` environment variable.

Example:
```sh
RUST_LOG=trace cargo test --features log
```

## Example

Resample a dummy audio file from 44100 to 48000 Hz.
See also the "process_f64" example that can be used to process a file from disk.
```rust
use rubato::{
    Resampler, Async, FixedAsync, Indexing,
    SincInterpolationType, SincInterpolationParameters,
    WindowFunction
};
use audioadapter_buffers::direct::InterleavedSlice;

let params = SincInterpolationParameters {
    sinc_len: 256,
    f_cutoff: 0.95,
    interpolation: SincInterpolationType::Linear,
    oversampling_factor: 256,
    window: WindowFunction::BlackmanHarris2,
};
let mut resampler = Async::<f64>::new_sinc(
    48000 as f64 / 44100 as f64,
    2.0,
    &params,
    1024,
    2,
    FixedAsync::Input,
).unwrap();

// create a short dummy audio clip, assuming it's stereo stored as interleaved f64 values
let audio_clip = vec![0.0; 2*10000];

// wrap it with an InterleavedSlice Adapter
let nbr_input_frames = audio_clip.len() / 2;
let input_adapter = InterleavedSlice::new(&audio_clip, 2, nbr_input_frames).unwrap();

// create a buffer for the output
let mut outdata = vec![0.0; 2*2*10000];
let outdata_capacity = outdata.len() / 2;
let mut output_adapter =
    InterleavedSlice::new_mut(&mut outdata, 2, outdata_capacity).unwrap();

// Preparations
let mut indexing = Indexing {
    input_offset: 0,
    output_offset: 0,
    active_channels_mask: None,
    partial_len: None,
};

let mut input_frames_left = nbr_input_frames;
let mut input_frames_next = resampler.input_frames_next();

// Loop over all full chunks.
// There will be some unprocessed input frames left after the last full chunk.
// see the `process_f64` example for how to handle those
// using `partial_len` of the indexing struct.
// It is also possible to use the `process_all_into_buffer` method
// to process the entire file (including any last partial chunk) with a single call.
while input_frames_left >= input_frames_next {
    let (frames_read, frames_written) = resampler
        .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
        .unwrap();

    indexing.input_offset += frames_read;
    indexing.output_offset += frames_written;
    input_frames_left -= frames_read;
    input_frames_next = resampler.input_frames_next();
}
```

## Included examples

The `examples` directory contains a few sample applications for testing the resamplers.
There are also Python scripts for generating simple test signals as well as analyzing the resampled results.

The examples read and write raw audio data in either 64-bit float of 16-bit integer format.
They can be used to process .wav files if the files are first converted to the right format.
Example, use `sox` to convert a .wav to 64-bit float raw samples:
```sh
sox some_file.wav -e floating-point -b 64 some_file_f64.raw
```

After processing with for instance the `process_f64` example,
the result can be converted back to a new .wav.
This command converts the 64-bit floats to 16-bits at 44.1 kHz:
```sh
sox -e floating-point -b 64 -r 44100 -c 2 resampler_output.raw -e signed-integer -b 16 some_file_resampled.wav
```

Many audio editors, for example Audacity, are also able to directly import and export the raw samples.

## Compatibility

The `rubato` crate requires rustc version 1.74 or newer.

## Changelog
- v1.0.0
  - New API using the AudioAdapter crate to handle different buffer layouts and sample formats.
  - Merged the FixedIn, FixedOut and FixedInOut resamplers into single types that supports all modes.
  - Merged the sinc and polynomial asynchronous resamplers into
    one type that supports both interpolation modes.
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
