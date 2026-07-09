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
[`Adapter`](https://docs.rs/audioadapter/4.0.0/audioadapter/trait.Adapter.html)
and [`AdapterMut`](https://docs.rs/audioadapter/4.0.0/audioadapter/trait.AdapterMut.html)
objects from the [audioadapter](https://crates.io/crates/audioadapter) crate.
By using a suitable adapter, any sample layout and format can be used.

The [audioadapter-buffers](https://crates.io/crates/audioadapter-buffers) crate
provides a selection of adapters for common data structures,
and the `audioadapter` traits are kept simple in order to make it easy to implement them
for new structures if needed.

For projects migrating from a previous version of `rubato`, the
[`SequentialSliceOfVecs`](https://docs.rs/audioadapter-buffers/4.0.0/audioadapter_buffers/direct/struct.SequentialSliceOfVecs.html)
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

Synchronous resampling is implemented via FFT (Fast Fourier Transform).
The audio data is transformed into the frequency domain,
the spectrum is scaled to match the target sample rate,
and then transformed back to produce the resampled output.
This type of resampler is considerably faster than sinc-based approaches
but doesn't support changing the resampling ratio.

## Choosing a resampler

Rubato provides two true resampler types. The synchronous `Fft` resampler is for
a fixed ratio, while the asynchronous `Async` resampler allows the ratio to vary
over time and comes in two interpolation modes. Which one to pick depends mainly
on whether the ratio is fixed, and on how you want to trade CPU time for quality.
In addition, the lightweight `Slip` clutch handles the special case of matching
two rates that are almost identical.

Some common tasks and a suitable resampler for each:

- **Convert an audio file from one fixed rate to another**, for example 44.1 to 48 kHz
  as an offline batch job: use `Fft` with `process_all()`.
- **Resample a real-time stream between two devices whose clocks drift**, such as a
  capture card feeding a sound card: use `Async::new_sinc` for the same high quality as `Fft`,
  and adjust the ratio to track the drift.
- **Do the same on a tight CPU budget** where a little aliasing is acceptable, for example in a
  game, a voice application such as a video call, or on an embedded target: use `Async::new_poly`.
- **Change playback speed or pitch on the fly** by sweeping the ratio while running: use an
  `Async` resampler, sinc for quality or polynomial for speed. The ratio is not limited to small
  adjustments; it can be swept over a wide range, up to the `max_resample_ratio_relative` factor
  chosen at construction.
- **Keep a stream aligned to a sound card that is a few ppm off**, without truly resampling:
  use `Slip`.
- **Feed a downstream stage where quality barely matters**, such as driving a VU meter or level
  display, or downsampling to a low rate to cut the cost of later processing: use `Async::new_poly`
  for the cheapest conversion.

A few notes to guide the choice:

- The `Fft` resampler requires the `fft_resampler` feature, which is enabled by default.
  It is both fast and high quality, so it is the natural choice whenever the ratio is fixed.
- Use an `Async` resampler when the input and output clocks may drift relative to each other,
  or when you otherwise need to adjust the ratio while running (see [Asynchronous resampling](#asynchronous-resampling)).
- `Async::new_sinc` matches the quality of the `Fft` resampler but is the most CPU-heavy option.
  `Async::new_poly` is much faster because it skips the sinc anti-aliasing filter; the quality
  loss is often subtle. See [Resampling quality](#resampling-quality) for the settings that tune
  each mode.
- `Slip` is **not** a true sample rate converter. It does not interpolate or filter; it just
  passes samples through and occasionally inserts or drops a single frame to keep two
  near-identical clocks in step, hiding each correction behind a short crossfade. In return it
  uses very little CPU, adds no delay, and leaves the passband untouched. Use it only for tracking
  tiny clock drift (a few ppm, ratio very close to `1.0`); for any real ratio change, use an
  `Async` resampler instead.

If you are unsure, `Fft` is a good default for offline or fixed-rate work,
and `Async::new_sinc` is a good default for real-time streams with clock drift.

## Usage
The resamplers provided by this library are intended to support processing streams of audio.
To enable this, they process audio in chunks.
The optimal chunk size is determined by the application,
but will likely end up somewhere between a few hundred to a few thousand frames.
This gives a good compromise between efficiency and memory usage.

### Chunk size and fixed size options

Rubato processes audio in chunks.
The `chunk_size` parameter given to the resampler constructor sets the target size
for the **fixed side** — the side that always has the same number of frames per call.
The other side is variable and will differ from call to call.

The resamplers allow specifying which side should have a fixed size.

*   **Fixed input** (`FixedAsync::Input` / `FixedSync::Input`):
    The input chunk size is fixed to `chunk_size` frames per call.
    The output chunk size varies depending on how many samples can be calculated
    from the available input data.
    This is convenient when the audio source delivers data in fixed-size chunks
    (e.g. a hardware capture callback).
*   **Fixed output** (`FixedAsync::Output` / `FixedSync::Output`):
    The output chunk size is fixed to `chunk_size` frames per call.
    The input chunk size varies depending on how many new samples the resampler
    needs to fill the output.
    This is useful when the audio destination consumes fixed-size chunks
    (e.g. a hardware playback callback).
*   **Both input and output fixed** (`FixedSync::Both`):
    Both input and output chunk sizes are fixed.
    This mode is only available for the synchronous resampler.
    In this mode, the `chunk_size` parameter is used as a hint,
    and the actual chunk sizes are calculated to fit the resampling ratio exactly.
    For example, a 44.1 kHz to 48 kHz resampler must use an input chunk size that is a multiple of 147,
    and an output chunk size that is a multiple of 160, in order to maintain the correct resampling ratio.

    This mode avoids some internal buffering compared to fixed input or fixed output modes,
    and is therefore somewhat more efficient.

    For asynchronous resamplers, fixing both input and output chunk sizes is not possible
    since the resampling ratio can change, requiring at least one side to be variable.

### Input and output sizes per call

The `chunk_size` constructor parameter is only the *target* size for the fixed side.
Always call `input_frames_next()` before providing data to `process_into_buffer`
to find out the exact number of input frames required for that call.
Similarly, call `output_frames_next()` to find the exact number of output frames that will be written.

*   With fixed input, `input_frames_next()` always returns `chunk_size`.
    `output_frames_next()` varies and must be checked each call.
*   With fixed output, `output_frames_next()` always returns `chunk_size`.
    `input_frames_next()` varies and must be checked each call.
*   With `FixedSync::Both`, both values are fixed, but they may differ from `chunk_size`
    because they are rounded to fit the exact sample-rate ratio.

The input and output buffers must be large enough to hold at least the number of frames
reported by `input_frames_next()` and `output_frames_next()` respectively.
Both buffers may be larger than required — only the needed frames are read or written.

### Resampling quality
The synchronous resampler has no quality settings; it always delivers the best quality.

When using cubic sinc interpolation, the quality of the asynchronous resampler
is equivalent to the synchronous resampler.
This mode is however computationally heavy, and therefore there are some settings
that can be used when a different balance between speed and quality is required.

When using sinc interpolation, the length of the sinc function can be reduced.
Each halving of the sinc function length nearly doubles the speed,
at the cost of increased roll-off at high frequencies.
It is also possible to lower the polynomial degree to quadratic or linear,
which also increases the speed while producing higher amounts of distortion.

The fastest option is to use plain polynomial interpolation.
This is significantly faster as it avoids the expensive sinc interpolation,
but does not provide any anti-alias filtering.
The effect of this is often subtle, and many applications can use this mode
to save a significant amount of CPU time with little or no perceived quality loss.

### Real-time considerations
Rubato is suitable for real-time applications when using the `Resampler::process_into_buffer()` method.
This stores the output in a pre-allocated output buffer, and performs no allocations or other
operations that may block the thread.
Ensure that the resampler instance and any needed input and output buffers are created
before entering time-sensitive parts of the application.

The [log feature](#log-enable-logging) is disabled by default,
and should not be enabled for real-time use.

### Resampling a given audio clip
To resample a full audio clip that is already in memory, use either
`Resampler::process_all()` or `Resampler::process_all_into_buffer()`.
Both process the clip in suitably sized chunks internally and trim off the
startup delay, so the result lines up with the input.
Create a resampler of a suitable type, for example `Fft` which is fast and gives good quality.
The chunk size can be chosen arbitrarily. Start with a chunk size of for example 1024.
In this application, the exact input or output chunk sizes are not important
and therefore the `FixedSync::Both` setting can be used for the `Fft` resampler.

- `process_all()` is the simplest option. It allocates the output, trims the delay,
  and returns an `InterleavedOwned` holding exactly the resampled frames.
- `process_all_into_buffer()` writes into a buffer you provide, avoiding the allocation.
  Call `Resampler::process_all_needed_output_len()` first to size the output buffer.

> **Do not** create a resampler with the chunk size set to the whole clip length and
> call `Resampler::process()` once. `process()` is meant for a single chunk and does not
> trim the startup delay, so the output would start with silence and be missing the tail.
> Use `process_all()` for whole clips instead.

If there is more than one clip to resample from and to the same sample rates,
the same resampler should be reused.
Creating a new resampler is an expensive task and should be avoided when possible.
`process_all()` resets the resampler for you; with `process_all_into_buffer()`,
call `Resampler::reset()` between clips to prepare it for a new job.

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
in case the loop gets blocked waiting, for example for disk access.

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

Note that outputting a log message allocates a `String`,
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

Resample stereo audio from 44100 to 48000 Hz, one chunk at a time.
The input is processed in a loop and can come from anywhere: a file read
chunk by chunk, or a live stream that keeps running indefinitely.
This uses the `Async` resampler with polynomial interpolation,
which is always available and needs no optional features.
See also the "process_f64" example that can be used to process a file from disk.
```rust
use rubato::{
    Resampler, Async, FixedAsync, PolynomialDegree, Indexing
};
use audioadapter_buffers::direct::InterleavedSlice;

let channels = 2;
let chunk_size = 1024;

let mut resampler = Async::<f64>::new_poly(
    48000.0 / 44100.0,
    1.1,
    PolynomialDegree::Cubic,
    chunk_size,
    channels,
    FixedAsync::Input,
).unwrap();

// Reusable buffers for a single chunk, assuming interleaved f64 samples.
// With `FixedAsync::Input` every call consumes `chunk_size` input frames and
// produces at most `output_frames_max()` output frames.
let mut indata = vec![0.0; channels * chunk_size];
let mut outdata = vec![0.0; channels * resampler.output_frames_max()];
let outdata_capacity = outdata.len() / channels;

let indexing = Indexing::new();

// Keep processing for as long as there is more audio to handle.
// Here the source is a dummy counter that stops after a few chunks;
// in a real application this stands in for "is there more data?".
let mut chunks_left = 10;
loop {
    // Fetch the next `input_frames_next()` frames from the source into `indata`.
    // For a file, break out of the loop once the end is reached (a shorter final
    // chunk is handled by setting `partial_len` on the indexing struct, see the
    // `process_f64` example). For an endless stream, simply never break.
    if chunks_left == 0 {
        break;
    }
    chunks_left -= 1;
    let frames_to_read = resampler.input_frames_next();
    // (read `frames_to_read` frames from the file or stream into `indata` here)

    let input_adapter = InterleavedSlice::new(&indata, channels, frames_to_read).unwrap();
    let mut output_adapter =
        InterleavedSlice::new_mut(&mut outdata, channels, outdata_capacity).unwrap();

    let (_frames_read, frames_written) = resampler
        .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
        .unwrap();

    // Write the `frames_written` output frames to the destination file or stream.
    let _ = frames_written;
}
```

## Included examples

The `examples` directory contains a few sample applications for testing the resamplers.
There are also Python scripts for generating simple test signals as well as analyzing the resampled results.

The examples read and write raw audio data in either 64-bit float or 16-bit integer format.
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

The `rubato` crate requires rustc version 1.85 or newer.

## Migrating from 3.x to 4.0

Version 4.0 has a handful of breaking changes. The common ones, with before/after:

**`audioadapter` 4.0.** The `Adapter` and `AdapterMut` traits no longer carry a lifetime
parameter. Update any explicit uses, and bump your own `audioadapter` / `audioadapter-buffers`
dependencies to `4.0` to match the versions `rubato` re-exports.

```rust,ignore
// before
fn resample(buf: &dyn Adapter<'_, f32>) { /* ... */ }
// after
fn resample(buf: &dyn Adapter<f32>) { /* ... */ }
```

**`Resampler::process` takes an `Option<&Indexing>`** instead of separate `input_offset` and
`active_channels_mask` arguments, matching `process_into_buffer`.

```rust,ignore
// before
let out = resampler.process(&input, 0, None)?;
// after
let out = resampler.process(&input, None)?;

// before: with an offset and mask
let out = resampler.process(&input, offset, Some(&mask))?;
// after
let indexing = Indexing::new().input_offset(offset).active_channels_mask(mask);
let out = resampler.process(&input, Some(&indexing))?;
```

**`SincInterpolationParameters::f_cutoff` is now an `Option<f32>`.** Prefer
`SincInterpolationParameters::new(sinc_len, window)`, which derives the cutoff automatically.
When building the struct directly, use `None` for the automatic cutoff or wrap a manual value
in `Some`.

```rust,ignore
// before
let params = SincInterpolationParameters {
    sinc_len: 256,
    f_cutoff: 0.95,
    oversampling_factor: 128,
    interpolation: SincInterpolationType::Cubic,
    window: WindowFunction::BlackmanHarris2,
};
// after (recommended): cutoff derived from sinc_len and window
let params = SincInterpolationParameters::new(256, WindowFunction::BlackmanHarris2);
// after (struct form): None for automatic, or Some(value) to override
let params = SincInterpolationParameters { f_cutoff: None, ..params };
```

**`Fft::new` no longer takes `sub_chunks`.** It picks a value automatically and uses a default
window. For control over either, use the new `Fft::new_custom`.

```rust,ignore
// before
let r = Fft::<f64>::new(rate_in, rate_out, chunk_size, sub_chunks, channels, fixed)?;
// after
let r = Fft::<f64>::new(rate_in, rate_out, chunk_size, channels, fixed)?;
// or, with explicit sub_chunks and window:
let r = Fft::<f64>::new_custom(rate_in, rate_out, chunk_size, sub_chunks, channels,
    WindowFunction::BlackmanHarris2, fixed)?;
```

**Ratio and chunk-size changes moved to capability traits.** `set_resample_ratio`,
`set_resample_ratio_relative` are now on the `Adjustable` trait, and `set_chunk_size` is on
`Resizable`. On a concrete resampler, just bring the trait into scope. On a `dyn Resampler`,
recover the capability with `as_adjustable()` / `as_resizable()` (this replaces the old
`SyncNotAdjustable` / `ChunkSizeNotAdjustable` errors, which are removed). To query the
capability through a shared `&dyn Resampler`, use `is_adjustable()` / `is_resizable()`.

```rust,ignore
// before: on a Box<dyn Resampler>, with a runtime error for synchronous resamplers
resampler.set_resample_ratio(new_ratio, true)?;
// after: None means "synchronous, nothing to adjust"
if let Some(adjustable) = resampler.as_adjustable() {
    adjustable.set_resample_ratio(new_ratio, true)?;
}

// before: on a concrete Async resampler
async_resampler.set_resample_ratio_relative(0.95, true)?;
// after: same call, but the Adjustable trait must be in scope
use rubato::Adjustable;
async_resampler.set_resample_ratio_relative(0.95, true)?;
```

**The error enums are now `#[non_exhaustive]`.** If you `match` on `ResampleError` or
`ResamplerConstructionError`, add a `_ => ...` arm.

## Changelog
- v4.0.0
  - Update to `audioadapter` 4.0, which removes the lifetime parameter from the
    `Adapter` and `AdapterMut` traits.
  - Add a fluent way to construct an `Indexing`: `Indexing::new` plus chainable setters,
    and implement `Default` for `Indexing`.
  - Add a fluent way to construct `SincInterpolationParameters`:
    `SincInterpolationParameters::new(sinc_len, window)` plus chainable setters.
  - Change `SincInterpolationParameters::f_cutoff` to an `Option<f32>`. Leave it `None`
    (the default) to let the resampler derive the cutoff from `sinc_len` and `window` with
    `calculate_cutoff`, or set `Some(value)` to override it.
  - Implement `Default` for `SincInterpolationParameters` (`sinc_len` 256, automatic cutoff,
    `oversampling_factor` 128, `Cubic` interpolation, `BlackmanHarris2` window).
  - Add `Resampler::process_all`, an allocating one-shot method for resampling a whole clip.
    It resets the resampler, trims the startup delay, and returns an `InterleavedOwned` holding
    exactly the resampled frames. This is the convenient counterpart to `process_all_into_buffer`.
  - Change `Resampler::process` to take an `Option<&Indexing>` instead of separate
    `input_offset` and `active_channels_mask` arguments, matching `process_into_buffer`. This
    adds `partial_len` support (for a short final chunk) and makes the common call
    `process(&input, None)`. The `Indexing` `output_offset` field is ignored here.
  - Let the synchronous `Fft` resampler choose the anti-aliasing window. `Fft::new` is
    simplified (it drops `sub_chunks`, picking a value automatically, and uses a default
    window), and a new `Fft::new_custom` exposes both `sub_chunks` and the window function.
  - Derive `Clone`, `Copy` and `PartialEq` for `ResampleError` and
    `ResamplerConstructionError`, and mark both `#[non_exhaustive]`.
  - Derive `PartialEq`, `Eq` and `Hash` for the configuration enums `WindowFunction`,
    `SincInterpolationType`, `PolynomialDegree`, `FixedSync` and `FixedAsync`.
  - Return `WrongNumberOfMaskChannels` instead of panicking when the
    `active_channels_mask` passed to a process method has the wrong length.
  - Split the capability-specific methods out of `Resampler` into the `Adjustable` trait
    (`set_resample_ratio`, `set_resample_ratio_relative`) and the `Resizable` trait
    (`set_chunk_size`). `Resampler` gains `as_adjustable()` and `as_resizable()` to recover
    these capabilities from a trait object, and `is_adjustable()` / `is_resizable()` to query
    them through a shared reference. The `SyncNotAdjustable` and `ChunkSizeNotAdjustable`
    error variants are removed, since calling these methods is now a compile-time capability.
  - Add `Slip`, a very cheap resampler for matching two almost-equal sample rates by occasionally
    slipping (inserting or dropping) a frame, hidden by a short crossfade, rather than running a
    full resampler. It is meant for compensating small clock differences: it adds no delay and no
    high-frequency roll-off, and its ratio is meant to be adjusted at runtime through `Adjustable`
    by a feedback loop.
- v3.0.0
  - Use separate lifetimes for `buffer_in` and `buffer_out` in `process_into_buffer`.
  - Improve sinc resampler performance with smarter dot product calculation.
  - Improve SIMD performance (AVX, SSE, NEON) using multiple accumulators.
  - Switch dot product strategy based on channel count.
  - More aggressive inlining of hot paths.
- v2.0.0
  - Update to `audioadapter` 3.0.
  - Add re-export of `audioadapter-buffers`.
- v1.0.1
  - Fix calculation in process_all_needed_output_len method.
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


## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
