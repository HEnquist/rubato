//! An audio sample rate conversion library for Rust.
//!
//! This library provides resamplers to process audio in chunks.
//!
//! The ratio between input and output sample rates is completely free.
//! Implementations are available that accept a fixed length input
//! while returning a variable length output, and vice versa.
//!
//! Rubato can be used in realtime applications without any allocation during
//! processing by preallocating a [Resampler] and using its
//! [input_buffer_allocate](Resampler::input_buffer_allocate) and
//! [output_buffer_allocate](Resampler::output_buffer_allocate) methods before
//! beginning processing. The [log feature](#log-enable-logging) feature should be disabled
//! for realtime use (it is disabled by default).
//!
//! # Input and output data format
//!
//! Input and output data are stored in a non-interleaved format.
//!
//! Input and output data are stored as slices of references, `&[AsRef<[f32]>]` or `&[AsRef<[f64]>]`.
//! The inner references (`AsRef<[f32]>` or `AsRef<[f64]>`) hold the sample values for one channel each.
//!
//! Since normal vectors implement the `AsRef` trait,
//! `Vec<Vec<f32>>` and `Vec<Vec<f64>>` can be used for both input and output.
//!
//! # Asynchronous resampling
//!
//! The asynchronous resamplers are available with and without anti-aliasing filters.
//!
//! Resampling with anti-aliasing is based on band-limited interpolation using sinc
//! interpolation filters. The sinc interpolation upsamples by an adjustable factor,
//! and then the new sample points are calculated by interpolating between these points.
//! The resampling ratio can be updated at any time.
//!
//! Resampling without anti-aliasing omits the cpu-heavy sinc interpolation.
//! This runs much faster but produces a lower quality result.
//!
//! # Synchronous resampling
//!
//! Synchronous resampling is implemented via FFT. The data is FFT:ed, the spectrum modified,
//! and then inverse FFT:ed to get the resampled data.
//! This type of resampler is considerably faster but doesn't support changing the resampling ratio.
//!
//! # SIMD acceleration
//!
//! ## Asynchronous resampling with anti-aliasing
//!
//! The asynchronous resampler supports SIMD on x86_64 and on aarch64.
//! The SIMD capabilities of the CPU are determined at runtime.
//! If no supported SIMD instruction set is available, it falls back to a scalar implementation.
//!
//! On x86_64, it will try to use AVX. If AVX isn't available, it will instead try SSE3.
//!
//! On aarch64 (64-bit Arm), it will use Neon if available.
//!
//! ## Synchronous resampling
//!
//! The synchronous resamplers benefit from the SIMD support of the RustFFT library.
//!
//! # Cargo features
//!
//! ## `log`: Enable logging
//!
//! This feature enables logging via the `log` crate. This is intended for debugging purposes.
//! Note that outputting logs allocates a [std::string::String] and most logging implementations involve various other system calls.
//! These calls may take some (unpredictable) time to return, during which the application is blocked.
//! This means that logging should be avoided if using this library in a realtime application.
//!
//! # Example
//!
//! Resample a single chunk of a dummy audio file from 44100 to 48000 Hz.
//! See also the "process_f64" example that can be used to process a file from disk.
//! ```
//! use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction};
//! let params = SincInterpolationParameters {
//!     sinc_len: 256,
//!     f_cutoff: 0.95,
//!     interpolation: SincInterpolationType::Linear,
//!     oversampling_factor: 256,
//!     window: WindowFunction::BlackmanHarris2,
//! };
//! let mut resampler = SincFixedIn::<f64>::new(
//!     48000 as f64 / 44100 as f64,
//!     2.0,
//!     params,
//!     1024,
//!     2,
//! ).unwrap();
//!
//! let waves_in = vec![vec![0.0f64; 1024];2];
//! let waves_out = resampler.process(&waves_in, None).unwrap();
//! ```
//!
//! # Included examples
//!
//! The `examples` directory contains a few sample applications for testing the resamplers.
//! There are also Python scripts for generating simple test signals as well as analyzing the resampled results.
//!
//! The examples read and write raw audio data in 64-bit float format.
//! They can be used to process .wav files if the files are first converted to the right format.
//! Use `sox` to convert a .wav to raw samples:
//! ```sh
//! sox some_file.wav -e floating-point -b 64 some_file_f64.raw
//! ```
//! After processing, the result can be converted back to new .wav. This examples converts to 16-bits at 44.1 kHz:
//! ```sh
//! sox -e floating-point -b 64 -r 44100 -c 2 resampler_output.raw -e signed-integer -b 16 some_file_resampled.wav
//! ```
//!
//! Many audio editors, for example Audacity, are also able to directly import and export the raw samples.
//!
//! # Compatibility
//!
//! The `rubato` crate requires rustc version 1.61 or newer.
//!
//! # Changelog
//!
//! - v0.14.1
//!   - More bugfixes for buffer allocation and max output length calculation.
//!   - Fix building with `log` feature.
//! - v0.14.0
//!   - Add argument to let `input/output_buffer_allocate()` optionally pre-fill buffers with zeros.
//!   - Add convenience methods for managing buffers.
//!   - Bugfixes for buffer allocation and max output length calculation.
//! - v0.13.0
//!   - Switch to slices of references for input and output data.
//!   - Add faster (lower quality) asynchronous resamplers.
//!   - Add a macro to help implement custom object safe resamplers.
//!   - Optional smooth ramping of ratio changes to avoid audible steps.
//!   - Add convenience methods for handling last frames in a stream.
//!   - Add resampler reset method.
//!   - Refactoring for a more logical structure.
//!   - Add helper function for calculating cutoff frequency.
//!   - Add quadratic interpolation for sinc resampler.
//!   - Add method to get the delay through a resampler as a number of output frames.
//! - v0.12.0
//!   - Always enable all simd acceleration (and remove the simd Cargo features).
//! - v0.11.0
//!   - New api to allow use in realtime applications.
//!   - Configurable adjust range of asynchronous resamplers.
//! - v0.10.1
//!   - Fix compiling with neon feature after changes in latest nightly.
//! - v0.10.0
//!   - Add an object-safe wrapper trait for Resampler.
//! - v0.9.0
//!   - Accept any AsRef<\[T\]> as input.
//!

#[cfg(feature = "log")]
extern crate log;

// Logging wrapper macros to avoid cluttering the code with conditionals.
#[allow(unused)]
macro_rules! trace { ($($x:tt)*) => (
    #[cfg(feature = "log")] {
        log::trace!($($x)*)
    }
) }
#[allow(unused)]
macro_rules! debug { ($($x:tt)*) => (
    #[cfg(feature = "log")] {
        log::debug!($($x)*)
    }
) }
#[allow(unused)]
macro_rules! info { ($($x:tt)*) => (
    #[cfg(feature = "log")] {
        log::info!($($x)*)
    }
) }
#[allow(unused)]
macro_rules! warn { ($($x:tt)*) => (
    #[cfg(feature = "log")] {
        log::warn!($($x)*)
    }
) }
#[allow(unused)]
macro_rules! error { ($($x:tt)*) => (
    #[cfg(feature = "log")] {
        log::error!($($x)*)
    }
) }

mod asynchro_fast;
mod asynchro_sinc;
mod error;
mod interpolation;
mod sample;
mod sinc;
mod synchro;
mod windows;

pub mod sinc_interpolator;

pub use crate::asynchro_fast::{FastFixedIn, FastFixedOut, PolynomialDegree};
pub use crate::asynchro_sinc::{
    SincFixedIn, SincFixedOut, SincInterpolationParameters, SincInterpolationType,
};
pub use crate::error::{
    CpuFeature, MissingCpuFeature, ResampleError, ResampleResult, ResamplerConstructionError,
};
pub use crate::sample::Sample;
pub use crate::synchro::{FftFixedIn, FftFixedInOut, FftFixedOut};
pub use crate::windows::{calculate_cutoff, WindowFunction};

/// A resampler that is used to resample a chunk of audio to a new sample rate.
/// For asynchronous resamplers, the rate can be adjusted as required.
///
/// This trait is not object safe. If you need an object safe resampler,
/// use the [VecResampler] wrapper trait.
pub trait Resampler<T>: Send
where
    T: Sample,
{
    /// This is a convenience wrapper for [process_into_buffer](Resampler::process_into_buffer)
    /// that allocates the output buffer with each call. For realtime applications, use
    /// [process_into_buffer](Resampler::process_into_buffer) with a buffer allocated by
    /// [output_buffer_allocate](Resampler::output_buffer_allocate) instead of this function.
    fn process<V: AsRef<[T]>>(
        &mut self,
        wave_in: &[V],
        active_channels_mask: Option<&[bool]>,
    ) -> ResampleResult<Vec<Vec<T>>> {
        let frames = self.output_frames_next();
        let channels = self.nbr_channels();
        let mut wave_out = Vec::with_capacity(channels);
        for chan in 0..channels {
            let chan_out = if active_channels_mask.map(|mask| mask[chan]).unwrap_or(true) {
                vec![T::zero(); frames]
            } else {
                vec![]
            };
            wave_out.push(chan_out);
        }
        let (_, out_len) =
            self.process_into_buffer(wave_in, &mut wave_out, active_channels_mask)?;
        for chan_out in wave_out.iter_mut() {
            chan_out.truncate(out_len);
        }
        Ok(wave_out)
    }

    /// Resample a buffer of audio to a pre-allocated output buffer.
    /// Use this in real-time applications where the unpredictable time required to allocate
    /// memory from the heap can cause glitches. If this is not a problem, you may use
    /// the [process](Resampler::process) method instead.
    ///
    /// The input and output buffers are used in a non-interleaved format.
    /// The input is a slice, where each element of the slice is itself referenceable
    /// as a slice ([AsRef<\[T\]>](AsRef)) which contains the samples for a single channel.
    /// Because `[Vec<T>]` implements [`AsRef<\[T\]>`](AsRef), the input may be [`Vec<Vec<T>>`](Vec).
    ///
    /// The output data is a slice, where each element of the slice is a `[T]` which contains
    /// the samples for a single channel. If the output channel slices do not have sufficient
    /// capacity for all output samples, the function will return an error with the expected
    /// size. You could allocate the required output buffer with
    /// [output_buffer_allocate](Resampler::output_buffer_allocate) before calling this function
    /// and reuse the same buffer for each call.
    ///
    /// The `active_channels_mask` is optional.
    /// Any channel marked as inactive by a false value will be skipped during processing
    /// and the corresponding output will be left unchanged.
    /// If `None` is given, all channels will be considered active.
    ///
    /// Before processing, it checks that the input and outputs are valid.
    /// If either has the wrong number of channels, or if the buffer for any channel is too short,
    /// a [ResampleError] is returned.
    /// Both input and output are allowed to be longer than required.
    /// The number of input samples consumed and the number output samples written
    /// per channel is returned in a tuple, `(input_frames, output_frames)`.
    fn process_into_buffer<Vin: AsRef<[T]>, Vout: AsMut<[T]>>(
        &mut self,
        wave_in: &[Vin],
        wave_out: &mut [Vout],
        active_channels_mask: Option<&[bool]>,
    ) -> ResampleResult<(usize, usize)>;

    /// This is a convenience method for processing the last frames at the end of a stream.
    /// Use this when there are fewer frames remaining than what the resampler requires as input.
    /// Calling this function is equivalent to padding the input buffer with zeros
    /// to make it the right input length, and then calling [process_into_buffer](Resampler::process_into_buffer).
    /// This method can also be called without any input frames, by providing `None` as input buffer.
    /// This can be utilized to push any remaining delayed frames out from the internal buffers.
    /// Note that this method allocates space for a temporary input buffer.
    /// Real-time applications should instead call `process_into_buffer` with a zero-padded pre-allocated input buffer.
    fn process_partial_into_buffer<Vin: AsRef<[T]>, Vout: AsMut<[T]>>(
        &mut self,
        wave_in: Option<&[Vin]>,
        wave_out: &mut [Vout],
        active_channels_mask: Option<&[bool]>,
    ) -> ResampleResult<(usize, usize)> {
        let frames = self.input_frames_next();
        let mut wave_in_padded = Vec::with_capacity(self.nbr_channels());
        for _ in 0..self.nbr_channels() {
            wave_in_padded.push(vec![T::zero(); frames]);
        }
        if let Some(input) = wave_in {
            for (ch_input, ch_padded) in input.iter().zip(wave_in_padded.iter_mut()) {
                let mut frames_in = ch_input.as_ref().len();
                if frames_in > frames {
                    frames_in = frames;
                }
                if frames_in > 0 {
                    ch_padded[..frames_in].copy_from_slice(&ch_input.as_ref()[..frames_in]);
                } else {
                    ch_padded.clear();
                }
            }
        }
        self.process_into_buffer(&wave_in_padded, wave_out, active_channels_mask)
    }

    /// This is a convenience method for processing the last frames at the end of a stream.
    /// It is similar to [process_partial_into_buffer](Resampler::process_partial_into_buffer)
    /// but allocates the output buffer with each call.
    /// Note that this method allocates space for both input and output.
    fn process_partial<V: AsRef<[T]>>(
        &mut self,
        wave_in: Option<&[V]>,
        active_channels_mask: Option<&[bool]>,
    ) -> ResampleResult<Vec<Vec<T>>> {
        let frames = self.output_frames_next();
        let channels = self.nbr_channels();
        let mut wave_out = Vec::with_capacity(channels);
        for chan in 0..channels {
            let chan_out = if active_channels_mask.map(|mask| mask[chan]).unwrap_or(true) {
                vec![T::zero(); frames]
            } else {
                vec![]
            };
            wave_out.push(chan_out);
        }
        let (_, out_len) =
            self.process_partial_into_buffer(wave_in, &mut wave_out, active_channels_mask)?;
        for chan_out in wave_out.iter_mut() {
            chan_out.truncate(out_len);
        }
        Ok(wave_out)
    }

    /// Convenience method for allocating an input buffer suitable for use with
    /// [process_into_buffer](Resampler::process_into_buffer). The buffer's capacity
    /// is big enough to prevent allocating additional heap memory before any call to
    /// [process_into_buffer](Resampler::process_into_buffer) regardless of the current
    /// resampling ratio.
    ///
    /// The `filled` argument determines if the vectors should be pre-filled with zeros or not.
    /// When false, the vectors are only allocated but returned empty.
    fn input_buffer_allocate(&self, filled: bool) -> Vec<Vec<T>> {
        let frames = self.input_frames_max();
        let channels = self.nbr_channels();
        make_buffer(channels, frames, filled)
    }

    /// Get the maximum number of input frames per channel the resampler could require.
    fn input_frames_max(&self) -> usize;

    /// Get the number of frames per channel needed for the next call to
    /// [process_into_buffer](Resampler::process_into_buffer) or [process](Resampler::process).
    fn input_frames_next(&self) -> usize;

    /// Get the maximum number of channels this Resampler is configured for.
    fn nbr_channels(&self) -> usize;

    /// Convenience method for allocating an output buffer suitable for use with
    /// [process_into_buffer](Resampler::process_into_buffer). The buffer's capacity
    /// is big enough to prevent allocating additional heap memory during any call to
    /// [process_into_buffer](Resampler::process_into_buffer) regardless of the current
    /// resampling ratio.
    ///
    /// The `filled` argument determines if the vectors should be pre-filled with zeros or not.
    /// When false, the vectors are only allocated but returned empty.
    fn output_buffer_allocate(&self, filled: bool) -> Vec<Vec<T>> {
        let frames = self.output_frames_max();
        let channels = self.nbr_channels();
        make_buffer(channels, frames, filled)
    }

    /// Get the max number of output frames per channel.
    fn output_frames_max(&self) -> usize;

    /// Get the number of frames per channel that will be output from the next call to
    /// [process_into_buffer](Resampler::process_into_buffer) or [process](Resampler::process).
    fn output_frames_next(&self) -> usize;

    /// Get the delay for the resampler, reported as a number of output frames.
    fn output_delay(&self) -> usize;

    /// Update the resample ratio.
    ///
    /// For asynchronous resamplers, the ratio must be within
    /// `original / maximum` to `original * maximum`, where the original and maximum are the
    /// resampling ratios that were provided to the constructor. Trying to set the ratio
    /// outside these bounds will return [ResampleError::RatioOutOfBounds].
    ///
    /// For synchronous resamplers, this will always return [ResampleError::SyncNotAdjustable].
    ///
    /// If the argument `ramp` is set to true, the ratio will be ramped from the old to the new value
    /// during processing of the next chunk. This allows smooth transitions from one ratio to another.
    /// If `ramp` is false, the new ratio will be applied from the start of the next chunk.
    fn set_resample_ratio(&mut self, new_ratio: f64, ramp: bool) -> ResampleResult<()>;

    /// Update the resample ratio as a factor relative to the original one.
    ///
    /// For asynchronous resamplers, the relative ratio must be within
    /// `1 / maximum` to `maximum`, where `maximum` is the maximum
    /// resampling ratio that was provided to the constructor. Trying to set the ratio
    /// outside these bounds will return [ResampleError::RatioOutOfBounds].
    ///
    /// Ratios above 1.0 slow down the output and lower the pitch, while ratios
    /// below 1.0 speed up the output and raise the pitch.
    ///
    /// For synchronous resamplers, this will always return [ResampleError::SyncNotAdjustable].
    fn set_resample_ratio_relative(&mut self, rel_ratio: f64, ramp: bool) -> ResampleResult<()>;

    /// Reset the resampler state and clear all internal buffers.
    fn reset(&mut self);
}

use crate as rubato;
/// A macro for implementing wrapper traits for when a [Resampler] must be object safe.
/// The wrapper trait locks the generic type parameters or the [Resampler] trait to specific types,
/// which is needed to make the trait into an object.
///
/// One wrapper trait, [VecResampler], is included per default.
/// It differs from [Resampler] by fixing the generic types
/// `&[AsRef<[T]>]` and `&mut [AsMut<[T]>]` to `&[Vec<T>]` and `&mut [Vec<T>]`.
/// This allows a [VecResampler] to be made into a trait object like this:
/// ```
/// # use rubato::{FftFixedIn, VecResampler};
/// let boxed: Box<dyn VecResampler<f64>> = Box::new(FftFixedIn::<f64>::new(44100, 88200, 1024, 2, 2).unwrap());
/// ```
/// Use this implementation as an example if you need to fix the input type to something else.
#[macro_export]
macro_rules! implement_resampler {
    ($trait_name:ident, $in_type:ty, $out_type:ty) => {
        #[doc = "This is an wrapper trait implemented via the [implement_resampler] macro."]
        #[doc = "The generic input and output types `&[AsRef<[T]>]` and `&mut [AsMut<[T]>]`"]
        #[doc = concat!("are locked to `", stringify!($in_type), "` and `", stringify!($out_type), "`.")]
        pub trait $trait_name<T>: Send {

            /// Refer to [Resampler::process].
            fn process(
                &mut self,
                wave_in: $in_type,
                active_channels_mask: Option<&[bool]>,
            ) -> rubato::ResampleResult<Vec<Vec<T>>>;

            /// Refer to [Resampler::process_into_buffer].
            fn process_into_buffer(
                &mut self,
                wave_in: $in_type,
                wave_out: $out_type,
                active_channels_mask: Option<&[bool]>,
            ) -> rubato::ResampleResult<(usize, usize)>;

            /// Refer to [Resampler::process_partial_into_buffer].
            fn process_partial_into_buffer(
                &mut self,
                wave_in: Option<$in_type>,
                wave_out: $out_type,
                active_channels_mask: Option<&[bool]>,
            ) -> rubato::ResampleResult<(usize, usize)>;

            /// Refer to [Resampler::process_partial].
            fn process_partial(
                &mut self,
                wave_in: Option<$in_type>,
                active_channels_mask: Option<&[bool]>,
            ) -> rubato::ResampleResult<Vec<Vec<T>>>;

            /// Refer to [Resampler::input_buffer_allocate].
            fn input_buffer_allocate(&self, filled: bool) -> Vec<Vec<T>>;

            /// Refer to [Resampler::input_frames_max].
            fn input_frames_max(&self) -> usize;

            /// Refer to [Resampler::input_frames_next].
            fn input_frames_next(&self) -> usize;

            /// Refer to [Resampler::nbr_channels].
            fn nbr_channels(&self) -> usize;

            /// Refer to [Resampler::output_buffer_allocate].
            fn output_buffer_allocate(&self, filled: bool) -> Vec<Vec<T>>;

            /// Refer to [Resampler::output_frames_max].
            fn output_frames_max(&self) -> usize;

            /// Refer to [Resampler::output_frames_next].
            fn output_frames_next(&self) -> usize;

            /// Refer to [Resampler::output_delay].
            fn output_delay(&self) -> usize;

            /// Refer to [Resampler::set_resample_ratio].
            fn set_resample_ratio(&mut self, new_ratio: f64, ramp: bool) -> rubato::ResampleResult<()>;

            /// Refer to [Resampler::set_resample_ratio_relative].
            fn set_resample_ratio_relative(&mut self, rel_ratio: f64, ramp: bool) -> rubato::ResampleResult<()>;
        }

        impl<T, U> $trait_name<T> for U
        where
            U: rubato::Resampler<T>,
            T: rubato::Sample,
        {
            fn process(
                &mut self,
                wave_in: $in_type,
                active_channels_mask: Option<&[bool]>,
            ) -> rubato::ResampleResult<Vec<Vec<T>>> {
                rubato::Resampler::process(self, wave_in, active_channels_mask)
            }

            fn process_into_buffer(
                &mut self,
                wave_in: $in_type,
                wave_out: $out_type,
                active_channels_mask: Option<&[bool]>,
            ) -> rubato::ResampleResult<(usize, usize)> {
                rubato::Resampler::process_into_buffer(self, wave_in, wave_out, active_channels_mask)
            }

            fn process_partial_into_buffer(
                &mut self,
                wave_in: Option<$in_type>,
                wave_out: $out_type,
                active_channels_mask: Option<&[bool]>,
            ) -> rubato::ResampleResult<(usize, usize)> {
                rubato::Resampler::process_partial_into_buffer(
                    self,
                    wave_in.map(AsRef::as_ref),
                    wave_out,
                    active_channels_mask,
                )
            }

            fn process_partial(
                &mut self,
                wave_in: Option<$in_type>,
                active_channels_mask: Option<&[bool]>,
            ) -> rubato::ResampleResult<Vec<Vec<T>>> {
                rubato::Resampler::process_partial(self, wave_in, active_channels_mask)
            }

            fn output_buffer_allocate(&self, filled: bool) -> Vec<Vec<T>> {
                rubato::Resampler::output_buffer_allocate(self, filled)
            }

            fn output_frames_next(&self) -> usize {
                rubato::Resampler::output_frames_next(self)
            }

            fn output_frames_max(&self) -> usize {
                rubato::Resampler::output_frames_max(self)
            }

            fn input_frames_next(&self) -> usize {
                rubato::Resampler::input_frames_next(self)
            }

            fn output_delay(&self) -> usize {
                rubato::Resampler::output_delay(self)
            }

            fn nbr_channels(&self) -> usize {
                rubato::Resampler::nbr_channels(self)
            }

            fn input_frames_max(&self) -> usize {
                rubato::Resampler::input_frames_max(self)
            }

            fn input_buffer_allocate(&self, filled: bool) -> Vec<Vec<T>> {
                rubato::Resampler::input_buffer_allocate(self, filled)
            }

            fn set_resample_ratio(&mut self, new_ratio: f64, ramp: bool) -> rubato::ResampleResult<()> {
                rubato::Resampler::set_resample_ratio(self, new_ratio, ramp)
            }

            fn set_resample_ratio_relative(&mut self, rel_ratio: f64, ramp: bool) -> rubato::ResampleResult<()> {
                rubato::Resampler::set_resample_ratio_relative(self, rel_ratio, ramp)
            }
        }
    }
}

implement_resampler!(VecResampler, &[Vec<T>], &mut [Vec<T>]);

/// Helper to make a mask where all channels are marked as active.
fn update_mask_from_buffers(mask: &mut [bool]) {
    mask.iter_mut().for_each(|v| *v = true);
}

pub(crate) fn validate_buffers<T, Vin: AsRef<[T]>, Vout: AsMut<[T]>>(
    wave_in: &[Vin],
    wave_out: &mut [Vout],
    mask: &[bool],
    channels: usize,
    min_input_len: usize,
    min_output_len: usize,
) -> ResampleResult<()> {
    if wave_in.len() != channels {
        return Err(ResampleError::WrongNumberOfInputChannels {
            expected: channels,
            actual: wave_in.len(),
        });
    }
    if mask.len() != channels {
        return Err(ResampleError::WrongNumberOfMaskChannels {
            expected: channels,
            actual: wave_in.len(),
        });
    }
    for (chan, wave_in) in wave_in.iter().enumerate().filter(|(chan, _)| mask[*chan]) {
        let actual_len = wave_in.as_ref().len();
        if actual_len < min_input_len {
            return Err(ResampleError::InsufficientInputBufferSize {
                channel: chan,
                expected: min_input_len,
                actual: actual_len,
            });
        }
    }
    if wave_out.len() != channels {
        return Err(ResampleError::WrongNumberOfOutputChannels {
            expected: channels,
            actual: wave_out.len(),
        });
    }
    for (chan, wave_out) in wave_out
        .iter_mut()
        .enumerate()
        .filter(|(chan, _)| mask[*chan])
    {
        let actual_len = wave_out.as_mut().len();
        if actual_len < min_output_len {
            return Err(ResampleError::InsufficientOutputBufferSize {
                channel: chan,
                expected: min_output_len,
                actual: actual_len,
            });
        }
    }
    Ok(())
}

/// Convenience method for allocating a buffer to hold a given number of channels and frames.
/// The `filled` argument determines if the vectors should be pre-filled with zeros or not.
/// When false, the vectors are only allocated but returned empty.
pub fn make_buffer<T: Sample>(channels: usize, frames: usize, filled: bool) -> Vec<Vec<T>> {
    let mut buffer = Vec::with_capacity(channels);
    for _ in 0..channels {
        buffer.push(Vec::with_capacity(frames));
    }
    if filled {
        resize_buffer(&mut buffer, frames)
    }
    buffer
}

/// Convenience method for resizing a buffer to a new number of frames.
/// If the new number of frames is no larger than the buffer capacity,
/// no reallocation will occur.
/// If the new length is smaller than the current, the excess elements are dropped.
/// If it is larger, zeros are inserted for the missing elements.
pub fn resize_buffer<T: Sample>(buffer: &mut [Vec<T>], frames: usize) {
    buffer.iter_mut().for_each(|v| v.resize(frames, T::zero()));
}

/// Convenience method for getting the current length of a buffer in frames.
/// Checks the [length](Vec::len) of the vector for each channel and returns the smallest.
pub fn buffer_length<T: Sample>(buffer: &[Vec<T>]) -> usize {
    return buffer.iter().map(|v| v.len()).min().unwrap_or_default();
}

/// Convenience method for getting the current allocated capacity of a buffer in frames.
/// Checks the [capacity](Vec::capacity) of the vector for each channel and returns the smallest.
pub fn buffer_capacity<T: Sample>(buffer: &[Vec<T>]) -> usize {
    return buffer
        .iter()
        .map(|v| v.capacity())
        .min()
        .unwrap_or_default();
}

#[cfg(test)]
pub mod tests {
    use crate::{buffer_capacity, buffer_length, make_buffer, resize_buffer, VecResampler};
    use crate::{FftFixedIn, FftFixedInOut, FftFixedOut};
    use crate::{SincFixedIn, SincFixedOut};

    // This tests that a VecResampler can be boxed.
    #[test]
    fn boxed_resampler() {
        let boxed: Box<dyn VecResampler<f64>> =
            Box::new(FftFixedIn::<f64>::new(44100, 88200, 1024, 2, 2).unwrap());
        let result = process_with_boxed(boxed);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 2048);
        assert_eq!(result[1].len(), 2048);
    }

    fn process_with_boxed(mut resampler: Box<dyn VecResampler<f64>>) -> Vec<Vec<f64>> {
        let frames = resampler.input_frames_next();
        let waves = vec![vec![0.0f64; frames]; 2];
        resampler.process(&waves, None).unwrap()
    }

    fn impl_send<T: Send>() {
        fn is_send<T: Send>() {}
        is_send::<SincFixedOut<T>>();
        is_send::<SincFixedIn<T>>();
        is_send::<FftFixedOut<T>>();
        is_send::<FftFixedIn<T>>();
        is_send::<FftFixedInOut<T>>();
    }

    // This tests that all resamplers are Send.
    #[test]
    fn test_impl_send() {
        impl_send::<f32>();
        impl_send::<f64>();
    }

    #[macro_export]
    macro_rules! check_output {
        ($name:ident, $resampler:ident) => {
            let mut val = 0.0;
            let mut prev_last = -0.1;
            let max_input_len = $resampler.input_frames_max();
            let max_output_len = $resampler.output_frames_max();
            for n in 0..50 {
                let frames = $resampler.input_frames_next();
                // Check that lengths are within the reported max values
                assert!(frames <= max_input_len);
                assert!($resampler.output_frames_next() <= max_output_len);
                let mut waves = vec![vec![0.0f64; frames]; 2];
                for m in 0..frames {
                    for ch in 0..2 {
                        waves[ch][m] = val;
                    }
                    val = val + 0.1;
                }
                let out = $resampler.process(&waves, None).unwrap();
                let frames_out = out[0].len();
                for ch in 0..2 {
                    assert!(
                        out[ch][0] > prev_last,
                        "Iteration {}, first value {} prev last value {}",
                        n,
                        out[ch][0],
                        prev_last
                    );
                    let expected_diff = frames as f64 * 0.1;
                    let diff = out[ch][frames_out - 1] - out[ch][0];
                    assert!(
                        diff < 1.5 * expected_diff && diff > 0.25 * expected_diff,
                        "Iteration {}, last value {} first value {}",
                        n,
                        out[ch][frames_out - 1],
                        out[ch][0]
                    );
                }
                prev_last = out[0][frames_out - 1];
                for m in 0..frames_out - 1 {
                    for ch in 0..2 {
                        let diff = out[ch][m + 1] - out[ch][m];
                        assert!(
                            diff < 0.15 && diff > -0.05,
                            "Frame {}:{} next value {} value {}",
                            n,
                            m,
                            out[ch][m + 1],
                            out[ch][m]
                        );
                    }
                }
            }
        };
    }

    #[test]
    fn test_buffer_helpers() {
        let buf1 = vec![vec![0.0f64; 7], vec![0.0f64; 5], vec![0.0f64; 10]];
        assert_eq!(buffer_length(&buf1), 5);
        let mut buf2 = vec![Vec::<f32>::with_capacity(5), Vec::<f32>::with_capacity(15)];
        assert_eq!(buffer_length(&buf2), 0);
        assert_eq!(buffer_capacity(&buf2), 5);

        resize_buffer(&mut buf2, 3);
        assert_eq!(buffer_length(&buf2), 3);
        assert_eq!(buffer_capacity(&buf2), 5);

        let buf3 = make_buffer::<f32>(4, 10, false);
        assert_eq!(buffer_length(&buf3), 0);
        assert_eq!(buffer_capacity(&buf3), 10);

        let buf4 = make_buffer::<f32>(4, 10, true);
        assert_eq!(buffer_length(&buf4), 10);
        assert_eq!(buffer_capacity(&buf4), 10);
    }
}
