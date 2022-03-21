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
//! ### Input and output data format
//!
//! Input and output data is stored non-interleaved.
//!
//! The output data is stored in a vector of vectors, `Vec<Vec<f32>>` or `Vec<Vec<f64>>`.
//! The inner vectors (`Vec<f32>` or `Vec<f64>`) hold the sample values for one channel each.
//!
//! The input data is similar, except that it allows the inner vectors to be `AsRef<[f32]>` or `AsRef<[f64]>`.
//! Normal vectors can be used since `Vec` implements the `AsRef` trait.
//!
//! ### Asynchronous resampling
//!
//! The resampling is based on band-limited interpolation using sinc
//! interpolation filters. The sinc interpolation upsamples by an adjustable factor,
//! and then the new sample points are calculated by interpolating between these points.
//! The resampling ratio can be updated at any time.
//!
//! ### Synchronous resampling
//!
//! Synchronous resampling is implemented via FFT. The data is FFT:ed, the spectrum modified,
//! and then inverse FFT:ed to get the resampled data.
//! This type of resampler is considerably faster but doesn't support changing the resampling ratio.
//!
//! ### SIMD acceleration
//!
//! #### Asynchronous resampling
//!
//! The asynchronous resampler supports SIMD on x86_64 and on aarch64.
//! The SIMD capabilities of the CPU are determined at runtime.
//! If no supported SIMD instruction set is available, it falls back to a scalar implementation.
//!
//! On x86_64 it will try to use AVX. If AVX isn't available, it will instead try SSE3.
//!
//! On aarch64 (64-bit Arm) it will use Neon if available.
//!
//! #### Synchronous resampling
//!
//! The synchronous resamplers benefit from the SIMD support of the RustFFT library.
//!
//! ### Cargo features
//!
//! ##### `log`: Enable logging
//!
//! This feature enables logging via the `log` crate. This is intended for debugging purposes.
//! Note that outputting logs allocates a [std::string::String] and most logging implementations involve various other system calls.
//! These calls may take some (unpredictable) time to return, during which the application is blocked.
//! This means that logging should be avoided if using this library in a realtime application.
//!
//! ## Example
//!
//! Resample a single chunk of a dummy audio file from 44100 to 48000 Hz.
//! See also the "fixedin64" example that can be used to process a file from disk.
//! ```
//! use rubato::{Resampler, SincFixedIn, InterpolationType, InterpolationParameters, WindowFunction};
//! let params = InterpolationParameters {
//!     sinc_len: 256,
//!     f_cutoff: 0.95,
//!     interpolation: InterpolationType::Linear,
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
//! ## Compatibility
//!
//! The `rubato` crate requires rustc version 1.61 or newer.
//!
//! ## Changelog
//!
//! - v0.12.0
//!   - Always enable all simd acceleration (and remove Cargo features).
//! - v0.11.0
//!   - New api to allow use in realtime applications.
//!   - Configurable adjust range of asynchronous resamplers.
//! - v0.10.1
//!   - Fix compiling with neon feature after changes in latest nightly.
//! - v0.10.0
//!   - Add an object-safe wrapper trait for Resampler.
//! - v0.9.0
//!   - Accept any AsRef<[T]> as input.

#[cfg(feature = "log")]
extern crate log;

// Logging wrapper macros to avoid cluttering the code with conditionals
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

mod asynchro;
mod error;
mod interpolation;
mod sample;
mod sinc;
mod synchro;
mod windows;

pub use crate::asynchro::{ScalarInterpolator, SincFixedIn, SincFixedOut};
pub use crate::error::{
    CpuFeature, MissingCpuFeature, ResampleError, ResampleResult, ResamplerConstructionError,
};
pub use crate::sample::Sample;
pub use crate::synchro::{FftFixedIn, FftFixedInOut, FftFixedOut};
pub use crate::windows::WindowFunction;

/// Helper macro to define a dummy implementation of the sample trait if a
/// feature is not supported.
macro_rules! interpolator {
    (
    #[cfg($($cond:tt)*)]
    mod $mod:ident;
    trait $trait:ident;
    ) => {
        #[cfg($($cond)*)]
        pub mod $mod;

        #[cfg($($cond)*)]
        use self::$mod::$trait;

        /// Dummy trait when not supported.
        #[cfg(not($($cond)*))]
        pub trait $trait {
        }

        /// Dummy impl of trait when not supported.
        #[cfg(not($($cond)*))]
        impl<T> $trait for T where T: Sample {
        }
    }
}

interpolator! {
    #[cfg(target_arch = "x86_64")]
    mod interpolator_avx;
    trait AvxSample;
}

interpolator! {
    #[cfg(target_arch = "x86_64")]
    mod interpolator_sse;
    trait SseSample;
}

interpolator! {
    #[cfg(target_arch = "aarch64")]
    mod interpolator_neon;
    trait NeonSample;
}

/// A struct holding the parameters for interpolation.
#[derive(Debug)]
pub struct InterpolationParameters {
    /// Length of the windowed sinc interpolation filter.
    /// Higher values can allow a higher cut-off frequency leading to less high frequency roll-off
    /// at the expense of higher cpu usage. 256 is a good starting point.
    /// The value will be rounded up to the nearest multiple of 8.
    pub sinc_len: usize,
    /// Relative cutoff frequency of the sinc interpolation filter
    /// (relative to the lowest one of fs_in/2 or fs_out/2). Start at 0.95, and increase if needed.
    pub f_cutoff: f32,
    /// The number of intermediate points to use for interpolation.
    /// Higher values use more memory for storing the sinc filters.
    /// Only the points actually needed are calculated during processing
    /// so a larger number does not directly lead to higher cpu usage.
    /// But keeping it down helps in keeping the sincs in the cpu cache. Start at 128.
    pub oversampling_factor: usize,
    /// Interpolation type, see `InterpolationType`
    pub interpolation: InterpolationType,
    /// Window function to use.
    pub window: WindowFunction,
}

/// Interpolation methods that can be selected. For asynchronous interpolation where the
/// ratio between input and output sample rates can be any number, it's not possible to
/// pre-calculate all the needed interpolation filters.
/// Instead they have to be computed as needed, which becomes impractical since the
/// sincs are very expensive to generate in terms of cpu time.
/// It's more efficient to combine the sinc filters with some other interpolation technique.
/// Then sinc filters are used to provide a fixed number of interpolated points between input samples,
/// and then the new value is calculated by interpolation between those points.
#[derive(Debug)]
pub enum InterpolationType {
    /// For cubic interpolation, the four nearest intermediate points are calculated
    /// using sinc interpolation.
    /// Then a cubic polynomial is fitted to these points, and is then used to calculate the new sample value.
    /// The computation time as about twice the one for linear interpolation,
    /// but it requires much fewer intermediate points for a good result.
    Cubic,
    /// With linear interpolation the new sample value is calculated by linear interpolation
    /// between the two nearest points.
    /// This requires two intermediate points to be calculated using sinc interpolation,
    /// and te output is a weighted average of these two.
    /// This is relatively fast, but needs a large number of intermediate points to
    /// push the resampling artefacts below the noise floor.
    Linear,
    /// The Nearest mode doesn't do any interpolation, but simply picks the nearest intermediate point.
    /// This is useful when the nearest point is actually the correct one, for example when upsampling by a factor 2,
    /// like 48kHz->96kHz.
    /// Then setting the oversampling_factor to 2, and using Nearest mode,
    /// no unnecessary computations are performed and the result is the same as for synchronous resampling.
    /// This also works for other ratios that can be expressed by a fraction. For 44.1kHz -> 48 kHz,
    /// setting oversampling_factor to 160 gives the desired result (since 48kHz = 160/147 * 44.1kHz).
    Nearest,
}

/// A resampler that us used to resample a chunk of audio to a new sample rate.
/// For asynchronous resamplers, the rate can be adjusted as required.
///
/// This trait is not object safe. If you need an object safe resampler,
/// use the [VecResampler] wrapper trait.
pub trait Resampler<T>: Send {
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
        for _ in 0..channels {
            wave_out.push(Vec::with_capacity(frames));
        }
        self.process_into_buffer(wave_in, &mut wave_out, active_channels_mask)?;
        Ok(wave_out)
    }

    /// Resample a buffer of audio to a pre-allocated output buffer.
    /// Use this in real-time applications where the unpredictable time required to allocate
    /// memory from the heap can cause glitches. If this is not a problem, you may use
    /// the [process](Resampler::process) method instead.
    ///
    /// The input and output buffers are noninterleaved.
    /// The input is a slice, where each element of the slice is itself referenceable
    /// as a slice ([AsRef<\[T\]>](AsRef)) which contains the samples for a single channel.
    /// Because [Vec<T>] implements [AsRef<\[T\]>](AsRef), the input may be [`Vec<Vec<T>>`](Vec).
    ///
    /// The output data is a slice, where each element of the slice is a [Vec] which contains
    /// the samples for a single channel. If the output channel vectors do not have sufficient
    /// capacity for all output samples, they will be resized by this function. To avoid these
    /// allocations during this function, allocate the output buffer with
    /// [output_buffer_allocate](Resampler::output_buffer_allocate) before calling this function
    /// and reuse the same buffer for each call.
    ///
    /// The `active_channels_mask` is optional.
    /// Any channel marked as inactive by a false value will be skipped during processing
    /// and the corresponding output will be left unchanged.
    /// If `None` is given, all channels will be considered active unless their length is 0.
    fn process_into_buffer<V: AsRef<[T]>>(
        &mut self,
        wave_in: &[V],
        wave_out: &mut [Vec<T>],
        active_channels_mask: Option<&[bool]>,
    ) -> ResampleResult<()>;

    /// Convenience method for allocating an input buffer suitable for use with
    /// [process_into_buffer](Resampler::process_into_buffer). The buffer's capacity
    /// is big enough to prevent allocating additional heap memory before any call to
    /// [process_into_buffer](Resampler::process_into_buffer) regardless of the current
    /// resampling ratio.
    fn input_buffer_allocate(&self) -> Vec<Vec<T>> {
        let frames = self.input_frames_max();
        let channels = self.nbr_channels();
        let mut buffer = Vec::with_capacity(channels);
        for _ in 0..channels {
            buffer.push(Vec::with_capacity(frames));
        }
        buffer
    }

    /// Get the maximum number of input frames per channel the resampler could require
    fn input_frames_max(&self) -> usize;

    /// Get the number of frames per channel needed for the next call to
    /// [process_into_buffer](Resampler::process_into_buffer) or [process](Resampler::process)
    fn input_frames_next(&self) -> usize;

    /// Get the maximum number of channels this Resampler is configured for
    fn nbr_channels(&self) -> usize;

    /// Convenience method for allocating an output buffer suitable for use with
    /// [process_into_buffer](Resampler::process_into_buffer). The buffer's capacity
    /// is big enough to prevent allocating additional heap memory during any call to
    /// [process_into_buffer](Resampler::process_into_buffer) regardless of the current
    /// resampling ratio.
    fn output_buffer_allocate(&self) -> Vec<Vec<T>> {
        let frames = self.output_frames_max();
        let channels = self.nbr_channels();
        let mut buffer = Vec::with_capacity(channels);
        for _ in 0..channels {
            buffer.push(Vec::with_capacity(frames));
        }
        buffer
    }

    /// Get the max number of output frames per channel
    fn output_frames_max(&self) -> usize;

    /// Get the number of frames per channel that will be output from the next call to
    /// [process_into_buffer](Resampler::process_into_buffer) or [process](Resampler::process)
    fn output_frames_next(&self) -> usize;

    /// Update the resample ratio
    ///
    /// For asynchronous resamplers, the ratio must be within
    /// `original / maximum` to `original * maximum`, where the original and maximum are the
    /// resampling ratios that were provided to the constructor. Trying to set the ratio
    /// outside these bounds will return [ResampleError::RatioOutOfBounds].
    ///
    /// For synchronous resamplers, this will always return [ResampleError::SyncNotAdjustable].
    fn set_resample_ratio(&mut self, new_ratio: f64) -> ResampleResult<()>;

    /// Update the resample ratio as a factor relative to the original one
    ///
    /// For asynchronous resamplers, the relative ratio must be within
    /// `1 / maximum` to `maximum`, where maximum is the maximum
    /// resampling ratio that was provided to the constructor. Trying to set the ratio
    /// outside these bounds will return [ResampleError::RatioOutOfBounds].
    ///
    /// Higher ratios above 1.0 slow down the output and lower the pitch. Lower ratios
    /// below 1.0 speed up the output and raise the pitch.
    ///
    /// For synchronous resamplers, this will always return [ResampleError::SyncNotAdjustable].
    fn set_resample_ratio_relative(&mut self, rel_ratio: f64) -> ResampleResult<()>;
}

/// This is a helper trait that can be used when a [Resampler] must be object safe.
///
/// It differs from [Resampler] only by fixing the type of the input of `process()`
/// and `process_into_buffer` to `&[Vec<T>]`.
/// This allows it to be made into a trait object like this:
/// ```
/// # use rubato::{FftFixedIn, VecResampler};
/// let boxed: Box<dyn VecResampler<f64>> = Box::new(FftFixedIn::<f64>::new(44100, 88200, 1024, 2, 2).unwrap());
/// ```
/// Use this implementation as an example if you need to fix the input type to something else.
pub trait VecResampler<T>: Send {
    /// Refer to [Resampler::process]
    fn process(
        &mut self,
        wave_in: &[Vec<T>],
        active_channels_mask: Option<&[bool]>,
    ) -> ResampleResult<Vec<Vec<T>>>;

    /// Refer to [Resampler::process_into_buffer]
    fn process_into_buffer(
        &mut self,
        wave_in: &[Vec<T>],
        wave_out: &mut [Vec<T>],
        active_channels_mask: Option<&[bool]>,
    ) -> ResampleResult<()>;

    /// Refer to [Resampler::input_buffer_allocate]
    fn input_buffer_allocate(&self) -> Vec<Vec<T>>;

    /// Refer to [Resampler::input_frames_max]
    fn input_frames_max(&self) -> usize;

    /// Refer to [Resampler::input_frames_next]
    fn input_frames_next(&self) -> usize;

    /// Refer to [Resampler::nbr_channels]
    fn nbr_channels(&self) -> usize;

    /// Refer to [Resampler::output_buffer_allocate]
    fn output_buffer_allocate(&self) -> Vec<Vec<T>>;

    /// Refer to [Resampler::output_frames_max]
    fn output_frames_max(&self) -> usize;

    /// Refer to [Resampler::output_frames_next]
    fn output_frames_next(&self) -> usize;

    /// Refer to [Resampler::set_resample_ratio]
    fn set_resample_ratio(&mut self, new_ratio: f64) -> ResampleResult<()>;

    /// Refer to [Resampler::set_resample_ratio_relative]
    fn set_resample_ratio_relative(&mut self, rel_ratio: f64) -> ResampleResult<()>;
}

impl<T, U> VecResampler<T> for U
where
    U: Resampler<T>,
{
    fn process(
        &mut self,
        wave_in: &[Vec<T>],
        active_channels_mask: Option<&[bool]>,
    ) -> ResampleResult<Vec<Vec<T>>> {
        Resampler::process(self, wave_in, active_channels_mask)
    }

    fn process_into_buffer(
        &mut self,
        wave_in: &[Vec<T>],
        wave_out: &mut [Vec<T>],
        active_channels_mask: Option<&[bool]>,
    ) -> ResampleResult<()> {
        Resampler::process_into_buffer(self, wave_in, wave_out, active_channels_mask)
    }

    fn output_buffer_allocate(&self) -> Vec<Vec<T>> {
        Resampler::output_buffer_allocate(self)
    }

    fn output_frames_next(&self) -> usize {
        Resampler::output_frames_next(self)
    }

    fn output_frames_max(&self) -> usize {
        Resampler::output_frames_max(self)
    }

    fn input_frames_next(&self) -> usize {
        Resampler::input_frames_next(self)
    }

    fn nbr_channels(&self) -> usize {
        Resampler::nbr_channels(self)
    }

    fn input_frames_max(&self) -> usize {
        Resampler::input_frames_max(self)
    }

    fn input_buffer_allocate(&self) -> Vec<Vec<T>> {
        Resampler::input_buffer_allocate(self)
    }

    fn set_resample_ratio(&mut self, new_ratio: f64) -> ResampleResult<()> {
        Resampler::set_resample_ratio(self, new_ratio)
    }

    fn set_resample_ratio_relative(&mut self, rel_ratio: f64) -> ResampleResult<()> {
        Resampler::set_resample_ratio_relative(self, rel_ratio)
    }
}

/// Helper to make a mask for the active channels based on which ones are empty.
fn update_mask_from_buffers<T, V: AsRef<[T]>>(wave_in: &[V], mask: &mut [bool]) {
    for (wave, active) in wave_in.iter().zip(mask.iter_mut()) {
        let wave = wave.as_ref();
        *active = !wave.is_empty();
    }
}

pub(crate) fn validate_buffers<T, V: AsRef<[T]>>(
    wave_in: &[V],
    wave_out: &mut [Vec<T>],
    mask: &[bool],
    channels: usize,
    needed_len: usize,
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
    for (chan, wave) in wave_in.iter().enumerate() {
        let wave = wave.as_ref();
        if wave.len() != needed_len && mask[chan] {
            return Err(ResampleError::WrongNumberOfInputFrames {
                channel: chan,
                expected: needed_len,
                actual: wave.len(),
            });
        }
    }
    if wave_out.len() != channels {
        return Err(ResampleError::WrongNumberOfOutputChannels {
            expected: channels,
            actual: wave_out.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::VecResampler;
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
}
