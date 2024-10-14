#![doc = include_str!("../README.md")]

#[cfg(feature = "log")]
extern crate log;
use audioadapter::owned::InterleavedOwned;
use audioadapter::{Adapter, AdapterMut};

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

mod asynchro;
mod asynchro_fast;
mod asynchro_sinc;
mod error;
mod interpolation;
mod sample;
mod sinc;
#[cfg(feature = "fft_resampler")]
mod synchro;
mod windows;

pub mod sinc_interpolator;

pub use crate::asynchro::{Async, FixedAsync};
pub use crate::asynchro_fast::PolynomialDegree;
pub use crate::asynchro_sinc::{SincInterpolationParameters, SincInterpolationType};
pub use crate::error::{
    CpuFeature, MissingCpuFeature, ResampleError, ResampleResult, ResamplerConstructionError,
};
pub use crate::sample::Sample;
#[cfg(feature = "fft_resampler")]
pub use crate::synchro::{Fft, FixedSync};
pub use crate::windows::{calculate_cutoff, WindowFunction};

/// A struct for providing optional parameters when calling
/// [process_into_buffer](Resampler::process_into_buffer).
#[derive(Debug)]
pub struct Indexing {
    pub input_offset: usize,
    pub output_offset: usize,
    pub partial_len: Option<usize>,
    pub active_channels_mask: Option<Vec<bool>>,
}

pub(crate) fn get_offsets(indexing: &Option<&Indexing>) -> (usize, usize) {
    indexing
        .as_ref()
        .map(|idx| (idx.input_offset, idx.output_offset))
        .unwrap_or((0, 0))
}

pub(crate) fn get_partial_len(indexing: &Option<&Indexing>) -> Option<usize> {
    indexing.as_ref().and_then(|idx| idx.partial_len)
}

/// Helper to update the mask from an optional Indexing struct
pub(crate) fn update_mask(indexing: &Option<&Indexing>, mask: &mut [bool]) {
    if let Some(idx) = indexing {
        if let Some(new_mask) = &idx.active_channels_mask {
            mask.copy_from_slice(new_mask);
            return;
        }
    }
    mask.iter_mut().for_each(|v| *v = true);
}

/// A resampler that is used to resample a chunk of audio to a new sample rate.
/// For asynchronous resamplers, the rate can be adjusted as required.
pub trait Resampler<T>: Send
where
    T: Sample,
{
    /// This is a convenience wrapper for [process_into_buffer](Resampler::process_into_buffer)
    /// that allocates the output buffer with each call. For realtime applications, use
    /// [process_into_buffer](Resampler::process_into_buffer) with a pre-allocated buffer
    /// instead of this function.
    ///
    /// The output is returned as an [InterleavedOwned] struct that wraps a `Vec<T>`
    /// of interleaved samples.
    ///
    /// The `input_offset` and `active_channels_mask` parameters have the same meaning as in
    /// [process_into_buffer](Resampler::process_into_buffer).
    fn process(
        &mut self,
        buffer_in: &dyn Adapter<'_, T>,
        input_offset: usize,
        active_channels_mask: Option<&[bool]>,
    ) -> ResampleResult<InterleavedOwned<T>> {
        let frames = self.output_frames_next();
        let channels = self.nbr_channels();
        let mut buffer_out = InterleavedOwned::<T>::new(T::coerce_from(0.0), channels, frames);

        let indexing = Indexing {
            input_offset,
            output_offset: 0,
            partial_len: None,
            active_channels_mask: active_channels_mask.map(|m| m.to_vec()),
        };
        self.process_into_buffer(buffer_in, &mut buffer_out, Some(&indexing))?;
        Ok(buffer_out)
    }

    /// Resample a buffer of audio to a pre-allocated output buffer.
    /// Use this in real-time applications where the unpredictable time required to allocate
    /// memory from the heap can cause glitches. If this is not a problem, you may use
    /// the [process](Resampler::process) method instead.
    ///
    /// The input and output buffers are buffers from the [audioadapter] crate.
    /// The input buffer must implement the [Adapter] trait,
    /// and the output the corresponding [AdapterMut] trait.
    /// This ensures that this method is able to both read and write
    /// audio data from and to buffers with different layout, as well as different sample formats.
    ///
    /// The `indexing` parameter is optional. When left out, the default values are used.
    ///  - `input_offset` and `output_offset`: these determine how many frames at the beginning
    ///    of the input and output buffers will be skipped before reading or writing starts.
    ///    See the `process_f64` example for how these may be used to process a longer sound clip.
    ///  - `partial_len`: If the input buffer has fewer frames than the required input length,
    ///    set `partial_len` to the available number.
    ///    The resampler will then insert silence in place of the missing frames.
    ///    This is useful for processing a longer buffer with repeated process calls,
    ///    where at the last iteration there may be fewer frames left than what the resampler needs.
    ///  - `active_channels_mask`: A vector of booleans determining what channels are to be processed.
    ///    Any channel marked as inactive by a false value will be skipped during processing
    ///    and the corresponding output will be left unchanged.
    ///    If `None` is given, all channels will be considered active.
    ///
    /// Before processing, the input and output buffer sizes are checked.
    /// If either has the wrong number of channels, or if the buffer can hold too few frames,
    /// a [ResampleError] is returned.
    /// Both input and output are allowed to be longer than required.
    /// The number of input samples consumed and the number output samples written
    /// per channel is returned in a tuple, `(input_frames, output_frames)`.
    fn process_into_buffer<'a>(
        &mut self,
        buffer_in: &dyn Adapter<'a, T>,
        buffer_out: &mut dyn AdapterMut<'a, T>,
        indexing: Option<&Indexing>,
    ) -> ResampleResult<(usize, usize)>;

    /// Get the maximum possible number of input frames per channel the resampler could require.
    fn input_frames_max(&self) -> usize;

    /// Get the number of frames per channel needed for the next call to
    /// [process_into_buffer](Resampler::process_into_buffer) or [process](Resampler::process).
    fn input_frames_next(&self) -> usize;

    /// Get the number of channels this Resampler is configured for.
    fn nbr_channels(&self) -> usize;

    /// Get the maximum possible number of output frames per channel.
    fn output_frames_max(&self) -> usize;

    /// Get the number of frames per channel that will be output from the next call to
    /// [process_into_buffer](Resampler::process_into_buffer) or [process](Resampler::process).
    fn output_frames_next(&self) -> usize;

    /// Get the delay for the resampler, reported as a number of output frames.
    /// This gives how many frames any event in the input is delayed before it appears in the output.
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

    /// Get the current resample ratio, defined as output sample rate divided by input sample rate.
    fn resample_ratio(&self) -> f64;

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

    /// Change the chunk size for the resampler.
    /// This is not supported by all resampler types.
    /// The value must be equal to or smaller than the chunk size the value
    /// that the resampler was created with.
    /// [ResampleError::InvalidChunkSize] is returned if the value is zero or too large.
    ///
    /// The meaning of chunk size depends on the resampler,
    /// it refers to the input size for resamplers with fixed input size,
    /// and output size for resamplers with fixed output size.
    ///
    /// Resamplers that do not support changing the chunk size
    /// return [ResampleError::ChunkSizeNotAdjustable].
    fn set_chunk_size(&mut self, _chunksize: usize) -> ResampleResult<()> {
        Err(ResampleError::ChunkSizeNotAdjustable)
    }
}

pub(crate) fn validate_buffers<'a, T: 'a>(
    wave_in: &dyn Adapter<'a, T>,
    wave_out: &dyn AdapterMut<'a, T>,
    mask: &[bool],
    channels: usize,
    min_input_len: usize,
    min_output_len: usize,
) -> ResampleResult<()> {
    if wave_in.channels() != channels {
        return Err(ResampleError::WrongNumberOfInputChannels {
            expected: channels,
            actual: wave_in.channels(),
        });
    }
    if mask.len() != channels {
        return Err(ResampleError::WrongNumberOfMaskChannels {
            expected: channels,
            actual: mask.len(),
        });
    }
    if wave_in.frames() < min_input_len {
        return Err(ResampleError::InsufficientInputBufferSize {
            expected: min_input_len,
            actual: wave_in.frames(),
        });
    }
    if wave_out.channels() != channels {
        return Err(ResampleError::WrongNumberOfOutputChannels {
            expected: channels,
            actual: wave_out.channels(),
        });
    }
    if wave_out.frames() < min_output_len {
        return Err(ResampleError::InsufficientOutputBufferSize {
            expected: min_output_len,
            actual: wave_out.frames(),
        });
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
    #[cfg(feature = "fft_resampler")]
    use crate::Fft;
    use crate::{buffer_capacity, buffer_length, make_buffer, resize_buffer, Resampler};
    use crate::{
        Async, FixedAsync, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    }; //, PolynomialDegree};
    use audioadapter::direct::SequentialSliceOfVecs;
    use test_log::test;

    // This tests that a Resampler can be boxed.
    #[test]
    fn boxed_resampler() {
        let mut boxed: Box<dyn Resampler<f64>> = Box::new(
            Async::<f64>::new_sinc(
                88200 as f64 / 44100 as f64,
                1.1,
                SincInterpolationParameters {
                    sinc_len: 64,
                    f_cutoff: 0.95,
                    interpolation: SincInterpolationType::Cubic,
                    oversampling_factor: 16,
                    window: WindowFunction::BlackmanHarris2,
                },
                1024,
                2,
                FixedAsync::Input,
            )
            .unwrap(),
        );
        let max_frames_out = boxed.output_frames_max();
        let nbr_frames_in_next = boxed.input_frames_next();
        let waves = vec![vec![0.0f64; nbr_frames_in_next]; 2];
        let mut waves_out = vec![vec![0.0f64; max_frames_out]; 2];
        let input = SequentialSliceOfVecs::new(&waves, 2, nbr_frames_in_next).unwrap();
        let mut output = SequentialSliceOfVecs::new_mut(&mut waves_out, 2, max_frames_out).unwrap();
        let _ = process_with_boxed(&mut boxed, &input, &mut output);
    }

    fn process_with_boxed<'a>(
        resampler: &mut Box<dyn Resampler<f64>>,
        input: &SequentialSliceOfVecs<&'a [Vec<f64>]>,
        output: &mut SequentialSliceOfVecs<&'a mut [Vec<f64>]>,
    ) {
        resampler.process_into_buffer(input, output, None).unwrap();
    }

    fn impl_send<T: Send>() {
        fn is_send<T: Send>() {}
        is_send::<Async<T>>();
        #[cfg(feature = "fft_resampler")]
        {
            is_send::<Fft<T>>();
        }
    }

    // This tests that all resamplers are Send.
    #[test]
    fn test_impl_send() {
        impl_send::<f32>();
        impl_send::<f64>();
    }

    #[macro_export]
    macro_rules! check_output {
        ($resampler:ident) => {
            let mut val = 0.0;
            let max_input_len = $resampler.input_frames_max();
            let max_output_len = $resampler.output_frames_max();
            let ratio = $resampler.resample_ratio();
            let mut delay = $resampler.output_delay();
            let mut prev_last = -0.1 / ratio;
            for n in 0..50 {
                let frames_in = $resampler.input_frames_next();
                let frames_out = $resampler.output_frames_next();
                // Check that lengths are within the reported max values
                assert!(frames_in <= max_input_len);
                assert!(frames_out <= max_output_len);
                let mut waves = vec![vec![0.0f64; frames_in]; 2];
                for m in 0..frames_in {
                    for ch in 0..2 {
                        waves[ch][m] = val;
                    }
                    val = val + 0.1;
                }
                let input = SequentialSliceOfVecs::new(&waves, 2, frames_in).unwrap();
                let mut waves_out = vec![vec![0.0f64; frames_out]; 2];
                let mut output =
                    SequentialSliceOfVecs::new_mut(&mut waves_out, 2, frames_out).unwrap();

                let (_input_frames, output_frames) = $resampler
                    .process_into_buffer(&input, &mut output, None)
                    .unwrap();

                for ch in 0..2 {
                    let diff = waves_out[ch][0] - prev_last;
                    assert!(
                        diff < 0.125 / ratio && diff > 0.075 / ratio,
                        "Iteration {}, first value {} prev last value {}",
                        n,
                        waves_out[ch][0],
                        prev_last
                    );
                    let expected_diff = (frames_out - delay) as f64 * 0.1 / ratio;
                    let first = waves_out[ch][0];
                    let last = waves_out[ch][output_frames - 1];
                    let diff = last - first;
                    assert!(
                        diff < 1.1 * expected_diff && diff > 0.9 * expected_diff,
                        "Iteration {}, last value {} first value {}, diff {}, expected {}",
                        n,
                        last,
                        first,
                        diff,
                        expected_diff,
                    );
                }

                prev_last = waves_out[0][output_frames - 1];
                for m in 0..output_frames - 1 {
                    let (upper, lower) = if m < delay {
                        // beginning of first iteration, allow a larger range here
                        (0.2 / ratio, -0.2 / ratio)
                    } else {
                        (0.125 / ratio, 0.075 / ratio)
                    };
                    for ch in 0..2 {
                        let diff = waves_out[ch][m + 1] - waves_out[ch][m];
                        assert!(
                            diff < upper && diff > lower,
                            "Too large diff, frame {}:{} next value {} value {}",
                            n,
                            m,
                            waves_out[ch][m + 1],
                            waves_out[ch][m]
                        );
                    }
                }
                // set delay to zero, the value is only needed for the first process call
                delay = 0;
            }
        };
    }

    #[macro_export]
    macro_rules! check_ratio {
        ($resampler:ident, $ratio:ident, $repetitions:literal, $fty:ty) => {
            let max_input_len = $resampler.input_frames_max();
            let max_output_len = $resampler.output_frames_max();
            let waves_in = vec![vec![0.0 as $fty; max_input_len]; 2];
            let input = SequentialSliceOfVecs::new(&waves_in, 2, max_input_len).unwrap();
            let mut waves_out = vec![vec![0.0 as $fty; max_output_len]; 2];
            let mut output =
                SequentialSliceOfVecs::new_mut(&mut waves_out, 2, max_output_len).unwrap();
            let mut total_in = 0;
            let mut total_out = 0;
            for _ in 0..$repetitions {
                let out = $resampler
                    .process_into_buffer(&input, &mut output, None)
                    .unwrap();
                total_in += out.0;
                total_out += out.1
            }
            let measured_ratio = total_out as f64 / total_in as f64;
            assert!(measured_ratio > 0.999 * $ratio);
            assert!(measured_ratio < 1.001 * $ratio);
        };
    }
    /*
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
    */
}
