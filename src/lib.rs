#![doc = include_str!("../README.md")]

#[cfg(feature = "log")]
extern crate log;
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

#[derive(Debug)]
pub struct Indexing {
    pub input_offset: usize,
    pub output_offset: usize,
    pub partial_len: Option<usize>,
    pub active_channels_mask: Option<Vec<bool>>,
}

pub(crate) fn get_offsets(indexing: &Option<&Indexing>) -> (usize, usize) {
    indexing.as_ref().map(|idx| (idx.input_offset, idx.output_offset)).unwrap_or((0, 0))
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
///
/// This trait is not object safe. If you need an object safe resampler,
/// use the [VecResampler] wrapper trait.
pub trait Resampler<T>: Send
where
    T: Sample,
{
    /*
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
    */
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
    fn process_into_buffer<'a>(
        &mut self,
        buffer_in: &dyn Adapter<'a, T>,
        buffer_out: &mut dyn AdapterMut<'a, T>,
        indexing: Option<&Indexing>,
    ) -> ResampleResult<(usize, usize)>;

    /*
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
    */

    /// Get the maximum possible number of input frames per channel the resampler could require.
    fn input_frames_max(&self) -> usize;

    /// Get the number of frames per channel needed for the next call to
    /// [process_into_buffer](Resampler::process_into_buffer) or [process](Resampler::process).
    fn input_frames_next(&self) -> usize;

    /// Get the number of channels this Resampler is configured for.
    fn nbr_channels(&self) -> usize;

    /*
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

    */
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
    //use crate::Fft;
    use crate::{buffer_capacity, buffer_length, make_buffer, resize_buffer, Resampler};
    use crate::{Async, FixedAsync, SincInterpolationParameters, SincInterpolationType, WindowFunction}; //, PolynomialDegree};
    use test_log::test;
    use audioadapter::direct::SequentialSliceOfVecs;

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

    fn process_with_boxed<'a>(resampler: &mut Box<dyn Resampler<f64>>, input: &SequentialSliceOfVecs<&'a [Vec<f64>]>, output: &mut SequentialSliceOfVecs<&'a mut [Vec<f64>]>) {
        resampler.process_into_buffer(input, output, None).unwrap();
    }

    fn impl_send<T: Send>() {
        fn is_send<T: Send>() {}
        is_send::<Async<T>>();
        //#[cfg(feature = "fft_resampler")]
        //{
        //    is_send::<Fft<T>>();
        //}
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
            let mut prev_last = -0.1;
            let max_input_len = $resampler.input_frames_max();
            let max_output_len = $resampler.output_frames_max();
            for n in 0..50 {
                let frames = $resampler.input_frames_next();
                let frames_out = $resampler.output_frames_next();
                // Check that lengths are within the reported max values
                assert!(frames <= max_input_len);
                assert!(frames_out <= max_output_len);
                let mut waves = vec![vec![0.0f64; frames]; 2];
                for m in 0..frames {
                    for ch in 0..2 {
                        waves[ch][m] = val;
                    }
                    val = val + 0.1;
                }
                let input = SequentialSliceOfVecs::new(&waves, 2, frames).unwrap();
                let mut waves_out = vec![vec![0.0f64; frames_out]; 2];
                let mut output = SequentialSliceOfVecs::new_mut(&mut waves_out, 2, frames_out).unwrap();
        
                let (_input_frames, output_frames) = $resampler.process_into_buffer(&input, &mut output, None).unwrap();
            
                for ch in 0..2 {
                    assert!(
                        waves_out[ch][0] > prev_last,
                        "Iteration {}, first value {} prev last value {}",
                        n,
                        waves_out[ch][0],
                        prev_last
                    );
                    let expected_diff = frames as f64 * 0.1;
                    let diff = waves_out[ch][output_frames - 1] - waves_out[ch][0];
                    assert!(
                        diff < 1.5 * expected_diff && diff > 0.25 * expected_diff,
                        "Iteration {}, last value {} first value {}, diff {}, expected {}",
                        n,
                        waves_out[ch][output_frames - 1],
                        waves_out[ch][0],
                        diff,
                        expected_diff,
                    );
                }
                prev_last = waves_out[0][output_frames - 1];
                for m in 0..output_frames - 1 {
                    for ch in 0..2 {
                        let diff = waves_out[ch][m + 1] - waves_out[ch][m];
                        assert!(
                            diff < 0.5 && diff > -0.05,
                            "Frame {}:{} next value {} value {}",
                            n,
                            m,
                            waves_out[ch][m + 1],
                            waves_out[ch][m]
                        );
                    }
                }
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
            let mut output = SequentialSliceOfVecs::new_mut(&mut waves_out, 2, max_output_len).unwrap();
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
