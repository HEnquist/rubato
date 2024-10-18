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

    /// Convenience method for processing longer audio clips.
    /// This method repeatedly calls [process_into_buffer](Resampler::process_into_buffer)
    /// until all frames of the input buffer have been processed.
    /// The processed frames are written to the output buffer,
    /// with the initial silence (caused by the resampler delay) trimmed off.
    ///
    /// Use [process_all_needed_output_len](Resampler::process_all_needed_output_len)
    /// to get the minimal length of the output buffer required
    /// to resample a clip of a given length.
    ///
    /// The `active_channels_mask` parameter has the same meaning as in
    /// [process_into_buffer](Resampler::process_into_buffer).
    fn process_all_into_buffer<'a>(
        &mut self,
        buffer_in: &dyn Adapter<'a, T>,
        buffer_out: &mut dyn AdapterMut<'a, T>,
        input_len: usize,
        active_channels_mask: Option<&[bool]>,
    ) -> ResampleResult<(usize, usize)> {
        let expected_output_len = (self.resample_ratio() * input_len as f64).ceil() as usize;

        let mut indexing = Indexing {
            input_offset: 0,
            output_offset: 0,
            active_channels_mask: active_channels_mask.map(|m| m.to_vec()),
            partial_len: None,
        };

        let mut frames_left = input_len;
        let mut output_len = 0;
        let mut frames_to_trim = self.output_delay();
        debug!(
            "resamping {} input frames to {} output frames, delay to trim off {} frames",
            input_len, expected_output_len, frames_to_trim
        );

        let next_nbr_input_frames = self.input_frames_next();
        while frames_left > next_nbr_input_frames {
            debug!("process, {} input frames left", frames_left);
            let (nbr_in, nbr_out) =
                self.process_into_buffer(buffer_in, buffer_out, Some(&indexing))?;
            frames_left -= nbr_in;
            output_len += nbr_out;
            indexing.input_offset += nbr_in;
            indexing.output_offset += nbr_out;
            if frames_to_trim > 0 && output_len > frames_to_trim {
                debug!(
                    "output, {} is longer  than delay to trim, {}, trimming..",
                    output_len, frames_to_trim
                );
                // move useful putput data to start of output buffer
                for chan in 0..self.nbr_channels() {
                    if let Some(mask) = active_channels_mask {
                        if !mask[chan] {
                            continue;
                        }
                    }
                    for frame in 0..(output_len - frames_to_trim) {
                        let val = buffer_out
                            .read_sample(chan, frame + frames_to_trim)
                            .unwrap();
                        buffer_out.write_sample(chan, frame, &val);
                    }
                }
                // update counters
                output_len -= frames_to_trim;
                indexing.output_offset -= frames_to_trim;
                frames_to_trim = 0;
            }
        }
        if frames_left > 0 {
            debug!("process the last partial chunk, len {}", frames_left);
            indexing.partial_len = Some(frames_left);
            let (_nbr_in, nbr_out) =
                self.process_into_buffer(buffer_in, buffer_out, Some(&indexing))?;
            output_len += nbr_out;
            indexing.output_offset += nbr_out;
        }
        indexing.partial_len = Some(0);
        while output_len < expected_output_len {
            debug!(
                "output is still too short, {} < {}, pump zeros..",
                output_len, expected_output_len
            );
            let (_nbr_in, nbr_out) =
                self.process_into_buffer(buffer_in, buffer_out, Some(&indexing))?;
            output_len += nbr_out;
            indexing.output_offset += nbr_out;
        }
        Ok((input_len, expected_output_len))
    }

    /// Calculate the minimal length of the output buffer
    /// needed to process a clip of length `input_len` using the
    /// [process_all_into_buffer](Resampler::process_all_into_buffer) method.
    fn process_all_needed_output_len(&mut self, input_len: usize) -> usize {
        let delay_frames = self.output_delay();
        let output_frames_next = self.output_frames_next();
        let expected_output_len = (self.resample_ratio() * input_len as f64).ceil() as usize;
        delay_frames + output_frames_next + expected_output_len
    }

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

#[cfg(test)]
pub mod tests {
    #[cfg(feature = "fft_resampler")]
    use crate::Fft;
    use crate::Resampler;
    use crate::{
        Async, FixedAsync, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };
    use audioadapter::direct::SequentialSliceOfVecs;
    use audioadapter::Adapter;
    use test_log::test;

    #[test]
    fn process_all() {
        let mut resampler = Async::<f64>::new_sinc(
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
        .unwrap();
        let input_len = 12345;
        let samples: Vec<f64> = (0..input_len).map(|v| v as f64 / 10.0).collect();
        let input_data = vec![samples; 2];
        // add a ramp to the input
        let input = SequentialSliceOfVecs::new(&input_data, 2, input_len).unwrap();
        let output_len = resampler.process_all_needed_output_len(input_len);
        let mut output_data = vec![vec![0.0f64; output_len]; 2];
        let mut output = SequentialSliceOfVecs::new_mut(&mut output_data, 2, output_len).unwrap();
        let (nbr_in, nbr_out) = resampler
            .process_all_into_buffer(&input, &mut output, input_len, None)
            .unwrap();
        assert_eq!(nbr_in, input_len);
        // This is a simple ratio, output should be twice as long as input
        assert_eq!(2 * nbr_in, nbr_out);

        // check that the output follows the input ramp, within suitable margins
        let increment = 0.1 / resampler.resample_ratio();
        let delay = resampler.output_delay();
        let margin = (delay as f64 * resampler.resample_ratio()) as usize;
        let mut expected = margin as f64 * increment;
        for frame in margin..(nbr_out - margin) {
            for chan in 0..2 {
                let val = output.read_sample(chan, frame).unwrap();
                assert!(
                    val - expected < 100.0 * increment,
                    "frame: {}, value: {}, expected: {}",
                    frame,
                    val,
                    expected
                );
                assert!(
                    expected - val < 100.0 * increment,
                    "frame: {}, value: {}, expected: {}",
                    frame,
                    val,
                    expected
                );
            }
            expected += increment;
        }
    }

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
}
