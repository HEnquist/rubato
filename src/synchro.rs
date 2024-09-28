use crate::error::ResamplerConstructionError;
use crate::sinc::make_sincs;
use crate::windows::WindowFunction;
use num_complex::Complex;
use num_integer as integer;
use num_traits::Zero;
use std::sync::Arc;

use crate::error::{ResampleError, ResampleResult};
use crate::{calculate_cutoff, update_mask_from_buffers, validate_buffers, Resampler, Sample};
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// A helper for resampling a single chunk of data.
struct FftResampler<T> {
    fft_size_in: usize,
    fft_size_out: usize,
    filter_f: Vec<Complex<T>>,
    fft: Arc<dyn RealToComplex<T>>,
    ifft: Arc<dyn ComplexToReal<T>>,
    scratch_fw: Vec<Complex<T>>,
    scratch_inv: Vec<Complex<T>>,
    input_buf: Vec<T>,
    input_f: Vec<Complex<T>>,
    output_f: Vec<Complex<T>>,
    output_buf: Vec<T>,
}

/// A synchronous resampler that needs a fixed number of audio frames for input
/// and returns a variable number of frames.
///
/// The resampling is done by FFT:ing the input data. The spectrum is then extended or
/// truncated as well as multiplied with an antialiasing filter
/// before it's inverse transformed to get the resampled waveforms.
pub struct FftFixedIn<T> {
    nbr_channels: usize,
    chunk_size_in: usize,
    fft_size_in: usize,
    fft_size_out: usize,
    overlaps: Vec<Vec<T>>,
    input_buffers: Vec<Vec<T>>,
    channel_mask: Vec<bool>,
    saved_frames: usize,
    resampler: FftResampler<T>,
}

/// A synchronous resampler that needs a varying number of audio frames for input
/// and returns a fixed number of frames.
///
/// The resampling is done by FFT:ing the input data. The spectrum is then extended or
/// truncated as well as multiplied with an antialiasing filter
/// before it's inverse transformed to get the resampled waveforms.
pub struct FftFixedOut<T> {
    nbr_channels: usize,
    chunk_size_out: usize,
    fft_size_in: usize,
    fft_size_out: usize,
    overlaps: Vec<Vec<T>>,
    output_buffers: Vec<Vec<T>>,
    channel_mask: Vec<bool>,
    saved_frames: usize,
    frames_needed: usize,
    resampler: FftResampler<T>,
}

/// A synchronous resampler that accepts a fixed number of audio frames for input
/// and returns a fixed number of frames.
///
/// The resampling is done by FFT:ing the input data. The spectrum is then extended or
/// truncated as well as multiplied with an antialiasing filter
/// before it's inverse transformed to get the resampled waveforms.
pub struct FftFixedInOut<T> {
    nbr_channels: usize,
    chunk_size_in: usize,
    chunk_size_out: usize,
    fft_size_in: usize,
    channel_mask: Vec<bool>,
    overlaps: Vec<Vec<T>>,
    resampler: FftResampler<T>,
}

fn validate_sample_rates(input: usize, output: usize) -> Result<(), ResamplerConstructionError> {
    if input == 0 || output == 0 {
        return Err(ResamplerConstructionError::InvalidSampleRate { input, output });
    }
    Ok(())
}

impl<T> FftResampler<T>
where
    T: Sample,
{
    //
    pub fn new(fft_size_in: usize, fft_size_out: usize) -> Self {
        // calculate antialiasing cutoff
        let cutoff = if fft_size_in > fft_size_out {
            calculate_cutoff::<f32>(fft_size_out, WindowFunction::BlackmanHarris2)
                * fft_size_out as f32
                / fft_size_in as f32
        } else {
            calculate_cutoff::<f32>(fft_size_in, WindowFunction::BlackmanHarris2)
        };
        debug!(
            "Create new FftResampler, fft_size_in: {}, fft_size_out: {}, cutoff: {}",
            fft_size_in, fft_size_out, cutoff
        );
        let sinc = make_sincs::<T>(fft_size_in, 1, cutoff, WindowFunction::BlackmanHarris2);
        let mut filter_t: Vec<T> = vec![T::zero(); 2 * fft_size_in];
        let mut filter_f: Vec<Complex<T>> = vec![Complex::zero(); fft_size_in + 1];
        for (n, f) in filter_t.iter_mut().enumerate().take(fft_size_in) {
            *f = sinc[0][n] / T::coerce(2 * fft_size_in);
        }

        let input_f: Vec<Complex<T>> = vec![Complex::zero(); fft_size_in + 1];
        let input_buf: Vec<T> = vec![T::zero(); 2 * fft_size_in];
        let output_f: Vec<Complex<T>> = vec![Complex::zero(); fft_size_out + 1];
        let output_buf: Vec<T> = vec![T::zero(); 2 * fft_size_out];
        let mut planner = RealFftPlanner::<T>::new();
        let fft = planner.plan_fft_forward(2 * fft_size_in);
        let ifft = planner.plan_fft_inverse(2 * fft_size_out);
        fft.process(&mut filter_t, &mut filter_f).unwrap();
        let scratch_fw = fft.make_scratch_vec();
        let scratch_inv = ifft.make_scratch_vec();

        FftResampler {
            fft_size_in,
            fft_size_out,
            filter_f,
            fft,
            ifft,
            scratch_fw,
            scratch_inv,
            input_buf,
            input_f,
            output_f,
            output_buf,
        }
    }

    /// Resample a small chunk.
    fn resample_unit(&mut self, wave_in: &[T], wave_out: &mut [T], overlap: &mut [T]) {
        // Copy to input buffer and clear padding area.
        self.input_buf[0..self.fft_size_in].copy_from_slice(wave_in);
        for item in self
            .input_buf
            .iter_mut()
            .skip(self.fft_size_in)
            .take(self.fft_size_in)
        {
            *item = T::zero();
        }

        // FFT and store result in history, update index.
        self.fft
            .process_with_scratch(&mut self.input_buf, &mut self.input_f, &mut self.scratch_fw)
            .unwrap();

        let new_len = if self.fft_size_in < self.fft_size_out {
            self.fft_size_in + 1
        } else {
            self.fft_size_out
        };

        // Multiply with filter FT.
        self.input_f
            .iter_mut()
            .take(new_len)
            .zip(self.filter_f.iter())
            .for_each(|(spec, filt)| *spec *= filt);

        // copy to modified spectrum
        self.output_f[0..new_len].copy_from_slice(&self.input_f[0..new_len]);
        for val in self.output_f[new_len..].iter_mut() {
            *val = Complex::zero();
        }
        // IFFT result, store result and overlap.
        self.ifft
            .process_with_scratch(
                &mut self.output_f,
                &mut self.output_buf,
                &mut self.scratch_inv,
            )
            .unwrap();
        for (n, item) in wave_out.iter_mut().enumerate().take(self.fft_size_out) {
            *item = self.output_buf[n] + overlap[n];
        }
        overlap.copy_from_slice(&self.output_buf[self.fft_size_out..]);
    }
}

impl<T> FftFixedInOut<T>
where
    T: Sample,
{
    /// Create a new FftFixedInOut.
    ///
    /// Parameters are:
    /// - `sample_rate_input`: Input sample rate, must be > 0.
    /// - `sample_rate_output`: Output sample rate, must be > 0.
    /// - `chunk_size_in`: desired length of input data in frames, actual value may be different.
    /// - `nbr_channels`: number of channels in input/output.
    pub fn new(
        sample_rate_input: usize,
        sample_rate_output: usize,
        chunk_size_in: usize,
        nbr_channels: usize,
    ) -> Result<Self, ResamplerConstructionError> {
        validate_sample_rates(sample_rate_input, sample_rate_output)?;

        debug!(
            "Create new FftFixedInOut, sample_rate_input: {}, sample_rate_output: {} chunk_size_in: {}, channels: {}",
            sample_rate_input, sample_rate_output, chunk_size_in, nbr_channels
        );

        let gcd = integer::gcd(sample_rate_input, sample_rate_output);
        let min_chunk_in = sample_rate_input / gcd;
        let fft_chunks = (chunk_size_in as f32 / min_chunk_in as f32).ceil() as usize;
        let fft_size_out = fft_chunks * sample_rate_output / gcd;
        let fft_size_in = fft_chunks * sample_rate_input / gcd;

        let resampler = FftResampler::<T>::new(fft_size_in, fft_size_out);

        let overlaps: Vec<Vec<T>> = vec![vec![T::zero(); fft_size_out]; nbr_channels];

        let channel_mask = vec![true; nbr_channels];

        Ok(FftFixedInOut {
            nbr_channels,
            chunk_size_in: fft_size_in,
            chunk_size_out: fft_size_out,
            fft_size_in,
            overlaps,
            resampler,
            channel_mask,
        })
    }
}

impl<T> Resampler<T> for FftFixedInOut<T>
where
    T: Sample,
{
    fn process_into_buffer<Vin: AsRef<[T]>, Vout: AsMut<[T]>>(
        &mut self,
        wave_in: &[Vin],
        wave_out: &mut [Vout],
        active_channels_mask: Option<&[bool]>,
    ) -> ResampleResult<(usize, usize)> {
        if let Some(mask) = active_channels_mask {
            self.channel_mask.copy_from_slice(mask);
        } else {
            update_mask_from_buffers(&mut self.channel_mask);
        };

        validate_buffers(
            wave_in,
            wave_out,
            &self.channel_mask,
            self.nbr_channels,
            self.chunk_size_in,
            self.chunk_size_out,
        )?;

        for (channel, active) in self.channel_mask.iter().enumerate() {
            if *active {
                self.resampler.resample_unit(
                    &wave_in[channel].as_ref()[..self.chunk_size_in],
                    &mut wave_out[channel].as_mut()[..self.chunk_size_out],
                    &mut self.overlaps[channel],
                )
            }
        }
        Ok((self.chunk_size_in, self.chunk_size_out))
    }

    fn input_frames_max(&self) -> usize {
        self.fft_size_in
    }

    fn input_frames_next(&self) -> usize {
        self.fft_size_in
    }

    fn nbr_channels(&self) -> usize {
        self.nbr_channels
    }

    fn output_frames_max(&self) -> usize {
        self.chunk_size_out
    }

    fn output_frames_next(&self) -> usize {
        self.output_frames_max()
    }

    fn output_delay(&self) -> usize {
        self.chunk_size_out / 2
    }

    /// Update the resample ratio. This is not supported by this resampler and
    /// always returns an [ResampleError::SyncNotAdjustable].
    fn set_resample_ratio(&mut self, _new_ratio: f64, _ramp: bool) -> ResampleResult<()> {
        Err(ResampleError::SyncNotAdjustable)
    }

    /// Update the resample ratio relative to the original one. This is not
    /// supported by this resampler and always returns an [ResampleError::SyncNotAdjustable].
    fn set_resample_ratio_relative(&mut self, _rel_ratio: f64, _ramp: bool) -> ResampleResult<()> {
        Err(ResampleError::SyncNotAdjustable)
    }

    fn reset(&mut self) {
        self.overlaps
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = T::zero()));
        self.channel_mask.iter_mut().for_each(|val| *val = true);
    }
}

impl<T> FftFixedOut<T>
where
    T: Sample,
{
    /// Create a new FftFixedOut.
    ///
    /// Parameters are:
    /// - `sample_rate_input`: Input sample rate, must be > 0.
    /// - `sample_rate_output`: Output sample rate, must be > 0.
    /// - `chunk_size_out`: length of output data in frames.
    /// - `sub_chunks`: desired number of subchunks for processing, actual number may be different.
    /// - `nbr_channels`: number of channels in input/output.
    pub fn new(
        sample_rate_input: usize,
        sample_rate_output: usize,
        chunk_size_out: usize,
        sub_chunks: usize,
        nbr_channels: usize,
    ) -> Result<Self, ResamplerConstructionError> {
        validate_sample_rates(sample_rate_input, sample_rate_output)?;

        let gcd = integer::gcd(sample_rate_input, sample_rate_output);
        let min_chunk_out = sample_rate_output / gcd;
        let wanted_subsize = chunk_size_out / sub_chunks;
        let fft_chunks = (wanted_subsize as f32 / min_chunk_out as f32).ceil() as usize;
        let fft_size_out = fft_chunks * sample_rate_output / gcd;
        let fft_size_in = fft_chunks * sample_rate_input / gcd;

        let resampler = FftResampler::<T>::new(fft_size_in, fft_size_out);

        debug!(
            "Create new FftFixedOut, sample_rate_input: {}, sample_rate_output: {} chunk_size_in: {}, channels: {}, fft_size_in: {}, fft_size_out: {}",
            sample_rate_input, sample_rate_output, chunk_size_out, nbr_channels, fft_size_in, fft_size_out
        );

        let overlaps: Vec<Vec<T>> = vec![vec![T::zero(); fft_size_out]; nbr_channels];
        let output_buffers: Vec<Vec<T>> =
            vec![vec![T::zero(); chunk_size_out + fft_size_out]; nbr_channels];

        let channel_mask = vec![true; nbr_channels];

        let saved_frames = 0;
        let chunks_needed = (chunk_size_out as f32 / fft_size_out as f32).ceil() as usize;
        let frames_needed = chunks_needed * fft_size_in;

        Ok(FftFixedOut {
            nbr_channels,
            chunk_size_out,
            fft_size_in,
            fft_size_out,
            overlaps,
            output_buffers,
            saved_frames,
            frames_needed,
            resampler,
            channel_mask,
        })
    }
}

impl<T> Resampler<T> for FftFixedOut<T>
where
    T: Sample,
{
    fn process_into_buffer<Vin: AsRef<[T]>, Vout: AsMut<[T]>>(
        &mut self,
        wave_in: &[Vin],
        wave_out: &mut [Vout],
        active_channels_mask: Option<&[bool]>,
    ) -> ResampleResult<(usize, usize)> {
        if let Some(mask) = active_channels_mask {
            self.channel_mask.copy_from_slice(mask);
        } else {
            update_mask_from_buffers(&mut self.channel_mask);
        };

        validate_buffers(
            wave_in,
            wave_out,
            &self.channel_mask,
            self.nbr_channels,
            self.frames_needed,
            self.chunk_size_out,
        )?;

        for (chan, active) in self.channel_mask.iter().enumerate() {
            if *active {
                debug_assert!(self.chunk_size_out <= wave_out[chan].as_mut().len());
                for (in_chunk, out_chunk) in wave_in[chan].as_ref()[..self.frames_needed]
                    .chunks(self.fft_size_in)
                    .zip(
                        self.output_buffers[chan][self.saved_frames..]
                            .chunks_mut(self.fft_size_out),
                    )
                {
                    self.resampler
                        .resample_unit(in_chunk, out_chunk, &mut self.overlaps[chan]);
                }
            }
        }
        let processed_frames =
            self.saved_frames + self.fft_size_out * (self.frames_needed / self.fft_size_in);

        // Copy to output, and save extra frames for next round.
        if processed_frames >= self.chunk_size_out {
            self.saved_frames = processed_frames - self.chunk_size_out;
            for (chan, active) in self.channel_mask.iter().enumerate() {
                if *active {
                    wave_out[chan].as_mut()[..self.chunk_size_out]
                        .copy_from_slice(&self.output_buffers[chan][..self.chunk_size_out]);
                    self.output_buffers[chan].copy_within(
                        self.chunk_size_out..(self.chunk_size_out + self.saved_frames),
                        0,
                    );
                }
            }
        } else {
            self.saved_frames = processed_frames;
        }
        // Calculate number of needed frames from next round.
        let frames_needed_out = if self.chunk_size_out > self.saved_frames {
            self.chunk_size_out - self.saved_frames
        } else {
            0
        };
        let input_frames_used = self.frames_needed;
        let chunks_needed = (frames_needed_out as f32 / self.fft_size_out as f32).ceil() as usize;
        self.frames_needed = chunks_needed * self.fft_size_in;
        Ok((input_frames_used, self.chunk_size_out))
    }

    fn input_frames_max(&self) -> usize {
        (self.chunk_size_out as f32 / self.fft_size_out as f32).ceil() as usize * self.fft_size_in
    }

    fn input_frames_next(&self) -> usize {
        self.frames_needed
    }

    fn nbr_channels(&self) -> usize {
        self.nbr_channels
    }

    fn output_frames_max(&self) -> usize {
        self.chunk_size_out
    }

    fn output_frames_next(&self) -> usize {
        self.output_frames_max()
    }

    fn output_delay(&self) -> usize {
        self.fft_size_out / 2
    }

    /// Update the resample ratio. This is not supported by this resampler and
    /// always returns [ResampleError::SyncNotAdjustable].
    fn set_resample_ratio(&mut self, _new_ratio: f64, _ramp: bool) -> ResampleResult<()> {
        Err(ResampleError::SyncNotAdjustable)
    }

    /// Update the resample ratio relative to the original one. This is not
    /// supported by this resampler and always returns [ResampleError::SyncNotAdjustable].
    fn set_resample_ratio_relative(&mut self, _rel_ratio: f64, _ramp: bool) -> ResampleResult<()> {
        Err(ResampleError::SyncNotAdjustable)
    }

    fn reset(&mut self) {
        self.overlaps
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = T::zero()));
        self.output_buffers
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = T::zero()));
        self.channel_mask.iter_mut().for_each(|val| *val = true);
        self.saved_frames = 0;
        let chunks_needed = (self.chunk_size_out as f32 / self.fft_size_out as f32).ceil() as usize;
        self.frames_needed = chunks_needed * self.fft_size_in;
    }
}

impl<T> FftFixedIn<T>
where
    T: Sample,
{
    /// Create a new FftFixedIn.
    ///
    /// Parameters are:
    /// - `sample_rate_input`: Input sample rate, must be > 0.
    /// - `sample_rate_output`: Output sample rate, must be > 0.
    /// - `chunk_size_in`: length of input data in frames.
    /// - `sub_chunks`: desired number of subchunks for processing, actual number used may be different.
    /// - `nbr_channels`: number of channels in input/output.
    pub fn new(
        sample_rate_input: usize,
        sample_rate_output: usize,
        chunk_size_in: usize,
        sub_chunks: usize,
        nbr_channels: usize,
    ) -> Result<Self, ResamplerConstructionError> {
        validate_sample_rates(sample_rate_input, sample_rate_output)?;

        let gcd = integer::gcd(sample_rate_input, sample_rate_output);
        let min_chunk_in = sample_rate_input / gcd;
        let wanted_subsize = chunk_size_in / sub_chunks;
        let fft_chunks = (wanted_subsize as f32 / min_chunk_in as f32).ceil() as usize;
        let fft_size_out = fft_chunks * sample_rate_output / gcd;
        let fft_size_in = fft_chunks * sample_rate_input / gcd;

        let resampler = FftResampler::<T>::new(fft_size_in, fft_size_out);
        debug!(
            "Create new FftFixedOut, sample_rate_input: {}, sample_rate_output: {} chunk_size_in: {}, channels: {}, fft_size_in: {}, fft_size_out: {}",
            sample_rate_input, sample_rate_output, chunk_size_in, nbr_channels, fft_size_in, fft_size_out
        );

        let overlaps: Vec<Vec<T>> = vec![vec![T::zero(); fft_size_out]; nbr_channels];
        let input_buffers: Vec<Vec<T>> =
            vec![vec![T::zero(); chunk_size_in + fft_size_in]; nbr_channels];

        let channel_mask = vec![true; nbr_channels];

        let saved_frames = 0;

        Ok(FftFixedIn {
            nbr_channels,
            chunk_size_in,
            fft_size_in,
            fft_size_out,
            overlaps,
            input_buffers,
            saved_frames,
            resampler,
            channel_mask,
        })
    }
}

impl<T> Resampler<T> for FftFixedIn<T>
where
    T: Sample,
{
    fn process_into_buffer<Vin: AsRef<[T]>, Vout: AsMut<[T]>>(
        &mut self,
        wave_in: &[Vin],
        wave_out: &mut [Vout],
        active_channels_mask: Option<&[bool]>,
    ) -> ResampleResult<(usize, usize)> {
        if let Some(mask) = active_channels_mask {
            self.channel_mask.copy_from_slice(mask);
        } else {
            update_mask_from_buffers(&mut self.channel_mask);
        };

        let next_saved_frames = self.saved_frames + self.chunk_size_in;
        let nbr_chunks_ready =
            (next_saved_frames as f32 / self.fft_size_in as f32).floor() as usize;
        let needed_len = nbr_chunks_ready * self.fft_size_out;

        validate_buffers(
            wave_in,
            wave_out,
            &self.channel_mask,
            self.nbr_channels,
            self.chunk_size_in,
            needed_len,
        )?;

        // Copy new samples to input buffer.
        for (chan, active) in self.channel_mask.iter().enumerate() {
            if *active {
                for (input, buffer) in wave_in[chan].as_ref().iter().zip(
                    self.input_buffers[chan]
                        .iter_mut()
                        .skip(self.saved_frames)
                        .take(self.chunk_size_in),
                ) {
                    *buffer = *input;
                }
            }
        }

        self.saved_frames = next_saved_frames;

        for (chan, active) in self.channel_mask.iter().enumerate() {
            if *active {
                debug_assert!(needed_len <= wave_out[chan].as_mut().len());
                for (in_chunk, out_chunk) in self.input_buffers[chan]
                    .chunks(self.fft_size_in)
                    .take(nbr_chunks_ready)
                    .zip(wave_out[chan].as_mut().chunks_mut(self.fft_size_out))
                {
                    self.resampler
                        .resample_unit(in_chunk, out_chunk, &mut self.overlaps[chan]);
                }
            }
        }

        // Save extra frames for next round.
        let frames_in_used = nbr_chunks_ready * self.fft_size_in;
        let extra = self.saved_frames - frames_in_used;

        if self.saved_frames > frames_in_used {
            for (chan, active) in self.channel_mask.iter().enumerate() {
                if *active {
                    self.input_buffers[chan].copy_within(frames_in_used..self.saved_frames, 0);
                }
            }
        }
        self.saved_frames = extra;
        Ok((self.chunk_size_in, needed_len))
    }

    fn input_frames_max(&self) -> usize {
        self.chunk_size_in
    }

    fn input_frames_next(&self) -> usize {
        self.chunk_size_in
    }

    fn nbr_channels(&self) -> usize {
        self.nbr_channels
    }

    fn output_frames_max(&self) -> usize {
        let max_stored_frames = self.fft_size_in - 1;
        let max_available_frames = max_stored_frames + self.chunk_size_in;
        let max_subchunks_to_process = max_available_frames / self.fft_size_in;
        max_subchunks_to_process * self.fft_size_out
    }

    fn output_frames_next(&self) -> usize {
        (((self.saved_frames + self.chunk_size_in) as f32) / self.fft_size_in as f32).floor()
            as usize
            * self.fft_size_out
    }

    fn output_delay(&self) -> usize {
        self.fft_size_out / 2
    }

    /// Update the resample ratio. This is not supported by this resampler and
    /// always returns [ResampleError::SyncNotAdjustable].
    fn set_resample_ratio(&mut self, _new_ratio: f64, _ramp: bool) -> ResampleResult<()> {
        Err(ResampleError::SyncNotAdjustable)
    }

    /// Update the resample ratio relative to the original one. This is not
    /// supported by this resampler and always returns [ResampleError::SyncNotAdjustable].
    fn set_resample_ratio_relative(&mut self, _rel_ratio: f64, _ramp: bool) -> ResampleResult<()> {
        Err(ResampleError::SyncNotAdjustable)
    }

    fn reset(&mut self) {
        self.overlaps
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = T::zero()));
        self.input_buffers
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = T::zero()));
        self.channel_mask.iter_mut().for_each(|val| *val = true);
        self.saved_frames = 0;
    }
}

#[cfg(test)]
mod tests {
    use crate::check_output;
    use crate::synchro::{FftFixedIn, FftFixedInOut, FftFixedOut, FftResampler};
    use crate::Resampler;
    use rand::Rng;
    use test_log::test;

    #[test]
    fn resample_unit() {
        let mut resampler = FftResampler::<f64>::new(147, 1000);
        let mut wave_in = vec![0.0; 147];

        wave_in[0] = 0.3;
        wave_in[1] = 0.7;
        wave_in[2] = 1.0;
        wave_in[3] = 1.0;
        wave_in[4] = 0.7;
        wave_in[5] = 0.3;

        let mut wave_out = vec![0.0; 1000];
        let mut overlap = vec![0.0; 1000];
        resampler.resample_unit(&wave_in, &mut wave_out, &mut overlap);
        let vecsum = wave_out.iter().sum::<f64>();
        let maxval = wave_out.iter().cloned().fold(0. / 0., f64::max);
        assert!((vecsum - 4.0 * 1000.0 / 147.0).abs() < 1.0e-6);
        assert!((maxval - 1.0).abs() < 0.1);
    }

    #[test]
    fn make_resampler_fio() {
        // asking for 1024 give the nearest which is 1029 -> 1120
        let mut resampler = FftFixedInOut::<f64>::new(44100, 48000, 1024, 2).unwrap();
        let frames = resampler.input_frames_next();
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1120);
    }

    #[test]
    fn reset_resampler_fio() {
        let mut resampler = FftFixedInOut::<f64>::new(44100, 48000, 1024, 2).unwrap();
        let frames = resampler.input_frames_next();

        let mut rng = rand::thread_rng();
        let mut waves = vec![vec![0.0f64; frames]; 2];
        waves
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = rng.gen()));
        let out1 = resampler.process(&waves, None).unwrap();
        resampler.reset();
        assert_eq!(
            frames,
            resampler.input_frames_next(),
            "Resampler requires different number of frames when new and after a reset."
        );
        let out2 = resampler.process(&waves, None).unwrap();
        assert_eq!(
            out1, out2,
            "Resampler gives different output when new and after a reset."
        );
    }

    #[test]
    fn make_resampler_fio_skipped() {
        // Asking for 1024 give the nearest which is 1029 -> 1120.
        let mut resampler = FftFixedInOut::<f64>::new(44100, 48000, 1024, 2).unwrap();
        let frames = resampler.input_frames_next();
        let waves = vec![vec![0.0f64; frames], Vec::new()];
        let mask = vec![true, false];
        let out = resampler.process(&waves, Some(&mask)).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1120);
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fo() {
        let mut resampler = FftFixedOut::<f64>::new(44100, 192000, 1024, 2, 2).unwrap();
        let frames = resampler.input_frames_next();
        assert_eq!(frames, 294);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn reset_resampler_fo() {
        let mut resampler = FftFixedOut::<f64>::new(44100, 192000, 1024, 2, 2).unwrap();
        let frames = resampler.input_frames_next();

        let mut rng = rand::thread_rng();
        let mut waves = vec![vec![0.0f64; frames]; 2];
        waves
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = rng.gen()));
        let out1 = resampler.process(&waves, None).unwrap();
        resampler.reset();
        assert_eq!(
            frames,
            resampler.input_frames_next(),
            "Resampler requires different number of frames when new and after a reset."
        );
        let out2 = resampler.process(&waves, None).unwrap();
        assert_eq!(
            out1, out2,
            "Resampler gives different output when new and after a reset."
        );
    }

    #[test]
    fn make_resampler_fo_skipped() {
        let mut resampler = FftFixedOut::<f64>::new(44100, 192000, 1024, 2, 2).unwrap();
        let frames = resampler.input_frames_next();
        assert_eq!(frames, 294);
        let waves = vec![vec![0.0f64; frames], Vec::new()];
        let mask = vec![true, false];
        let out = resampler.process(&waves, Some(&mask)).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fo_empty() {
        let mut resampler = FftFixedOut::<f64>::new(44100, 192000, 1024, 2, 2).unwrap();
        let frames = resampler.input_frames_next();
        assert_eq!(frames, 294);
        let waves = vec![Vec::new(); 2];
        let mask = vec![false; 2];
        let out = resampler.process(&waves, Some(&mask)).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].is_empty());
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fi() {
        let mut resampler = FftFixedIn::<f64>::new(44100, 48000, 1024, 2, 2).unwrap();
        let frames = resampler.input_frames_next();
        assert_eq!(frames, 1024);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 640);
    }

    #[test]
    fn reset_resampler_fi() {
        let mut resampler = FftFixedIn::<f64>::new(44100, 48000, 1024, 2, 2).unwrap();

        let mut rng = rand::thread_rng();
        let mut waves = vec![vec![0.0f64; 1024]; 2];
        waves
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = rng.gen()));
        let out1 = resampler.process(&waves, None).unwrap();
        resampler.reset();
        let out2 = resampler.process(&waves, None).unwrap();
        assert_eq!(
            out1, out2,
            "Resampler gives different output when new and after a reset."
        );
    }

    #[test]
    fn make_resampler_fi_noalloc() {
        let mut resampler = FftFixedIn::<f64>::new(44100, 48000, 1024, 2, 2).unwrap();
        let frames = resampler.input_frames_next();
        assert_eq!(frames, 1024);
        let waves = vec![vec![0.0f64; frames]; 2];
        let mut out = vec![vec![0.0f64; 2 * frames]; 2];
        let allocated_out_len = out[0].len();
        assert_eq!(allocated_out_len, out[1].len());
        let mask = vec![true; 2];
        let (consumed_in_len, processed_out_len) = resampler
            .process_into_buffer(&waves, &mut out, Some(&mask))
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(consumed_in_len, frames);
        assert_eq!(processed_out_len, 640);
        // The vectors are not truncated during processing.
        assert_eq!(allocated_out_len, out[0].len());
        assert_eq!(allocated_out_len, out[1].len());
    }

    #[test]
    fn make_resampler_fi_downsample() {
        let mut resampler = FftFixedIn::<f64>::new(48000, 16000, 1200, 2, 2).unwrap();
        let frames = resampler.input_frames_next();
        assert_eq!(frames, 1200);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 400);
    }

    #[test]
    fn make_resampler_fi_skipped() {
        let mut resampler = FftFixedIn::<f64>::new(44100, 48000, 1024, 2, 2).unwrap();
        let frames = resampler.input_frames_next();
        assert_eq!(frames, 1024);
        let waves = vec![vec![0.0f64; frames], Vec::new()];
        let mask = vec![true, false];
        let out = resampler.process(&waves, Some(&mask)).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 640);
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fi_empty() {
        let mut resampler = FftFixedIn::<f64>::new(44100, 48000, 1024, 2, 2).unwrap();
        let frames = resampler.input_frames_next();
        assert_eq!(frames, 1024);
        let waves = vec![Vec::new(); 2];
        let mask = vec![false; 2];
        let out = resampler.process(&waves, Some(&mask)).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].is_empty());
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fio_unusualratio() {
        // Asking for 1024 give the nearest which is 1029 -> 1120.
        let mut resampler = FftFixedInOut::<f64>::new(44100, 44110, 1024, 2).unwrap();
        let frames = resampler.input_frames_next();
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 4411);
    }

    #[test]
    fn make_resampler_fo_unusualratio() {
        let mut resampler = FftFixedOut::<f64>::new(44100, 44110, 1024, 2, 2).unwrap();
        let frames = resampler.input_frames_next();
        assert_eq!(frames, 4410);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn check_fo_output() {
        let mut resampler = FftFixedOut::<f64>::new(44100, 48000, 4096, 4, 2).unwrap();
        check_output!(resampler);
    }

    #[test]
    fn check_fi_output() {
        let mut resampler = FftFixedIn::<f64>::new(44100, 48000, 4096, 4, 2).unwrap();
        check_output!(resampler);
    }

    #[test]
    fn check_fio_output() {
        let mut resampler = FftFixedInOut::<f64>::new(44100, 48000, 4096, 2).unwrap();
        check_output!(resampler);
    }

    #[test]
    fn check_fi_max_output_length() {
        // parameters:
        // - rate in
        // - rate out
        // - requested chunksize
        // - requested number of subchunks
        // - expected fft input length
        // - expected fft output length
        let params_to_test = [
            // fft sizes < chunksize
            [44100, 48000, 1024, 4, 294, 320],
            [48000, 44100, 1024, 4, 320, 294],
            // fft sizes << chunksize
            [44000, 48000, 1024, 100, 11, 12],
            // fft sizes > chunksize
            [32728, 32000, 1024, 4, 4091, 4000],
            [32000, 32728, 1024, 4, 4000, 4091],
            // fft sizes >> chunksize
            [37199, 39119, 1024, 4, 37199, 39119],
            [39119, 37199, 1024, 4, 39119, 37199],
        ];
        for params in params_to_test {
            println!("params: {:?}", params);
            let [rate_in, rate_out, chunksize, subchunks, fft_in_len, fft_out_len] = params;
            let resampler =
                FftFixedIn::<f64>::new(rate_in, rate_out, chunksize, subchunks, 1).unwrap();
            assert_eq!(resampler.fft_size_in, fft_in_len);
            assert_eq!(resampler.fft_size_out, fft_out_len);
            let resampler_max_output_len = resampler.output_frames_max();
            println!(
                "Resampler reports max output length: {}",
                resampler_max_output_len
            );
            assert!(resampler.output_frames_max() >= fft_out_len);
            // expected length
            let max_stored_frames = fft_in_len - 1;
            let max_available_samples = max_stored_frames + chunksize;
            let max_subchunks_to_process = max_available_samples / fft_in_len;
            let expected_max_out_len = max_subchunks_to_process * fft_out_len;
            println!("Max stored frames: {}, max avail frames: {}, max ready subchunks: {}, expected max output len: {}", max_stored_frames, max_available_samples, max_subchunks_to_process, expected_max_out_len);
            assert_eq!(resampler.output_frames_max(), expected_max_out_len);
        }
    }

    #[test]
    fn check_fo_max_input_length() {
        // parameters:
        // - rate in
        // - rate out
        // - requested chunksize
        // - requested number of subchunks
        // - expected fft input length
        // - expected fft output length
        let params_to_test = [
            // fft sizes < chunksize
            [44100, 48000, 1024, 4, 294, 320],
            [48000, 44100, 1024, 4, 320, 294],
            // fft sizes << chunksize
            [44000, 48000, 1024, 100, 11, 12],
            // fft sizes > chunksize
            [32728, 32000, 1024, 4, 4091, 4000],
            [32000, 32728, 1024, 4, 4000, 4091],
            // fft sizes >> chunksize
            [37199, 39119, 1024, 4, 37199, 39119],
            [39119, 37199, 1024, 4, 39119, 37199],
        ];
        for params in params_to_test {
            println!("params: {:?}", params);
            let [rate_in, rate_out, chunksize, subchunks, fft_in_len, fft_out_len] = params;
            let resampler =
                FftFixedOut::<f64>::new(rate_in, rate_out, chunksize, subchunks, 1).unwrap();
            assert_eq!(resampler.fft_size_in, fft_in_len);
            assert_eq!(resampler.fft_size_out, fft_out_len);
            let resampler_max_input_len = resampler.input_frames_max();
            println!(
                "Resampler reports max input length: {}",
                resampler_max_input_len
            );
            assert!(resampler.input_frames_max() >= fft_in_len);
            // max needed is when we have none stored
            let max_frames_needed = chunksize;
            let max_subchunks_needed =
                (max_frames_needed as f32 / fft_out_len as f32).ceil() as usize;
            let expected_max_in_len = max_subchunks_needed * fft_in_len;
            println!(
                "Max frames needed: {}, max subchunks_needed: {}, expected max input len: {}",
                max_frames_needed, max_subchunks_needed, expected_max_in_len
            );
            assert_eq!(resampler.input_frames_max(), expected_max_in_len);
        }
    }
}
