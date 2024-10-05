use crate::error::ResamplerConstructionError;
use crate::sinc::make_sincs;
use crate::windows::WindowFunction;
use num_complex::Complex;
use num_integer as integer;
use num_traits::Zero;
use std::fmt;
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

/// An enum for specifying which side of the resampler should be fixed size.
#[derive(Debug, PartialEq)]
pub enum FftFixed {
    /// Input size is fixed, output size varies.
    Input,
    /// Output size is fixed, input size varies.
    Output,
    /// Both input and output size is fixed.
    Both,
}

/// A synchronous resampler that uses FFT.
///
/// The resampling is done by FFT:ing the input data. The spectrum is then extended or
/// truncated as well as multiplied with an antialiasing filter
/// before it's inverse transformed to get the resampled waveforms.
pub struct Fft<T> {
    nbr_channels: usize,
    chunk_size_in: usize,
    chunk_size_out: usize,
    fft_size_in: usize,
    fft_size_out: usize,
    overlaps: Vec<Vec<T>>,
    buffers: Vec<Vec<T>>,
    channel_mask: Vec<bool>,
    saved_frames: usize,
    resampler: FftResampler<T>,
    fixed: FftFixed,
}

impl<T> fmt::Debug for Fft<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Fast")
            .field("nbr_channels", &self.nbr_channels)
            .field("chunk_size_in,", &self.chunk_size_in)
            .field("chunk_size_out,", &self.chunk_size_out)
            .field("fft_size_in,", &self.fft_size_in)
            .field("fft_size_out,", &self.fft_size_out)
            .field("overlaps[0].len()", &self.overlaps[0].len())
            .field("buffer[0].len()", &self.buffers[0].len())
            .field("channel_mask", &self.channel_mask)
            .field("saved_frames", &self.saved_frames)
            .field("fixed", &self.fixed)
            .finish()
    }
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

impl<T> Fft<T>
where
    T: Sample,
{
    /// Create a new Fft.
    ///
    /// Parameters are:
    /// - `sample_rate_input`: Input sample rate, must be > 0.
    /// - `sample_rate_output`: Output sample rate, must be > 0.
    /// - `chunk_size`: desired chunk size in frames.
    /// - `sub_chunks`: desired number of subchunks for processing, actual number may be different.
    /// - `nbr_channels`: number of channels in input/output.
    pub fn new(
        sample_rate_input: usize,
        sample_rate_output: usize,
        chunk_size: usize,
        sub_chunks: usize,
        nbr_channels: usize,
        fixed: FftFixed,
    ) -> Result<Self, ResamplerConstructionError> {
        validate_sample_rates(sample_rate_input, sample_rate_output)?;

        let (fft_size_in, fft_size_out) = match fixed {
            FftFixed::Input => {
                let gcd = integer::gcd(sample_rate_input, sample_rate_output);
                let min_chunk_in = sample_rate_input / gcd;
                let wanted_subsize = chunk_size / sub_chunks;
                let fft_chunks = (wanted_subsize as f32 / min_chunk_in as f32).ceil() as usize;
                let size_out = fft_chunks * sample_rate_output / gcd;
                let size_in = fft_chunks * sample_rate_input / gcd;
                (size_in, size_out)
            }
            FftFixed::Output => {
                let gcd = integer::gcd(sample_rate_input, sample_rate_output);
                let min_chunk_out = sample_rate_output / gcd;
                let wanted_subsize = chunk_size / sub_chunks;
                let fft_chunks = (wanted_subsize as f32 / min_chunk_out as f32).ceil() as usize;
                let size_out = fft_chunks * sample_rate_output / gcd;
                let size_in = fft_chunks * sample_rate_input / gcd;
                (size_in, size_out)
            }
            FftFixed::Both => {
                let gcd = integer::gcd(sample_rate_input, sample_rate_output);
                let min_chunk_in = sample_rate_input / gcd;
                let fft_chunks = (chunk_size as f32 / min_chunk_in as f32).ceil() as usize;
                let size_out = fft_chunks * sample_rate_output / gcd;
                let size_in = fft_chunks * sample_rate_input / gcd;
                (size_in, size_out)
            }
        };

        let resampler = FftResampler::<T>::new(fft_size_in, fft_size_out);

        debug!(
            "Create new Fft with fixed {:?}, sample_rate_input: {}, sample_rate_output: {} chunk_size: {}, channels: {}, fft_size_in: {}, fft_size_out: {}",
            fixed, sample_rate_input, sample_rate_output, chunk_size, nbr_channels, fft_size_in, fft_size_out
        );

        let overlaps: Vec<Vec<T>> = vec![vec![T::zero(); fft_size_out]; nbr_channels];

        let needed_buffer_size = match fixed {
            FftFixed::Input => chunk_size + fft_size_in,
            FftFixed::Output => chunk_size + fft_size_out,
            FftFixed::Both => 0,
        };
        let buffers: Vec<Vec<T>> = vec![vec![T::zero(); needed_buffer_size]; nbr_channels];

        let channel_mask = vec![true; nbr_channels];

        let saved_frames = 0;

        let (chunk_size_in, chunk_size_out) =
            Self::calc_chunk_sizes(fft_size_in, fft_size_out, chunk_size, saved_frames, &fixed);

        Ok(Fft {
            nbr_channels,
            chunk_size_in,
            chunk_size_out,
            fft_size_in,
            fft_size_out,
            overlaps,
            buffers,
            saved_frames,
            resampler,
            channel_mask,
            fixed,
        })
    }

    fn calc_chunk_sizes(
        fft_size_in: usize,
        fft_size_out: usize,
        chunk_size: usize,
        saved_frames: usize,
        fixed: &FftFixed,
    ) -> (usize, usize) {
        match fixed {
            FftFixed::Input => {
                let subchunks_available: f32 =
                    ((chunk_size + saved_frames) as f32 / fft_size_in as f32).floor();
                let frames_available = (subchunks_available as usize) * fft_size_out;
                (chunk_size, frames_available)
            }
            FftFixed::Output => {
                let subchunks_needed = ((chunk_size as f32 - saved_frames as f32)
                    / fft_size_out as f32)
                    .ceil()
                    .max(0.0);
                let frames_needed = (subchunks_needed as usize) * fft_size_in;
                (frames_needed, chunk_size)
            }
            FftFixed::Both => {
                let subchunks_needed = (chunk_size as f32 / fft_size_in as f32).ceil() as usize;
                let frames_needed_in = subchunks_needed * fft_size_in;
                let frames_needed_out = subchunks_needed * fft_size_out;
                (frames_needed_in, frames_needed_out)
            }
        }
    }

    fn update_chunk_sizes(&mut self) {
        match self.fixed {
            FftFixed::Input => {
                (self.chunk_size_in, self.chunk_size_out) = Self::calc_chunk_sizes(
                    self.fft_size_in,
                    self.fft_size_out,
                    self.chunk_size_in,
                    self.saved_frames,
                    &self.fixed,
                )
            }
            FftFixed::Output => {
                (self.chunk_size_in, self.chunk_size_out) = Self::calc_chunk_sizes(
                    self.fft_size_in,
                    self.fft_size_out,
                    self.chunk_size_out,
                    self.saved_frames,
                    &self.fixed,
                )
            }
            FftFixed::Both => {}
        }
    }
}

impl<T> Resampler<T> for Fft<T>
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
        trace!("Start processing, {:?}", self);

        validate_buffers(
            wave_in,
            wave_out,
            &self.channel_mask,
            self.nbr_channels,
            self.chunk_size_in,
            self.chunk_size_out,
        )?;

        match self.fixed {
            FftFixed::Input => {
                // Fixed input. Buffer input in the internal buffer, and resample directly to output
                let available_input_frames = self.saved_frames + self.chunk_size_in;
                let nbr_chunks_ready =
                    (available_input_frames as f32 / self.fft_size_in as f32).floor() as usize;
                let input_frames_to_process = nbr_chunks_ready * self.fft_size_in;

                // Copy new samples to internal buffer.
                for (chan, active) in self.channel_mask.iter().enumerate() {
                    if *active {
                        for (input, buffer) in wave_in[chan].as_ref().iter().zip(
                            self.buffers[chan]
                                .iter_mut()
                                .skip(self.saved_frames)
                                .take(self.chunk_size_in),
                        ) {
                            *buffer = *input;
                        }
                    }
                }
                self.saved_frames = available_input_frames - input_frames_to_process;

                // Resample from internal buffer to output
                for (chan, active) in self.channel_mask.iter().enumerate() {
                    if *active {
                        debug_assert!(self.chunk_size_out <= wave_out[chan].as_mut().len());
                        for (in_chunk, out_chunk) in self.buffers[chan]
                            .chunks(self.fft_size_in)
                            .take(nbr_chunks_ready)
                            .zip(wave_out[chan].as_mut().chunks_mut(self.fft_size_out))
                        {
                            self.resampler.resample_unit(
                                in_chunk,
                                out_chunk,
                                &mut self.overlaps[chan],
                            );
                        }
                    }
                }

                // Copy saved frames to start of internal buffer
                for (chan, active) in self.channel_mask.iter().enumerate() {
                    if *active {
                        self.buffers[chan].copy_within(
                            input_frames_to_process..(input_frames_to_process + self.saved_frames),
                            0,
                        );
                    }
                }
            }

            FftFixed::Output => {
                // Fixed Output. Buffer output in the internal buffer, and resample directly from input

                // Resample from input to internal buffer
                for (chan, active) in self.channel_mask.iter().enumerate() {
                    if *active {
                        debug_assert!(self.chunk_size_out <= wave_out[chan].as_mut().len());
                        for (in_chunk, out_chunk) in wave_in[chan].as_ref()[..self.chunk_size_in]
                            .chunks(self.fft_size_in)
                            .zip(
                                self.buffers[chan][self.saved_frames..]
                                    .chunks_mut(self.fft_size_out),
                            )
                        {
                            self.resampler.resample_unit(
                                in_chunk,
                                out_chunk,
                                &mut self.overlaps[chan],
                            );
                        }
                    }
                }

                // Copy to output, and copy saved frames to start of internal buffer for next round.
                let available_output_frames =
                    self.saved_frames + self.fft_size_out * (self.chunk_size_in / self.fft_size_in);
                self.saved_frames = available_output_frames - self.chunk_size_out;
                for (chan, active) in self.channel_mask.iter().enumerate() {
                    if *active {
                        wave_out[chan].as_mut()[..self.chunk_size_out]
                            .copy_from_slice(&self.buffers[chan][..self.chunk_size_out]);
                        self.buffers[chan].copy_within(
                            self.chunk_size_out..(self.chunk_size_out + self.saved_frames),
                            0,
                        );
                    }
                }
            }

            FftFixed::Both => {
                // Fixed Input and Output. Resample directly from input to output.
                for (channel, active) in self.channel_mask.iter().enumerate() {
                    if *active {
                        self.resampler.resample_unit(
                            &wave_in[channel].as_ref()[..self.chunk_size_in],
                            &mut wave_out[channel].as_mut()[..self.chunk_size_out],
                            &mut self.overlaps[channel],
                        )
                    }
                }
            }
        };

        let input_size = self.chunk_size_in;
        let output_size = self.chunk_size_out;
        self.update_chunk_sizes();
        Ok((input_size, output_size))
    }

    fn input_frames_max(&self) -> usize {
        match self.fixed {
            FftFixed::Both | FftFixed::Input => self.chunk_size_in,
            FftFixed::Output => {
                (self.chunk_size_out as f32 / self.fft_size_out as f32).ceil() as usize
                    * self.fft_size_in
            }
        }
    }

    fn input_frames_next(&self) -> usize {
        self.chunk_size_in
    }

    fn nbr_channels(&self) -> usize {
        self.nbr_channels
    }

    fn output_frames_max(&self) -> usize {
        match self.fixed {
            FftFixed::Both | FftFixed::Output => self.chunk_size_out,
            FftFixed::Input => {
                let max_stored_frames = self.fft_size_in - 1;
                let max_available_frames = max_stored_frames + self.chunk_size_in;
                let max_subchunks_to_process = max_available_frames / self.fft_size_in;
                max_subchunks_to_process * self.fft_size_out
            }
        }
    }

    fn output_frames_next(&self) -> usize {
        self.chunk_size_out
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
        self.buffers
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = T::zero()));
        self.channel_mask.iter_mut().for_each(|val| *val = true);
        self.saved_frames = 0;
        self.update_chunk_sizes();
    }
}

#[cfg(test)]
mod tests {
    use crate::check_output;
    use crate::synchro::{Fft, FftFixed, FftResampler};
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
        let mut resampler = Fft::<f64>::new(44100, 48000, 1024, 1, 2, FftFixed::Both).unwrap();
        let frames = resampler.input_frames_next();
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1120);
    }

    #[test]
    fn reset_resampler_fio() {
        let mut resampler = Fft::<f64>::new(44100, 48000, 1024, 1, 2, FftFixed::Both).unwrap();
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
        let mut resampler = Fft::<f64>::new(44100, 48000, 1024, 1, 2, FftFixed::Both).unwrap();
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
        let mut resampler = Fft::<f64>::new(44100, 192000, 1024, 2, 2, FftFixed::Output).unwrap();
        let frames = resampler.input_frames_next();
        assert_eq!(frames, 294);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn reset_resampler_fo() {
        let mut resampler = Fft::<f64>::new(44100, 192000, 1024, 2, 2, FftFixed::Output).unwrap();
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
        let mut resampler = Fft::<f64>::new(44100, 192000, 1024, 2, 2, FftFixed::Output).unwrap();
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
        let mut resampler = Fft::<f64>::new(44100, 192000, 1024, 2, 2, FftFixed::Output).unwrap();
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
        let mut resampler = Fft::<f64>::new(44100, 48000, 1024, 2, 2, FftFixed::Input).unwrap();
        let frames = resampler.input_frames_next();
        assert_eq!(frames, 1024);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 640);
    }

    #[test]
    fn reset_resampler_fi() {
        let mut resampler = Fft::<f64>::new(44100, 48000, 1024, 2, 2, FftFixed::Input).unwrap();

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
        let mut resampler = Fft::<f64>::new(44100, 48000, 1024, 2, 2, FftFixed::Input).unwrap();
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
        let mut resampler = Fft::<f64>::new(48000, 16000, 1200, 2, 2, FftFixed::Input).unwrap();
        let frames = resampler.input_frames_next();
        assert_eq!(frames, 1200);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 400);
    }

    #[test]
    fn make_resampler_fi_skipped() {
        let mut resampler = Fft::<f64>::new(44100, 48000, 1024, 2, 2, FftFixed::Input).unwrap();
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
        let mut resampler = Fft::<f64>::new(44100, 48000, 1024, 2, 2, FftFixed::Input).unwrap();
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
        let mut resampler = Fft::<f64>::new(44100, 44110, 1024, 1, 2, FftFixed::Both).unwrap();
        let frames = resampler.input_frames_next();
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 4411);
    }

    #[test]
    fn make_resampler_fo_unusualratio() {
        let mut resampler = Fft::<f64>::new(44100, 44110, 1024, 2, 2, FftFixed::Output).unwrap();
        let frames = resampler.input_frames_next();
        assert_eq!(frames, 4410);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn check_fo_output() {
        let mut resampler = Fft::<f64>::new(44100, 48000, 4096, 4, 2, FftFixed::Output).unwrap();
        check_output!(resampler);
    }

    #[test]
    fn check_fi_output() {
        let mut resampler = Fft::<f64>::new(44100, 48000, 4096, 4, 2, FftFixed::Input).unwrap();
        check_output!(resampler);
    }

    #[test]
    fn check_fio_output() {
        let mut resampler = Fft::<f64>::new(44100, 48000, 4096, 1, 2, FftFixed::Both).unwrap();
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
                Fft::<f64>::new(rate_in, rate_out, chunksize, subchunks, 1, FftFixed::Input)
                    .unwrap();
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
                Fft::<f64>::new(rate_in, rate_out, chunksize, subchunks, 1, FftFixed::Output)
                    .unwrap();
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
