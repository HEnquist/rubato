use num_traits::Zero;
use std::marker::PhantomData;

use crate::error::{ResampleError, ResampleResult};
use crate::{Resampler, Sample};

fn expand_by_factor<T: Sample>(indata: &[T], outdata: &mut [T], factor: usize) {
    for (invalue, outvalues) in indata.iter().zip(outdata.chunks_exact_mut(factor)) {
        outvalues[0] = *invalue;
        for outval in outvalues.iter_mut().skip(1) {
            *outval = T::zero();
        }
    }
}

/// A synchronous resampler that needs a fixed number of audio frames for input
/// and returns a variable number of frames.
///
/// The resampling is done by FFT:ing the input data. The spectrum is then extended or
/// truncated as well as multiplied with an antialiasing filter
/// before it's inverse transformed to get the resampled waveforms.
pub struct ExpandFixedIn<T> {
    nbr_channels: usize,
    chunk_size_in: usize,
    factor: usize,
    input_buffers: Vec<Vec<T>>,
    saved_frames: usize,
}

/// A synchronous resampler that needs a varying number of audio frames for input
/// and returns a fixed number of frames.
///
/// The resampling is done by FFT:ing the input data. The spectrum is then extended or
/// truncated as well as multiplied with an antialiasing filter
/// before it's inverse transformed to get the resampled waveforms.
pub struct ExpandFixedOut<T> {
    nbr_channels: usize,
    chunk_size_out: usize,
    factor: usize,
    output_buffers: Vec<Vec<T>>,
    saved_frames: usize,
    frames_needed: usize,
}

/// A synchronous resampler that accepts a fixed number of audio frames for input
/// and returns a fixed number of frames.
///
/// The resampling is done by FFT:ing the input data. The spectrum is then extended or
/// truncated as well as multiplied with an antialiasing filter
/// before it's inverse transformed to get the resampled waveforms.
pub struct ExpandFixedInOut<T> {
    nbr_channels: usize,
    chunk_size_in: usize,
    factor: usize,
    _phantom: PhantomData<T>,
}

impl<T> ExpandFixedInOut<T>
where
    T: Sample,
{
    /// Create a new FftFixedInOut
    ///
    /// Parameters are:
    /// - `fs_in`: Input sample rate.
    /// - `fs_out`: Output sample rate.
    /// - `chunk_size_in`: desired length of input data in frames, actual value may be different.
    /// - `nbr_channels`: number of channels in input/output.
    pub fn new(factor: usize, chunk_size_in: usize, nbr_channels: usize) -> Self {
        debug!(
            "Create new ExpandFixedInOut, factor: {}, chunk_size_in: {}, channels: {}",
            factor, chunk_size_in, nbr_channels
        );

        ExpandFixedInOut {
            nbr_channels,
            chunk_size_in,
            factor,
            _phantom: PhantomData,
        }
    }
}

impl<T> Resampler<T> for ExpandFixedInOut<T>
where
    T: Sample,
{
    /// Query for the number of frames needed for the next call to "process".
    fn nbr_frames_needed(&self) -> usize {
        self.chunk_size_in
    }

    /// Resample a chunk of audio. The input and output lengths are fixed.
    /// If the waveform for a channel is empty, this channel will be ignored and produce a
    /// corresponding empty output waveform.
    /// # Errors
    ///
    /// The function returns an error if the size of the input data is not equal
    /// to the number of channels and input size defined when creating the instance.
    fn process<V: AsRef<[T]>>(&mut self, wave_in: &[V]) -> ResampleResult<Vec<Vec<T>>> {
        if wave_in.len() != self.nbr_channels {
            return Err(ResampleError::WrongNumberOfChannels {
                expected: self.nbr_channels,
                actual: wave_in.len(),
            });
        }
        let mut used_channels = Vec::new();
        for (chan, wave) in wave_in.iter().enumerate() {
            let wave = wave.as_ref();
            if !wave.is_empty() {
                used_channels.push(chan);
                if wave.len() != self.chunk_size_in {
                    return Err(ResampleError::WrongNumberOfFrames {
                        channel: chan,
                        expected: self.chunk_size_in,
                        actual: wave.len(),
                    });
                }
            }
        }
        let mut wave_out = vec![Vec::new(); self.nbr_channels];
        for chan in used_channels.iter() {
            wave_out[*chan] = vec![T::zero(); self.chunk_size_in*self.factor];
        }

        for n in used_channels.iter() {
            expand_by_factor(wave_in[*n].as_ref(), &mut wave_out[*n], self.factor);
        }
        Ok(wave_out)
    }

    /// Update the resample ratio. This is not supported by this resampler and
    /// always returns an error.
    fn set_resample_ratio(&mut self, _new_ratio: f64) -> ResampleResult<()> {
        Err(ResampleError::SyncNotAdjustable)
    }

    /// Update the resample ratio relative to the original one. This is not
    /// supported by this resampler and always returns an error.
    fn set_resample_ratio_relative(&mut self, _rel_ratio: f64) -> ResampleResult<()> {
        Err(ResampleError::SyncNotAdjustable)
    }
}

/*
impl<T> FftFixedOut<T>
where
    T: Sample,
{
    /// Create a new FftFixedOut
    ///
    /// Parameters are:
    /// - `fs_in`: Input sample rate.
    /// - `fs_out`: Output sample rate.
    /// - `chunk_size_out`: length of output data in frames.
    /// - `sub_chunks`: desired number of subchunks for processing, actual number may be different.
    /// - `nbr_channels`: number of channels in input/output.
    pub fn new(
        fs_in: usize,
        fs_out: usize,
        chunk_size_out: usize,
        sub_chunks: usize,
        nbr_channels: usize,
    ) -> Self {
        let gcd = integer::gcd(fs_in, fs_out);
        let min_chunk_out = fs_out / gcd;
        let wanted_subsize = chunk_size_out / sub_chunks;
        let fft_chunks = (wanted_subsize as f32 / min_chunk_out as f32).ceil() as usize;
        let fft_size_out = fft_chunks * fs_out / gcd;
        let fft_size_in = fft_chunks * fs_in / gcd;

        let resampler = FftResampler::<T>::new(fft_size_in, fft_size_out);

        debug!(
            "Create new FftFixedOut, fs_in: {}, fs_out: {} chunk_size_in: {}, channels: {}, fft_size_in: {}, fft_size_out: {}",
            fs_in, fs_out, chunk_size_out, nbr_channels, fft_size_in, fft_size_out
        );

        let overlaps: Vec<Vec<T>> = vec![vec![T::zero(); fft_size_out]; nbr_channels];
        let output_buffers: Vec<Vec<T>> =
            vec![vec![T::zero(); chunk_size_out + fft_size_out]; nbr_channels];

        let saved_frames = 0;
        let chunks_needed = (chunk_size_out as f32 / fft_size_out as f32).ceil() as usize;
        let frames_needed = chunks_needed * fft_size_in;

        FftFixedOut {
            nbr_channels,
            chunk_size_out,
            fft_size_in,
            fft_size_out,
            overlaps,
            output_buffers,
            saved_frames,
            frames_needed,
            resampler,
        }
    }
}

impl<T> Resampler<T> for FftFixedOut<T>
where
    T: Sample,
{
    /// Query for the number of frames needed for the next call to "process".
    fn nbr_frames_needed(&self) -> usize {
        self.frames_needed
    }

    /// Resample a chunk of audio. The required input length is provided by
    /// the "nbr_frames_needed" function, and the output length is fixed.
    /// If the waveform for a channel is empty, this channel will be ignored and produce a
    /// corresponding empty output waveform.
    /// # Errors
    ///
    /// The function returns an error if the length of the input data is not
    /// equal to the number of channels defined when creating the instance,
    /// and the number of audio frames given by "nbr_frames_needed".
    fn process<V: AsRef<[T]>>(&mut self, wave_in: &[V]) -> ResampleResult<Vec<Vec<T>>> {
        if wave_in.len() != self.nbr_channels {
            return Err(ResampleError::WrongNumberOfChannels {
                expected: self.nbr_channels,
                actual: wave_in.len(),
            });
        }
        let mut used_channels = Vec::new();
        for (chan, wave) in wave_in.iter().enumerate() {
            let wave = wave.as_ref();
            if !wave.is_empty() {
                used_channels.push(chan);
                if wave.len() != self.frames_needed {
                    return Err(ResampleError::WrongNumberOfFrames {
                        channel: chan,
                        expected: self.frames_needed,
                        actual: wave.len(),
                    });
                }
            }
        }

        let mut wave_out = vec![Vec::new(); self.nbr_channels];
        for chan in used_channels.iter() {
            wave_out[*chan] = self.output_buffers[*chan].clone();
        }

        for n in used_channels.iter() {
            for (in_chunk, out_chunk) in wave_in[*n]
                .as_ref()
                .chunks(self.fft_size_in)
                .zip(wave_out[*n][self.saved_frames..].chunks_mut(self.fft_size_out))
            {
                self.resampler
                    .resample_unit(in_chunk, out_chunk, &mut self.overlaps[*n]);
            }
        }
        let processed_frames =
            self.saved_frames + self.fft_size_out * (self.frames_needed / self.fft_size_in);

        // save extra frames for next round
        self.saved_frames = processed_frames - self.chunk_size_out;
        if processed_frames > self.chunk_size_out {
            for n in used_channels.iter() {
                self.output_buffers[*n][0..self.saved_frames].copy_from_slice(
                    &wave_out[*n][self.chunk_size_out..(self.chunk_size_out + self.saved_frames)],
                );
            }
        }
        for n in used_channels.iter() {
            wave_out[*n].truncate(self.chunk_size_out);
        }
        //calculate number of needed frames from next round
        let frames_needed_out = if self.chunk_size_out > self.saved_frames {
            self.chunk_size_out - self.saved_frames
        } else {
            0
        };
        let chunks_needed = (frames_needed_out as f32 / self.fft_size_out as f32).ceil() as usize;
        self.frames_needed = chunks_needed * self.fft_size_in;
        Ok(wave_out)
    }

    /// Update the resample ratio. This is not supported by this resampler and
    /// always returns an error.
    fn set_resample_ratio(&mut self, _new_ratio: f64) -> ResampleResult<()> {
        Err(ResampleError::SyncNotAdjustable)
    }

    /// Update the resample ratio relative to the original one. This is not
    /// supported by this resampler and always returns an error.
    fn set_resample_ratio_relative(&mut self, _rel_ratio: f64) -> ResampleResult<()> {
        Err(ResampleError::SyncNotAdjustable)
    }
}

impl<T> FftFixedIn<T>
where
    T: Sample,
{
    /// Create a new FftFixedOut
    ///
    /// Parameters are:
    /// - `fs_in`: Input sample rate.
    /// - `fs_out`: Output sample rate.
    /// - `chunk_size_out`: length of output data in frames.
    /// - `sub_chunks`: desired number of subchunks for processing, actual number used may be different.
    /// - `nbr_channels`: number of channels in input/output.
    pub fn new(
        fs_in: usize,
        fs_out: usize,
        chunk_size_in: usize,
        sub_chunks: usize,
        nbr_channels: usize,
    ) -> Self {
        let gcd = integer::gcd(fs_in, fs_out);
        let min_chunk_in = fs_in / gcd;
        let wanted_subsize = chunk_size_in / sub_chunks;
        let fft_chunks = (wanted_subsize as f32 / min_chunk_in as f32).ceil() as usize;
        let fft_size_out = fft_chunks * fs_out / gcd;
        let fft_size_in = fft_chunks * fs_in / gcd;

        let resampler = FftResampler::<T>::new(fft_size_in, fft_size_out);
        debug!(
            "Create new FftFixedOut, fs_in: {}, fs_out: {} chunk_size_in: {}, channels: {}, fft_size_in: {}, fft_size_out: {}",
            fs_in, fs_out, chunk_size_in, nbr_channels, fft_size_in, fft_size_out
        );

        let overlaps: Vec<Vec<T>> = vec![vec![T::zero(); fft_size_out]; nbr_channels];
        let input_buffers: Vec<Vec<T>> =
            vec![vec![T::zero(); chunk_size_in + fft_size_out]; nbr_channels];

        let saved_frames = 0;

        FftFixedIn {
            nbr_channels,
            chunk_size_in,
            fft_size_in,
            fft_size_out,
            overlaps,
            input_buffers,
            saved_frames,
            resampler,
        }
    }
}

impl<T> Resampler<T> for FftFixedIn<T>
where
    T: Sample,
{
    /// Query for the number of frames needed for the next call to "process".
    fn nbr_frames_needed(&self) -> usize {
        self.chunk_size_in
    }

    /// Resample a chunk of audio. The required input length is provided by
    /// the "nbr_frames_needed" function, and the output length is fixed.
    /// If the waveform for a channel is empty, this channel will be ignored and produce a
    /// corresponding empty output waveform.
    /// # Errors
    ///
    /// The function returns an error if the length of the input data is not
    /// equal to the number of channels defined when creating the instance,
    /// and the number of audio frames given by "nbr_frames_needed".
    fn process<V: AsRef<[T]>>(&mut self, wave_in: &[V]) -> ResampleResult<Vec<Vec<T>>> {
        if wave_in.len() != self.nbr_channels {
            return Err(ResampleError::WrongNumberOfChannels {
                expected: self.nbr_channels,
                actual: wave_in.len(),
            });
        }
        let mut used_channels = Vec::new();
        for (chan, wave) in wave_in.iter().enumerate() {
            let wave = wave.as_ref();
            if !wave.is_empty() {
                used_channels.push(chan);
                if wave.len() != self.chunk_size_in {
                    return Err(ResampleError::WrongNumberOfFrames {
                        channel: chan,
                        expected: self.chunk_size_in,
                        actual: wave.len(),
                    });
                }
            }
        }

        let mut input_temp = vec![Vec::new(); self.nbr_channels];
        for chan in used_channels.iter() {
            input_temp[*chan] = vec![T::zero(); self.saved_frames + self.chunk_size_in];
        }

        // copy new samples to input buffer
        for n in used_channels.iter() {
            for (input, buffer) in self.input_buffers[*n]
                .iter()
                .take(self.saved_frames)
                .zip(input_temp[*n].iter_mut())
            {
                *buffer = *input;
            }
        }
        for n in used_channels.iter() {
            for (input, buffer) in wave_in[*n].as_ref().iter().zip(
                input_temp[*n]
                    .iter_mut()
                    .skip(self.saved_frames)
                    .take(self.chunk_size_in),
            ) {
                *buffer = *input;
            }
        }
        self.saved_frames += self.chunk_size_in;

        let nbr_chunks_ready =
            (self.saved_frames as f32 / self.fft_size_in as f32).floor() as usize;
        let mut wave_out = vec![Vec::new(); self.nbr_channels];
        for chan in used_channels.iter() {
            wave_out[*chan] = vec![T::zero(); nbr_chunks_ready * self.fft_size_out];
        }
        for n in used_channels.iter() {
            for (in_chunk, out_chunk) in input_temp[*n]
                .chunks(self.fft_size_in)
                .take(nbr_chunks_ready)
                .zip(wave_out[*n].chunks_mut(self.fft_size_out))
            {
                self.resampler
                    .resample_unit(in_chunk, out_chunk, &mut self.overlaps[*n]);
            }
        }

        // save extra frames for next round
        let frames_in_used = nbr_chunks_ready * self.fft_size_in;
        let extra = self.saved_frames - frames_in_used;

        if self.saved_frames > frames_in_used {
            for n in used_channels.iter() {
                for (input, buffer) in input_temp[*n]
                    .iter()
                    .skip(frames_in_used)
                    .take(extra)
                    .zip(self.input_buffers[*n].iter_mut())
                {
                    *buffer = *input;
                }
            }
        }
        self.saved_frames = extra;
        Ok(wave_out)
    }

    /// Update the resample ratio. This is not supported by this resampler and
    /// always returns an error.
    fn set_resample_ratio(&mut self, _new_ratio: f64) -> ResampleResult<()> {
        Err(ResampleError::SyncNotAdjustable)
    }

    /// Update the resample ratio relative to the original one. This is not
    /// supported by this resampler and always returns an error.
    fn set_resample_ratio_relative(&mut self, _rel_ratio: f64) -> ResampleResult<()> {
        Err(ResampleError::SyncNotAdjustable)
    }
}
*/

#[cfg(test)]
mod tests {
    use crate::synchro::{FftFixedIn, FftFixedInOut, FftFixedOut, FftResampler};
    use crate::Resampler;

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
        let mut resampler = FftFixedInOut::<f64>::new(44100, 48000, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1120);
    }

    #[test]
    fn make_resampler_fio_skipped() {
        // asking for 1024 give the nearest which is 1029 -> 1120
        let mut resampler = FftFixedInOut::<f64>::new(44100, 48000, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        let waves = vec![vec![0.0f64; frames], Vec::new()];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1120);
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fo() {
        let mut resampler = FftFixedOut::<f64>::new(44100, 192000, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 294);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn make_resampler_fo_skipped() {
        let mut resampler = FftFixedOut::<f64>::new(44100, 192000, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 294);
        let waves = vec![vec![0.0f64; frames], Vec::new()];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fo_empty() {
        let mut resampler = FftFixedOut::<f64>::new(44100, 192000, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 294);
        let waves = vec![Vec::new(); 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].is_empty());
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fi() {
        let mut resampler = FftFixedIn::<f64>::new(44100, 48000, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 1024);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 640);
    }

    #[test]
    fn make_resampler_fi_downsample() {
        let mut resampler = FftFixedIn::<f64>::new(48000, 16000, 1200, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 1200);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 400);
    }

    #[test]
    fn make_resampler_fi_skipped() {
        let mut resampler = FftFixedIn::<f64>::new(44100, 48000, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 1024);
        let waves = vec![vec![0.0f64; frames], Vec::new()];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 640);
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fi_empty() {
        let mut resampler = FftFixedIn::<f64>::new(44100, 48000, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 1024);
        let waves = vec![Vec::new(); 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].is_empty());
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fio_unusualratio() {
        // asking for 1024 give the nearest which is 1029 -> 1120
        let mut resampler = FftFixedInOut::<f64>::new(44100, 44110, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 4411);
    }

    #[test]
    fn make_resampler_fo_unusualratio() {
        let mut resampler = FftFixedOut::<f64>::new(44100, 44110, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 4410);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }
}
