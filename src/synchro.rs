use crate::sinc::make_sincs;
use crate::windows::WindowFunction;
use num::integer;
use num::traits::Zero;
use num::Complex;
use std::error;

type Res<T> = Result<T, Box<dyn error::Error>>;

use crate::Resampler;
use crate::ResamplerError;
use realfft::{ComplexToReal, RealToComplex};

/// A helper for resampling a single chunk of data.
struct FftResampler<T> {
    fft_size_in: usize,
    fft_size_out: usize,
    filter_f: Vec<Complex<T>>,
    fft: RealToComplex<T>,
    ifft: ComplexToReal<T>,
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
    overlaps: Vec<Vec<T>>,
    resampler: FftResampler<T>,
}

macro_rules! impl_resampler {
    ($ft:ty, $rt:ty) => {
        impl $rt {
            //
            pub fn new(fft_size_in: usize, fft_size_out: usize) -> Self {
                // calculate antialiasing cutoff
                let cutoff = if fft_size_in > fft_size_out {
                    0.4f32.powf(16.0 / fft_size_in as f32) * fft_size_out as f32
                        / fft_size_in as f32
                } else {
                    0.4f32.powf(16.0 / fft_size_in as f32)
                };
                debug!(
                    "Create new FftResampler, fft_size_in: {}, fft_size_out: {}, cutoff: {}",
                    fft_size_in, fft_size_out, cutoff
                );
                let sinc =
                    make_sincs::<$ft>(fft_size_in, 1, cutoff, WindowFunction::BlackmanHarris2);
                let mut filter_t: Vec<$ft> = vec![0.0; 2 * fft_size_in];
                let mut filter_f: Vec<Complex<$ft>> = vec![Complex::zero(); fft_size_in + 1];
                for n in 0..fft_size_in {
                    filter_t[n] = sinc[0][n] / (fft_size_in as $ft);
                }

                let input_f: Vec<Complex<$ft>> = vec![Complex::zero(); fft_size_in + 1];
                let input_buf: Vec<$ft> = vec![0.0; 2 * fft_size_in];
                let output_f: Vec<Complex<$ft>> = vec![Complex::zero(); fft_size_out + 1];
                let output_buf: Vec<$ft> = vec![0.0; 2 * fft_size_out];
                let mut fft = RealToComplex::<$ft>::new(2 * fft_size_in).unwrap();
                let ifft = ComplexToReal::<$ft>::new(2 * fft_size_out).unwrap();
                fft.process(&mut filter_t, &mut filter_f).unwrap();

                FftResampler {
                    fft_size_in,
                    fft_size_out,
                    filter_f,
                    fft,
                    ifft,
                    input_buf,
                    input_f,
                    output_f,
                    output_buf,
                }
            }

            /// Resample a small chunk
            fn resample_unit(
                &mut self,
                wave_in: &[$ft],
                wave_out: &mut [$ft],
                overlap: &mut [$ft],
            ) {
                // Copy to inut buffer and convert to complex
                for (n, item) in wave_in.iter().enumerate().take(self.fft_size_in) {
                    self.input_buf[n] = *item;
                    self.input_buf[n+self.fft_size_in] = 0.0;
                }

                // FFT and store result in history, update index
                self.fft
                    .process(&mut self.input_buf, &mut self.input_f)
                    .unwrap();

                // multiply with filter FT
                for n in 0..(self.fft_size_in + 1) {
                    self.input_f[n] *= self.filter_f[n];
                }

                let new_len = if self.fft_size_in < self.fft_size_out {
                    self.fft_size_in
                } else {
                    self.fft_size_out
                };

                // copy to modified spectrum
                for n in 0..new_len {
                    self.output_f[n] = self.input_f[n];
                }
                self.output_f[self.fft_size_out] = self.input_f[self.fft_size_in];

                // IFFT result, store result and overlap
                self.ifft
                    .process(&self.output_f, &mut self.output_buf)
                    .unwrap();
                for (n, item) in wave_out.iter_mut().enumerate().take(self.fft_size_out) {
                    *item = self.output_buf[n] + overlap[n];
                    overlap[n] = self.output_buf[n + self.fft_size_out];
                }
            }
        }
    };
}
impl_resampler!(f32, FftResampler<f32>);
impl_resampler!(f64, FftResampler<f64>);

macro_rules! impl_fixedinout {
    ($ft:ty) => {
        impl FftFixedInOut<$ft> {
            /// Create a new FftFixedInOut
            ///
            /// Parameters are:
            /// - `fs_in`: Input sample rate.
            /// - `fs_out`: Output sample rate.
            /// - `chunk_size_in`: desired length of input data in frames, actual value may be different.
            /// - `nbr_channels`: number of channels in input/output.
            pub fn new(
                fs_in: usize,
                fs_out: usize,
                chunk_size_in: usize,
                nbr_channels: usize,
            ) -> Self {
                debug!(
                    "Create new FftFixedInOut, fs_in: {}, fs_out: {} chunk_size_in: {}, channels: {}",
                    fs_in, fs_out, chunk_size_in, nbr_channels
                );

                let gcd = integer::gcd(fs_in, fs_out);
                let min_chunk_out = fs_out/gcd;
                let wanted = chunk_size_in;
                let fft_chunks = (wanted as f32 / min_chunk_out as f32).ceil() as usize;
                let fft_size_out = fft_chunks * fs_out / gcd;
                let fft_size_in = fft_chunks * fs_in / gcd;

                let resampler = FftResampler::<$ft>::new(fft_size_in, fft_size_out);

                let overlaps: Vec<Vec<$ft>> = vec![vec![0.0; fft_size_out]; nbr_channels];

                FftFixedInOut {
                    nbr_channels,
                    chunk_size_in: fft_size_in,
                    chunk_size_out: fft_size_out,
                    fft_size_in,
                    overlaps,
                    resampler,
                }
            }
        }
    }
}
impl_fixedinout!(f64);
impl_fixedinout!(f32);

macro_rules! resampler_FftFixedinout {
    ($t:ty) => {
        impl Resampler<$t> for FftFixedInOut<$t> {
            /// Query for the number of frames needed for the next call to "process".
            fn nbr_frames_needed(&self) -> usize {
                self.fft_size_in
            }

            /// Update the resample ratio. This is not supported by this resampler and always returns an error.
            fn set_resample_ratio(&mut self, _new_ratio: f64) -> Res<()> {
                Err(Box::new(ResamplerError::new(
                    "Not possible to adjust a synchronous resampler)",
                )))
            }

            /// Update the resample ratio relative to the original one.
            /// This is not supported by this resampler and always returns an error.
            fn set_resample_ratio_relative(&mut self, _rel_ratio: f64) -> Res<()> {
                Err(Box::new(ResamplerError::new(
                    "Not possible to adjust a synchronous resampler)",
                )))
            }

            /// Resample a chunk of audio. The input and output lengths are fixed.
            /// # Errors
            ///
            /// The function returns an error if the size of the input data is not equal
            /// to the number of channels and input size defined when creating the instance.
            fn process(&mut self, wave_in: &[Vec<$t>]) -> Res<Vec<Vec<$t>>> {
                if wave_in.len() != self.nbr_channels {
                    return Err(Box::new(ResamplerError::new(
                        "Wrong number of channels in input",
                    )));
                }
                if wave_in[0].len() != self.chunk_size_in {
                    return Err(Box::new(ResamplerError::new(
                        format!(
                            "Wrong number of frames in input, expected {}, got {}",
                            self.chunk_size_in,
                            wave_in[0].len()
                        )
                        .as_str(),
                    )));
                }
                let mut wave_out = vec![vec![0.0 as $t; self.chunk_size_out]; self.nbr_channels];
                for n in 0..self.nbr_channels {
                    self.resampler.resample_unit(
                        &wave_in[n],
                        &mut wave_out[n],
                        &mut self.overlaps[n],
                    )
                }
                Ok(wave_out)
            }
        }
    };
}
resampler_FftFixedinout!(f32);
resampler_FftFixedinout!(f64);

macro_rules! impl_fixedout {
    ($ft:ty) => {
        impl FftFixedOut<$ft> {
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
                let min_chunk_out = fs_out/gcd;
                let wanted_subsize = chunk_size_out/sub_chunks;
                let fft_chunks = (wanted_subsize as f32 / min_chunk_out as f32).ceil() as usize;
                let fft_size_out = fft_chunks * fs_out / gcd;
                let fft_size_in = fft_chunks * fs_in / gcd;

                let resampler = FftResampler::<$ft>::new(fft_size_in, fft_size_out);

                debug!(
                    "Create new FftFixedOut, fs_in: {}, fs_out: {} chunk_size_in: {}, channels: {}, fft_size_in: {}, fft_size_out: {}",
                    fs_in, fs_out, chunk_size_out, nbr_channels, fft_size_in, fft_size_out
                );

                let overlaps: Vec<Vec<$ft>> = vec![vec![0.0; fft_size_out]; nbr_channels];
                let output_buffers: Vec<Vec<$ft>> = vec![vec![0.0; chunk_size_out+fft_size_out]; nbr_channels];

                let saved_frames = 0;
                let chunks_needed = (chunk_size_out as f32 / fft_size_out as f32).ceil() as usize;
                let frames_needed = chunks_needed*fft_size_in;

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
    }
}
impl_fixedout!(f64);
impl_fixedout!(f32);

macro_rules! resampler_FftFixedout {
    ($t:ty) => {
        impl Resampler<$t> for FftFixedOut<$t> {
            /// Query for the number of frames needed for the next call to "process".
            fn nbr_frames_needed(&self) -> usize {
                self.frames_needed
            }

            /// Update the resample ratio. This is not supported by this resampler and always returns an error.
            fn set_resample_ratio(&mut self, _new_ratio: f64) -> Res<()> {
                Err(Box::new(ResamplerError::new(
                    "Not possible to adjust a synchronous resampler)",
                )))
            }

            /// Update the resample ratio relative to the original one.
            /// This is not supported by this resampler and always returns an error.
            fn set_resample_ratio_relative(&mut self, _rel_ratio: f64) -> Res<()> {
                Err(Box::new(ResamplerError::new(
                    "Not possible to adjust a synchronous resampler)",
                )))
            }

            /// Resample a chunk of audio. The required input length is provided by
            /// the "nbr_frames_needed" function, and the output length is fixed.
            /// # Errors
            ///
            /// The function returns an error if the length of the input data is not
            /// equal to the number of channels defined when creating the instance,
            /// and the number of audio frames given by "nbr_frames_needed".
            fn process(&mut self, wave_in: &[Vec<$t>]) -> Res<Vec<Vec<$t>>> {
                if wave_in.len() != self.nbr_channels {
                    return Err(Box::new(ResamplerError::new(
                        "Wrong number of channels in input",
                    )));
                }
                if wave_in[0].len() != self.frames_needed {
                    return Err(Box::new(ResamplerError::new(
                        format!(
                            "Wrong number of frames in input, expected {}, got {}",
                            self.frames_needed,
                            wave_in[0].len()
                        )
                        .as_str(),
                    )));
                }

                let mut wave_out = self.output_buffers.clone();
                let mut processed_samples = self.saved_frames * self.nbr_channels;
                for n in 0..self.nbr_channels {
                    for (in_chunk, out_chunk) in wave_in[n]
                        .chunks(self.fft_size_in)
                        .zip(wave_out[n][self.saved_frames..].chunks_mut(self.fft_size_out))
                    {
                        self.resampler
                            .resample_unit(in_chunk, out_chunk, &mut self.overlaps[n]);
                        processed_samples += self.fft_size_out;
                    }
                }
                let processed_frames = processed_samples / self.nbr_channels;

                // save extra frames for next round
                self.saved_frames = processed_frames - self.chunk_size_out;
                if processed_frames > self.chunk_size_out {
                    for n in 0..self.nbr_channels {
                        for (extra, saved) in wave_out[n]
                            .iter()
                            .skip(self.chunk_size_out)
                            .take(self.saved_frames)
                            .zip(self.output_buffers[n].iter_mut().take(self.saved_frames))
                        {
                            *saved = *extra;
                        }
                    }
                }
                for n in 0..self.nbr_channels {
                    wave_out[n].truncate(self.chunk_size_out);
                }
                //calculate number of needed frames from next round
                let frames_needed_out = self.chunk_size_out - self.saved_frames;
                let chunks_needed =
                    (frames_needed_out as f32 / self.fft_size_out as f32).ceil() as usize;
                self.frames_needed = chunks_needed * self.fft_size_in;
                Ok(wave_out)
            }
        }
    };
}
resampler_FftFixedout!(f32);
resampler_FftFixedout!(f64);

macro_rules! impl_fixedin {
    ($ft:ty) => {
        impl FftFixedIn<$ft> {
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
                let min_chunk_in = fs_in/gcd;
                let wanted_subsize = chunk_size_in/sub_chunks;
                let fft_chunks = (wanted_subsize as f32 / min_chunk_in as f32).ceil() as usize;
                let fft_size_out = fft_chunks * fs_out / gcd;
                let fft_size_in = fft_chunks * fs_in / gcd;

                let resampler = FftResampler::<$ft>::new(fft_size_in, fft_size_out);
                debug!(
                    "Create new FftFixedOut, fs_in: {}, fs_out: {} chunk_size_in: {}, channels: {}, fft_size_in: {}, fft_size_out: {}",
                    fs_in, fs_out, chunk_size_in, nbr_channels, fft_size_in, fft_size_out
                );

                let overlaps: Vec<Vec<$ft>> = vec![vec![0.0; fft_size_out]; nbr_channels];
                let input_buffers: Vec<Vec<$ft>> = vec![vec![0.0; chunk_size_in+fft_size_out]; nbr_channels];

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
    }
}
impl_fixedin!(f64);
impl_fixedin!(f32);

macro_rules! resampler_FftFixedin {
    ($t:ty) => {
        impl Resampler<$t> for FftFixedIn<$t> {
            /// Query for the number of frames needed for the next call to "process".
            fn nbr_frames_needed(&self) -> usize {
                self.chunk_size_in
            }

            /// Update the resample ratio. This is not supported by this resampler and always returns an error.
            fn set_resample_ratio(&mut self, _new_ratio: f64) -> Res<()> {
                Err(Box::new(ResamplerError::new(
                    "Not possible to adjust a synchronous resampler)",
                )))
            }

            /// Update the resample ratio relative to the original one.
            /// This is not supported by this resampler and always returns an error.
            fn set_resample_ratio_relative(&mut self, _rel_ratio: f64) -> Res<()> {
                Err(Box::new(ResamplerError::new(
                    "Not possible to adjust a synchronous resampler)",
                )))
            }

            /// Resample a chunk of audio. The required input length is provided by
            /// the "nbr_frames_needed" function, and the output length is fixed.
            /// # Errors
            ///
            /// The function returns an error if the length of the input data is not
            /// equal to the number of channels defined when creating the instance,
            /// and the number of audio frames given by "nbr_frames_needed".
            fn process(&mut self, wave_in: &[Vec<$t>]) -> Res<Vec<Vec<$t>>> {
                if wave_in.len() != self.nbr_channels {
                    return Err(Box::new(ResamplerError::new(
                        "Wrong number of channels in input",
                    )));
                }
                if wave_in[0].len() != self.chunk_size_in {
                    return Err(Box::new(ResamplerError::new(
                        format!(
                            "Wrong number of frames in input, expected {}, got {}",
                            self.chunk_size_in,
                            wave_in[0].len()
                        )
                        .as_str(),
                    )));
                }

                // copy new samples to input buffer
                let mut input_temp =
                    vec![vec![0.0; self.saved_frames + self.chunk_size_in]; self.nbr_channels];
                for n in 0..self.nbr_channels {
                    for (input, buffer) in self.input_buffers[n]
                        .iter()
                        .take(self.saved_frames)
                        .zip(input_temp[n].iter_mut())
                    {
                        *buffer = *input;
                    }
                }
                for n in 0..self.nbr_channels {
                    for (input, buffer) in wave_in[n].iter().zip(
                        input_temp[n]
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
                let mut wave_out =
                    vec![vec![0.0; nbr_chunks_ready * self.fft_size_out]; self.nbr_channels];
                for n in 0..self.nbr_channels {
                    for (in_chunk, out_chunk) in input_temp[n]
                        .chunks(self.fft_size_in)
                        .take(nbr_chunks_ready)
                        .zip(wave_out[n].chunks_mut(self.fft_size_out))
                    {
                        self.resampler
                            .resample_unit(in_chunk, out_chunk, &mut self.overlaps[n]);
                    }
                }

                // save extra frames for next round
                let frames_in_used = nbr_chunks_ready * self.fft_size_in;
                let extra = self.saved_frames - frames_in_used;

                if self.saved_frames > frames_in_used {
                    for n in 0..self.nbr_channels {
                        for (input, buffer) in input_temp[n]
                            .iter()
                            .skip(frames_in_used)
                            .take(extra)
                            .zip(self.input_buffers[n].iter_mut())
                        {
                            *buffer = *input;
                        }
                    }
                }
                self.saved_frames = extra;
                Ok(wave_out)
            }
        }
    };
}
resampler_FftFixedin!(f32);
resampler_FftFixedin!(f64);

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
    fn make_resampler_fi() {
        let mut resampler = FftFixedIn::<f64>::new(44100, 48000, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 1024);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 640);
    }
}
