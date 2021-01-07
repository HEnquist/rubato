use crate::windows::WindowFunction;

use crate::interpolation::*;
use crate::{InterpolationParameters, InterpolationType};
use crate::sinc::make_sincs;
use num_traits::Float;
use std::error;
use std::fmt;
use core::arch::x86_64::{__m128d, __m128};
use std::any::TypeId;


type Res<T> = Result<T, Box<dyn error::Error>>;

use crate::Resampler;
use crate::ResamplerError;

enum SincBuffer {
    //SseSingle(Vec<Vec<__m128>>),
    //SseDouble(Vec<Vec<__m128d>>),
    SseSingle(Vec<Vec<f32>>),
    SseDouble(Vec<Vec<f64>>),
}


/// An asynchronous resampler that accepts a fixed number of audio frames for input
/// and returns a variable number of frames.
///
/// The resampling is done by creating a number of intermediate points (defined by oversampling_factor)
/// by sinc interpolation. The new samples are then calculated by interpolating between these points.
pub struct SseSincFixedIn<T> {
    nbr_channels: usize,
    chunk_size: usize,
    oversampling_factor: usize,
    last_index: f64,
    resample_ratio: f64,
    resample_ratio_original: f64,
    sinc_len: usize,
    sincs: SincBuffer,
    buffer: Vec<Vec<T>>,
    interpolation: InterpolationType,
}
/// An asynchronous resampler that return a fixed number of audio frames.
/// The number of input frames required is given by the frames_needed function.
///
/// The resampling is done by creating a number of intermediate points (defined by oversampling_factor)
/// by sinc interpolation. The new samples are then calculated by interpolating between these points.
pub struct SseSincFixedOut<T> {
    nbr_channels: usize,
    chunk_size: usize,
    needed_input_size: usize,
    oversampling_factor: usize,
    last_index: f64,
    current_buffer_fill: usize,
    resample_ratio: f64,
    resample_ratio_original: f64,
    sinc_len: usize,
    sincs: SincBuffer,
    buffer: Vec<Vec<T>>,
    interpolation: InterpolationType,
}

macro_rules! impl_resampler_single {
    ($rt:ty) => {
        impl $rt {
            /// Calculate the scalar produt of an input wave and the selected sinc filter
            fn get_sinc_interpolated(&self, wave: &[f32], index: usize, subindex: usize) -> f32 {
                match &self.sincs {
                    SincBuffer::SseSingle(sincs) => {
                        let wave_cut = &wave[index..(index + sincs[subindex].len())];
                        wave_cut
                            .chunks(8)
                            .zip(sincs[subindex].chunks(8))
                            .fold([0.0; 8], |acc, (x, y)| {
                                [
                                    acc[0] + x[0] * y[0],
                                    acc[1] + x[1] * y[1],
                                    acc[2] + x[2] * y[2],
                                    acc[3] + x[3] * y[3],
                                    acc[4] + x[4] * y[4],
                                    acc[5] + x[5] * y[5],
                                    acc[6] + x[6] * y[6],
                                    acc[7] + x[7] * y[7],
                                ]
                            })
                            .iter()
                            .sum()
                        }
                    _ => panic!("Wrong sinc data type")
                }
            }

            /// Perform cubic polynomial interpolation to get value at x.
            /// Input points are assumed to be at x = -1, 0, 1, 2
            unsafe fn interp_cubic(&self, x: f32, yvals: &[f32]) -> f32 {
                let a0 = yvals.get_unchecked(1);
                let a1 = -(1.0 / 3.0) * yvals.get_unchecked(0) - 0.5 * yvals.get_unchecked(1)
                    + yvals.get_unchecked(2)
                    - (1.0 / 6.0) * yvals.get_unchecked(3);
                let a2 = 0.5 * (yvals.get_unchecked(0) + yvals.get_unchecked(2))
                    - yvals.get_unchecked(1);
                let a3 = 0.5 * (yvals.get_unchecked(1) - yvals.get_unchecked(2))
                    + (1.0 / 6.0) * (yvals.get_unchecked(3) - yvals.get_unchecked(0));
                a0 + a1 * x + a2 * x.powi(2) + a3 * x.powi(3)
            }

            /// Linear interpolation between two points at x=0 and x=1
            unsafe fn interp_lin(&self, x: f32, yvals: &[f32]) -> f32 {
                (1.0 - x) * yvals.get_unchecked(0) + x * yvals.get_unchecked(1)
            }

            /// Prepare sinc buffer
            fn pack_sincs(sincs: Vec<Vec<f32>>) -> SincBuffer {
                SincBuffer::SseSingle(sincs)
            }
        }
    };
}


macro_rules! impl_resampler_double {
    ($rt:ty) => {
        impl $rt {
            /// Calculate the scalar produt of an input wave and the selected sinc filter
            fn get_sinc_interpolated(&self, wave: &[f64], index: usize, subindex: usize) -> f64 {
                match &self.sincs {
                    SincBuffer::SseDouble(sincs) => {
                        let wave_cut = &wave[index..(index + sincs[subindex].len())];
                        wave_cut
                            .chunks(8)
                            .zip(sincs[subindex].chunks(8))
                            .fold([0.0; 8], |acc, (x, y)| {
                                [
                                    acc[0] + x[0] * y[0],
                                    acc[1] + x[1] * y[1],
                                    acc[2] + x[2] * y[2],
                                    acc[3] + x[3] * y[3],
                                    acc[4] + x[4] * y[4],
                                    acc[5] + x[5] * y[5],
                                    acc[6] + x[6] * y[6],
                                    acc[7] + x[7] * y[7],
                                ]
                            })
                            .iter()
                            .sum()
                        }
                    _ => panic!("Wrong type")
                }
            }

            /// Perform cubic polynomial interpolation to get value at x.
            /// Input points are assumed to be at x = -1, 0, 1, 2
            unsafe fn interp_cubic(&self, x: f64, yvals: &[f64]) -> f64 {
                let a0 = yvals.get_unchecked(1);
                let a1 = -(1.0 / 3.0) * yvals.get_unchecked(0) - 0.5 * yvals.get_unchecked(1)
                    + yvals.get_unchecked(2)
                    - (1.0 / 6.0) * yvals.get_unchecked(3);
                let a2 = 0.5 * (yvals.get_unchecked(0) + yvals.get_unchecked(2))
                    - yvals.get_unchecked(1);
                let a3 = 0.5 * (yvals.get_unchecked(1) - yvals.get_unchecked(2))
                    + (1.0 / 6.0) * (yvals.get_unchecked(3) - yvals.get_unchecked(0));
                a0 + a1 * x + a2 * x.powi(2) + a3 * x.powi(3)
            }

            /// Linear interpolation between two points at x=0 and x=1
            unsafe fn interp_lin(&self, x: f64, yvals: &[f64]) -> f64 {
                (1.0 - x) * yvals.get_unchecked(0) + x * yvals.get_unchecked(1)
            }

            /// Prepare sinc buffer
            fn pack_sincs(sincs: Vec<Vec<f64>>) -> SincBuffer {
                SincBuffer::SseDouble(sincs)
            }
        }
    };
}

impl_resampler_single!(SseSincFixedIn<f32>);
impl_resampler_double!(SseSincFixedIn<f64>);
impl_resampler_single!(SseSincFixedOut<f32>);
impl_resampler_double!(SseSincFixedOut<f64>);

macro_rules! impl_new_ssesincfixedin {
    ($t:ty) => {
        impl SseSincFixedIn<$t> {
            /// Create a new SincFixedIn
            ///
            /// Parameters are:
            /// - `resample_ratio`: Ratio between output and input sample rates.
            /// - `parameters`: Parameters for interpolation, see `InterpolationParameters`
            /// - `chunk_size`: size of input data in frames
            /// - `nbr_channels`: number of channels in input/output
            pub fn new(
                resample_ratio: f64,
                parameters: InterpolationParameters,
                chunk_size: usize,
                nbr_channels: usize,
            ) -> Self {
                debug!(
                    "Create new SincFixedIn, ratio: {}, chunk_size: {}, channels: {}, parameters: {:?}",
                    resample_ratio, chunk_size, nbr_channels, parameters
                );
                let sinc_cutoff = if resample_ratio >= 1.0 {
                    parameters.f_cutoff
                } else {
                    parameters.f_cutoff * resample_ratio as f32
                };
                let sinc_len = 8 * (((parameters.sinc_len as f32) / 8.0).ceil() as usize);
                debug!("sinc_len rounded up to {}", sinc_len);
                let sincs = make_sincs(
                    sinc_len,
                    parameters.oversampling_factor,
                    sinc_cutoff,
                    parameters.window,
                );
                let buffer = vec![vec![0.0; chunk_size + 2 * sinc_len]; nbr_channels];

                let sincs = Self::pack_sincs(sincs);

                SseSincFixedIn {
                    nbr_channels,
                    chunk_size,
                    oversampling_factor: parameters.oversampling_factor,
                    last_index: -((sinc_len / 2) as f64),
                    resample_ratio,
                    resample_ratio_original: resample_ratio,
                    sinc_len,
                    sincs,
                    buffer,
                    interpolation: parameters.interpolation,
                }
            }
        }
    }
}
impl_new_ssesincfixedin!(f32);
impl_new_ssesincfixedin!(f64);

macro_rules! resampler_ssesincfixedin {
    ($t:ty) => {
        impl Resampler<$t> for SseSincFixedIn<$t> {
            /// Resample a chunk of audio. The input length is fixed, and the output varies in length.
            /// If the waveform for a channel is empty, this channel will be ignored and produce a
            /// corresponding empty output waveform.
            /// # Errors
            ///
            /// The function returns an error if the length of the input data is not equal
            /// to the number of channels and chunk size defined when creating the instance.
            fn process(&mut self, wave_in: &[Vec<$t>]) -> Res<Vec<Vec<$t>>> {
                if wave_in.len() != self.nbr_channels {
                    return Err(Box::new(ResamplerError::new(
                        "Wrong number of channels in input",
                    )));
                }
                let mut used_channels = Vec::new();
                for (chan, wave) in wave_in.iter().enumerate() {
                    if !wave.is_empty() {
                        used_channels.push(chan);
                        if wave.len() != self.chunk_size {
                            return Err(Box::new(ResamplerError::new(
                                "Wrong number of frames in input",
                            )));
                        }
                    }
                }
                let end_idx = self.chunk_size as isize - (self.sinc_len as isize + 1);
                //update buffer with new data
                for wav in self.buffer.iter_mut() {
                    for idx in 0..(2 * self.sinc_len) {
                        wav[idx] = wav[idx + self.chunk_size];
                    }
                }

                let mut wave_out = vec![Vec::new(); self.nbr_channels];

                for chan in used_channels.iter() {
                    for (idx, sample) in wave_in[*chan].iter().enumerate() {
                        self.buffer[*chan][idx + 2 * self.sinc_len] = *sample;
                    }
                    wave_out[*chan] = vec![
                        0.0 as $t;
                        (self.chunk_size as f64 * self.resample_ratio + 10.0)
                            as usize
                    ];
                }

                let mut idx = self.last_index;
                let t_ratio = 1.0 / self.resample_ratio as f64;

                let mut n = 0;

                match self.interpolation {
                    InterpolationType::Cubic => {
                        let mut points = vec![0.0 as $t; 4];
                        let mut nearest = vec![(0isize, 0isize); 4];
                        while idx < end_idx as f64 {
                            idx += t_ratio;
                            get_nearest_times_4(
                                idx,
                                self.oversampling_factor as isize,
                                &mut nearest,
                            );
                            let frac = idx * self.oversampling_factor as f64
                                - (idx * self.oversampling_factor as f64).floor();
                            let frac_offset = frac as $t;
                            for chan in used_channels.iter() {
                                let buf = &self.buffer[*chan];
                                for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                    *p = self.get_sinc_interpolated(
                                        &buf,
                                        (n.0 + 2 * self.sinc_len as isize) as usize,
                                        n.1 as usize,
                                    );
                                }
                                unsafe {
                                    wave_out[*chan][n] = self.interp_cubic(frac_offset, &points);
                                }
                            }
                            n += 1;
                        }
                    }
                    InterpolationType::Linear => {
                        let mut points = vec![0.0 as $t; 2];
                        let mut nearest = vec![(0isize, 0isize); 2];
                        while idx < end_idx as f64 {
                            idx += t_ratio;
                            get_nearest_times_2(
                                idx,
                                self.oversampling_factor as isize,
                                &mut nearest,
                            );
                            let frac = idx * self.oversampling_factor as f64
                                - (idx * self.oversampling_factor as f64).floor();
                            let frac_offset = frac as $t;
                            for chan in used_channels.iter() {
                                let buf = &self.buffer[*chan];
                                for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                    *p = self.get_sinc_interpolated(
                                        &buf,
                                        (n.0 + 2 * self.sinc_len as isize) as usize,
                                        n.1 as usize,
                                    );
                                }
                                unsafe {
                                    wave_out[*chan][n] = self.interp_lin(frac_offset, &points);
                                }
                            }
                            n += 1;
                        }
                    }
                    InterpolationType::Nearest => {
                        let mut point;
                        let mut nearest;
                        while idx < end_idx as f64 {
                            idx += t_ratio;
                            nearest = get_nearest_time(idx, self.oversampling_factor as isize);
                            for chan in used_channels.iter() {
                                let buf = &self.buffer[*chan];
                                point = self.get_sinc_interpolated(
                                    &buf,
                                    (nearest.0 + 2 * self.sinc_len as isize) as usize,
                                    nearest.1 as usize,
                                );
                                wave_out[*chan][n] = point;
                            }
                            n += 1;
                        }
                    }
                }

                // store last index for next iteration
                self.last_index = idx - self.chunk_size as f64;
                for chan in used_channels.iter() {
                    //for w in wave_out.iter_mut() {
                    wave_out[*chan].truncate(n);
                }
                trace!(
                    "Resampling channels {:?}, {} frames in, {} frames out",
                    used_channels,
                    self.chunk_size,
                    n,
                );
                Ok(wave_out)
            }

            /// Update the resample ratio. New value must be within +-10% of the original one
            fn set_resample_ratio(&mut self, new_ratio: f64) -> Res<()> {
                trace!("Change resample ratio to {}", new_ratio);
                if (new_ratio / self.resample_ratio_original > 0.9)
                    && (new_ratio / self.resample_ratio_original < 1.1)
                {
                    self.resample_ratio = new_ratio;
                    Ok(())
                } else {
                    Err(Box::new(ResamplerError::new(
                        "New resample ratio is too far off from original",
                    )))
                }
            }
            /// Update the resample ratio relative to the original one
            fn set_resample_ratio_relative(&mut self, rel_ratio: f64) -> Res<()> {
                let new_ratio = self.resample_ratio_original * rel_ratio;
                self.set_resample_ratio(new_ratio)
            }

            /// Query for the number of frames needed for the next call to "process".
            /// Will always return the chunk_size defined when creating the instance.
            fn nbr_frames_needed(&self) -> usize {
                self.chunk_size
            }
        }
    };
}
resampler_ssesincfixedin!(f32);
resampler_ssesincfixedin!(f64);

macro_rules! impl_new_ssesincfixedout {
    ($t:ty) => {
        impl SseSincFixedOut<$t> {
            /// Create a new SincFixedOut
            ///
            /// Parameters are:
            /// - `resample_ratio`: Ratio between output and input sample rates.
            /// - `parameters`: Parameters for interpolation, see `InterpolationParameters`
            /// - `chunk_size`: size of output data in frames
            /// - `nbr_channels`: number of channels in input/output
            pub fn new(
                resample_ratio: f64,
                parameters: InterpolationParameters,
                chunk_size: usize,
                nbr_channels: usize,
            ) -> Self {
                debug!(
                    "Create new SincFixedOut, ratio: {}, chunk_size: {}, channels: {}, parameters: {:?}",
                    resample_ratio, chunk_size, nbr_channels, parameters
                );
                let sinc_cutoff = if resample_ratio >= 1.0 {
                    parameters.f_cutoff
                } else {
                    parameters.f_cutoff * resample_ratio as f32
                };
                let sinc_len = 8 * (((parameters.sinc_len as f32) / 8.0).ceil() as usize);
                debug!("sinc_len rounded up to {}", sinc_len);
                let sincs = make_sincs(
                    sinc_len,
                    parameters.oversampling_factor,
                    sinc_cutoff,
                    parameters.window,
                );

                let sincs = Self::pack_sincs(sincs);
                
                let needed_input_size =
                    (chunk_size as f64 / resample_ratio).ceil() as usize + 2 + sinc_len / 2;
                let buffer = vec![vec![0.0; 3 * needed_input_size / 2 + 2 * sinc_len]; nbr_channels];
                SseSincFixedOut {
                    nbr_channels,
                    chunk_size,
                    needed_input_size,
                    oversampling_factor: parameters.oversampling_factor,
                    last_index: -((sinc_len / 2) as f64),
                    current_buffer_fill: needed_input_size,
                    resample_ratio,
                    resample_ratio_original: resample_ratio,
                    sinc_len,
                    sincs,
                    buffer,
                    interpolation: parameters.interpolation,
                }
            }
        }
    }
}
impl_new_ssesincfixedout!(f32);
impl_new_ssesincfixedout!(f64);

macro_rules! resampler_ssesincfixedout {
    ($t:ty) => {
        impl Resampler<$t> for SseSincFixedOut<$t> {
            /// Query for the number of frames needed for the next call to "process".
            fn nbr_frames_needed(&self) -> usize {
                self.needed_input_size
            }

            /// Update the resample ratio. New value must be within +-10% of the original one
            fn set_resample_ratio(&mut self, new_ratio: f64) -> Res<()> {
                trace!("Change resample ratio to {}", new_ratio);
                if (new_ratio / self.resample_ratio_original > 0.9)
                    && (new_ratio / self.resample_ratio_original < 1.1)
                {
                    self.resample_ratio = new_ratio;
                    self.needed_input_size = (self.last_index as f32
                        + self.chunk_size as f32 / self.resample_ratio as f32
                        + self.sinc_len as f32)
                        .ceil() as usize
                        + 2;
                    Ok(())
                } else {
                    Err(Box::new(ResamplerError::new(
                        "New resample ratio is too far off from original",
                    )))
                }
            }

            /// Update the resample ratio relative to the original one
            fn set_resample_ratio_relative(&mut self, rel_ratio: f64) -> Res<()> {
                let new_ratio = self.resample_ratio_original * rel_ratio;
                self.set_resample_ratio(new_ratio)
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
            fn process(&mut self, wave_in: &[Vec<$t>]) -> Res<Vec<Vec<$t>>> {
                //update buffer with new data
                if wave_in.len() != self.nbr_channels {
                    return Err(Box::new(ResamplerError::new(
                        "Wrong number of channels in input",
                    )));
                }

                let mut used_channels = Vec::new();
                for (chan, wave) in wave_in.iter().enumerate() {
                    if !wave.is_empty() {
                        used_channels.push(chan);
                        if wave.len() != self.needed_input_size {
                            return Err(Box::new(ResamplerError::new(
                                "Wrong number of frames in input",
                            )));
                        }
                    }
                }
                for wav in self.buffer.iter_mut() {
                    for idx in 0..(2 * self.sinc_len) {
                        wav[idx] = wav[idx + self.current_buffer_fill];
                    }
                }
                self.current_buffer_fill = self.needed_input_size;

                let mut wave_out = vec![Vec::new(); self.nbr_channels];

                for chan in used_channels.iter() {
                    for (idx, sample) in wave_in[*chan].iter().enumerate() {
                        self.buffer[*chan][idx + 2 * self.sinc_len] = *sample;
                    }
                    wave_out[*chan] = vec![0.0 as $t; self.chunk_size];
                }

                let mut idx = self.last_index;
                let t_ratio = 1.0 / self.resample_ratio as f64;

                match self.interpolation {
                    InterpolationType::Cubic => {
                        let mut points = vec![0.0 as $t; 4];
                        let mut nearest = vec![(0isize, 0isize); 4];
                        for n in 0..self.chunk_size {
                            idx += t_ratio;
                            get_nearest_times_4(idx, self.oversampling_factor as isize, &mut nearest);
                            let frac = idx * self.oversampling_factor as f64
                                - (idx * self.oversampling_factor as f64).floor();
                            let frac_offset = frac as $t;
                            for chan in used_channels.iter() {
                                let buf = &self.buffer[*chan];
                                for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                    *p = self.get_sinc_interpolated(
                                        &buf,
                                        (n.0 + 2 * self.sinc_len as isize) as usize,
                                        n.1 as usize,
                                    );
                                }
                                unsafe {
                                    wave_out[*chan][n] = self.interp_cubic(frac_offset, &points);
                                }
                            }
                        }
                    }
                    InterpolationType::Linear => {
                        let mut points = vec![0.0 as $t; 2];
                        let mut nearest = vec![(0isize, 0isize); 2];
                        for n in 0..self.chunk_size {
                            idx += t_ratio;
                            get_nearest_times_2(idx, self.oversampling_factor as isize, &mut nearest);
                            let frac = idx * self.oversampling_factor as f64
                                - (idx * self.oversampling_factor as f64).floor();
                            let frac_offset = frac as $t;
                            for chan in used_channels.iter() {
                                let buf = &self.buffer[*chan];
                                for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                    *p = self.get_sinc_interpolated(
                                        &buf,
                                        (n.0 + 2 * self.sinc_len as isize) as usize,
                                        n.1 as usize,
                                    );
                                }
                                unsafe {
                                    wave_out[*chan][n] = self.interp_lin(frac_offset, &points);
                                }
                            }
                        }
                    }
                    InterpolationType::Nearest => {
                        let mut point;
                        let mut nearest;
                        for n in 0..self.chunk_size {
                            idx += t_ratio;
                            nearest = get_nearest_time(idx, self.oversampling_factor as isize);
                            for chan in used_channels.iter() {
                                let buf = &self.buffer[*chan];
                                point = self.get_sinc_interpolated(
                                    &buf,
                                    (nearest.0 + 2 * self.sinc_len as isize) as usize,
                                    nearest.1 as usize,
                                );
                                wave_out[*chan][n] = point;
                            }
                        }
                    }
                }

                let prev_input_len = self.needed_input_size;
                // store last index for next iteration
                self.last_index = idx - self.current_buffer_fill as f64;
                self.needed_input_size = (self.last_index as f32
                    + self.chunk_size as f32 / self.resample_ratio as f32
                    + self.sinc_len as f32)
                    .ceil() as usize
                    + 2;
                trace!(
                    "Resampling channels {:?}, {} frames in, {} frames out. Next needed length: {} frames, last index {}",
                    used_channels,
                    prev_input_len,
                    self.chunk_size,
                    self.needed_input_size,
                    self.last_index
                );
                Ok(wave_out)
            }
        }
    }
}
resampler_ssesincfixedout!(f32);
resampler_ssesincfixedout!(f64);

#[cfg(test)]
mod tests {
    use crate::InterpolationParameters;
    use crate::InterpolationType;
    use crate::Resampler;
    use crate::WindowFunction;
    use crate::{SseSincFixedIn, SseSincFixedOut};

    #[test]
    fn int_cubic() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let resampler = SseSincFixedIn::<f64>::new(1.2, params, 1024, 2);
        let yvals = vec![0.0f64, 2.0f64, 4.0f64, 6.0f64];
        unsafe {
            let interp = resampler.interp_cubic(0.5f64, &yvals);
            assert_eq!(interp, 3.0f64);
        }
    }

    #[test]
    fn int_lin_32() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let resampler = SseSincFixedIn::<f32>::new(1.2, params, 1024, 2);
        let yvals = vec![1.0f32, 5.0f32];
        unsafe {
            let interp = resampler.interp_lin(0.25f32, &yvals);
            assert_eq!(interp, 2.0f32);
        }
    }

    #[test]
    fn int_cubic_32() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let resampler = SseSincFixedIn::<f32>::new(1.2, params, 1024, 2);
        let yvals = vec![0.0f32, 2.0f32, 4.0f32, 6.0f32];
        unsafe {
            let interp = resampler.interp_cubic(0.5f32, &yvals);
            assert_eq!(interp, 3.0f32);
        }
    }

    #[test]
    fn int_lin() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let resampler = SseSincFixedIn::<f64>::new(1.2, params, 1024, 2);
        let yvals = vec![1.0f64, 5.0f64];
        unsafe {
            let interp = resampler.interp_lin(0.25f64, &yvals);
            assert_eq!(interp, 2.0f64);
        }
    }

    #[test]
    fn make_resampler_fi() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SseSincFixedIn::<f64>::new(1.2, params, 1024, 2);
        let waves = vec![vec![0.0f64; 1024]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].len() > 1150 && out[0].len() < 1250);
    }

    #[test]
    fn make_resampler_fi_32() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SseSincFixedIn::<f32>::new(1.2, params, 1024, 2);
        let waves = vec![vec![0.0f32; 1024]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].len() > 1150 && out[0].len() < 1250);
    }

    #[test]
    fn make_resampler_fi_skipped() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SseSincFixedIn::<f64>::new(1.2, params, 1024, 2);
        let waves = vec![vec![0.0f64; 1024], Vec::new()];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].len() > 1150 && out[0].len() < 1250);
        assert!(out[1].is_empty());
        let waves = vec![Vec::new(), vec![0.0f64; 1024]];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[1].len() > 1150 && out[0].len() < 1250);
        assert!(out[0].is_empty());
    }

    #[test]
    fn make_resampler_fo() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SseSincFixedOut::<f64>::new(1.2, params, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn make_resampler_fo_32() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SseSincFixedOut::<f32>::new(1.2, params, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let waves = vec![vec![0.0f32; frames]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn make_resampler_fo_skipped() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SseSincFixedOut::<f64>::new(1.2, params, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let mut waves = vec![vec![0.0f64; frames], Vec::new()];
        waves[0][10] = 3.0;
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
        assert!(out[1].is_empty());
        assert!(out[0].iter().sum::<f64>() > 2.0);

        let frames = resampler.nbr_frames_needed();
        let mut waves = vec![Vec::new(), vec![0.0f64; frames]];
        waves[1][10] = 3.0;
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].len(), 1024);
        assert!(out[0].is_empty());
        assert!(out[1].iter().sum::<f64>() > 2.0);
    }
}
