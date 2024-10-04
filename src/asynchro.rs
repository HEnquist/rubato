use std::fmt;
use std::marker::PhantomData;

use crate::asynchro_fast::{
    interp_cubic, interp_lin, interp_quintic, interp_septic, PolynomialDegree,
};
use crate::asynchro_sinc::{
    interp_cubic as interp_cubic_array, interp_lin as interp_lin_array,
    interp_quad as interp_quad_array, make_interpolator, SincInterpolationParameters,
    SincInterpolationType,
};
use crate::error::{ResampleError, ResampleResult, ResamplerConstructionError};
use crate::interpolation::*;
use crate::sinc_interpolator::SincInterpolator;
use crate::{update_mask_from_buffers, validate_buffers, Fixed, Resampler, Sample};

macro_rules! t {
    // Shorter form of T::coerce(value)
    ($expression:expr) => {
        T::coerce($expression)
    };
}

/// Functions for making the scalar product with a sinc.
pub trait InnerResampler<T>: Send {
    /// Make the scalar product between the waveform starting at `index` and the sinc of `subindex`.
    fn process(
        &self,
        index: f64,
        nbr_frames: usize,
        channel_mask: &[bool],
        t_ratio: f64,
        t_ratio_increment: f64,
        wave_in: &[Vec<T>],
        wave_out: &mut [&mut [T]],
    ) -> f64;

    /// Get interpolator length.
    fn len(&self) -> usize;
}

pub struct InnerSinc<T> {
    interpolator: Box<dyn SincInterpolator<T>>,
    interpolation: SincInterpolationType,
}

impl<T> InnerResampler<T> for InnerSinc<T>
where
    T: Sample,
{
    fn process(
        &self,
        idx: f64,
        nbr_frames: usize,
        channel_mask: &[bool],
        t_ratio: f64,
        t_ratio_increment: f64,
        wave_in: &[Vec<T>],
        wave_out: &mut [&mut [T]],
    ) -> f64 {
        let mut t_ratio = t_ratio;
        let mut idx = idx;
        let interpolator_len = self.interpolator.len();
        match self.interpolation {
            SincInterpolationType::Cubic => {
                let oversampling_factor = self.interpolator.nbr_sincs();
                let mut points = [T::zero(); 4];
                let mut nearest = [(0isize, 0isize); 4];
                for n in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    get_nearest_times_4(idx, oversampling_factor as isize, &mut nearest);
                    let frac = idx * oversampling_factor as f64
                        - (idx * oversampling_factor as f64).floor();
                    let frac_offset = T::coerce(frac);
                    for (chan, active) in channel_mask.iter().enumerate() {
                        if *active {
                            let buf = &wave_in[chan];
                            for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                *p = self.interpolator.get_sinc_interpolated(
                                    buf,
                                    (n.0 + 2 * interpolator_len as isize) as usize,
                                    n.1 as usize,
                                );
                            }
                            wave_out[chan][n] = interp_cubic_array(frac_offset, &points);
                        }
                    }
                }
            }
            SincInterpolationType::Quadratic => {
                let oversampling_factor = self.interpolator.nbr_sincs();
                let mut points = [T::zero(); 3];
                let mut nearest = [(0isize, 0isize); 3];
                for n in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    get_nearest_times_3(idx, oversampling_factor as isize, &mut nearest);
                    let frac = idx * oversampling_factor as f64
                        - (idx * oversampling_factor as f64).floor();
                    let frac_offset = T::coerce(frac);
                    for (chan, active) in channel_mask.iter().enumerate() {
                        if *active {
                            let buf = &wave_in[chan];
                            for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                *p = self.interpolator.get_sinc_interpolated(
                                    buf,
                                    (n.0 + 2 * interpolator_len as isize) as usize,
                                    n.1 as usize,
                                );
                            }
                            wave_out[chan][n] = interp_quad_array(frac_offset, &points);
                        }
                    }
                }
            }
            SincInterpolationType::Linear => {
                let oversampling_factor = self.interpolator.nbr_sincs();
                let mut points = [T::zero(); 2];
                let mut nearest = [(0isize, 0isize); 2];
                for n in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    get_nearest_times_2(idx, oversampling_factor as isize, &mut nearest);
                    let frac = idx * oversampling_factor as f64
                        - (idx * oversampling_factor as f64).floor();
                    let frac_offset = T::coerce(frac);
                    for (chan, active) in channel_mask.iter().enumerate() {
                        if *active {
                            let buf = &wave_in[chan];
                            for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                *p = self.interpolator.get_sinc_interpolated(
                                    buf,
                                    (n.0 + 2 * interpolator_len as isize) as usize,
                                    n.1 as usize,
                                );
                            }
                            wave_out[chan][n] = interp_lin_array(frac_offset, &points);
                        }
                    }
                }
            }
            SincInterpolationType::Nearest => {
                let oversampling_factor = self.interpolator.nbr_sincs();
                let mut point;
                let mut nearest;
                for n in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    nearest = get_nearest_time(idx, oversampling_factor as isize);
                    for (chan, active) in channel_mask.iter().enumerate() {
                        if *active {
                            let buf = &wave_in[chan];
                            point = self.interpolator.get_sinc_interpolated(
                                buf,
                                (nearest.0 + 2 * interpolator_len as isize) as usize,
                                nearest.1 as usize,
                            );
                            wave_out[chan][n] = point;
                        }
                    }
                }
            }
        }
        idx
    }

    fn len(&self) -> usize {
        self.interpolator.len()
    }
}

pub struct InnerPoly<T> {
    _phantom: PhantomData<T>,
    interpolation: PolynomialDegree,
}

impl<T> InnerResampler<T> for InnerPoly<T>
where
    T: Sample,
{
    fn process(
        &self,
        idx: f64,
        nbr_frames: usize,
        channel_mask: &[bool],
        t_ratio: f64,
        t_ratio_increment: f64,
        wave_in: &[Vec<T>],
        wave_out: &mut [&mut [T]],
    ) -> f64 {
        let interpolator_len = self.len();
        let mut t_ratio = t_ratio;
        let mut idx = idx;
        match self.interpolation {
            PolynomialDegree::Septic => {
                for n in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let idx_floor = idx.floor();
                    let start_idx = idx_floor as isize - 3;
                    let frac = idx - idx_floor;
                    let frac_offset = t!(frac);
                    for (chan, active) in channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let buf = wave_in.get_unchecked(chan).get_unchecked(
                                    (start_idx + 2 * interpolator_len as isize) as usize
                                        ..(start_idx + 2 * interpolator_len as isize + 8) as usize,
                                );
                                *wave_out.get_unchecked_mut(chan).get_unchecked_mut(n) =
                                    interp_septic(frac_offset, buf);
                            }
                        }
                    }
                }
            }
            PolynomialDegree::Quintic => {
                for n in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let idx_floor = idx.floor();
                    let start_idx = idx_floor as isize - 2;
                    let frac = idx - idx_floor;
                    let frac_offset = t!(frac);
                    for (chan, active) in channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let buf = wave_in.get_unchecked(chan).get_unchecked(
                                    (start_idx + 2 * interpolator_len as isize) as usize
                                        ..(start_idx + 2 * interpolator_len as isize + 6) as usize,
                                );
                                *wave_out.get_unchecked_mut(chan).get_unchecked_mut(n) =
                                    interp_quintic(frac_offset, buf);
                            }
                        }
                    }
                }
            }
            PolynomialDegree::Cubic => {
                for n in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let idx_floor = idx.floor();
                    let start_idx = idx_floor as isize - 1;
                    let frac = idx - idx_floor;
                    let frac_offset = t!(frac);
                    for (chan, active) in channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let buf = wave_in.get_unchecked(chan).get_unchecked(
                                    (start_idx + 2 * interpolator_len as isize) as usize
                                        ..(start_idx + 2 * interpolator_len as isize + 4) as usize,
                                );
                                *wave_out.get_unchecked_mut(chan).get_unchecked_mut(n) =
                                    interp_cubic(frac_offset, buf);
                            }
                        }
                    }
                }
            }
            PolynomialDegree::Linear => {
                for n in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let idx_floor = idx.floor();
                    let start_idx = idx_floor as isize;
                    let frac = idx - idx_floor;
                    let frac_offset = t!(frac);
                    for (chan, active) in channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let buf = wave_in.get_unchecked(chan).get_unchecked(
                                    (start_idx + 2 * interpolator_len as isize) as usize
                                        ..(start_idx + 2 * interpolator_len as isize + 2) as usize,
                                );
                                *wave_out.get_unchecked_mut(chan).get_unchecked_mut(n) =
                                    interp_lin(frac_offset, buf);
                            }
                        }
                    }
                }
            }
            PolynomialDegree::Nearest => {
                for n in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let start_idx = idx.floor() as isize;
                    for (chan, active) in channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let point = wave_in.get_unchecked(chan).get_unchecked(
                                    (start_idx + 2 * interpolator_len as isize) as usize,
                                );
                                *wave_out.get_unchecked_mut(chan).get_unchecked_mut(n) = *point;
                            }
                        }
                    }
                }
            }
        }
        idx
    }

    fn len(&self) -> usize {
        self.interpolation.len()
    }
}

/// An asynchronous resampler that either takes a fixed number of input frames,
/// or returns a fixed number of audio frames.
///
/// The `fixed` argument determines if input of output should be fixed size.
/// When the input size is fixed, the output size varies from call to call,
/// and when output size is fixed, the input size varies.
///
/// The number of frames on the fixed side is determined by the chunk size argument to the constructor.
/// This value can be changed by the `set_chunk_size()` method,
/// to let the resampler process smaller chunks of audio data.
/// Note that the chunk size cannot exceed the value given at creation time.
///
/// When the input size is fixed, the maximum value can be retrieved using the `input_size_max()` method,
/// and `input_frames_next()` gives the current value.
/// When the output size is fixed, the corresponding values are instead provided by the `output_size_max()`
/// and `output_size_next()` methods.
///
/// Polynomial
/// The resampling is done by interpolating between the input samples.
/// The polynomial degree can be selected, see [PolynomialDegree] for the available options.
///
/// Note that no anti-aliasing filter is used.
/// This makes it run considerably faster than the corresponding Sinc resampler, which performs anti-aliasing filtering.
/// The price is that the resampling creates some artefacts in the output, mainly at higher frequencies.
/// Use a Sinc resampler if this can not be tolerated.
///
/// Sinc
/// The resampling is done by creating a number of intermediate points (defined by oversampling_factor)
/// by sinc interpolation. The new samples are then calculated by interpolating between these points.
///
/// The resampling ratio can be freely adjusted within the range specified to the constructor.
/// Adjusting the ratio does not recalculate the sinc functions used by the anti-aliasing filter.
/// This causes no issue when increasing the ratio (which slows down the output).
/// However, when decreasing more than a few percent (or speeding up the output),
/// the filters can no longer suppress all aliasing and this may lead to some artefacts.
///
/// The resampling ratio can be freely adjusted within the range specified to the constructor.
/// Higher maximum ratios require more memory to be allocated by
/// [input_buffer_allocate](Resampler::input_buffer_allocate),
/// [output_buffer_allocate](Resampler::output_buffer_allocate), and an internal buffer.
pub struct Async<'a, T> {
    nbr_channels: usize,
    chunk_size: usize,
    max_chunk_size: usize,
    needed_input_size: usize,
    needed_output_size: usize,
    last_index: f64,
    current_buffer_fill: usize,
    resample_ratio: f64,
    resample_ratio_original: f64,
    target_ratio: f64,
    max_relative_ratio: f64,
    buffer: Vec<Vec<T>>,
    inner_resampler: Box<dyn InnerResampler<T>>,
    channel_mask: Vec<bool>,
    fixed: Fixed,
    temp_output: Vec<&'a mut [T]>,
}

impl<'a, T> fmt::Debug for Async<'a, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Fast")
            .field("nbr_channels", &self.nbr_channels)
            .field("chunk_size,", &self.chunk_size)
            .field("max_chunk_size,", &self.max_chunk_size)
            .field("needed_input_size,", &self.needed_input_size)
            .field("needed_output_size,", &self.needed_output_size)
            .field("last_index", &self.last_index)
            .field("current_buffer_fill", &self.current_buffer_fill)
            .field("resample_ratio", &self.resample_ratio)
            .field("resample_ratio_original", &self.resample_ratio_original)
            .field("target_ratio", &self.target_ratio)
            .field("max_relative_ratio", &self.max_relative_ratio)
            .field("buffer[0].len()", &self.buffer[0].len())
            .field("channel_mask", &self.channel_mask)
            .field("fixed", &self.fixed)
            .finish()
    }
}

fn validate_ratios(
    resample_ratio: f64,
    max_resample_ratio_relative: f64,
) -> Result<(), ResamplerConstructionError> {
    if resample_ratio <= 0.0 {
        return Err(ResamplerConstructionError::InvalidRatio(resample_ratio));
    }
    if max_resample_ratio_relative < 1.0 {
        return Err(ResamplerConstructionError::InvalidRelativeRatio(
            max_resample_ratio_relative,
        ));
    }
    Ok(())
}

impl<'a, T> Async<'a, T>
where
    T: Sample,
{
    /// Create a new Async with polynomial interpolation.
    ///
    /// Parameters are:
    /// - `resample_ratio`: Starting ratio between output and input sample rates, must be > 0.
    /// - `max_resample_ratio_relative`: Maximum ratio that can be set with [Resampler::set_resample_ratio] relative to `resample_ratio`, must be >= 1.0. The minimum relative ratio is the reciprocal of the maximum. For example, with `max_resample_ratio_relative` of 10.0, the ratio can be set between `resample_ratio * 10.0` and `resample_ratio / 10.0`.
    /// - `interpolation_type`: Degree of polynomial used for interpolation, see [PolynomialDegree].
    /// - `chunk_size`: Size of input data in frames.
    /// - `nbr_channels`: Number of channels in input/output.
    /// - `fixed`: Deciding whether input or output size is fixed.
    pub fn new_poly(
        resample_ratio: f64,
        max_resample_ratio_relative: f64,
        interpolation_type: PolynomialDegree,
        chunk_size: usize,
        nbr_channels: usize,
        fixed: Fixed,
    ) -> Result<Self, ResamplerConstructionError> {
        debug!(
            "Create new Fast with fixed {:?}, ratio: {}, chunk_size: {}, channels: {}",
            fixed, resample_ratio, chunk_size, nbr_channels,
        );

        validate_ratios(resample_ratio, max_resample_ratio_relative)?;

        let channel_mask = vec![true; nbr_channels];

        let interpolator_len = interpolation_type.len();

        let last_index = -(interpolator_len as f64 / 2.0);
        let needed_input_size = Self::calculate_input_size(
            chunk_size,
            resample_ratio,
            resample_ratio,
            last_index,
            interpolator_len,
            &fixed,
        );
        let needed_output_size = Self::calculate_output_size(
            chunk_size,
            resample_ratio,
            resample_ratio,
            last_index,
            interpolator_len,
            &fixed,
        );

        let buffer_len = Self::calculate_max_input_size(
            chunk_size,
            resample_ratio,
            max_resample_ratio_relative,
            interpolator_len,
            &fixed,
        ) + 2 * interpolator_len;
        let buffer = vec![vec![T::zero(); buffer_len]; nbr_channels];

        let inner_resampler = InnerPoly {
            interpolation: interpolation_type,
            _phantom: PhantomData,
        };

        let temp_output = Vec::with_capacity(nbr_channels);

        Ok(Async {
            nbr_channels,
            chunk_size,
            max_chunk_size: chunk_size,
            needed_input_size,
            needed_output_size,
            current_buffer_fill: needed_input_size,
            last_index,
            resample_ratio,
            resample_ratio_original: resample_ratio,
            target_ratio: resample_ratio,
            max_relative_ratio: max_resample_ratio_relative,
            buffer,
            inner_resampler: Box::new(inner_resampler),
            channel_mask,
            fixed,
            temp_output,
        })
    }

    /// Create a new Async with sinc interpolation.
    ///
    /// Parameters are:
    /// - `resample_ratio`: Starting ratio between output and input sample rates, must be > 0.
    /// - `max_resample_ratio_relative`: Maximum ratio that can be set with [Resampler::set_resample_ratio] relative to `resample_ratio`, must be >= 1.0. The minimum relative ratio is the reciprocal of the maximum. For example, with `max_resample_ratio_relative` of 10.0, the ratio can be set between `resample_ratio * 10.0` and `resample_ratio / 10.0`.
    /// - `parameters`: Parameters for interpolation, see `SincInterpolationParameters`.
    /// - `chunk_size`: Size of input data in frames.
    /// - `nbr_channels`: Number of channels in input/output.
    pub fn new_sinc(
        resample_ratio: f64,
        max_resample_ratio_relative: f64,
        parameters: SincInterpolationParameters,
        chunk_size: usize,
        nbr_channels: usize,
        fixed: Fixed,
    ) -> Result<Self, ResamplerConstructionError> {
        debug!(
            "Create new Sinc fixed {:?}, ratio: {}, chunk_size: {}, channels: {}, parameters: {:?}",
            fixed, resample_ratio, chunk_size, nbr_channels, parameters
        );

        let interpolator = make_interpolator(
            parameters.sinc_len,
            resample_ratio,
            parameters.f_cutoff,
            parameters.oversampling_factor,
            parameters.window,
        );

        Self::new_with_sinc_interpolator(
            resample_ratio,
            max_resample_ratio_relative,
            parameters.interpolation,
            interpolator,
            chunk_size,
            nbr_channels,
            fixed,
        )
    }

    /// Create a new Sinc using an existing Interpolator.
    ///
    /// Parameters are:
    /// - `resample_ratio`: Starting ratio between output and input sample rates, must be > 0.
    /// - `max_resample_ratio_relative`: Maximum ratio that can be set with [Resampler::set_resample_ratio] relative to `resample_ratio`, must be >= 1.0. The minimum relative ratio is the reciprocal of the maximum. For example, with `max_resample_ratio_relative` of 10.0, the ratio can be set between `resample_ratio` * 10.0 and `resample_ratio` / 10.0.
    /// - `interpolation_type`: Parameters for interpolation, see `SincInterpolationParameters`.
    /// - `interpolator`: The interpolator to use.
    /// - `chunk_size`: Size of output data in frames.
    /// - `nbr_channels`: Number of channels in input/output.
    pub fn new_with_sinc_interpolator(
        resample_ratio: f64,
        max_resample_ratio_relative: f64,
        interpolation_type: SincInterpolationType,
        interpolator: Box<dyn SincInterpolator<T>>,
        chunk_size: usize,
        nbr_channels: usize,
        fixed: Fixed,
    ) -> Result<Self, ResamplerConstructionError> {
        validate_ratios(resample_ratio, max_resample_ratio_relative)?;

        let interpolator_len = interpolator.len();

        let last_index = -(interpolator_len as f64) / 2.0;
        let needed_input_size = Self::calculate_input_size(
            chunk_size,
            resample_ratio,
            resample_ratio,
            last_index,
            interpolator_len,
            &fixed,
        );
        let needed_output_size = Self::calculate_output_size(
            chunk_size,
            resample_ratio,
            resample_ratio,
            last_index,
            interpolator_len,
            &fixed,
        );

        let buffer_len = Self::calculate_max_input_size(
            chunk_size,
            resample_ratio,
            max_resample_ratio_relative,
            interpolator_len,
            &fixed,
        ) + 2 * interpolator_len;

        let buffer = vec![vec![T::zero(); buffer_len]; nbr_channels];

        let channel_mask = vec![true; nbr_channels];
        let last_index = -((interpolator.len() / 2) as f64);

        let inner_resampler = InnerSinc {
            interpolator,
            interpolation: interpolation_type,
        };

        let temp_output = Vec::with_capacity(nbr_channels);

        Ok(Async {
            nbr_channels,
            chunk_size,
            max_chunk_size: chunk_size,
            needed_input_size,
            needed_output_size,
            last_index,
            current_buffer_fill: needed_input_size,
            resample_ratio,
            resample_ratio_original: resample_ratio,
            target_ratio: resample_ratio,
            max_relative_ratio: max_resample_ratio_relative,
            inner_resampler: Box::new(inner_resampler),
            buffer,
            channel_mask,
            fixed,
            temp_output,
        })
    }

    fn calculate_input_size(
        chunk_size: usize,
        resample_ratio: f64,
        target_ratio: f64,
        last_index: f64,
        interpolator_len: usize,
        fixed: &Fixed,
    ) -> usize {
        match fixed {
            Fixed::Input => chunk_size,
            Fixed::Output => (last_index
                + chunk_size as f64 / (0.5 * resample_ratio + 0.5 * target_ratio)
                + interpolator_len as f64)
                .ceil() as usize,
        }
    }

    fn calculate_output_size(
        chunk_size: usize,
        resample_ratio: f64,
        target_ratio: f64,
        last_index: f64,
        interpolator_len: usize,
        fixed: &Fixed,
    ) -> usize {
        match fixed {
            Fixed::Output => chunk_size,
            Fixed::Input => ((chunk_size as f64 - (interpolator_len + 1) as f64 - last_index)
                * (0.5 * resample_ratio + 0.5 * target_ratio))
                .floor() as usize,
        }
    }

    fn calculate_max_input_size(
        chunk_size: usize,
        resample_ratio_original: f64,
        max_relative_ratio: f64,
        interpolator_len: usize,
        fixed: &Fixed,
    ) -> usize {
        match fixed {
            Fixed::Input => chunk_size,
            Fixed::Output => {
                (chunk_size as f64 / resample_ratio_original * max_relative_ratio).ceil() as usize
                    + 2
                    + interpolator_len / 2
            }
        }
    }

    fn calculate_max_output_size(
        chunk_size: usize,
        resample_ratio_original: f64,
        max_relative_ratio: f64,
        fixed: &Fixed,
    ) -> usize {
        match fixed {
            Fixed::Output => chunk_size,
            Fixed::Input => {
                (chunk_size as f64 * resample_ratio_original * max_relative_ratio + 10.0) as usize
            }
        }
    }

    fn update_lengths(&mut self) {
        self.needed_input_size = Async::<T>::calculate_input_size(
            self.chunk_size,
            self.resample_ratio,
            self.target_ratio,
            self.last_index,
            self.inner_resampler.len(),
            &self.fixed,
        );
        self.needed_output_size = Async::<T>::calculate_output_size(
            self.chunk_size,
            self.resample_ratio,
            self.target_ratio,
            self.last_index,
            self.inner_resampler.len(),
            &self.fixed,
        );
        trace!(
            "Updated lengths, input: {}, output: {}",
            self.needed_input_size,
            self.needed_output_size
        );
    }
}

impl<'a, T> Resampler<T> for Async<'a, T>
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
            self.needed_input_size,
            self.needed_output_size,
        )?;

        let interpolator_len = self.inner_resampler.len();

        let t_ratio = 1.0 / self.resample_ratio;
        let t_ratio_end = 1.0 / self.target_ratio;

        let t_ratio_increment = (t_ratio_end - t_ratio) / self.needed_output_size as f64;

        // Update buffer with new data.
        for buf in self.buffer.iter_mut() {
            buf.copy_within(
                self.current_buffer_fill..self.current_buffer_fill + 2 * interpolator_len,
                0,
            );
        }

        for (chan, wave_in) in wave_in
            .iter()
            .enumerate()
            .filter(|(chan, _)| self.channel_mask[*chan])
        {
            self.buffer[chan][2 * interpolator_len..2 * interpolator_len + self.needed_input_size]
                .copy_from_slice(&wave_in.as_ref()[..self.needed_input_size]);
        }

        self.current_buffer_fill = self.needed_input_size;

        let mut idx = self.last_index;

        // Get references to the channel data, store in the temp vec.
        // This is to avoid allocating a vector.
        // This is safe because we drop them again right after the processing step.
        for wave in wave_out {
            // Transmute to work around the different lifetimes of the data and the vector.
            let slice = unsafe { std::mem::transmute(wave.as_mut()) };
            self.temp_output.push(slice);
        }
        // Process
        idx = self.inner_resampler.process(
            idx,
            self.needed_output_size,
            &self.channel_mask,
            t_ratio,
            t_ratio_increment,
            &self.buffer,
            &mut self.temp_output,
        );

        // Drop all the references
        self.temp_output.clear();

        // Store last index for next iteration.
        self.last_index = idx - self.needed_input_size as f64;
        self.resample_ratio = self.target_ratio;
        trace!(
            "Resampling channels {:?}, {} frames in, {} frames out",
            active_channels_mask,
            self.needed_input_size,
            self.needed_output_size,
        );
        let input_size = self.needed_input_size;
        let output_size = self.needed_output_size;
        self.update_lengths();
        Ok((input_size, output_size))
    }

    fn output_frames_max(&self) -> usize {
        Async::<T>::calculate_max_output_size(
            self.max_chunk_size,
            self.resample_ratio_original,
            self.max_relative_ratio,
            &self.fixed,
        )
    }

    fn output_frames_next(&self) -> usize {
        self.needed_output_size
    }

    fn output_delay(&self) -> usize {
        (self.inner_resampler.len() as f64 * self.resample_ratio / 2.0) as usize
    }

    fn nbr_channels(&self) -> usize {
        self.nbr_channels
    }

    fn input_frames_max(&self) -> usize {
        Async::<T>::calculate_max_input_size(
            self.max_chunk_size,
            self.resample_ratio_original,
            self.max_relative_ratio,
            self.inner_resampler.len(),
            &self.fixed,
        )
    }

    fn input_frames_next(&self) -> usize {
        self.needed_input_size
    }

    fn set_resample_ratio(&mut self, new_ratio: f64, ramp: bool) -> ResampleResult<()> {
        trace!("Change resample ratio to {}", new_ratio);
        if (new_ratio / self.resample_ratio_original >= 1.0 / self.max_relative_ratio)
            && (new_ratio / self.resample_ratio_original <= self.max_relative_ratio)
        {
            if !ramp {
                self.resample_ratio = new_ratio;
            }
            self.target_ratio = new_ratio;
            self.update_lengths();
            Ok(())
        } else {
            Err(ResampleError::RatioOutOfBounds {
                provided: new_ratio,
                original: self.resample_ratio_original,
                max_relative_ratio: self.max_relative_ratio,
            })
        }
    }

    fn set_resample_ratio_relative(&mut self, rel_ratio: f64, ramp: bool) -> ResampleResult<()> {
        let new_ratio = self.resample_ratio_original * rel_ratio;
        self.set_resample_ratio(new_ratio, ramp)
    }

    fn reset(&mut self) {
        self.buffer
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = T::zero()));
        self.channel_mask.iter_mut().for_each(|val| *val = true);
        self.last_index = -(self.inner_resampler.len() as f64 / 2.0);
        self.resample_ratio = self.resample_ratio_original;
        self.target_ratio = self.resample_ratio_original;
        self.chunk_size = self.max_chunk_size;
        self.update_lengths();
    }

    fn set_chunk_size(&mut self, chunksize: usize) -> ResampleResult<()> {
        if chunksize > self.max_chunk_size || chunksize == 0 {
            return Err(ResampleError::InvalidChunkSize {
                max: self.max_chunk_size,
                requested: chunksize,
            });
        }
        self.chunk_size = chunksize;
        self.update_lengths();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::PolynomialDegree;
    use crate::Resampler;
    use crate::{check_output, check_ratio};
    use crate::{Async, Fixed};
    use rand::Rng;
    use test_log::test;

    #[test]
    fn make_poly_resampler_fi() {
        let mut resampler =
            Async::<f64>::new_poly(1.2, 1.0, PolynomialDegree::Cubic, 1024, 2, Fixed::Input)
                .unwrap();
        let waves = vec![vec![0.0f64; 1024]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert!(
            out[0].len() > 1150 && out[0].len() < 1229,
            "expected {} - {} samples, got {}",
            1150,
            1229,
            out[0].len()
        );
        let out2 = resampler.process(&waves, None).unwrap();
        assert_eq!(out2.len(), 2, "Expected {} channels, got {}", 2, out2.len());
        assert!(
            out2[0].len() > 1226 && out2[0].len() < 1232,
            "expected {} - {} samples, got {}",
            1226,
            1232,
            out2[0].len()
        );
    }

    #[test]
    fn reset_poly_resampler_fi() {
        let mut resampler =
            Async::<f64>::new_poly(1.2, 1.0, PolynomialDegree::Cubic, 1024, 2, Fixed::Input)
                .unwrap();

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
    fn make_poly_resampler_fi_32() {
        let mut resampler =
            Async::<f32>::new_poly(1.2, 1.0, PolynomialDegree::Cubic, 1024, 2, Fixed::Input)
                .unwrap();
        let waves = vec![vec![0.0f32; 1024]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert!(
            out[0].len() > 1150 && out[0].len() < 1229,
            "expected {} - {} samples, got {}",
            1150,
            1229,
            out[0].len()
        );
        let out2 = resampler.process(&waves, None).unwrap();
        assert_eq!(out2.len(), 2, "Expected {} channels, got {}", 2, out2.len());
        assert!(
            out2[0].len() > 1226 && out2[0].len() < 1232,
            "expected {} - {} samples, got {}",
            1226,
            1232,
            out2[0].len()
        );
    }

    #[test]
    fn make_poly_resampler_fi_skipped() {
        let mut resampler =
            Async::<f64>::new_poly(1.2, 1.0, PolynomialDegree::Cubic, 1024, 2, Fixed::Input)
                .unwrap();
        let waves = vec![vec![0.0f64; 1024], Vec::new()];
        let mask = vec![true, false];
        let out = resampler.process(&waves, Some(&mask)).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].len() > 1150 && out[0].len() < 1250);
        assert!(out[1].is_empty());
        let waves = vec![Vec::new(), vec![0.0f64; 1024]];
        let mask = vec![false, true];
        let out = resampler.process(&waves, Some(&mask)).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[1].len() > 1150 && out[0].len() < 1250);
        assert!(out[0].is_empty());
    }

    #[test]
    fn make_poly_resampler_fi_downsample() {
        // Replicate settings from reported issue.
        let mut resampler = Async::<f64>::new_poly(
            16000 as f64 / 96000 as f64,
            1.0,
            PolynomialDegree::Cubic,
            1024,
            2,
            Fixed::Input,
        )
        .unwrap();
        let waves = vec![vec![0.0f64; 1024]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert!(
            out[0].len() > 140 && out[0].len() < 200,
            "expected {} - {} samples, got {}",
            140,
            200,
            out[0].len()
        );
        let out2 = resampler.process(&waves, None).unwrap();
        assert_eq!(out2.len(), 2, "Expected {} channels, got {}", 2, out2.len());
        assert!(
            out2[0].len() > 167 && out2[0].len() < 173,
            "expected {} - {} samples, got {}",
            167,
            173,
            out2[0].len()
        );
    }

    #[test]
    fn make_poly_resampler_fi_upsample() {
        // Replicate settings from reported issue.
        let mut resampler = Async::<f64>::new_poly(
            192000 as f64 / 44100 as f64,
            1.0,
            PolynomialDegree::Cubic,
            1024,
            2,
            Fixed::Input,
        )
        .unwrap();
        let waves = vec![vec![0.0f64; 1024]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert!(
            out[0].len() > 3800 && out[0].len() < 4458,
            "expected {} - {} samples, got {}",
            3800,
            4458,
            out[0].len()
        );
        let out2 = resampler.process(&waves, None).unwrap();
        assert_eq!(out2.len(), 2, "Expected {} channels, got {}", 2, out2.len());
        assert!(
            out2[0].len() > 4455 && out2[0].len() < 4461,
            "expected {} - {} samples, got {}",
            4455,
            4461,
            out2[0].len()
        );
    }

    #[test]
    fn make_poly_resampler_fo() {
        let mut resampler =
            Async::<f64>::new_poly(1.2, 1.0, PolynomialDegree::Cubic, 1024, 2, Fixed::Output)
                .unwrap();
        let frames = resampler.input_frames_next();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn reset_poly_resampler_fo() {
        let mut resampler =
            Async::<f64>::new_poly(1.2, 1.0, PolynomialDegree::Cubic, 1024, 2, Fixed::Output)
                .unwrap();
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
    fn make_poly_resampler_fo_32() {
        let mut resampler =
            Async::<f32>::new_poly(1.2, 1.0, PolynomialDegree::Cubic, 1024, 2, Fixed::Output)
                .unwrap();
        let frames = resampler.input_frames_next();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let waves = vec![vec![0.0f32; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn make_poly_resampler_fo_skipped() {
        let mut resampler =
            Async::<f64>::new_poly(1.2, 1.0, PolynomialDegree::Cubic, 1024, 2, Fixed::Output)
                .unwrap();
        let frames = resampler.input_frames_next();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let mut waves = vec![vec![0.0f64; frames], Vec::new()];
        let mask = vec![true, false];
        waves[0][100] = 3.0;
        let out = resampler.process(&waves, Some(&mask)).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
        assert!(out[1].is_empty());
        println!("{:?}", out[0]);
        let summed = out[0].iter().sum::<f64>();
        println!("sum: {}", summed);
        assert!(summed < 4.0);
        assert!(summed > 2.0);

        let frames = resampler.input_frames_next();
        let mut waves = vec![Vec::new(), vec![0.0f64; frames]];
        let mask = vec![false, true];
        waves[1][10] = 3.0;
        let out = resampler.process(&waves, Some(&mask)).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].len(), 1024);
        assert!(out[0].is_empty());
        let summed = out[1].iter().sum::<f64>();
        assert!(summed < 4.0);
        assert!(summed > 2.0);
    }

    #[test]
    fn make_poly_resampler_fo_downsample() {
        let mut resampler =
            Async::<f64>::new_poly(0.125, 1.0, PolynomialDegree::Cubic, 1024, 2, Fixed::Output)
                .unwrap();
        let frames = resampler.input_frames_next();
        println!("{}", frames);
        assert!(
            frames > 8192 && frames < 9000,
            "expected {}..{} samples, got {}",
            8192,
            9000,
            frames
        );
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert_eq!(
            out[0].len(),
            1024,
            "Expected {} frames, got {}",
            1024,
            out[0].len()
        );
        let frames2 = resampler.input_frames_next();
        assert!(
            frames2 > 8189 && frames2 < 8195,
            "expected {}..{} samples, got {}",
            8189,
            8195,
            frames2
        );
        let waves2 = vec![vec![0.0f64; frames2]; 2];
        let out2 = resampler.process(&waves2, None).unwrap();
        assert_eq!(
            out2[0].len(),
            1024,
            "Expected {} frames, got {}",
            1024,
            out2[0].len()
        );
    }

    #[test]
    fn make_poly_resampler_fo_upsample() {
        let mut resampler =
            Async::<f64>::new_poly(8.0, 1.0, PolynomialDegree::Cubic, 1024, 2, Fixed::Output)
                .unwrap();
        let frames = resampler.input_frames_next();
        println!("{}", frames);
        assert!(
            frames > 128 && frames < 300,
            "expected {}..{} samples, got {}",
            140,
            200,
            frames
        );
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert_eq!(
            out[0].len(),
            1024,
            "Expected {} frames, got {}",
            1024,
            out[0].len()
        );
        let frames2 = resampler.input_frames_next();
        assert!(
            frames2 > 125 && frames2 < 131,
            "expected {}..{} samples, got {}",
            125,
            131,
            frames2
        );
        let waves2 = vec![vec![0.0f64; frames2]; 2];
        let out2 = resampler.process(&waves2, None).unwrap();
        assert_eq!(
            out2[0].len(),
            1024,
            "Expected {} frames, got {}",
            1024,
            out2[0].len()
        );
    }

    #[test]
    fn check_poly_fo_output_up() {
        let mut resampler =
            Async::<f64>::new_poly(8.0, 1.0, PolynomialDegree::Cubic, 1024, 2, Fixed::Output)
                .unwrap();
        check_output!(resampler);
    }

    #[test]
    fn check_poly_fo_output_down() {
        let mut resampler =
            Async::<f64>::new_poly(0.8, 1.0, PolynomialDegree::Cubic, 1024, 2, Fixed::Output)
                .unwrap();
        check_output!(resampler);
    }

    #[test]
    fn check_poly_fi_output_up() {
        let mut resampler =
            Async::<f64>::new_poly(8.0, 1.0, PolynomialDegree::Cubic, 1024, 2, Fixed::Input)
                .unwrap();
        check_output!(resampler);
    }

    #[test]
    fn check_poly_fi_output_down() {
        let mut resampler =
            Async::<f64>::new_poly(0.8, 1.0, PolynomialDegree::Cubic, 1024, 2, Fixed::Input)
                .unwrap();
        check_output!(resampler);
    }

    #[test]
    fn resample_poly_small_fo_up() {
        let ratio = 96000.0 / 44100.0;
        let mut resampler =
            Async::<f32>::new_poly(ratio, 100.0, PolynomialDegree::Cubic, 1, 2, Fixed::Output)
                .unwrap();
        check_ratio!(resampler, ratio, 1000000);
    }

    #[test]
    fn resample_poly_big_fo_up() {
        let ratio = 96000.0 / 44100.0;
        let mut resampler = Async::<f32>::new_poly(
            ratio,
            100.0,
            PolynomialDegree::Cubic,
            1024,
            2,
            Fixed::Output,
        )
        .unwrap();
        check_ratio!(resampler, ratio, 1000);
    }

    #[test]
    fn resample_poly_small_fo_down() {
        let ratio = 44100.0 / 96000.0;
        let mut resampler =
            Async::<f32>::new_poly(ratio, 100.0, PolynomialDegree::Cubic, 1, 2, Fixed::Output)
                .unwrap();
        check_ratio!(resampler, ratio, 1000000);
    }

    #[test]
    fn resample_poly_big_fo_down() {
        let ratio = 44100.0 / 96000.0;
        let mut resampler = Async::<f32>::new_poly(
            ratio,
            100.0,
            PolynomialDegree::Cubic,
            1024,
            2,
            Fixed::Output,
        )
        .unwrap();
        check_ratio!(resampler, ratio, 1000);
    }

    #[test]
    fn resample_poly_small_fi_up() {
        let ratio = 96000.0 / 44100.0;
        let mut resampler =
            Async::<f32>::new_poly(ratio, 100.0, PolynomialDegree::Cubic, 1, 2, Fixed::Input)
                .unwrap();
        check_ratio!(resampler, ratio, 1000000);
    }

    #[test]
    fn resample_poly_big_fi_up() {
        let ratio = 96000.0 / 44100.0;
        let mut resampler =
            Async::<f32>::new_poly(ratio, 100.0, PolynomialDegree::Cubic, 1024, 2, Fixed::Input)
                .unwrap();
        check_ratio!(resampler, ratio, 1000);
    }

    #[test]
    fn resample_poly_small_fi_down() {
        let ratio = 44100.0 / 96000.0;
        let mut resampler =
            Async::<f32>::new_poly(ratio, 100.0, PolynomialDegree::Cubic, 1, 2, Fixed::Input)
                .unwrap();
        check_ratio!(resampler, ratio, 1000000);
    }

    #[test]
    fn resample_poly_big_fi_down() {
        let ratio = 44100.0 / 96000.0;
        let mut resampler =
            Async::<f32>::new_poly(ratio, 100.0, PolynomialDegree::Cubic, 1024, 2, Fixed::Input)
                .unwrap();
        check_ratio!(resampler, ratio, 1000);
    }

    #[test]
    fn check_poly_fo_output_resize() {
        let mut resampler =
            Async::<f64>::new_poly(1.2, 100.0, PolynomialDegree::Cubic, 1024, 2, Fixed::Output)
                .unwrap();
        assert_eq!(resampler.output_frames_next(), 1024);
        resampler.set_chunk_size(256).unwrap();
        assert_eq!(resampler.output_frames_next(), 256);
        check_output!(resampler);
    }

    #[test]
    fn check_poly_fi_output_resize() {
        let mut resampler =
            Async::<f64>::new_poly(1.2, 100.0, PolynomialDegree::Cubic, 1024, 2, Fixed::Input)
                .unwrap();
        assert_eq!(resampler.input_frames_next(), 1024);
        resampler.set_chunk_size(256).unwrap();
        assert_eq!(resampler.input_frames_next(), 256);
        check_output!(resampler);
    }

    // ------ Sinc tests ------
    use super::{interp_cubic, interp_lin};
    use crate::SincInterpolationParameters;
    use crate::SincInterpolationType;
    use crate::WindowFunction;

    fn basic_params() -> SincInterpolationParameters {
        SincInterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        }
    }

    #[test]
    fn int_cubic() {
        let params = basic_params();
        let _resampler = Async::<f64>::new_sinc(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        let yvals = [0.0f64, 2.0f64, 4.0f64, 6.0f64];
        let interp = interp_cubic(0.5f64, &yvals);
        assert_eq!(interp, 3.0f64);
    }

    #[test]
    fn int_lin_32() {
        let params = basic_params();
        let _resampler = Async::<f32>::new_sinc(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        let yvals = [1.0f32, 5.0f32];
        let interp = interp_lin(0.25f32, &yvals);
        assert_eq!(interp, 2.0f32);
    }

    #[test]
    fn int_cubic_32() {
        let params = basic_params();
        let _resampler = Async::<f32>::new_sinc(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        let yvals = [0.0f32, 2.0f32, 4.0f32, 6.0f32];
        let interp = interp_cubic(0.5f32, &yvals);
        assert_eq!(interp, 3.0f32);
    }

    #[test]
    fn int_lin() {
        let params = basic_params();
        let _resampler = Async::<f64>::new_sinc(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        let yvals = [1.0f64, 5.0f64];
        let interp = interp_lin(0.25f64, &yvals);
        assert_eq!(interp, 2.0f64);
    }

    #[test]
    fn make_sinc_resampler_fi() {
        let params = basic_params();
        let mut resampler =
            Async::<f64>::new_sinc(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        let waves = vec![vec![0.0f64; 1024]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert!(
            out[0].len() > 1150 && out[0].len() < 1229,
            "expected {} - {} samples, got {}",
            1150,
            1229,
            out[0].len()
        );
        let out2 = resampler.process(&waves, None).unwrap();
        assert_eq!(out2.len(), 2, "Expected {} channels, got {}", 2, out2.len());
        assert!(
            out2[0].len() > 1226 && out2[0].len() < 1232,
            "expected {} - {} samples, got {}",
            1226,
            1232,
            out2[0].len()
        );
    }

    #[test]
    fn reset_sinc_resampler_fi() {
        let params = basic_params();
        let mut resampler =
            Async::<f64>::new_sinc(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();

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
    fn make_sinc_resampler_fi_32() {
        let params = basic_params();
        let mut resampler =
            Async::<f32>::new_sinc(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        let waves = vec![vec![0.0f32; 1024]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert!(
            out[0].len() > 1150 && out[0].len() < 1229,
            "expected {} - {} samples, got {}",
            1150,
            1229,
            out[0].len()
        );
        let out2 = resampler.process(&waves, None).unwrap();
        assert_eq!(out2.len(), 2, "Expected {} channels, got {}", 2, out2.len());
        assert!(
            out2[0].len() > 1226 && out2[0].len() < 1232,
            "expected {} - {} samples, got {}",
            1226,
            1232,
            out2[0].len()
        );
    }

    #[test]
    fn make_sinc_resampler_fi_skipped() {
        let params = basic_params();
        let mut resampler =
            Async::<f64>::new_sinc(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        let waves = vec![vec![0.0f64; 1024], Vec::new()];
        let mask = vec![true, false];
        let out = resampler.process(&waves, Some(&mask)).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].len() > 1150 && out[0].len() < 1250);
        assert!(out[1].is_empty());
        let waves = vec![Vec::new(), vec![0.0f64; 1024]];
        let mask = vec![false, true];
        let out = resampler.process(&waves, Some(&mask)).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[1].len() > 1150 && out[0].len() < 1250);
        assert!(out[0].is_empty());
    }

    #[test]
    fn make_sinc_resampler_fi_downsample() {
        // Replicate settings from reported issue
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 160,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = Async::<f64>::new_sinc(
            16000 as f64 / 96000 as f64,
            1.0,
            params,
            1024,
            2,
            Fixed::Input,
        )
        .unwrap();
        let waves = vec![vec![0.0f64; 1024]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert!(
            out[0].len() > 140 && out[0].len() < 200,
            "expected {} - {} samples, got {}",
            140,
            200,
            out[0].len()
        );
        let out2 = resampler.process(&waves, None).unwrap();
        assert_eq!(out2.len(), 2, "Expected {} channels, got {}", 2, out2.len());
        assert!(
            out2[0].len() > 167 && out2[0].len() < 173,
            "expected {} - {} samples, got {}",
            167,
            173,
            out2[0].len()
        );
    }

    #[test]
    fn make_sinc_resampler_fi_upsample() {
        // Replicate settings from reported issue
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 160,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = Async::<f64>::new_sinc(
            192000 as f64 / 44100 as f64,
            1.0,
            params,
            1024,
            2,
            Fixed::Input,
        )
        .unwrap();
        let waves = vec![vec![0.0f64; 1024]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert!(
            out[0].len() > 3800 && out[0].len() < 4458,
            "expected {} - {} samples, got {}",
            3800,
            4458,
            out[0].len()
        );
        let out2 = resampler.process(&waves, None).unwrap();
        assert_eq!(out2.len(), 2, "Expected {} channels, got {}", 2, out2.len());
        assert!(
            out2[0].len() > 4455 && out2[0].len() < 4461,
            "expected {} - {} samples, got {}",
            4455,
            4461,
            out2[0].len()
        );
    }

    #[test]
    fn make_sinc_resampler_fo() {
        let params = basic_params();
        let mut resampler =
            Async::<f64>::new_sinc(1.2, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        let frames = resampler.input_frames_next();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn reset_sinc_resampler_fo() {
        let params = basic_params();
        let mut resampler =
            Async::<f64>::new_sinc(1.2, 1.0, params, 1024, 2, Fixed::Output).unwrap();
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
    fn make_sinc_resampler_fo_32() {
        let params = basic_params();
        let mut resampler =
            Async::<f32>::new_sinc(1.2, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        let frames = resampler.input_frames_next();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let waves = vec![vec![0.0f32; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn make_sinc_resampler_fo_skipped() {
        let params = basic_params();
        let mut resampler =
            Async::<f64>::new_sinc(1.2, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        let frames = resampler.input_frames_next();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let mut waves = vec![vec![0.0f64; frames], Vec::new()];
        let mask = vec![true, false];
        waves[0][100] = 3.0;
        let out = resampler.process(&waves, Some(&mask)).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
        assert!(out[1].is_empty());
        println!("{:?}", out[0]);
        let summed = out[0].iter().sum::<f64>();
        println!("sum: {}", summed);
        assert!(summed < 4.0);
        assert!(summed > 2.0);

        let frames = resampler.input_frames_next();
        let mut waves = vec![Vec::new(), vec![0.0f64; frames]];
        let mask = vec![false, true];
        waves[1][10] = 3.0;
        let out = resampler.process(&waves, Some(&mask)).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].len(), 1024);
        assert!(out[0].is_empty());
        let summed = out[1].iter().sum::<f64>();
        assert!(summed < 4.0);
        assert!(summed > 2.0);
    }

    #[test]
    fn make_sinc_resampler_fo_downsample() {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 160,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler =
            Async::<f64>::new_sinc(0.125, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        let frames = resampler.input_frames_next();
        println!("{}", frames);
        assert!(
            frames > 8192 && frames < 9000,
            "expected {}..{} samples, got {}",
            8192,
            9000,
            frames
        );
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert_eq!(
            out[0].len(),
            1024,
            "Expected {} frames, got {}",
            1024,
            out[0].len()
        );
        let frames2 = resampler.input_frames_next();
        assert!(
            frames2 > 8189 && frames2 < 8195,
            "expected {}..{} samples, got {}",
            8189,
            8195,
            frames2
        );
        let waves2 = vec![vec![0.0f64; frames2]; 2];
        let out2 = resampler.process(&waves2, None).unwrap();
        assert_eq!(
            out2[0].len(),
            1024,
            "Expected {} frames, got {}",
            1024,
            out2[0].len()
        );
    }

    #[test]
    fn make_sinc_resampler_fo_upsample() {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 160,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler =
            Async::<f64>::new_sinc(8.0, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        let frames = resampler.input_frames_next();
        println!("{}", frames);
        assert!(
            frames > 128 && frames < 300,
            "expected {}..{} samples, got {}",
            140,
            200,
            frames
        );
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert_eq!(
            out[0].len(),
            1024,
            "Expected {} frames, got {}",
            1024,
            out[0].len()
        );
        let frames2 = resampler.input_frames_next();
        assert!(
            frames2 > 125 && frames2 < 131,
            "expected {}..{} samples, got {}",
            125,
            131,
            frames2
        );
        let waves2 = vec![vec![0.0f64; frames2]; 2];
        let out2 = resampler.process(&waves2, None).unwrap();
        assert_eq!(
            out2[0].len(),
            1024,
            "Expected {} frames, got {}",
            1024,
            out2[0].len()
        );
    }

    #[test]
    fn check_sinc_fo_output_up() {
        let params = basic_params();
        let mut resampler =
            Async::<f64>::new_sinc(1.2, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        check_output!(resampler);
    }

    #[test]
    fn check_sinc_fo_output_down() {
        let params = basic_params();
        let mut resampler =
            Async::<f64>::new_sinc(0.8, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        check_output!(resampler);
    }

    #[test]
    fn check_sinc_fi_output_up() {
        let params = basic_params();
        let mut resampler =
            Async::<f64>::new_sinc(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        check_output!(resampler);
    }

    #[test]
    fn check_sinc_fi_output_down() {
        let params = basic_params();
        let mut resampler =
            Async::<f64>::new_sinc(0.8, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        check_output!(resampler);
    }

    #[test]
    fn resample_sinc_small_fo_up() {
        let ratio = 96000.0 / 44100.0;
        let params = basic_params();
        let mut resampler =
            Async::<f32>::new_sinc(ratio, 1.0, params, 1, 2, Fixed::Output).unwrap();
        check_ratio!(resampler, ratio, 100000);
    }

    #[test]
    fn resample_sinc_big_fo_up() {
        let ratio = 96000.0 / 44100.0;
        let params = basic_params();
        let mut resampler =
            Async::<f32>::new_sinc(ratio, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        check_ratio!(resampler, ratio, 100);
    }

    #[test]
    fn resample_sinc_small_fo_down() {
        let ratio = 44100.0 / 96000.0;
        let params = basic_params();
        let mut resampler =
            Async::<f32>::new_sinc(ratio, 1.0, params, 1, 2, Fixed::Output).unwrap();
        check_ratio!(resampler, ratio, 100000);
    }

    #[test]
    fn resample_sinc_big_fo_down() {
        let ratio = 44100.0 / 96000.0;
        let params = basic_params();
        let mut resampler =
            Async::<f32>::new_sinc(ratio, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        check_ratio!(resampler, ratio, 100);
    }

    #[test]
    fn resample_sinc_small_fi_up() {
        let ratio = 96000.0 / 44100.0;
        let params = basic_params();
        let mut resampler = Async::<f32>::new_sinc(ratio, 1.0, params, 1, 2, Fixed::Input).unwrap();
        check_ratio!(resampler, ratio, 100000);
    }

    #[test]
    fn resample_sinc_big_fi_up() {
        let ratio = 96000.0 / 44100.0;
        let params = basic_params();
        let mut resampler =
            Async::<f32>::new_sinc(ratio, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        check_ratio!(resampler, ratio, 100);
    }

    #[test]
    fn resample_sinc_small_fi_down() {
        let ratio = 44100.0 / 96000.0;
        let params = basic_params();
        let mut resampler = Async::<f32>::new_sinc(ratio, 1.0, params, 1, 2, Fixed::Input).unwrap();
        check_ratio!(resampler, ratio, 100000);
    }

    #[test]
    fn resample_sinc_big_fi_down() {
        let ratio = 44100.0 / 96000.0;
        let params = basic_params();
        let mut resampler =
            Async::<f32>::new_sinc(ratio, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        check_ratio!(resampler, ratio, 100);
    }

    #[test]
    fn check_sinc_fo_output_resize() {
        let params = basic_params();
        let mut resampler =
            Async::<f64>::new_sinc(1.2, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        assert_eq!(resampler.output_frames_next(), 1024);
        resampler.set_chunk_size(256).unwrap();
        assert_eq!(resampler.output_frames_next(), 256);
        check_output!(resampler);
    }

    #[test]
    fn check_sinc_fi_output_resize() {
        let params = basic_params();
        let mut resampler =
            Async::<f64>::new_sinc(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        assert_eq!(resampler.input_frames_next(), 1024);
        resampler.set_chunk_size(256).unwrap();
        assert_eq!(resampler.input_frames_next(), 256);
        check_output!(resampler);
    }
}
