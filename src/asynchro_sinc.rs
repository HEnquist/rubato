use crate::error::{ResampleError, ResampleResult, ResamplerConstructionError};
use crate::interpolation::*;
use crate::sinc_interpolator::{ScalarInterpolator, SincInterpolator};
#[cfg(target_arch = "x86_64")]
use crate::sinc_interpolator_avx::AvxInterpolator;
#[cfg(target_arch = "aarch64")]
use crate::sinc_interpolator_neon::NeonInterpolator;
#[cfg(target_arch = "x86_64")]
use crate::sinc_interpolator_sse::SseInterpolator;
use crate::windows::WindowFunction;
use crate::{update_mask_from_buffers, validate_buffers, Resampler, Sample};
use crate::{InterpolationParameters, InterpolationType};

/// An asynchronous resampler that accepts a fixed number of audio frames for input
/// and returns a variable number of frames.
///
/// The resampling is done by creating a number of intermediate points (defined by oversampling_factor)
/// by sinc interpolation. The new samples are then calculated by interpolating between these points.
///
/// The resampling ratio can be freely adjusted within the range specified to the constructor.
/// Adjusting the ratio does not recalculate the sinc functions used by the anti-aliasing filter.
/// This causes no issue when increasing the ratio (which slows down the output).
/// However when decreasing more than a few percent (or speeding up the output),
/// the filters can no longer suppress all aliasing and this may lead to some artefacts.
/// Higher maximum ratios require more memory to be allocated by [Resampler::output_buffer_allocate].
pub struct SincFixedIn<T> {
    nbr_channels: usize,
    chunk_size: usize,
    last_index: f64,
    resample_ratio: f64,
    resample_ratio_original: f64,
    target_ratio: f64,
    max_relative_ratio: f64,
    interpolator: Box<dyn SincInterpolator<T>>,
    buffer: Vec<Vec<T>>,
    interpolation: InterpolationType,
    channel_mask: Vec<bool>,
}

/// An asynchronous resampler that return a fixed number of audio frames.
/// The number of input frames required is given by the
/// [input_frames_next](Resampler::input_frames_next) function.
///
/// The resampling is done by creating a number of intermediate points (defined by oversampling_factor)
/// by sinc interpolation. The new samples are then calculated by interpolating between these points.
///
/// The resampling ratio can be freely adjusted within the range specified to the constructor.
/// Adjusting the ratio does not recalculate the sinc functions used by the anti-aliasing filter.
/// This causes no issue when increasing the ratio (which slows down the output).
/// However when decreasing more than a few percent (i.e. speeding up the output),
/// the filters can no longer suppress all aliasing and this may lead to some artefacts.
/// Higher maximum ratios require more memory to be allocated by
/// [input_buffer_allocate](Resampler::input_buffer_allocate) and an internal buffer.
pub struct SincFixedOut<T> {
    nbr_channels: usize,
    chunk_size: usize,
    needed_input_size: usize,
    last_index: f64,
    current_buffer_fill: usize,
    resample_ratio: f64,
    resample_ratio_original: f64,
    max_relative_ratio: f64,
    interpolator: Box<dyn SincInterpolator<T>>,
    buffer: Vec<Vec<T>>,
    interpolation: InterpolationType,
    channel_mask: Vec<bool>,
}

pub fn make_interpolator<T>(
    sinc_len: usize,
    resample_ratio: f64,
    f_cutoff: f32,
    oversampling_factor: usize,
    window: WindowFunction,
) -> Box<dyn SincInterpolator<T>>
where
    T: Sample,
{
    let sinc_len = 8 * (((sinc_len as f32) / 8.0).ceil() as usize);
    let f_cutoff = if resample_ratio >= 1.0 {
        f_cutoff
    } else {
        f_cutoff * resample_ratio as f32
    };

    #[cfg(target_arch = "x86_64")]
    if let Ok(interpolator) =
        AvxInterpolator::<T>::new(sinc_len, oversampling_factor, f_cutoff, window)
    {
        return Box::new(interpolator);
    }

    #[cfg(target_arch = "x86_64")]
    if let Ok(interpolator) =
        SseInterpolator::<T>::new(sinc_len, oversampling_factor, f_cutoff, window)
    {
        return Box::new(interpolator);
    }

    #[cfg(target_arch = "aarch64")]
    if let Ok(interpolator) =
        NeonInterpolator::<T>::new(sinc_len, oversampling_factor, f_cutoff, window)
    {
        return Box::new(interpolator);
    }

    Box::new(ScalarInterpolator::<T>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ))
}

/// Perform cubic polynomial interpolation to get value at x.
/// Input points are assumed to be at x = -1, 0, 1, 2
fn interp_cubic<T>(x: T, yvals: &[T; 4]) -> T
where
    T: Sample,
{
    let a0 = yvals[1];
    let a1 = -(T::one() / T::coerce(3.0)) * yvals[0] - T::coerce(0.5) * yvals[1] + yvals[2]
        - (T::one() / T::coerce(6.0)) * yvals[3];
    let a2 = T::coerce(0.5) * (yvals[0] + yvals[2]) - yvals[1];
    let a3 = T::coerce(0.5) * (yvals[1] - yvals[2])
        + (T::one() / T::coerce(6.0)) * (yvals[3] - yvals[0]);
    let x2 = x * x;
    let x3 = x2 * x;
    a0 + a1 * x + a2 * x2 + a3 * x3
}

/// Linear interpolation between two points at x=0 and x=1
fn interp_lin<T>(x: T, yvals: &[T; 2]) -> T
where
    T: Sample,
{
    yvals[0] + x * (yvals[1] - yvals[0])
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

impl<T> SincFixedIn<T>
where
    T: Sample,
{
    /// Create a new SincFixedIn
    ///
    /// Parameters are:
    /// - `resample_ratio`: Starting ratio between output and input sample rates, must be > 0.
    /// - `max_resample_ratio_relative`: Maximum ratio that can be set with [Resampler::set_resample_ratio] relative to `resample_ratio`, must be >= 1.0. The minimum relative ratio is the reciprocal of the maximum. For example, with `max_resample_ratio_relative` of 10.0, the ratio can be set between `resample_ratio * 10.0` and `resample_ratio / 10.0`.
    /// - `parameters`: Parameters for interpolation, see `InterpolationParameters`.
    /// - `chunk_size`: Size of input data in frames.
    /// - `nbr_channels`: Number of channels in input/output.
    pub fn new(
        resample_ratio: f64,
        max_resample_ratio_relative: f64,
        parameters: InterpolationParameters,
        chunk_size: usize,
        nbr_channels: usize,
    ) -> Result<Self, ResamplerConstructionError> {
        debug!(
            "Create new SincFixedIn, ratio: {}, chunk_size: {}, channels: {}, parameters: {:?}",
            resample_ratio, chunk_size, nbr_channels, parameters
        );

        let interpolator = make_interpolator(
            parameters.sinc_len,
            resample_ratio,
            parameters.f_cutoff,
            parameters.oversampling_factor,
            parameters.window,
        );

        Self::new_with_interpolator(
            resample_ratio,
            max_resample_ratio_relative,
            parameters.interpolation,
            interpolator,
            chunk_size,
            nbr_channels,
        )
    }

    /// Create a new SincFixedIn using an existing Interpolator
    ///
    /// Parameters are:
    /// - `resample_ratio`: Starting ratio between output and input sample rates, must be > 0.
    /// - `max_resample_ratio_relative`: Maximum ratio that can be set with [Resampler::set_resample_ratio] relative to `resample_ratio`, must be >= 1.0. The minimum relative ratio is the reciprocal of the maximum. For example, with `max_resample_ratio_relative` of 10.0, the ratio can be set between `resample_ratio` * 10.0 and `resample_ratio` / 10.0.
    /// - `interpolation_type`: Parameters for interpolation, see `InterpolationParameters`.
    /// - `interpolator`: The interpolator to use.
    /// - `chunk_size`: Size of output data in frames.
    /// - `nbr_channels`: Number of channels in input/output.
    pub fn new_with_interpolator(
        resample_ratio: f64,
        max_resample_ratio_relative: f64,
        interpolation_type: InterpolationType,
        interpolator: Box<dyn SincInterpolator<T>>,
        chunk_size: usize,
        nbr_channels: usize,
    ) -> Result<Self, ResamplerConstructionError> {
        validate_ratios(resample_ratio, max_resample_ratio_relative)?;
        let buffer = vec![vec![T::zero(); chunk_size + 2 * interpolator.len()]; nbr_channels];

        let channel_mask = vec![true; nbr_channels];

        Ok(SincFixedIn {
            nbr_channels,
            chunk_size,
            last_index: -((interpolator.len() / 2) as f64),
            resample_ratio,
            resample_ratio_original: resample_ratio,
            target_ratio: resample_ratio,
            max_relative_ratio: max_resample_ratio_relative,
            interpolator,
            buffer,
            interpolation: interpolation_type,
            channel_mask,
        })
    }
}

impl<T> Resampler<T> for SincFixedIn<T>
where
    T: Sample,
{
    fn process_into_buffer<V: AsRef<[T]>>(
        &mut self,
        wave_in: &[V],
        wave_out: &mut [Vec<T>],
        active_channels_mask: Option<&[bool]>,
    ) -> ResampleResult<()> {
        if let Some(mask) = active_channels_mask {
            self.channel_mask.copy_from_slice(mask);
        } else {
            update_mask_from_buffers(wave_in, &mut self.channel_mask);
        };

        validate_buffers(
            wave_in,
            wave_out,
            &self.channel_mask,
            self.nbr_channels,
            self.chunk_size,
        )?;

        let sinc_len = self.interpolator.len();
        let oversampling_factor = self.interpolator.nbr_sincs();
        let mut t_ratio = 1.0 / self.resample_ratio as f64;
        let t_ratio_end = 1.0 / self.target_ratio as f64;
        let approximate_nbr_frames =
            self.chunk_size as f64 * (0.5 * self.resample_ratio + 0.5 * self.target_ratio);
        let t_ratio_increment = (t_ratio_end - t_ratio) / approximate_nbr_frames;
        let end_idx =
            self.chunk_size as isize - (sinc_len as isize + 1) - t_ratio_end.ceil() as isize;

        //update buffer with new data
        for buf in self.buffer.iter_mut() {
            buf.copy_within(self.chunk_size..self.chunk_size + 2 * sinc_len, 0);
        }

        let needed_len = (self.chunk_size as f64
            * (0.5 * self.resample_ratio + 0.5 * self.target_ratio)
            + 10.0) as usize;
        for (chan, active) in self.channel_mask.iter().enumerate() {
            if *active {
                self.buffer[chan][2 * sinc_len..2 * sinc_len + self.chunk_size]
                    .copy_from_slice(wave_in[chan].as_ref());
                // Set length to chunksize*ratio plus a safety margin of 10 elements.
                if needed_len > wave_out[chan].capacity() {
                    trace!(
                        "Allocating more space for channel {}, old capacity: {}, new: {}",
                        chan,
                        wave_out[chan].capacity(),
                        needed_len
                    );
                }
                wave_out[chan].resize(needed_len, T::zero());
            }
        }

        let mut idx = self.last_index;

        let mut n = 0;

        match self.interpolation {
            InterpolationType::Cubic => {
                let mut points = [T::zero(); 4];
                let mut nearest = [(0isize, 0isize); 4];
                while idx < end_idx as f64 {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    get_nearest_times_4(idx, oversampling_factor as isize, &mut nearest);
                    let frac = idx * oversampling_factor as f64
                        - (idx * oversampling_factor as f64).floor();
                    let frac_offset = T::coerce(frac);
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            let buf = &self.buffer[chan];
                            for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                *p = self.interpolator.get_sinc_interpolated(
                                    buf,
                                    (n.0 + 2 * sinc_len as isize) as usize,
                                    n.1 as usize,
                                );
                            }
                            wave_out[chan][n] = interp_cubic(frac_offset, &points);
                        }
                    }
                    n += 1;
                }
            }
            InterpolationType::Linear => {
                let mut points = [T::zero(); 2];
                let mut nearest = [(0isize, 0isize); 2];
                while idx < end_idx as f64 {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    get_nearest_times_2(idx, oversampling_factor as isize, &mut nearest);
                    let frac = idx * oversampling_factor as f64
                        - (idx * oversampling_factor as f64).floor();
                    let frac_offset = T::coerce(frac);
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            let buf = &self.buffer[chan];
                            for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                *p = self.interpolator.get_sinc_interpolated(
                                    buf,
                                    (n.0 + 2 * sinc_len as isize) as usize,
                                    n.1 as usize,
                                );
                            }
                            wave_out[chan][n] = interp_lin(frac_offset, &points);
                        }
                    }
                    n += 1;
                }
            }
            InterpolationType::Nearest => {
                let mut point;
                let mut nearest;
                while idx < end_idx as f64 {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    nearest = get_nearest_time(idx, oversampling_factor as isize);
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            let buf = &self.buffer[chan];
                            point = self.interpolator.get_sinc_interpolated(
                                buf,
                                (nearest.0 + 2 * sinc_len as isize) as usize,
                                nearest.1 as usize,
                            );
                            wave_out[chan][n] = point;
                        }
                    }
                    n += 1;
                }
            }
        }

        // store last index for next iteration
        self.last_index = idx - self.chunk_size as f64;
        for (chan, active) in self.channel_mask.iter().enumerate() {
            if *active {
                wave_out[chan].truncate(n);
            }
        }
        self.resample_ratio = self.target_ratio;
        trace!(
            "Resampling channels {:?}, {} frames in, {} frames out",
            active_channels_mask,
            self.chunk_size,
            n,
        );
        Ok(())
    }

    fn output_frames_max(&self) -> usize {
        // Set length to chunksize*ratio plus a safety margin of 10 elements.
        (self.chunk_size as f64 * self.resample_ratio_original * self.max_relative_ratio + 10.0)
            as usize
    }

    fn output_frames_next(&self) -> usize {
        (self.chunk_size as f64 * (0.5 * self.resample_ratio + 0.5 * self.target_ratio) + 10.0)
            as usize
    }

    fn nbr_channels(&self) -> usize {
        self.nbr_channels
    }

    fn input_frames_max(&self) -> usize {
        self.chunk_size
    }

    fn input_frames_next(&self) -> usize {
        self.chunk_size
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
}

impl<T> SincFixedOut<T>
where
    T: Sample,
{
    /// Create a new SincFixedOut
    ///
    /// Parameters are:
    /// - `resample_ratio`: Starting ratio between output and input sample rates, must be > 0.
    /// - `max_resample_ratio_relative`: Maximum ratio that can be set with [Resampler::set_resample_ratio] relative to `resample_ratio`, must be >= 1.0. The minimum relative ratio is the reciprocal of the maximum. For example, with `max_resample_ratio_relative` of 10.0, the ratio can be set between `resample_ratio * 10.0` and `resample_ratio / 10.0`.
    /// - `parameters`: Parameters for interpolation, see `InterpolationParameters`.
    /// - `chunk_size`: Size of output data in frames.
    /// - `nbr_channels`: Number of channels in input/output.
    pub fn new(
        resample_ratio: f64,
        max_resample_ratio_relative: f64,
        parameters: InterpolationParameters,
        chunk_size: usize,
        nbr_channels: usize,
    ) -> Result<Self, ResamplerConstructionError> {
        debug!(
            "Create new SincFixedIn, ratio: {}, chunk_size: {}, channels: {}, parameters: {:?}",
            resample_ratio, chunk_size, nbr_channels, parameters
        );
        let interpolator = make_interpolator(
            parameters.sinc_len,
            resample_ratio,
            parameters.f_cutoff,
            parameters.oversampling_factor,
            parameters.window,
        );

        Self::new_with_interpolator(
            resample_ratio,
            max_resample_ratio_relative,
            parameters.interpolation,
            interpolator,
            chunk_size,
            nbr_channels,
        )
    }

    /// Create a new SincFixedOut using an existing Interpolator
    ///
    /// Parameters are:
    /// - `resample_ratio`: Starting ratio between output and input sample rates, must be > 0.
    /// - `max_resample_ratio_relative`: Maximum ratio that can be set with [Resampler::set_resample_ratio] relative to `resample_ratio`, must be >= 1.0. The minimum relative ratio is the reciprocal of the maximum. For example, with `max_resample_ratio_relative` of 10.0, the ratio can be set between `resample_ratio` * 10.0 and `resample_ratio` / 10.0.
    /// - `interpolation_type`: Parameters for interpolation, see `InterpolationParameters`.
    /// - `interpolator`: The interpolator to use.
    /// - `chunk_size`: Size of output data in frames.
    /// - `nbr_channels`: Number of channels in input/output.
    pub fn new_with_interpolator(
        resample_ratio: f64,
        max_resample_ratio_relative: f64,
        interpolation_type: InterpolationType,
        interpolator: Box<dyn SincInterpolator<T>>,
        chunk_size: usize,
        nbr_channels: usize,
    ) -> Result<Self, ResamplerConstructionError> {
        validate_ratios(resample_ratio, max_resample_ratio_relative)?;

        let needed_input_size =
            (chunk_size as f64 / resample_ratio).ceil() as usize + 2 + interpolator.len() / 2;
        let buffer_channel_length = ((max_resample_ratio_relative + 1.0) * needed_input_size as f64)
            as usize
            + 2 * interpolator.len();
        let buffer = vec![vec![T::zero(); buffer_channel_length]; nbr_channels];
        let channel_mask = vec![true; nbr_channels];

        Ok(SincFixedOut {
            nbr_channels,
            chunk_size,
            needed_input_size,
            last_index: -((interpolator.len() / 2) as f64),
            current_buffer_fill: needed_input_size,
            resample_ratio,
            resample_ratio_original: resample_ratio,
            max_relative_ratio: max_resample_ratio_relative,
            interpolator,
            buffer,
            interpolation: interpolation_type,
            channel_mask,
        })
    }
}

impl<T> Resampler<T> for SincFixedOut<T>
where
    T: Sample,
{
    fn process_into_buffer<V: AsRef<[T]>>(
        &mut self,
        wave_in: &[V],
        wave_out: &mut [Vec<T>],
        active_channels_mask: Option<&[bool]>,
    ) -> ResampleResult<()> {
        if let Some(mask) = active_channels_mask {
            self.channel_mask.copy_from_slice(mask);
        } else {
            update_mask_from_buffers(wave_in, &mut self.channel_mask);
        };

        validate_buffers(
            wave_in,
            wave_out,
            &self.channel_mask,
            self.nbr_channels,
            self.needed_input_size,
        )?;
        let sinc_len = self.interpolator.len();
        let oversampling_factor = self.interpolator.nbr_sincs();

        for buf in self.buffer.iter_mut() {
            buf.copy_within(
                self.current_buffer_fill..self.current_buffer_fill + 2 * sinc_len,
                0,
            );
        }
        self.current_buffer_fill = self.needed_input_size;

        for (chan, active) in self.channel_mask.iter().enumerate() {
            if *active {
                self.buffer[chan][2 * sinc_len..2 * sinc_len + wave_in[chan].as_ref().len()]
                    .copy_from_slice(wave_in[chan].as_ref());
                if self.chunk_size > wave_out[chan].capacity() {
                    trace!(
                        "Allocating more space for channel {}, old capacity: {}, new: {}",
                        chan,
                        wave_out[chan].capacity(),
                        self.chunk_size
                    );
                }
                wave_out[chan].resize(self.chunk_size, T::zero());
            }
        }

        let mut idx = self.last_index;
        let t_ratio = 1.0 / self.resample_ratio as f64;

        match self.interpolation {
            InterpolationType::Cubic => {
                let mut points = [T::zero(); 4];
                let mut nearest = [(0isize, 0isize); 4];
                for n in 0..self.chunk_size {
                    idx += t_ratio;
                    get_nearest_times_4(idx, oversampling_factor as isize, &mut nearest);
                    let frac = idx * oversampling_factor as f64
                        - (idx * oversampling_factor as f64).floor();
                    let frac_offset = T::coerce(frac);
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            let buf = &self.buffer[chan];
                            for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                *p = self.interpolator.get_sinc_interpolated(
                                    buf,
                                    (n.0 + 2 * sinc_len as isize) as usize,
                                    n.1 as usize,
                                );
                            }
                            wave_out[chan][n] = interp_cubic(frac_offset, &points);
                        }
                    }
                }
            }
            InterpolationType::Linear => {
                let mut points = [T::zero(); 2];
                let mut nearest = [(0isize, 0isize); 2];
                for n in 0..self.chunk_size {
                    idx += t_ratio;
                    get_nearest_times_2(idx, oversampling_factor as isize, &mut nearest);
                    let frac = idx * oversampling_factor as f64
                        - (idx * oversampling_factor as f64).floor();
                    let frac_offset = T::coerce(frac);
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            let buf = &self.buffer[chan];
                            for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                *p = self.interpolator.get_sinc_interpolated(
                                    buf,
                                    (n.0 + 2 * sinc_len as isize) as usize,
                                    n.1 as usize,
                                );
                            }
                            wave_out[chan][n] = interp_lin(frac_offset, &points);
                        }
                    }
                }
            }
            InterpolationType::Nearest => {
                let mut point;
                let mut nearest;
                for n in 0..self.chunk_size {
                    idx += t_ratio;
                    nearest = get_nearest_time(idx, oversampling_factor as isize);
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            let buf = &self.buffer[chan];
                            point = self.interpolator.get_sinc_interpolated(
                                buf,
                                (nearest.0 + 2 * sinc_len as isize) as usize,
                                nearest.1 as usize,
                            );
                            wave_out[chan][n] = point;
                        }
                    }
                }
            }
        }

        // store last index for next iteration
        self.last_index = idx - self.current_buffer_fill as f64;
        self.needed_input_size = (self.last_index as f32
            + self.chunk_size as f32 / self.resample_ratio as f32
            + sinc_len as f32)
            .ceil() as usize
            + 2;
        trace!(
            "Resampling channels {:?}, {} frames in, {} frames out. Next needed length: {} frames, last index {}",
            active_channels_mask,
            self.current_buffer_fill,
            self.chunk_size,
            self.needed_input_size,
            self.last_index
        );
        Ok(())
    }

    fn input_frames_max(&self) -> usize {
        (self.chunk_size as f64 * self.resample_ratio_original * self.max_relative_ratio).ceil()
            as usize
            + 2
            + self.interpolator.len() / 2
    }

    fn input_frames_next(&self) -> usize {
        self.needed_input_size
    }

    fn nbr_channels(&self) -> usize {
        self.nbr_channels
    }

    fn output_frames_max(&self) -> usize {
        self.chunk_size
    }

    fn output_frames_next(&self) -> usize {
        self.chunk_size
    }

    fn set_resample_ratio(&mut self, new_ratio: f64, ramp: bool) -> ResampleResult<()> {
        trace!("Change resample ratio to {}", new_ratio);
        if (new_ratio / self.resample_ratio_original >= 1.0 / self.max_relative_ratio)
            && (new_ratio / self.resample_ratio_original <= self.max_relative_ratio)
        {
            self.resample_ratio = new_ratio;
            self.needed_input_size = (self.last_index as f32
                + self.chunk_size as f32 / self.resample_ratio as f32
                + self.interpolator.len() as f32)
                .ceil() as usize
                + 2;
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
}

#[cfg(test)]
mod tests {
    use super::{interp_cubic, interp_lin};
    use crate::InterpolationParameters;
    use crate::InterpolationType;
    use crate::Resampler;
    use crate::WindowFunction;
    use crate::{SincFixedIn, SincFixedOut};

    #[test]
    fn int_cubic() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let _resampler = SincFixedIn::<f64>::new(1.2, 1.0, params, 1024, 2).unwrap();
        let yvals = [0.0f64, 2.0f64, 4.0f64, 6.0f64];
        let interp = interp_cubic(0.5f64, &yvals);
        assert_eq!(interp, 3.0f64);
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
        let _resampler = SincFixedIn::<f32>::new(1.2, 1.0, params, 1024, 2).unwrap();
        let yvals = [1.0f32, 5.0f32];
        let interp = interp_lin(0.25f32, &yvals);
        assert_eq!(interp, 2.0f32);
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
        let _resampler = SincFixedIn::<f32>::new(1.2, 1.0, params, 1024, 2).unwrap();
        let yvals = [0.0f32, 2.0f32, 4.0f32, 6.0f32];
        let interp = interp_cubic(0.5f32, &yvals);
        assert_eq!(interp, 3.0f32);
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
        let _resampler = SincFixedIn::<f64>::new(1.2, 1.0, params, 1024, 2).unwrap();
        let yvals = [1.0f64, 5.0f64];
        let interp = interp_lin(0.25f64, &yvals);
        assert_eq!(interp, 2.0f64);
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
        let mut resampler = SincFixedIn::<f64>::new(1.2, 1.0, params, 1024, 2).unwrap();
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
    fn make_resampler_fi_32() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedIn::<f32>::new(1.2, 1.0, params, 1024, 2).unwrap();
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
    fn make_resampler_fi_skipped() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedIn::<f64>::new(1.2, 1.0, params, 1024, 2).unwrap();
        let waves = vec![vec![0.0f64; 1024], Vec::new()];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].len() > 1150 && out[0].len() < 1250);
        assert!(out[1].is_empty());
        let waves = vec![Vec::new(), vec![0.0f64; 1024]];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[1].len() > 1150 && out[0].len() < 1250);
        assert!(out[0].is_empty());
    }

    #[test]
    fn make_resampler_fi_downsample() {
        // Replicate settings from reported issue
        let params = InterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 160,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler =
            SincFixedIn::<f64>::new(16000 as f64 / 96000 as f64, 1.0, params, 1024, 2).unwrap();
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
    fn make_resampler_fi_upsample() {
        // Replicate settings from reported issue
        let params = InterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 160,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler =
            SincFixedIn::<f64>::new(192000 as f64 / 44100 as f64, 1.0, params, 1024, 2).unwrap();
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
    fn make_resampler_fo() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedOut::<f64>::new(1.2, 1.0, params, 1024, 2).unwrap();
        let frames = resampler.input_frames_next();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
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
        let mut resampler = SincFixedOut::<f32>::new(1.2, 1.0, params, 1024, 2).unwrap();
        let frames = resampler.input_frames_next();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let waves = vec![vec![0.0f32; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
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
        let mut resampler = SincFixedOut::<f64>::new(1.2, 1.0, params, 1024, 2).unwrap();
        let frames = resampler.input_frames_next();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let mut waves = vec![vec![0.0f64; frames], Vec::new()];
        waves[0][100] = 3.0;
        let out = resampler.process(&waves, None).unwrap();
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
        waves[1][10] = 3.0;
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].len(), 1024);
        assert!(out[0].is_empty());
        let summed = out[1].iter().sum::<f64>();
        assert!(summed < 4.0);
        assert!(summed > 2.0);
    }

    #[test]
    fn make_resampler_fo_downsample() {
        let params = InterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 160,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedOut::<f64>::new(0.125, 1.0, params, 1024, 2).unwrap();
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
    fn make_resampler_fo_upsample() {
        let params = InterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 160,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedOut::<f64>::new(8.0, 1.0, params, 1024, 2).unwrap();
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
}
