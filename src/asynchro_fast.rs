use crate::error::{ResampleError, ResampleResult, ResamplerConstructionError};
use crate::{update_mask_from_buffers, validate_buffers, Resampler, Sample};

const POLYNOMIAL_LEN_U: usize = 8;
const POLYNOMIAL_LEN_I: isize = 8;

macro_rules! t {
    // Shorter form of T::coerce(value)
    ($expression:expr) => {
        T::coerce($expression)
    };
}

/// Degree of the polynomial used for interpolation.
/// A higher degree gives a higher quality result, while taking longer to compute.
#[derive(Debug)]
pub enum PolynomialDegree {
    /// Septic polynomial, fitted using 8 sample points.
    Septic,
    /// Quintic polynomial, fitted using 6 sample points.
    Quintic,
    /// Cubic polynomial, fitted using 4 sample points.
    Cubic,
    /// Linear polynomial, fitted using 2 sample points.
    Linear,
    /// Nearest, uses the nearest sample point without any fitting.
    Nearest,
}

/// An asynchronous resampler that accepts a fixed number of audio frames for input
/// and returns a variable number of frames.
///
/// The resampling is done by interpolating between the input samples by fitting polynomials.
/// The polynomial degree can selected, see [PolynomialDegree] for the available options.
///
/// Note that no anti-aliasing filter is used.
/// This makes it run considerably faster than the corresponding SincFixedIn, which performs anti-aliasing filtering.
/// The price is that the resampling creates some artefacts in the output, mainly at higher frequencies.
/// Use SincFixedIn if this can not be tolerated.
///
/// The resampling ratio can be freely adjusted within the range specified to the constructor.
/// Higher maximum ratios require more memory to be allocated by [Resampler::output_buffer_allocate].
pub struct FastFixedIn<T> {
    nbr_channels: usize,
    chunk_size: usize,
    last_index: f64,
    resample_ratio: f64,
    resample_ratio_original: f64,
    target_ratio: f64,
    max_relative_ratio: f64,
    buffer: Vec<Vec<T>>,
    interpolation: PolynomialDegree,
    channel_mask: Vec<bool>,
}

/// An asynchronous resampler that returns a fixed number of audio frames.
/// The number of input frames required is given by the
/// [input_frames_next](Resampler::input_frames_next) function.
///
/// The resampling is done by interpolating between the input samples.
/// The polynomial degree can be selected, see [PolynomialDegree] for the available options.
///
/// Note that no anti-aliasing filter is used.
/// This makes it run considerably faster than the corresponding SincFixedOut, which performs anti-aliasing filtering.
/// The price is that the resampling creates some artefacts in the output, mainly at higher frequencies.
/// Use SincFixedOut if this can not be tolerated.
///
/// The resampling ratio can be freely adjusted within the range specified to the constructor.
/// Higher maximum ratios require more memory to be allocated by
/// [input_buffer_allocate](Resampler::input_buffer_allocate) and an internal buffer.
pub struct FastFixedOut<T> {
    nbr_channels: usize,
    chunk_size: usize,
    needed_input_size: usize,
    last_index: f64,
    current_buffer_fill: usize,
    resample_ratio: f64,
    resample_ratio_original: f64,
    target_ratio: f64,
    max_relative_ratio: f64,
    buffer: Vec<Vec<T>>,
    interpolation: PolynomialDegree,
    channel_mask: Vec<bool>,
}

/// Perform septic polynomial interpolation to get value at x.
/// Input points are assumed to be at x = -3, -2, -1, 0, 1, 2, 3, 4.
fn interp_septic<T>(x: T, yvals: &[T]) -> T
where
    T: Sample,
{
    let a = yvals[0];
    let b = yvals[1];
    let c = yvals[2];
    let d = yvals[3];
    let e = yvals[4];
    let f = yvals[5];
    let g = yvals[6];
    let h = yvals[7];
    let k7 = -a + t!(7.0) * b - t!(21.0) * c + t!(35.0) * d - t!(35.0) * e + t!(21.0) * f
        - t!(7.0) * g
        + h;
    let k6 = t!(7.0) * a - t!(42.0) * b + t!(105.0) * c - t!(140.0) * d + t!(105.0) * e
        - t!(42.0) * f
        + t!(7.0) * g;
    let k5 = -t!(7.0) * a - t!(14.0) * b + t!(189.0) * c - t!(490.0) * d + t!(595.0) * e
        - t!(378.0) * f
        + t!(119.0) * g
        - t!(14.0) * h;
    let k4 = -t!(35.0) * a + t!(420.0) * b - t!(1365.0) * c + t!(1960.0) * d - t!(1365.0) * e
        + t!(420.0) * f
        - t!(35.0) * g;
    let k3 = t!(56.0) * a - t!(497.0) * b + t!(336.0) * c + t!(1715.0) * d - t!(3080.0) * e
        + t!(1869.0) * f
        - t!(448.0) * g
        + t!(49.0) * h;
    let k2 = t!(28.0) * a - t!(378.0) * b + t!(3780.0) * c - t!(6860.0) * d + t!(3780.0) * e
        - t!(378.0) * f
        + t!(28.0) * g;
    let k1 = -t!(48.0) * a + t!(504.0) * b - t!(3024.0) * c - t!(1260.0) * d + t!(5040.0) * e
        - t!(1512.0) * f
        + t!(336.0) * g
        - t!(36.0) * h;
    let k0 = t!(5040.0) * d;
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    let x5 = x2 * x3;
    let x6 = x3 * x3;
    let x7 = x3 * x4;
    let val = k7 * x7 + k6 * x6 + k5 * x5 + k4 * x4 + k3 * x3 + k2 * x2 + k1 * x + k0;
    t!(1.0 / 5040.0) * val
}

/// Perform quintic polynomial interpolation to get value at x.
/// Input points are assumed to be at x = -2, -1, 0, 1, 2, 3.
fn interp_quintic<T>(x: T, yvals: &[T]) -> T
where
    T: Sample,
{
    let a = yvals[0];
    let b = yvals[1];
    let c = yvals[2];
    let d = yvals[3];
    let e = yvals[4];
    let f = yvals[5];
    let k5 = -a + t!(5.0) * b - t!(10.0) * c + t!(10.0) * d - t!(5.0) * e + f;
    let k4 = t!(5.0) * a - t!(20.0) * b + t!(30.0) * c - t!(20.0) * d + t!(5.0) * e;
    let k3 = -t!(5.0) * a - t!(5.0) * b + t!(50.0) * c - t!(70.0) * d + t!(35.0) * e - t!(5.0) * f;
    let k2 = -t!(5.0) * a + t!(80.0) * b - t!(150.0) * c + t!(80.0) * d - t!(5.0) * e;
    let k1 = t!(6.0) * a - t!(60.0) * b - t!(40.0) * c + t!(120.0) * d - t!(30.0) * e + t!(4.0) * f;
    let k0 = t!(120.0) * c;
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    let x5 = x2 * x3;
    let val = k5 * x5 + k4 * x4 + k3 * x3 + k2 * x2 + k1 * x + k0;
    t!(1.0 / 120.0) * val
}

/// Perform cubic polynomial interpolation to get value at x.
/// Input points are assumed to be at x = -1, 0, 1, 2.
fn interp_cubic<T>(x: T, yvals: &[T]) -> T
where
    T: Sample,
{
    let a0 = yvals[1];
    let a1 = -t!(1.0 / 3.0) * yvals[0] - t!(0.5) * yvals[1] + yvals[2] - t!(1.0 / 6.0) * yvals[3];
    let a2 = t!(0.5) * (yvals[0] + yvals[2]) - yvals[1];
    let a3 = t!(0.5) * (yvals[1] - yvals[2]) + t!(1.0 / 6.0) * (yvals[3] - yvals[0]);
    let x2 = x * x;
    let x3 = x2 * x;
    a0 + a1 * x + a2 * x2 + a3 * x3
}

/// Linear interpolation between two points at x=0 and x=1.
fn interp_lin<T>(x: T, yvals: &[T]) -> T
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

impl<T> FastFixedIn<T>
where
    T: Sample,
{
    /// Create a new FastFixedIn.
    ///
    /// Parameters are:
    /// - `resample_ratio`: Starting ratio between output and input sample rates, must be > 0.
    /// - `max_resample_ratio_relative`: Maximum ratio that can be set with [Resampler::set_resample_ratio] relative to `resample_ratio`, must be >= 1.0. The minimum relative ratio is the reciprocal of the maximum. For example, with `max_resample_ratio_relative` of 10.0, the ratio can be set between `resample_ratio * 10.0` and `resample_ratio / 10.0`.
    /// - `interpolation_type`: Degree of polynomial used for interpolation, see [PolynomialDegree].
    /// - `chunk_size`: Size of input data in frames.
    /// - `nbr_channels`: Number of channels in input/output.
    pub fn new(
        resample_ratio: f64,
        max_resample_ratio_relative: f64,
        interpolation_type: PolynomialDegree,
        chunk_size: usize,
        nbr_channels: usize,
    ) -> Result<Self, ResamplerConstructionError> {
        debug!(
            "Create new FastFixedIn, ratio: {}, chunk_size: {}, channels: {}",
            resample_ratio, chunk_size, nbr_channels,
        );

        validate_ratios(resample_ratio, max_resample_ratio_relative)?;

        let buffer = vec![vec![T::zero(); chunk_size + 2 * POLYNOMIAL_LEN_U]; nbr_channels];

        let channel_mask = vec![true; nbr_channels];

        Ok(FastFixedIn {
            nbr_channels,
            chunk_size,
            last_index: -(POLYNOMIAL_LEN_I / 2) as f64,
            resample_ratio,
            resample_ratio_original: resample_ratio,
            target_ratio: resample_ratio,
            max_relative_ratio: max_resample_ratio_relative,
            buffer,
            interpolation: interpolation_type,
            channel_mask,
        })
    }
}

impl<T> Resampler<T> for FastFixedIn<T>
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

        // Set length to chunksize*ratio plus a safety margin of 10 elements.
        let needed_len = (self.chunk_size as f64
            * (0.5 * self.resample_ratio + 0.5 * self.target_ratio)
            + 10.0) as usize;

        validate_buffers(
            wave_in,
            wave_out,
            &self.channel_mask,
            self.nbr_channels,
            self.chunk_size,
            needed_len,
        )?;

        // Update buffer with new data.
        for buf in self.buffer.iter_mut() {
            buf.copy_within(self.chunk_size..self.chunk_size + 2 * POLYNOMIAL_LEN_U, 0);
        }

        for (chan, active) in self.channel_mask.iter().enumerate() {
            if *active {
                self.buffer[chan][2 * POLYNOMIAL_LEN_U..2 * POLYNOMIAL_LEN_U + self.chunk_size]
                    .copy_from_slice(&wave_in[chan].as_ref()[..self.chunk_size]);
            }
        }

        let mut t_ratio = 1.0 / self.resample_ratio;
        let t_ratio_end = 1.0 / self.target_ratio;
        let approximate_nbr_frames =
            self.chunk_size as f64 * (0.5 * self.resample_ratio + 0.5 * self.target_ratio);
        let t_ratio_increment = (t_ratio_end - t_ratio) / approximate_nbr_frames;
        let end_idx =
            self.chunk_size as isize - (POLYNOMIAL_LEN_I + 1) - t_ratio_end.ceil() as isize;

        //println!(
        //    "start ratio {}, end_ratio {}, frames {}, t_increment {}",
        //    t_ratio,
        //    t_ratio_end,
        //    approximate_nbr_frames,
        //    t_ratio_increment
        //);

        let mut idx = self.last_index;

        let mut n = 0;

        match self.interpolation {
            PolynomialDegree::Septic => {
                while idx < end_idx as f64 {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let idx_floor = idx.floor();
                    let start_idx = idx_floor as isize - 3;
                    let frac = idx - idx_floor;
                    let frac_offset = T::coerce(frac);
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let buf = self.buffer.get_unchecked(chan).get_unchecked(
                                    (start_idx + 2 * POLYNOMIAL_LEN_I) as usize
                                        ..(start_idx + 2 * POLYNOMIAL_LEN_I + 8) as usize,
                                );
                                *wave_out
                                    .get_unchecked_mut(chan)
                                    .as_mut()
                                    .get_unchecked_mut(n) = interp_septic(frac_offset, buf);
                            }
                        }
                    }
                    n += 1;
                }
            }
            PolynomialDegree::Quintic => {
                while idx < end_idx as f64 {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let idx_floor = idx.floor();
                    let start_idx = idx_floor as isize - 2;
                    let frac = idx - idx_floor;
                    let frac_offset = T::coerce(frac);
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let buf = self.buffer.get_unchecked(chan).get_unchecked(
                                    (start_idx + 2 * POLYNOMIAL_LEN_I) as usize
                                        ..(start_idx + 2 * POLYNOMIAL_LEN_I + 6) as usize,
                                );
                                *wave_out
                                    .get_unchecked_mut(chan)
                                    .as_mut()
                                    .get_unchecked_mut(n) = interp_quintic(frac_offset, buf);
                            }
                        }
                    }
                    n += 1;
                }
            }
            PolynomialDegree::Cubic => {
                while idx < end_idx as f64 {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let idx_floor = idx.floor();
                    let start_idx = idx_floor as isize - 1;
                    let frac = idx - idx_floor;
                    let frac_offset = T::coerce(frac);
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let buf = self.buffer.get_unchecked(chan).get_unchecked(
                                    (start_idx + 2 * POLYNOMIAL_LEN_I) as usize
                                        ..(start_idx + 2 * POLYNOMIAL_LEN_I + 4) as usize,
                                );
                                *wave_out
                                    .get_unchecked_mut(chan)
                                    .as_mut()
                                    .get_unchecked_mut(n) = interp_cubic(frac_offset, buf);
                            }
                        }
                    }
                    n += 1;
                }
            }
            PolynomialDegree::Linear => {
                while idx < end_idx as f64 {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let idx_floor = idx.floor();
                    let start_idx = idx_floor as isize;
                    let frac = idx - idx_floor;
                    let frac_offset = T::coerce(frac);
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let buf = self.buffer.get_unchecked(chan).get_unchecked(
                                    (start_idx + 2 * POLYNOMIAL_LEN_I) as usize
                                        ..(start_idx + 2 * POLYNOMIAL_LEN_I + 2) as usize,
                                );
                                *wave_out
                                    .get_unchecked_mut(chan)
                                    .as_mut()
                                    .get_unchecked_mut(n) = interp_lin(frac_offset, buf);
                            }
                        }
                    }
                    n += 1;
                }
            }
            PolynomialDegree::Nearest => {
                while idx < end_idx as f64 {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let start_idx = idx.floor() as isize;
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let point = self
                                    .buffer
                                    .get_unchecked(chan)
                                    .get_unchecked((start_idx + 2 * POLYNOMIAL_LEN_I) as usize);
                                *wave_out
                                    .get_unchecked_mut(chan)
                                    .as_mut()
                                    .get_unchecked_mut(n) = *point;
                            }
                        }
                    }
                    n += 1;
                }
            }
        }

        // Store last index for next iteration.
        self.last_index = idx - self.chunk_size as f64;
        self.resample_ratio = self.target_ratio;
        trace!(
            "Resampling channels {:?}, {} frames in, {} frames out",
            active_channels_mask,
            self.chunk_size,
            n,
        );
        Ok((self.chunk_size, n))
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

    fn output_delay(&self) -> usize {
        (POLYNOMIAL_LEN_U as f64 * self.resample_ratio / 2.0) as usize
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

    fn reset(&mut self) {
        self.buffer
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = T::zero()));
        self.channel_mask.iter_mut().for_each(|val| *val = true);
        self.last_index = -(POLYNOMIAL_LEN_I / 2) as f64;
        self.resample_ratio = self.resample_ratio_original;
        self.target_ratio = self.resample_ratio_original;
    }
}

impl<T> FastFixedOut<T>
where
    T: Sample,
{
    /// Create a new FastFixedOut.
    ///
    /// Parameters are:
    /// - `resample_ratio`: Starting ratio between output and input sample rates, must be > 0.
    /// - `max_resample_ratio_relative`: Maximum ratio that can be set with [Resampler::set_resample_ratio] relative to `resample_ratio`, must be >= 1.0. The minimum relative ratio is the reciprocal of the maximum. For example, with `max_resample_ratio_relative` of 10.0, the ratio can be set between `resample_ratio * 10.0` and `resample_ratio / 10.0`.
    /// - `interpolation_type`: Degree of polynomial used for interpolation, see [PolynomialDegree].
    /// - `chunk_size`: Size of output data in frames.
    /// - `nbr_channels`: Number of channels in input/output.
    pub fn new(
        resample_ratio: f64,
        max_resample_ratio_relative: f64,
        interpolation_type: PolynomialDegree,
        chunk_size: usize,
        nbr_channels: usize,
    ) -> Result<Self, ResamplerConstructionError> {
        debug!(
            "Create new FastFixedOut, ratio: {}, chunk_size: {}, channels: {}",
            resample_ratio, chunk_size, nbr_channels,
        );
        validate_ratios(resample_ratio, max_resample_ratio_relative)?;

        let needed_input_size =
            (chunk_size as f64 / resample_ratio).ceil() as usize + 2 + POLYNOMIAL_LEN_U / 2;
        let buffer_channel_length = ((max_resample_ratio_relative + 1.0) * needed_input_size as f64)
            as usize
            + 2 * POLYNOMIAL_LEN_U;
        let buffer = vec![vec![T::zero(); buffer_channel_length]; nbr_channels];
        let channel_mask = vec![true; nbr_channels];

        Ok(FastFixedOut {
            nbr_channels,
            chunk_size,
            needed_input_size,
            last_index: -(POLYNOMIAL_LEN_I / 2) as f64,
            current_buffer_fill: needed_input_size,
            resample_ratio,
            resample_ratio_original: resample_ratio,
            target_ratio: resample_ratio,
            max_relative_ratio: max_resample_ratio_relative,
            buffer,
            interpolation: interpolation_type,
            channel_mask,
        })
    }
}

impl<T> Resampler<T> for FastFixedOut<T>
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
            self.needed_input_size,
            self.chunk_size,
        )?;
        for buf in self.buffer.iter_mut() {
            buf.copy_within(
                self.current_buffer_fill..self.current_buffer_fill + 2 * POLYNOMIAL_LEN_U,
                0,
            );
        }
        self.current_buffer_fill = self.needed_input_size;

        for (chan, wave_in) in wave_in
            .iter()
            .enumerate()
            .filter(|(chan, _)| self.channel_mask[*chan])
        {
            debug_assert!(self.chunk_size <= wave_out[chan].as_mut().len());
            self.buffer[chan][2 * POLYNOMIAL_LEN_U..2 * POLYNOMIAL_LEN_U + self.needed_input_size]
                .copy_from_slice(&wave_in.as_ref()[..self.needed_input_size]);
        }

        let mut idx = self.last_index;
        let mut t_ratio = 1.0 / self.resample_ratio;
        let t_ratio_end = 1.0 / self.target_ratio;
        let t_ratio_increment = (t_ratio_end - t_ratio) / self.chunk_size as f64;

        match self.interpolation {
            PolynomialDegree::Septic => {
                for n in 0..self.chunk_size {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let idx_floor = idx.floor();
                    let start_idx = idx_floor as isize - 3;
                    let frac = idx - idx_floor;
                    let frac_offset = T::coerce(frac);
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let buf = self.buffer.get_unchecked(chan).get_unchecked(
                                    (start_idx + 2 * POLYNOMIAL_LEN_I) as usize
                                        ..(start_idx + 2 * POLYNOMIAL_LEN_I + 8) as usize,
                                );
                                *wave_out
                                    .get_unchecked_mut(chan)
                                    .as_mut()
                                    .get_unchecked_mut(n) = interp_septic(frac_offset, buf);
                            }
                        }
                    }
                }
            }
            PolynomialDegree::Quintic => {
                for n in 0..self.chunk_size {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let idx_floor = idx.floor();
                    let start_idx = idx_floor as isize - 2;
                    let frac = idx - idx_floor;
                    let frac_offset = T::coerce(frac);
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let buf = self.buffer.get_unchecked(chan).get_unchecked(
                                    (start_idx + 2 * POLYNOMIAL_LEN_I) as usize
                                        ..(start_idx + 2 * POLYNOMIAL_LEN_I + 6) as usize,
                                );
                                *wave_out
                                    .get_unchecked_mut(chan)
                                    .as_mut()
                                    .get_unchecked_mut(n) = interp_quintic(frac_offset, buf);
                            }
                        }
                    }
                }
            }
            PolynomialDegree::Cubic => {
                for n in 0..self.chunk_size {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let idx_floor = idx.floor();
                    let start_idx = idx_floor as isize - 1;
                    let frac = idx - idx_floor;
                    let frac_offset = T::coerce(frac);
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let buf = self.buffer.get_unchecked(chan).get_unchecked(
                                    (start_idx + 2 * POLYNOMIAL_LEN_I) as usize
                                        ..(start_idx + 2 * POLYNOMIAL_LEN_I + 4) as usize,
                                );
                                *wave_out
                                    .get_unchecked_mut(chan)
                                    .as_mut()
                                    .get_unchecked_mut(n) = interp_cubic(frac_offset, buf);
                            }
                        }
                    }
                }
            }
            PolynomialDegree::Linear => {
                for n in 0..self.chunk_size {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let idx_floor = idx.floor();
                    let start_idx = idx_floor as isize;
                    let frac = idx - idx_floor;
                    let frac_offset = T::coerce(frac);
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let buf = self.buffer.get_unchecked(chan).get_unchecked(
                                    (start_idx + 2 * POLYNOMIAL_LEN_I) as usize
                                        ..(start_idx + 2 * POLYNOMIAL_LEN_I + 2) as usize,
                                );
                                *wave_out
                                    .get_unchecked_mut(chan)
                                    .as_mut()
                                    .get_unchecked_mut(n) = interp_lin(frac_offset, buf);
                            }
                        }
                    }
                }
            }
            PolynomialDegree::Nearest => {
                for n in 0..self.chunk_size {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let start_idx = idx.floor() as isize;
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let point = self
                                    .buffer
                                    .get_unchecked(chan)
                                    .get_unchecked((start_idx + 2 * POLYNOMIAL_LEN_I) as usize);
                                *wave_out
                                    .get_unchecked_mut(chan)
                                    .as_mut()
                                    .get_unchecked_mut(n) = *point;
                            }
                        }
                    }
                }
            }
        }

        // Store last index for next iteration.
        let input_frames_used = self.needed_input_size;
        self.last_index = idx - self.current_buffer_fill as f64;
        self.resample_ratio = self.target_ratio;
        self.needed_input_size = (self.last_index as f32
            + self.chunk_size as f32 / self.resample_ratio as f32
            + POLYNOMIAL_LEN_U as f32)
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
        Ok((input_frames_used, self.chunk_size))
    }

    fn input_frames_max(&self) -> usize {
        (self.chunk_size as f64 * self.resample_ratio_original * self.max_relative_ratio).ceil()
            as usize
            + 2
            + POLYNOMIAL_LEN_U / 2
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

    fn output_delay(&self) -> usize {
        (POLYNOMIAL_LEN_U as f64 * self.resample_ratio / 2.0) as usize
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
            self.needed_input_size = (self.last_index as f32
                + self.chunk_size as f32
                    / (0.5 * self.resample_ratio as f32 + 0.5 * self.target_ratio as f32))
                .ceil() as usize
                + POLYNOMIAL_LEN_U
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

    fn reset(&mut self) {
        self.buffer
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = T::zero()));
        self.needed_input_size = (self.chunk_size as f64 / self.resample_ratio_original).ceil()
            as usize
            + 2
            + POLYNOMIAL_LEN_U / 2;
        self.current_buffer_fill = self.needed_input_size;
        self.last_index = -(POLYNOMIAL_LEN_I / 2) as f64;
        self.channel_mask.iter_mut().for_each(|val| *val = true);
        self.resample_ratio = self.resample_ratio_original;
        self.target_ratio = self.resample_ratio_original;
    }
}

#[cfg(test)]
mod tests {
    use crate::check_output;
    use crate::PolynomialDegree;
    use crate::Resampler;
    use crate::{FastFixedIn, FastFixedOut};
    use rand::Rng;

    #[test]
    fn make_resampler_fi() {
        let mut resampler =
            FastFixedIn::<f64>::new(1.2, 1.0, PolynomialDegree::Cubic, 1024, 2).unwrap();
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
    fn reset_resampler_fi() {
        let mut resampler =
            FastFixedIn::<f64>::new(1.2, 1.0, PolynomialDegree::Cubic, 1024, 2).unwrap();

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
    fn make_resampler_fi_32() {
        let mut resampler =
            FastFixedIn::<f32>::new(1.2, 1.0, PolynomialDegree::Cubic, 1024, 2).unwrap();
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
        let mut resampler =
            FastFixedIn::<f64>::new(1.2, 1.0, PolynomialDegree::Cubic, 1024, 2).unwrap();
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
    fn make_resampler_fi_downsample() {
        // Replicate settings from reported issue.
        let mut resampler = FastFixedIn::<f64>::new(
            16000 as f64 / 96000 as f64,
            1.0,
            PolynomialDegree::Cubic,
            1024,
            2,
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
    fn make_resampler_fi_upsample() {
        // Replicate settings from reported issue.
        let mut resampler = FastFixedIn::<f64>::new(
            192000 as f64 / 44100 as f64,
            1.0,
            PolynomialDegree::Cubic,
            1024,
            2,
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
    fn make_resampler_fo() {
        let mut resampler =
            FastFixedOut::<f64>::new(1.2, 1.0, PolynomialDegree::Cubic, 1024, 2).unwrap();
        let frames = resampler.input_frames_next();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn reset_resampler_fo() {
        let mut resampler =
            FastFixedOut::<f64>::new(1.2, 1.0, PolynomialDegree::Cubic, 1024, 2).unwrap();
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
    fn make_resampler_fo_32() {
        let mut resampler =
            FastFixedOut::<f32>::new(1.2, 1.0, PolynomialDegree::Cubic, 1024, 2).unwrap();
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
        let mut resampler =
            FastFixedOut::<f64>::new(1.2, 1.0, PolynomialDegree::Cubic, 1024, 2).unwrap();
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
    fn make_resampler_fo_downsample() {
        let mut resampler =
            FastFixedOut::<f64>::new(0.125, 1.0, PolynomialDegree::Cubic, 1024, 2).unwrap();
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
        let mut resampler =
            FastFixedOut::<f64>::new(8.0, 1.0, PolynomialDegree::Cubic, 1024, 2).unwrap();
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
    fn check_fo_output() {
        let mut resampler =
            FastFixedOut::<f64>::new(8.0, 1.0, PolynomialDegree::Cubic, 1024, 2).unwrap();
        check_output!(check_fo_output, resampler);
    }

    #[test]
    fn check_fi_output() {
        let mut resampler =
            FastFixedIn::<f64>::new(8.0, 1.0, PolynomialDegree::Cubic, 1024, 2).unwrap();
        check_output!(check_fo_output, resampler);
    }
}
