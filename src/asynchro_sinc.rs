use crate::asynchro::InnerResampler;
use crate::interpolation::*;
#[cfg(target_arch = "x86_64")]
use crate::sinc_interpolator::sinc_interpolator_avx::AvxInterpolator;
#[cfg(target_arch = "aarch64")]
use crate::sinc_interpolator::sinc_interpolator_neon::NeonInterpolator;
#[cfg(target_arch = "x86_64")]
use crate::sinc_interpolator::sinc_interpolator_sse::SseInterpolator;
use crate::sinc_interpolator::{
    AlignedBuf, AnyInterpolator, AvxSample, NeonSample, ScalarInterpolator, SincInterpolator,
    SseSample,
};
use crate::windows::WindowFunction;
use crate::Sample;
use audioadapter::AdapterMut;

macro_rules! t {
    // Shorter form of T::coerce(value)
    ($expression:expr) => {
        T::coerce($expression)
    };
}

/// A struct holding the parameters for sinc interpolation.
#[derive(Debug, Clone, Copy)]
pub struct SincInterpolationParameters {
    /// Length of the windowed sinc interpolation filter.
    /// Higher values can allow a higher cut-off frequency leading to less high frequency roll-off
    /// at the expense of higher CPU usage. 256 is a good starting point.
    /// The value will be rounded up to the nearest multiple of 8.
    pub sinc_len: usize,
    /// Relative cutoff frequency of the sinc interpolation filter
    /// (relative to the lowest one of fs_in/2 or fs_out/2). Start at 0.95, and increase if needed.
    pub f_cutoff: f32,
    /// The number of intermediate points to use for interpolation.
    /// Higher values use more memory for storing the sinc filters.
    /// Only the points actually needed are calculated during processing
    /// so a larger number does not directly lead to higher CPU usage.
    /// A lower value helps in keeping the sincs in the CPU cache. Start at 128.
    pub oversampling_factor: usize,
    /// Interpolation type, see `SincInterpolationType`
    pub interpolation: SincInterpolationType,
    /// Window function to use.
    pub window: WindowFunction,
}

/// Interpolation methods that can be selected. For asynchronous interpolation where the
/// ratio between input and output sample rates can be any number, it's not possible to
/// pre-calculate all the needed interpolation filters.
/// Instead they have to be computed as needed, which becomes impractical since the
/// sincs are very expensive to generate in terms of CPU time.
/// It's more efficient to combine the sinc filters with some other interpolation technique.
/// Then, sinc filters are used to provide a fixed number of interpolated points between input samples,
/// and then, the new value is calculated by interpolation between those points.
#[derive(Debug, Clone, Copy)]
pub enum SincInterpolationType {
    /// For cubic interpolation, the four nearest intermediate points are calculated
    /// using sinc interpolation.
    /// Then, a cubic polynomial is fitted to these points, and is used to calculate the new sample value.
    /// The computation time is approximately twice as long as that of linear interpolation,
    /// but it requires much fewer intermediate points for a good result.
    Cubic,
    /// For quadratic interpolation, the three nearest intermediate points are calculated
    /// using sinc interpolation.
    /// Then, a quadratic polynomial is fitted to these points, and is used to calculate the new sample value.
    /// The computation time lies approximately halfway between that of linear and quadratic interpolation.
    Quadratic,
    /// For linear interpolation, the new sample value is calculated by linear interpolation
    /// between the two nearest points.
    /// This requires two intermediate points to be calculated using sinc interpolation,
    /// and the output is obtained by taking a weighted average of these two points.
    /// This is relatively fast, but needs a large number of intermediate points to
    /// push the resampling artefacts below the noise floor.
    Linear,
    /// The Nearest mode doesn't do any interpolation, but simply picks the nearest intermediate point.
    /// This is useful when the nearest point is actually the correct one, for example when upsampling by a factor 2,
    /// like 48kHz->96kHz.
    /// Then, when setting the oversampling_factor to 2 and using Nearest mode,
    /// no unnecessary computations are performed and the result is equivalent to that of synchronous resampling.
    /// This also works for other ratios that can be expressed by a fraction. For 44.1kHz -> 48 kHz,
    /// setting oversampling_factor to 160 gives the desired result (since 48kHz = 160/147 * 44.1kHz).
    Nearest,
}

pub fn make_interpolator<T>(
    sinc_len: usize,
    resample_ratio: f64,
    f_cutoff: f32,
    oversampling_factor: usize,
    window: WindowFunction,
) -> AnyInterpolator<T>
where
    T: AvxSample + SseSample + NeonSample + Sample,
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
        return AnyInterpolator::Avx(interpolator);
    }

    #[cfg(target_arch = "x86_64")]
    if let Ok(interpolator) =
        SseInterpolator::<T>::new(sinc_len, oversampling_factor, f_cutoff, window)
    {
        return AnyInterpolator::Sse(interpolator);
    }

    #[cfg(target_arch = "aarch64")]
    if let Ok(interpolator) =
        NeonInterpolator::<T>::new(sinc_len, oversampling_factor, f_cutoff, window)
    {
        return AnyInterpolator::Neon(interpolator);
    }

    AnyInterpolator::Scalar(ScalarInterpolator::<T>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ))
}

/// Perform cubic polynomial interpolation to get value at x.
/// Input points are assumed to be at x = -1, 0, 1, 2.
#[allow(dead_code)]
pub fn interp_cubic<T>(x: T, yvals: &[T; 4]) -> T
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

/// Compute the four blending weights for cubic interpolation at fractional position x.
/// These are the per-point coefficients such that interp_cubic(x, pts) == dot(weights, pts).
/// Input points are assumed to be at x = -1, 0, 1, 2.
pub fn interp_cubic_weights<T>(x: T) -> [T; 4]
where
    T: Sample,
{
    let x2 = x * x;
    let x3 = x2 * x;
    [
        t!(-1.0 / 3.0) * x + t!(0.5) * x2 - t!(1.0 / 6.0) * x3,
        t!(1.0) - t!(0.5) * x - x2 + t!(0.5) * x3,
        x + t!(0.5) * x2 - t!(0.5) * x3,
        -t!(1.0 / 6.0) * x + t!(1.0 / 6.0) * x3,
    ]
}

/// Perform quadratic polynomial interpolation to get value at x.
/// Input points are assumed to be at x = 0, 1, 2.
#[allow(dead_code)]
pub fn interp_quad<T>(x: T, yvals: &[T; 3]) -> T
where
    T: Sample,
{
    let a2 = yvals[0] - t!(2.0) * yvals[1] + yvals[2];
    let a1 = -t!(3.0) * yvals[0] + t!(4.0) * yvals[1] - yvals[2];
    let a0 = t!(2.0) * yvals[0];
    let x2 = x * x;
    t!(0.5) * (a0 + a1 * x + a2 * x2)
}

/// Compute the three blending weights for quadratic interpolation at fractional position x.
/// These are the per-point coefficients such that interp_quad(x, pts) == dot(weights, pts).
/// Input points are assumed to be at x = 0, 1, 2.
pub fn interp_quad_weights<T>(x: T) -> [T; 3]
where
    T: Sample,
{
    let x2 = x * x;
    [
        t!(0.5) * (t!(2.0) - t!(3.0) * x + x2),
        t!(0.5) * (t!(4.0) * x - t!(2.0) * x2),
        t!(0.5) * (x2 - x),
    ]
}

/// Perform linear interpolation between two points at x=0 and x=1.
#[allow(dead_code)]
pub fn interp_lin<T>(x: T, yvals: &[T; 2]) -> T
where
    T: Sample,
{
    yvals[0] + x * (yvals[1] - yvals[0])
}

/// Compute the two blending weights for linear interpolation at fractional position x.
/// These are the per-point coefficients such that interp_lin(x, pts) == dot(weights, pts).
pub fn interp_lin_weights<T>(x: T) -> [T; 2]
where
    T: Sample,
{
    [t!(1.0) - x, x]
}

pub(crate) struct InnerSinc<T>
where
    T: AvxSample + SseSample + NeonSample + Sample,
{
    pub interpolator: AnyInterpolator<T>,
    pub interpolation: SincInterpolationType,
    // Pre-allocated buffer for the combined sinc (used by the >2 channel path).
    // Length is interpolator.nbr_points() + 1. 32-byte aligned so 256-bit AVX
    // loads on it never cross a cache line boundary.
    combined: AlignedBuf<T>,
}

impl<T> InnerSinc<T>
where
    T: AvxSample + SseSample + NeonSample + Sample,
{
    pub(crate) fn new(
        interpolator: AnyInterpolator<T>,
        interpolation: SincInterpolationType,
    ) -> Self {
        let len = interpolator.nbr_points() + 1;
        Self {
            interpolator,
            interpolation,
            combined: AlignedBuf::zeroed(len),
        }
    }
}

impl<T> InnerResampler<T> for InnerSinc<T>
where
    T: AvxSample + SseSample + NeonSample + Sample,
{
    fn process(
        &mut self,
        idx: f64,
        nbr_frames: usize,
        channel_mask: &[bool],
        t_ratio: f64,
        t_ratio_increment: f64,
        wave_in: &[Vec<T>],
        wave_out: &mut dyn AdapterMut<'_, T>,
        output_offset: usize,
    ) -> f64 {
        let mut t_ratio = t_ratio;
        let mut idx = idx;
        let interpolator_len = self.interpolator.nbr_points();
        let active_count = channel_mask.iter().filter(|&&a| a).count();
        match self.interpolation {
            SincInterpolationType::Cubic => {
                // Cubic has 4 nearest points so the build cost breaks even at 2 channels.
                let use_combined = active_count >= 2;
                let oversampling_factor = self.interpolator.nbr_sincs();
                let mut nearest = [(0isize, 0isize); 4];
                for frame in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    get_nearest_times_4(idx, oversampling_factor as isize, &mut nearest);
                    let frac = idx * oversampling_factor as f64
                        - (idx * oversampling_factor as f64).floor();
                    let frac_offset = t!(frac);
                    if use_combined {
                        let weights = interp_cubic_weights(frac_offset);
                        for n in &nearest {
                            self.interpolator.prefetch_sinc(n.1 as usize);
                        }
                        let min_idx = self.interpolator.make_combined_sinc(
                            &nearest,
                            &weights,
                            &mut self.combined,
                        );
                        let base = (min_idx + 2 * interpolator_len as isize) as usize;
                        for (chan, active) in channel_mask.iter().enumerate() {
                            if *active {
                                let buf = &wave_in[chan];
                                let result = self.interpolator.get_sinc_dot_product(
                                    buf,
                                    base,
                                    &self.combined[..interpolator_len],
                                ) + self.combined[interpolator_len]
                                    * buf[base + interpolator_len];
                                wave_out.write_sample(chan, frame + output_offset, &result);
                            }
                        }
                    } else {
                        for (chan, active) in channel_mask.iter().enumerate() {
                            if *active {
                                let buf = &wave_in[chan];
                                let mut points = [T::zero(); 4];
                                for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                    *p = self.interpolator.get_sinc_interpolated(
                                        buf,
                                        (n.0 + 2 * interpolator_len as isize) as usize,
                                        n.1 as usize,
                                    );
                                }
                                let result = interp_cubic(frac_offset, &points);
                                wave_out.write_sample(chan, frame + output_offset, &result);
                            }
                        }
                    }
                }
            }
            SincInterpolationType::Quadratic => {
                // Quadratic has 3 nearest points; combined sinc pays off from 3 channels.
                let use_combined = active_count > 2;
                let oversampling_factor = self.interpolator.nbr_sincs();
                let mut nearest = [(0isize, 0isize); 3];
                for frame in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    get_nearest_times_3(idx, oversampling_factor as isize, &mut nearest);
                    let frac = idx * oversampling_factor as f64
                        - (idx * oversampling_factor as f64).floor();
                    let frac_offset = t!(frac);
                    if use_combined {
                        let weights = interp_quad_weights(frac_offset);
                        for n in &nearest {
                            self.interpolator.prefetch_sinc(n.1 as usize);
                        }
                        let min_idx = self.interpolator.make_combined_sinc(
                            &nearest,
                            &weights,
                            &mut self.combined,
                        );
                        let base = (min_idx + 2 * interpolator_len as isize) as usize;
                        for (chan, active) in channel_mask.iter().enumerate() {
                            if *active {
                                let buf = &wave_in[chan];
                                let result = self.interpolator.get_sinc_dot_product(
                                    buf,
                                    base,
                                    &self.combined[..interpolator_len],
                                ) + self.combined[interpolator_len]
                                    * buf[base + interpolator_len];
                                wave_out.write_sample(chan, frame + output_offset, &result);
                            }
                        }
                    } else {
                        for (chan, active) in channel_mask.iter().enumerate() {
                            if *active {
                                let buf = &wave_in[chan];
                                let mut points = [T::zero(); 3];
                                for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                    *p = self.interpolator.get_sinc_interpolated(
                                        buf,
                                        (n.0 + 2 * interpolator_len as isize) as usize,
                                        n.1 as usize,
                                    );
                                }
                                let result = interp_quad(frac_offset, &points);
                                wave_out.write_sample(chan, frame + output_offset, &result);
                            }
                        }
                    }
                }
            }
            SincInterpolationType::Linear => {
                // Linear has 2 nearest points; combined sinc pays off from 3 channels.
                let use_combined = active_count > 2;
                let oversampling_factor = self.interpolator.nbr_sincs();
                let mut nearest = [(0isize, 0isize); 2];
                for frame in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    get_nearest_times_2(idx, oversampling_factor as isize, &mut nearest);
                    let frac = idx * oversampling_factor as f64
                        - (idx * oversampling_factor as f64).floor();
                    let frac_offset = t!(frac);
                    if use_combined {
                        let weights = interp_lin_weights(frac_offset);
                        for n in &nearest {
                            self.interpolator.prefetch_sinc(n.1 as usize);
                        }
                        let min_idx = self.interpolator.make_combined_sinc(
                            &nearest,
                            &weights,
                            &mut self.combined,
                        );
                        let base = (min_idx + 2 * interpolator_len as isize) as usize;
                        for (chan, active) in channel_mask.iter().enumerate() {
                            if *active {
                                let buf = &wave_in[chan];
                                let result = self.interpolator.get_sinc_dot_product(
                                    buf,
                                    base,
                                    &self.combined[..interpolator_len],
                                ) + self.combined[interpolator_len]
                                    * buf[base + interpolator_len];
                                wave_out.write_sample(chan, frame + output_offset, &result);
                            }
                        }
                    } else {
                        for (chan, active) in channel_mask.iter().enumerate() {
                            if *active {
                                let buf = &wave_in[chan];
                                let mut points = [T::zero(); 2];
                                for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                    *p = self.interpolator.get_sinc_interpolated(
                                        buf,
                                        (n.0 + 2 * interpolator_len as isize) as usize,
                                        n.1 as usize,
                                    );
                                }
                                let result = interp_lin(frac_offset, &points);
                                wave_out.write_sample(chan, frame + output_offset, &result);
                            }
                        }
                    }
                }
            }
            SincInterpolationType::Nearest => {
                let oversampling_factor = self.interpolator.nbr_sincs();
                let mut point;
                let mut nearest;
                for frame in 0..nbr_frames {
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
                            wave_out.write_sample(chan, frame + output_offset, &point);
                        }
                    }
                }
            }
        }
        idx
    }

    fn nbr_points(&self) -> usize {
        self.interpolator.nbr_points()
    }

    fn init_last_index(&self) -> f64 {
        -(self.interpolator.nbr_points() as f64 - 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        interp_cubic, interp_cubic_weights, interp_lin, interp_lin_weights, interp_quad,
        interp_quad_weights,
    };

    #[test]
    fn int_cubic() {
        let yvals = [0.0f64, 2.0f64, 4.0f64, 6.0f64];
        let interp = interp_cubic(0.5f64, &yvals);
        assert_eq!(interp, 3.0f64);
    }

    #[test]
    fn int_lin_32() {
        let yvals = [1.0f32, 5.0f32];
        let interp = interp_lin(0.25f32, &yvals);
        assert_eq!(interp, 2.0f32);
    }

    #[test]
    fn int_cubic_32() {
        let yvals = [0.0f32, 2.0f32, 4.0f32, 6.0f32];
        let interp = interp_cubic(0.5f32, &yvals);
        assert_eq!(interp, 3.0f32);
    }

    /// Verify that interp_cubic_weights produces the same result as interp_cubic.
    #[test]
    fn cubic_weights_match_cubic() {
        let yvals = [1.3f64, -0.7f64, 2.1f64, 0.4f64];
        for x_int in 0..=10 {
            let x = x_int as f64 / 10.0;
            let direct = interp_cubic(x, &yvals);
            let w = interp_cubic_weights(x);
            let blended = w[0] * yvals[0] + w[1] * yvals[1] + w[2] * yvals[2] + w[3] * yvals[3];
            assert!((direct - blended).abs() < 1.0e-12, "mismatch at x={x}");
        }
    }

    /// Verify that interp_quad_weights produces the same result as interp_quad.
    #[test]
    fn quad_weights_match_quad() {
        let yvals = [1.3f64, -0.7f64, 2.1f64];
        for x_int in 0..=10 {
            let x = x_int as f64 / 10.0;
            let direct = interp_quad(x, &yvals);
            let w = interp_quad_weights(x);
            let blended = w[0] * yvals[0] + w[1] * yvals[1] + w[2] * yvals[2];
            assert!((direct - blended).abs() < 1.0e-12, "mismatch at x={x}");
        }
    }

    /// Verify that interp_lin_weights produces the same result as interp_lin.
    #[test]
    fn lin_weights_match_lin() {
        let yvals = [1.3f64, -0.7f64];
        for x_int in 0..=10 {
            let x = x_int as f64 / 10.0;
            let direct = interp_lin(x, &yvals);
            let w = interp_lin_weights(x);
            let blended = w[0] * yvals[0] + w[1] * yvals[1];
            assert!((direct - blended).abs() < 1.0e-12, "mismatch at x={x}");
        }
    }
}
