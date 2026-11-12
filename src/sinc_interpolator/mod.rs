//! Sinc interpolators, usable for reading an in-memory wave at arbitrary fractional positions.
//!
//! [make_interpolator] builds the fastest interpolator available on the running CPU
//! (AVX, SSE or NEON, falling back to a scalar implementation).
//! [SincInterpolator::get_sinc_interpolated] returns the wave at position
//! `index + nbr_points / 2 - 1 + (subindex + 1) / nbr_sincs`.
//! To read at an arbitrary position `pos`, map it to `t = pos - nbr_points / 2 + 1 - 1 / nbr_sincs`,
//! find the four nearest table points with
//! [get_nearest_times_4](crate::interpolation::get_nearest_times_4),
//! evaluate each, and blend the results with
//! [interp_cubic](crate::interpolation::interp_cubic).
//!
//! The wave must be padded so that `index + nbr_points` stays inside it.
//! The cutoff is fixed when the interpolator is built.
//! Reading slower than the ratio used for construction while downsampling can therefore alias.
//!
//! ```
//! use rubato::interpolation::{get_nearest_times_4, interp_cubic};
//! use rubato::sinc_interpolator::{make_interpolator, SincInterpolator};
//! use rubato::WindowFunction;
//!
//! // Build an interpolator for ratio 1.0, with no downsampling cutoff reduction.
//! let interpolator = make_interpolator::<f64>(128, 1.0, None, 256, WindowFunction::BlackmanHarris2);
//! let nbr_points = interpolator.nbr_points();
//! let nbr_sincs = interpolator.nbr_sincs();
//!
//! // A slow sine, padded with zeros at both ends.
//! let pad = nbr_points;
//! let mut wave = vec![0.0f64; 2000 + 2 * pad];
//! for (n, v) in wave[pad..pad + 2000].iter_mut().enumerate() {
//!     *v = (n as f64 * 0.05).sin();
//! }
//!
//! // Read at a fractional position, here 1000.37 frames into the unpadded wave.
//! let pos = 1000.37 + pad as f64;
//! let t = pos - (nbr_points / 2) as f64 + 1.0 - 1.0 / nbr_sincs as f64;
//! let mut nearest = [(0isize, 0isize); 4];
//! get_nearest_times_4(t, nbr_sincs as isize, &mut nearest);
//! let mut points = [0.0f64; 4];
//! for (p, (index, subindex)) in points.iter_mut().zip(nearest) {
//!     *p = interpolator.get_sinc_interpolated(&wave, index as usize, subindex as usize);
//! }
//! // Fractional position between the first and second nearest point.
//! let frac = (t * nbr_sincs as f64).fract();
//! let value = interp_cubic(frac, &points);
//! assert!((value - (1000.37f64 * 0.05).sin()).abs() < 1e-4);
//! ```

use crate::sinc::make_sincs;
use crate::windows::WindowFunction;
use crate::Sample;

pub mod aligned_buf;
pub use crate::asynchro_sinc::make_interpolator;
pub use aligned_buf::AlignedBuf;

/// Helper macro to define a dummy implementation of the sample trait if a
/// feature is not supported.
macro_rules! interpolator {
    (
    #[cfg($($cond:tt)*)]
    mod $mod:ident;
    trait $trait:ident;
    ) => {
        #[cfg($($cond)*)]
        pub mod $mod;

        #[cfg(not($($cond)*))]
        pub mod $mod {
            use crate::Sample;

            /// Dummy trait when not supported.
            pub trait $trait {
            }

            /// Dummy impl of trait when not supported.
            impl<T> $trait for T where T: Sample {
            }
        }

        pub use self::$mod::$trait;
    }
}

interpolator! {
    #[cfg(target_arch = "x86_64")]
    mod sinc_interpolator_avx;
    trait AvxSample;
}

interpolator! {
    #[cfg(target_arch = "x86_64")]
    mod sinc_interpolator_sse;
    trait SseSample;
}

interpolator! {
    #[cfg(target_arch = "aarch64")]
    mod sinc_interpolator_neon;
    trait NeonSample;
}

/// Functions for making the scalar product with a sinc.
pub trait SincInterpolator<T>: Send {
    /// Compute the dot product of the wave starting at `index` with the provided sinc slice.
    fn get_sinc_dot_product(&self, wave: &[T], index: usize, sinc: &[T]) -> T;

    /// Expose the raw sinc table for use in combined-sinc building.
    fn get_sincs(&self) -> &[AlignedBuf<T>];

    /// Get sinc length.
    fn nbr_points(&self) -> usize;

    /// Get number of sincs used for oversampling.
    fn nbr_sincs(&self) -> usize;

    /// Warm the CPU cache with the sinc row at `subindex`.
    /// The default is a no-op; SIMD backends may override this to hide L2 latency.
    #[inline]
    fn prefetch_sinc(&self, _subindex: usize) {}

    /// Make the scalar product between the waveform starting at `index` and the sinc of `subindex`.
    #[inline]
    fn get_sinc_interpolated(&self, wave: &[T], index: usize, subindex: usize) -> T {
        assert!(
            (index + self.nbr_points()) < wave.len(),
            "Tried to interpolate for index {}, max for the given input is {}",
            index,
            wave.len() - self.nbr_points() - 1
        );
        assert!(
            subindex < self.nbr_sincs(),
            "Tried to use sinc subindex {}, max is {}",
            subindex,
            self.nbr_sincs() - 1
        );
        self.get_sinc_dot_product(wave, index, &self.get_sincs()[subindex])
    }

    /// Build a combined sinc by blending the sincs indicated by `nearest` with `weights`.
    ///
    /// The 4 (or fewer) nearest points may span two consecutive integer indices, so `combined`
    /// must have length `nbr_points() + 1`. The extra element at position `nbr_points()` holds
    /// the contribution of any points at the higher index, which the caller adds as a scalar
    /// multiply after the main SIMD dot-product loop.
    ///
    /// Returns the minimum integer index found in `nearest`, used by the caller to compute the
    /// base buffer offset.
    fn make_combined_sinc(
        &self,
        nearest: &[(isize, isize)],
        weights: &[T],
        combined: &mut [T],
    ) -> isize
    where
        T: Sample,
    {
        debug_assert_eq!(
            combined.len(),
            self.nbr_points() + 1,
            "combined must be nbr_points()+1: the extra element holds any spillover \
             from nearest points at the higher integer index"
        );
        let min_idx = nearest.iter().map(|n| n.0).min().unwrap();
        // memset to zero — valid for f32/f64 since all-zero bits represent 0.0.
        unsafe {
            std::ptr::write_bytes(combined.as_mut_ptr(), 0, combined.len());
        }
        for (n, &w) in nearest.iter().zip(weights.iter()) {
            let shift = (n.0 - min_idx) as usize;
            for (k, &s) in self.get_sincs()[n.1 as usize].iter().enumerate() {
                combined[shift + k] += w * s;
            }
        }
        min_idx
    }
}

/// A concrete enum over every sinc-interpolator implementation.
///
/// Replaces `Box<dyn SincInterpolator<T>>` so the hot path can dispatch via a
/// `match` on a small closed set of variants, enabling the compiler to inline
/// `get_sinc_dot_product` and fully unroll the inner FMA loop.
///
/// The bounds `T: AvxSample + SseSample + NeonSample` collapse to `T: Sample`
/// on each platform because the non-native SIMD traits have blanket impls.
pub enum AnyInterpolator<T>
where
    T: AvxSample + SseSample + NeonSample + Sample,
{
    #[cfg(target_arch = "x86_64")]
    Avx(sinc_interpolator_avx::AvxInterpolator<T>),
    #[cfg(target_arch = "x86_64")]
    Sse(sinc_interpolator_sse::SseInterpolator<T>),
    #[cfg(target_arch = "aarch64")]
    Neon(sinc_interpolator_neon::NeonInterpolator<T>),
    Scalar(ScalarInterpolator<T>),
}

/// Dispatch a method call to the active interpolator variant.
macro_rules! dispatch {
    ($self:expr, $method:ident ($($arg:expr),*)) => {
        match $self {
            #[cfg(target_arch = "x86_64")]
            AnyInterpolator::Avx(i) => i.$method($($arg),*),
            #[cfg(target_arch = "x86_64")]
            AnyInterpolator::Sse(i) => i.$method($($arg),*),
            #[cfg(target_arch = "aarch64")]
            AnyInterpolator::Neon(i) => i.$method($($arg),*),
            AnyInterpolator::Scalar(i) => i.$method($($arg),*),
        }
    };
}

impl<T> SincInterpolator<T> for AnyInterpolator<T>
where
    T: AvxSample + SseSample + NeonSample + Sample,
{
    #[inline]
    fn get_sinc_dot_product(&self, wave: &[T], index: usize, sinc: &[T]) -> T {
        dispatch!(self, get_sinc_dot_product(wave, index, sinc))
    }

    #[inline]
    fn get_sincs(&self) -> &[AlignedBuf<T>] {
        dispatch!(self, get_sincs())
    }

    #[inline]
    fn nbr_points(&self) -> usize {
        dispatch!(self, nbr_points())
    }

    #[inline]
    fn nbr_sincs(&self) -> usize {
        dispatch!(self, nbr_sincs())
    }

    #[inline]
    fn prefetch_sinc(&self, subindex: usize) {
        dispatch!(self, prefetch_sinc(subindex))
    }

    #[inline]
    fn make_combined_sinc(
        &self,
        nearest: &[(isize, isize)],
        weights: &[T],
        combined: &mut [T],
    ) -> isize
    where
        T: Sample,
    {
        dispatch!(self, make_combined_sinc(nearest, weights, combined))
    }
}

/// A plain scalar interpolator.
pub struct ScalarInterpolator<T> {
    sincs: Vec<AlignedBuf<T>>,
    length: usize,
    nbr_sincs: usize,
}

impl<T> SincInterpolator<T> for ScalarInterpolator<T>
where
    T: Sample,
{
    fn get_sinc_dot_product(&self, wave: &[T], index: usize, sinc: &[T]) -> T {
        let wave_cut = &wave[index..(index + sinc.len())];
        unsafe {
            let mut acc0 = T::zero();
            let mut acc1 = T::zero();
            let mut acc2 = T::zero();
            let mut acc3 = T::zero();
            let mut acc4 = T::zero();
            let mut acc5 = T::zero();
            let mut acc6 = T::zero();
            let mut acc7 = T::zero();
            let mut idx = 0;
            for _ in 0..wave_cut.len() / 8 {
                acc0 += *wave_cut.get_unchecked(idx) * *sinc.get_unchecked(idx);
                acc1 += *wave_cut.get_unchecked(idx + 1) * *sinc.get_unchecked(idx + 1);
                acc2 += *wave_cut.get_unchecked(idx + 2) * *sinc.get_unchecked(idx + 2);
                acc3 += *wave_cut.get_unchecked(idx + 3) * *sinc.get_unchecked(idx + 3);
                acc4 += *wave_cut.get_unchecked(idx + 4) * *sinc.get_unchecked(idx + 4);
                acc5 += *wave_cut.get_unchecked(idx + 5) * *sinc.get_unchecked(idx + 5);
                acc6 += *wave_cut.get_unchecked(idx + 6) * *sinc.get_unchecked(idx + 6);
                acc7 += *wave_cut.get_unchecked(idx + 7) * *sinc.get_unchecked(idx + 7);
                idx += 8;
            }
            acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7
        }
    }

    fn get_sincs(&self) -> &[AlignedBuf<T>] {
        &self.sincs
    }

    fn nbr_points(&self) -> usize {
        self.length
    }

    fn nbr_sincs(&self) -> usize {
        self.nbr_sincs
    }
}

impl<T> ScalarInterpolator<T>
where
    T: Sample,
{
    /// Create a new ScalarInterpolator.
    ///
    /// Parameters are:
    /// - `sinc_len`: Length of sinc functions.
    /// - `oversampling_factor`: Number of intermediate sincs (oversampling factor).
    /// - `f_cutoff`: Relative cutoff frequency.
    /// - `window`: Window function to use.
    pub fn new(
        sinc_len: usize,
        oversampling_factor: usize,
        f_cutoff: f32,
        window: WindowFunction,
    ) -> Self {
        assert!(
            sinc_len.is_multiple_of(8),
            "Sinc length must be a multiple of 8"
        );
        let raw_sincs: Vec<Vec<T>> = make_sincs(sinc_len, oversampling_factor, f_cutoff, window);
        let sincs = raw_sincs
            .into_iter()
            .map(|row| AlignedBuf::from_slice(&row))
            .collect();
        Self {
            sincs,
            length: sinc_len,
            nbr_sincs: oversampling_factor,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ScalarInterpolator;
    use super::SincInterpolator;
    use crate::WindowFunction;
    use num_traits::Float;
    use test_log::test;

    fn get_sinc_interpolated<T: Float>(wave: &[T], index: usize, sinc: &[T]) -> T {
        let wave_cut = &wave[index..(index + sinc.len())];
        wave_cut
            .iter()
            .zip(sinc.iter())
            .fold(T::zero(), |acc, (x, y)| acc + *x * *y)
    }

    #[test]
    fn test_scalar_interpolator_64() {
        let mut wave = Vec::new();
        for _ in 0..2048 {
            wave.push(rand::random::<f64>());
        }
        let sinc_len = 256;
        let f_cutoff = 0.94733715;
        let oversampling_factor = 256;
        let window = WindowFunction::BlackmanHarris2;

        let interpolator =
            ScalarInterpolator::<f64>::new(sinc_len, oversampling_factor, f_cutoff, window);
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &interpolator.sincs[123]);
        assert!((value - check).abs() < 1.0e-9);
    }

    #[test]
    fn test_scalar_interpolator_32() {
        let mut wave = Vec::new();
        for _ in 0..2048 {
            wave.push(rand::random::<f32>());
        }
        let sinc_len = 256;
        let f_cutoff = 0.94733715;
        let oversampling_factor = 256;
        let window = WindowFunction::BlackmanHarris2;

        let interpolator =
            ScalarInterpolator::<f32>::new(sinc_len, oversampling_factor, f_cutoff, window);
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &interpolator.sincs[123]);
        assert!((value - check).abs() < 1.0e-6);
    }
}
