use crate::sinc::make_sincs;
use crate::windows::WindowFunction;
use crate::Sample;

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
#[cfg_attr(feature = "bench_asyncro", visibility::make(pub))]
pub(crate) trait SincInterpolator<T>: Send {
    /// Make the scalar product between the waveform starting at `index` and the sinc of `subindex`.
    fn get_sinc_interpolated(&self, wave: &[T], index: usize, subindex: usize) -> T;

    /// Get sinc length.
    fn nbr_points(&self) -> usize;

    /// Get number of sincs used for oversampling.
    fn nbr_sincs(&self) -> usize;

    /// Issue a hardware prefetch hint for the sinc row at `subindex`.
    /// The default is a no-op; SIMD implementations may override this to bring
    /// the next sinc row into cache while the current one is being processed.
    #[inline]
    fn prefetch_sinc(&self, _subindex: usize) {}
}

/// A concrete enum over every sinc interpolator implementation.
///
/// Replaces a `Box<dyn SincInterpolator<T>>` so that the hot path can dispatch
/// via a `match` on a known-small set of variants. This enables the compiler to
/// inline `get_sinc_interpolated`, which in turn unlocks unrolling of the inner
/// FMA loop and cross-call register reuse.
#[cfg_attr(feature = "bench_asyncro", visibility::make(pub))]
pub(crate) enum AnyInterpolator<T>
where
    T: Sample,
{
    #[cfg(target_arch = "x86_64")]
    Avx(sinc_interpolator_avx::AvxInterpolator<T>),
    #[cfg(target_arch = "x86_64")]
    Sse(sinc_interpolator_sse::SseInterpolator<T>),
    #[cfg(target_arch = "aarch64")]
    Neon(sinc_interpolator_neon::NeonInterpolator<T>),
    Scalar(ScalarInterpolator<T>),
}

impl<T> SincInterpolator<T> for AnyInterpolator<T>
where
    T: Sample,
{
    #[inline]
    fn get_sinc_interpolated(&self, wave: &[T], index: usize, subindex: usize) -> T {
        match self {
            #[cfg(target_arch = "x86_64")]
            AnyInterpolator::Avx(i) => i.get_sinc_interpolated(wave, index, subindex),
            #[cfg(target_arch = "x86_64")]
            AnyInterpolator::Sse(i) => i.get_sinc_interpolated(wave, index, subindex),
            #[cfg(target_arch = "aarch64")]
            AnyInterpolator::Neon(i) => i.get_sinc_interpolated(wave, index, subindex),
            AnyInterpolator::Scalar(i) => i.get_sinc_interpolated(wave, index, subindex),
        }
    }

    #[inline]
    fn nbr_points(&self) -> usize {
        match self {
            #[cfg(target_arch = "x86_64")]
            AnyInterpolator::Avx(i) => i.nbr_points(),
            #[cfg(target_arch = "x86_64")]
            AnyInterpolator::Sse(i) => i.nbr_points(),
            #[cfg(target_arch = "aarch64")]
            AnyInterpolator::Neon(i) => i.nbr_points(),
            AnyInterpolator::Scalar(i) => i.nbr_points(),
        }
    }

    #[inline]
    fn nbr_sincs(&self) -> usize {
        match self {
            #[cfg(target_arch = "x86_64")]
            AnyInterpolator::Avx(i) => i.nbr_sincs(),
            #[cfg(target_arch = "x86_64")]
            AnyInterpolator::Sse(i) => i.nbr_sincs(),
            #[cfg(target_arch = "aarch64")]
            AnyInterpolator::Neon(i) => i.nbr_sincs(),
            AnyInterpolator::Scalar(i) => i.nbr_sincs(),
        }
    }

    #[inline]
    fn prefetch_sinc(&self, subindex: usize) {
        match self {
            #[cfg(target_arch = "x86_64")]
            AnyInterpolator::Avx(i) => i.prefetch_sinc(subindex),
            #[cfg(target_arch = "x86_64")]
            AnyInterpolator::Sse(i) => i.prefetch_sinc(subindex),
            #[cfg(target_arch = "aarch64")]
            AnyInterpolator::Neon(i) => i.prefetch_sinc(subindex),
            AnyInterpolator::Scalar(i) => i.prefetch_sinc(subindex),
        }
    }
}

impl<T> From<ScalarInterpolator<T>> for AnyInterpolator<T>
where
    T: Sample,
{
    fn from(value: ScalarInterpolator<T>) -> Self {
        AnyInterpolator::Scalar(value)
    }
}

#[cfg(target_arch = "x86_64")]
impl<T> From<sinc_interpolator_avx::AvxInterpolator<T>> for AnyInterpolator<T>
where
    T: Sample,
{
    fn from(value: sinc_interpolator_avx::AvxInterpolator<T>) -> Self {
        AnyInterpolator::Avx(value)
    }
}

#[cfg(target_arch = "x86_64")]
impl<T> From<sinc_interpolator_sse::SseInterpolator<T>> for AnyInterpolator<T>
where
    T: Sample,
{
    fn from(value: sinc_interpolator_sse::SseInterpolator<T>) -> Self {
        AnyInterpolator::Sse(value)
    }
}

#[cfg(target_arch = "aarch64")]
impl<T> From<sinc_interpolator_neon::NeonInterpolator<T>> for AnyInterpolator<T>
where
    T: Sample,
{
    fn from(value: sinc_interpolator_neon::NeonInterpolator<T>) -> Self {
        AnyInterpolator::Neon(value)
    }
}

/// A plain scalar interpolator.
#[cfg_attr(feature = "bench_asyncro", visibility::make(pub))]
pub(crate) struct ScalarInterpolator<T> {
    sincs: Vec<Vec<T>>,
    length: usize,
    nbr_sincs: usize,
}

impl<T> SincInterpolator<T> for ScalarInterpolator<T>
where
    T: Sample,
{
    /// Calculate the scalar produt of an input wave and the selected sinc filter.
    fn get_sinc_interpolated(&self, wave: &[T], index: usize, subindex: usize) -> T {
        assert!(
            (index + self.length) < wave.len(),
            "Tried to interpolate for index {}, max for the given input is {}",
            index,
            wave.len() - self.length - 1
        );
        assert!(
            subindex < self.nbr_sincs,
            "Tried to use sinc subindex {}, max is {}",
            subindex,
            self.nbr_sincs - 1
        );
        let wave_cut = &wave[index..(index + self.sincs[subindex].len())];
        let sinc = &self.sincs[subindex];
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
        assert!(sinc_len % 8 == 0, "Sinc length must be a multiple of 8");
        let sincs = make_sincs(sinc_len, oversampling_factor, f_cutoff, window);
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
