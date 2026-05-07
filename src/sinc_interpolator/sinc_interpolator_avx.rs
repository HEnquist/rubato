use crate::error::{CpuFeature, MissingCpuFeature};
use crate::sinc::make_sincs;
use crate::sinc_interpolator::{AlignedBuf, SincInterpolator};
use crate::windows::WindowFunction;
use crate::Sample;
use core::arch::x86_64::{
    __m256, __m256d, _mm256_castpd256_pd128, _mm256_castps256_ps128, _mm256_extractf128_pd,
    _mm256_extractf128_ps,
};
use core::arch::x86_64::{
    _mm256_add_pd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_setzero_pd,
    _mm256_storeu_pd, _mm_add_pd, _mm_hadd_pd, _mm_store_sd,
};
use core::arch::x86_64::{
    _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps, _mm_add_ps, _mm_hadd_ps, _mm_store_ss,
};
use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

/// Collection of CPU features required for this interpolator.
static FEATURES: &[CpuFeature] = &[CpuFeature::Avx, CpuFeature::Fma];

/// Runtime-length fallback with 4 accumulators; handles sinc lengths not in {64, 128, 256}.
#[target_feature(enable = "avx", enable = "fma")]
unsafe fn dot_avx_f32_dyn(wave: &[f32], index: usize, sinc: &[f32], length: usize) -> f32 {
    let wave_cut = &wave[index..(index + length)];
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut idx = 0;
    for _ in 0..length / 32 {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(wave_cut.get_unchecked(idx)),      _mm256_loadu_ps(sinc.get_unchecked(idx)),      acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(wave_cut.get_unchecked(idx + 8)),  _mm256_loadu_ps(sinc.get_unchecked(idx + 8)),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(wave_cut.get_unchecked(idx + 16)), _mm256_loadu_ps(sinc.get_unchecked(idx + 16)), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(wave_cut.get_unchecked(idx + 24)), _mm256_loadu_ps(sinc.get_unchecked(idx + 24)), acc3);
        idx += 32;
    }
    for _ in 0..(length % 32) / 8 {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(wave_cut.get_unchecked(idx)), _mm256_loadu_ps(sinc.get_unchecked(idx)), acc0);
        idx += 8;
    }
    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let acc_high = _mm256_extractf128_ps(acc, 1);
    let acc_low = _mm_add_ps(acc_high, _mm256_castps256_ps128(acc));
    let temp2 = _mm_hadd_ps(acc_low, acc_low);
    let temp1 = _mm_hadd_ps(temp2, temp2);
    let mut result = 0.0f32;
    _mm_store_ss(&mut result, temp1);
    result
}

/// Runtime-length fallback for f64; handles sinc lengths not in {64, 128, 256}.
#[target_feature(enable = "avx", enable = "fma")]
unsafe fn dot_avx_f64_dyn(wave: &[f64], index: usize, sinc: &[f64], length: usize) -> f64 {
    let wave_cut = &wave[index..(index + length)];
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    let mut acc2 = _mm256_setzero_pd();
    let mut acc3 = _mm256_setzero_pd();
    let mut idx = 0;
    for _ in 0..length / 16 {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(wave_cut.get_unchecked(idx)),      _mm256_loadu_pd(sinc.get_unchecked(idx)),      acc0);
        acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(wave_cut.get_unchecked(idx + 4)),  _mm256_loadu_pd(sinc.get_unchecked(idx + 4)),  acc1);
        acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(wave_cut.get_unchecked(idx + 8)),  _mm256_loadu_pd(sinc.get_unchecked(idx + 8)),  acc2);
        acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(wave_cut.get_unchecked(idx + 12)), _mm256_loadu_pd(sinc.get_unchecked(idx + 12)), acc3);
        idx += 16;
    }
    for _ in 0..(length % 16) / 4 {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(wave_cut.get_unchecked(idx)), _mm256_loadu_pd(sinc.get_unchecked(idx)), acc0);
        idx += 4;
    }
    let acc = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
    let acc_high = _mm256_extractf128_pd(acc, 1);
    let temp2 = _mm_add_pd(acc_high, _mm256_castpd256_pd128(acc));
    let temp1 = _mm_hadd_pd(temp2, temp2);
    let mut result = 0.0f64;
    _mm_store_sd(&mut result, temp1);
    result
}

/// Trait governing what can be done with an AvxSample.
pub trait AvxSample: Sized + Send {
    /// Compute the dot product of `wave[index..]` with `sinc` using AVX instructions.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `wave[index..index+length]` and `sinc[..length]` are
    /// valid, and that `length` is a multiple of 8.
    unsafe fn get_sinc_dot_product_unsafe(
        wave: &[Self],
        index: usize,
        sinc: &[Self],
        length: usize,
    ) -> Self;

    /// Compute `out[..length] += scale * input[..length]` using AVX instructions.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `out[..length]` and `input[..length]` are valid,
    /// and that `length` is a multiple of 8.
    unsafe fn accumulate_scaled_unsafe(out: &mut [Self], scale: Self, input: &[Self], length: usize);
}

impl AvxSample for f32 {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn accumulate_scaled_unsafe(out: &mut [f32], scale: f32, input: &[f32], length: usize) {
        let scale_vec = _mm256_set1_ps(scale);
        let mut idx = 0;
        for _ in 0..length / 8 {
            let x = _mm256_loadu_ps(input.get_unchecked(idx));
            let y = _mm256_loadu_ps(out.get_unchecked(idx));
            _mm256_storeu_ps(
                out.get_unchecked_mut(idx) as *mut f32,
                _mm256_fmadd_ps(scale_vec, x, y),
            );
            idx += 8;
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn get_sinc_dot_product_unsafe(
        wave: &[f32],
        index: usize,
        sinc: &[f32],
        length: usize,
    ) -> f32 {
        dot_avx_f32_dyn(wave, index, sinc, length)
    }
}

impl AvxSample for f64 {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn accumulate_scaled_unsafe(out: &mut [f64], scale: f64, input: &[f64], length: usize) {
        let scale_vec = _mm256_set1_pd(scale);
        let mut idx = 0;
        for _ in 0..length / 4 {
            let x = _mm256_loadu_pd(input.get_unchecked(idx));
            let y = _mm256_loadu_pd(out.get_unchecked(idx));
            _mm256_storeu_pd(
                out.get_unchecked_mut(idx) as *mut f64,
                _mm256_fmadd_pd(scale_vec, x, y),
            );
            idx += 4;
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn get_sinc_dot_product_unsafe(
        wave: &[f64],
        index: usize,
        sinc: &[f64],
        length: usize,
    ) -> f64 {
        // Use the looped _dyn path for f64. The const-generic versions
        // fully unroll the inner loop (~64–128 sequential FMAs for typical
        // sinc lengths), which cannot benefit from the Loop Stream Detector
        // and pays a larger μOp-cache delivery cost than the compact loop
        // body. The _dyn variant has the same 4-accumulator structure but
        // stays short enough for the LSD to replay at near-peak throughput.
        dot_avx_f64_dyn(wave, index, sinc, length)
    }
}

/// An AVX accelerated interpolator.
#[cfg_attr(feature = "bench_asyncro", visibility::make(pub))]
pub(crate) struct AvxInterpolator<T>
where
    T: AvxSample,
{
    sincs: Vec<AlignedBuf<T>>,
    length: usize,
    nbr_sincs: usize,
}

impl<T> SincInterpolator<T> for AvxInterpolator<T>
where
    T: AvxSample,
{
    fn get_sinc_dot_product(&self, wave: &[T], index: usize, sinc: &[T]) -> T {
        unsafe { T::get_sinc_dot_product_unsafe(wave, index, sinc, self.length) }
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

    #[inline]
    fn prefetch_sinc(&self, subindex: usize) {
        if subindex < self.nbr_sincs {
            unsafe {
                let row = self.sincs.get_unchecked(subindex);
                _mm_prefetch::<_MM_HINT_T0>(row.as_ptr() as *const i8);
            }
        }
    }

    fn make_combined_sinc(
        &self,
        nearest: &[(isize, isize)],
        weights: &[T],
        combined: &mut [T],
    ) -> isize
    where
        T: crate::Sample,
    {
        debug_assert_eq!(
            combined.len(),
            self.length + 1,
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
            unsafe {
                <T as AvxSample>::accumulate_scaled_unsafe(
                    &mut combined[shift..shift + self.length],
                    w,
                    &self.sincs[n.1 as usize],
                    self.length,
                );
            }
        }
        min_idx
    }
}

impl<T> AvxInterpolator<T>
where
    T: AvxSample + Sample,
{
    /// Create a new AvxInterpolator.
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
    ) -> Result<Self, MissingCpuFeature> {
        if let Some(feature) = FEATURES.iter().find(|f| !f.is_detected()) {
            return Err(MissingCpuFeature(*feature));
        }

        assert!(sinc_len % 8 == 0, "Sinc length must be a multiple of 8.");
        let raw_sincs: Vec<Vec<T>> = make_sincs(sinc_len, oversampling_factor, f_cutoff, window);
        let sincs = raw_sincs
            .into_iter()
            .map(|row| AlignedBuf::from_slice(&row))
            .collect();

        Ok(Self {
            sincs,
            length: sinc_len,
            nbr_sincs: oversampling_factor,
        })
    }
}

// __m256/__m256d are used only inside #[target_feature]-gated functions. The compiler
// performs dead-code analysis before monomorphising those gates, so it cannot see the
// uses and would emit an unused-import warning without this suppression.
#[allow(dead_code)]
const _: () = {
    let _ = core::mem::size_of::<__m256>();
    let _ = core::mem::size_of::<__m256d>();
};

#[cfg(test)]
mod tests {
    use crate::sinc::make_sincs;
    use crate::sinc_interpolator::sinc_interpolator_avx::AvxInterpolator;
    use crate::sinc_interpolator::SincInterpolator;
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
    fn test_avx_interpolator_64() {
        let mut wave = Vec::new();
        for _ in 0..2048 {
            wave.push(rand::random::<f64>());
        }
        let sinc_len = 256;
        let f_cutoff = 0.94733715;
        let oversampling_factor = 256;
        let window = WindowFunction::BlackmanHarris2;
        let sincs = make_sincs::<f64>(sinc_len, oversampling_factor, f_cutoff, window);

        let interpolator =
            match AvxInterpolator::<f64>::new(sinc_len, oversampling_factor, f_cutoff, window) {
                Ok(interpolator) => interpolator,
                Err(..) => {
                    assert!(!(is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma")));
                    return;
                }
            };

        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &sincs[123]);
        assert!((value - check).abs() < 1.0e-9);
    }

    #[test]
    fn test_avx_interpolator_32() {
        let mut wave = Vec::new();
        for _ in 0..2048 {
            wave.push(rand::random::<f32>());
        }
        let sinc_len = 256;
        let f_cutoff = 0.94733715;
        let oversampling_factor = 256;
        let window = WindowFunction::BlackmanHarris2;
        let sincs = make_sincs::<f32>(sinc_len, oversampling_factor, f_cutoff, window);

        let interpolator =
            match AvxInterpolator::<f32>::new(sinc_len, oversampling_factor, f_cutoff, window) {
                Ok(interpolator) => interpolator,
                Err(..) => {
                    assert!(!(is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma")));
                    return;
                }
            };

        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &sincs[123]);
        assert!((value - check).abs() < 1.0e-6);
    }
}
