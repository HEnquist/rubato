use crate::error::{CpuFeature, MissingCpuFeature};
use crate::sinc::make_sincs;
use crate::sinc_interpolator::{AlignedBuf, SincInterpolator};
use crate::windows::WindowFunction;
use crate::Sample;
use core::arch::x86_64::{__m128, __m128d};
use core::arch::x86_64::{
    _mm_add_pd, _mm_hadd_pd, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_setzero_pd,
    _mm_store_sd, _mm_storeu_pd,
};
use core::arch::x86_64::{
    _mm_add_ps, _mm_hadd_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps,
    _mm_store_ss, _mm_storeu_ps,
};
use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

/// Runtime-length f32 fallback with 4 accumulators.
#[target_feature(enable = "sse3")]
unsafe fn dot_sse_f32_dyn(wave: &[f32], index: usize, sinc: &[f32], length: usize) -> f32 {
    let wave_cut = &wave[index..(index + length)];
    let mut acc0 = _mm_setzero_ps();
    let mut acc1 = _mm_setzero_ps();
    let mut acc2 = _mm_setzero_ps();
    let mut acc3 = _mm_setzero_ps();
    let mut idx = 0;
    for _ in 0..length / 16 {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(wave_cut.get_unchecked(idx)),      _mm_loadu_ps(sinc.get_unchecked(idx))));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(wave_cut.get_unchecked(idx + 4)),  _mm_loadu_ps(sinc.get_unchecked(idx + 4))));
        acc2 = _mm_add_ps(acc2, _mm_mul_ps(_mm_loadu_ps(wave_cut.get_unchecked(idx + 8)),  _mm_loadu_ps(sinc.get_unchecked(idx + 8))));
        acc3 = _mm_add_ps(acc3, _mm_mul_ps(_mm_loadu_ps(wave_cut.get_unchecked(idx + 12)), _mm_loadu_ps(sinc.get_unchecked(idx + 12))));
        idx += 16;
    }
    for _ in 0..(length % 16) / 4 {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(wave_cut.get_unchecked(idx)), _mm_loadu_ps(sinc.get_unchecked(idx))));
        idx += 4;
    }
    let temp4 = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
    let temp2 = _mm_hadd_ps(temp4, temp4);
    let temp1 = _mm_hadd_ps(temp2, temp2);
    let mut result = 0.0f32;
    _mm_store_ss(&mut result, temp1);
    result
}

/// Runtime-length f64 with 8 accumulators; processes 16 f64 per iteration (16 iterations
/// for the typical length=256), matching the trip count of dot_sse_f32_dyn and dot_avx_f64_dyn.
#[target_feature(enable = "sse3")]
unsafe fn dot_sse_f64_dyn(wave: &[f64], index: usize, sinc: &[f64], length: usize) -> f64 {
    let wave_cut = &wave[index..(index + length)];
    let mut acc0 = _mm_setzero_pd();
    let mut acc1 = _mm_setzero_pd();
    let mut acc2 = _mm_setzero_pd();
    let mut acc3 = _mm_setzero_pd();
    let mut acc4 = _mm_setzero_pd();
    let mut acc5 = _mm_setzero_pd();
    let mut acc6 = _mm_setzero_pd();
    let mut acc7 = _mm_setzero_pd();
    let mut idx = 0;
    for _ in 0..length / 16 {
        acc0 = _mm_add_pd(acc0, _mm_mul_pd(_mm_loadu_pd(wave_cut.get_unchecked(idx)),      _mm_loadu_pd(sinc.get_unchecked(idx))));
        acc1 = _mm_add_pd(acc1, _mm_mul_pd(_mm_loadu_pd(wave_cut.get_unchecked(idx + 2)),  _mm_loadu_pd(sinc.get_unchecked(idx + 2))));
        acc2 = _mm_add_pd(acc2, _mm_mul_pd(_mm_loadu_pd(wave_cut.get_unchecked(idx + 4)),  _mm_loadu_pd(sinc.get_unchecked(idx + 4))));
        acc3 = _mm_add_pd(acc3, _mm_mul_pd(_mm_loadu_pd(wave_cut.get_unchecked(idx + 6)),  _mm_loadu_pd(sinc.get_unchecked(idx + 6))));
        acc4 = _mm_add_pd(acc4, _mm_mul_pd(_mm_loadu_pd(wave_cut.get_unchecked(idx + 8)),  _mm_loadu_pd(sinc.get_unchecked(idx + 8))));
        acc5 = _mm_add_pd(acc5, _mm_mul_pd(_mm_loadu_pd(wave_cut.get_unchecked(idx + 10)), _mm_loadu_pd(sinc.get_unchecked(idx + 10))));
        acc6 = _mm_add_pd(acc6, _mm_mul_pd(_mm_loadu_pd(wave_cut.get_unchecked(idx + 12)), _mm_loadu_pd(sinc.get_unchecked(idx + 12))));
        acc7 = _mm_add_pd(acc7, _mm_mul_pd(_mm_loadu_pd(wave_cut.get_unchecked(idx + 14)), _mm_loadu_pd(sinc.get_unchecked(idx + 14))));
        idx += 16;
    }
    for _ in 0..(length % 16) / 2 {
        acc0 = _mm_add_pd(acc0, _mm_mul_pd(_mm_loadu_pd(wave_cut.get_unchecked(idx)), _mm_loadu_pd(sinc.get_unchecked(idx))));
        idx += 2;
    }
    let s01 = _mm_add_pd(acc0, acc1);
    let s23 = _mm_add_pd(acc2, acc3);
    let s45 = _mm_add_pd(acc4, acc5);
    let s67 = _mm_add_pd(acc6, acc7);
    let temp2 = _mm_hadd_pd(_mm_add_pd(s01, s23), _mm_add_pd(s45, s67));
    let temp1 = _mm_hadd_pd(temp2, temp2);
    let mut result = 0.0f64;
    _mm_store_sd(&mut result, temp1);
    result
}

/// Collection of CPU features required for this interpolator.
static FEATURES: &[CpuFeature] = &[CpuFeature::Sse3];

/// Trait governing what can be done with an SseSample.
pub trait SseSample: Sized + Send {
    /// Compute the dot product of `wave[index..]` with `sinc` using SSE instructions.
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

    /// Compute `out[..length] += scale * input[..length]` using SSE instructions.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `out[..length]` and `input[..length]` are valid,
    /// and that `length` is a multiple of 8.
    unsafe fn saxpy_unsafe(out: &mut [Self], scale: Self, input: &[Self], length: usize);
}

impl SseSample for f32 {
    #[target_feature(enable = "sse3")]
    unsafe fn saxpy_unsafe(out: &mut [f32], scale: f32, input: &[f32], length: usize) {
        let scale_vec = _mm_set1_ps(scale);
        let mut idx = 0;
        for _ in 0..length / 4 {
            let x = _mm_loadu_ps(input.get_unchecked(idx));
            let y = _mm_loadu_ps(out.get_unchecked(idx));
            _mm_storeu_ps(
                out.get_unchecked_mut(idx) as *mut f32,
                _mm_add_ps(y, _mm_mul_ps(scale_vec, x)),
            );
            idx += 4;
        }
    }

    #[target_feature(enable = "sse3")]
    unsafe fn get_sinc_dot_product_unsafe(
        wave: &[f32],
        index: usize,
        sinc: &[f32],
        length: usize,
    ) -> f32 {
        dot_sse_f32_dyn(wave, index, sinc, length)
    }
}

impl SseSample for f64 {
    #[target_feature(enable = "sse3")]
    unsafe fn saxpy_unsafe(out: &mut [f64], scale: f64, input: &[f64], length: usize) {
        let scale_vec = _mm_set1_pd(scale);
        let mut idx = 0;
        for _ in 0..length / 2 {
            let x = _mm_loadu_pd(input.get_unchecked(idx));
            let y = _mm_loadu_pd(out.get_unchecked(idx));
            _mm_storeu_pd(
                out.get_unchecked_mut(idx) as *mut f64,
                _mm_add_pd(y, _mm_mul_pd(scale_vec, x)),
            );
            idx += 2;
        }
    }

    #[target_feature(enable = "sse3")]
    unsafe fn get_sinc_dot_product_unsafe(
        wave: &[f64],
        index: usize,
        sinc: &[f64],
        length: usize,
    ) -> f64 {
        dot_sse_f64_dyn(wave, index, sinc, length)
    }
}

/// A SSE accelerated interpolator.
#[cfg_attr(feature = "bench_asyncro", visibility::make(pub))]
pub(crate) struct SseInterpolator<T>
where
    T: SseSample,
{
    sincs: Vec<AlignedBuf<T>>,
    length: usize,
    nbr_sincs: usize,
}

impl<T> SincInterpolator<T> for SseInterpolator<T>
where
    T: SseSample,
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
        let min_idx = nearest.iter().map(|n| n.0).min().unwrap();
        // memset to zero — valid for f32/f64 since all-zero bits represent 0.0.
        unsafe {
            std::ptr::write_bytes(combined.as_mut_ptr(), 0, combined.len());
        }
        for (n, &w) in nearest.iter().zip(weights.iter()) {
            let shift = (n.0 - min_idx) as usize;
            unsafe {
                <T as SseSample>::saxpy_unsafe(
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

impl<T> SseInterpolator<T>
where
    T: SseSample + Sample,
{
    /// Create a new SseInterpolator.
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

// Suppress dead_code warnings for __m128/__m128d used only in target_feature-gated functions.
#[allow(dead_code)]
const _: () = {
    let _ = core::mem::size_of::<__m128>();
    let _ = core::mem::size_of::<__m128d>();
};

#[cfg(test)]
mod tests {
    use crate::sinc::make_sincs;
    use crate::sinc_interpolator::sinc_interpolator_sse::SseInterpolator;
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
    fn test_sse_interpolator_64() {
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
            SseInterpolator::<f64>::new(sinc_len, oversampling_factor, f_cutoff, window).unwrap();
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &sincs[123]);
        assert!((value - check).abs() < 1.0e-9);
    }

    #[test]
    fn test_sse_interpolator_32() {
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
            SseInterpolator::<f32>::new(sinc_len, oversampling_factor, f_cutoff, window).unwrap();
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &sincs[123]);
        assert!((value - check).abs() < 1.0e-5);
    }
}
