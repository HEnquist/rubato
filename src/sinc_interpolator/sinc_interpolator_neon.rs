use crate::error::{CpuFeature, MissingCpuFeature};
use crate::sinc::make_sincs;
use crate::sinc_interpolator::{AlignedBuf, SincInterpolator};
use crate::windows::WindowFunction;
use crate::Sample;
use core::arch::aarch64::{float32x4_t, float64x2_t};
use core::arch::aarch64::{
    vadd_f32, vaddq_f32, vfmaq_f32, vget_high_f32, vget_low_f32, vld1q_f32, vmovq_n_f32,
    vst1_f32, vst1q_f32,
};
use core::arch::aarch64::{vaddq_f64, vfmaq_f64, vld1q_f64, vmovq_n_f64, vst1q_f64};

/// Collection of CPU features required for this interpolator.
static FEATURES: &[CpuFeature] = &[CpuFeature::Neon];

/// Runtime-length f32 fallback with 4 accumulators.
#[target_feature(enable = "neon")]
unsafe fn dot_neon_f32_dyn(wave: &[f32], index: usize, sinc: &[f32], length: usize) -> f32 {
    let wave_cut = &wave[index..(index + length)];
    let mut acc0 = vmovq_n_f32(0.0);
    let mut acc1 = vmovq_n_f32(0.0);
    let mut acc2 = vmovq_n_f32(0.0);
    let mut acc3 = vmovq_n_f32(0.0);
    let mut idx = 0;
    for _ in 0..length / 16 {
        acc0 = vfmaq_f32(acc0, vld1q_f32(wave_cut.get_unchecked(idx)),      vld1q_f32(sinc.get_unchecked(idx)));
        acc1 = vfmaq_f32(acc1, vld1q_f32(wave_cut.get_unchecked(idx + 4)),  vld1q_f32(sinc.get_unchecked(idx + 4)));
        acc2 = vfmaq_f32(acc2, vld1q_f32(wave_cut.get_unchecked(idx + 8)),  vld1q_f32(sinc.get_unchecked(idx + 8)));
        acc3 = vfmaq_f32(acc3, vld1q_f32(wave_cut.get_unchecked(idx + 12)), vld1q_f32(sinc.get_unchecked(idx + 12)));
        idx += 16;
    }
    for _ in 0..(length % 16) / 8 {
        acc0 = vfmaq_f32(acc0, vld1q_f32(wave_cut.get_unchecked(idx)),     vld1q_f32(sinc.get_unchecked(idx)));
        acc1 = vfmaq_f32(acc1, vld1q_f32(wave_cut.get_unchecked(idx + 4)), vld1q_f32(sinc.get_unchecked(idx + 4)));
        idx += 8;
    }
    let sum4 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    let high = vget_high_f32(sum4);
    let low = vget_low_f32(sum4);
    let sum2 = vadd_f32(high, low);
    let mut array = [0.0f32, 0.0f32];
    vst1_f32(array.as_mut_ptr(), sum2);
    array[0] + array[1]
}

/// Runtime-length f64 with 8 accumulators; processes 16 f64 per iteration (16 iterations
/// for the typical length=256), matching the trip count of dot_neon_f32_dyn.
#[target_feature(enable = "neon")]
unsafe fn dot_neon_f64_dyn(wave: &[f64], index: usize, sinc: &[f64], length: usize) -> f64 {
    let wave_cut = &wave[index..(index + length)];
    let mut acc0 = vmovq_n_f64(0.0);
    let mut acc1 = vmovq_n_f64(0.0);
    let mut acc2 = vmovq_n_f64(0.0);
    let mut acc3 = vmovq_n_f64(0.0);
    let mut acc4 = vmovq_n_f64(0.0);
    let mut acc5 = vmovq_n_f64(0.0);
    let mut acc6 = vmovq_n_f64(0.0);
    let mut acc7 = vmovq_n_f64(0.0);
    let mut idx = 0;
    for _ in 0..length / 16 {
        let w0 = vld1q_f64(wave_cut.get_unchecked(idx));
        let w1 = vld1q_f64(wave_cut.get_unchecked(idx + 2));
        let w2 = vld1q_f64(wave_cut.get_unchecked(idx + 4));
        let w3 = vld1q_f64(wave_cut.get_unchecked(idx + 6));
        let w4 = vld1q_f64(wave_cut.get_unchecked(idx + 8));
        let w5 = vld1q_f64(wave_cut.get_unchecked(idx + 10));
        let w6 = vld1q_f64(wave_cut.get_unchecked(idx + 12));
        let w7 = vld1q_f64(wave_cut.get_unchecked(idx + 14));
        acc0 = vfmaq_f64(acc0, w0, vld1q_f64(sinc.get_unchecked(idx)));
        acc1 = vfmaq_f64(acc1, w1, vld1q_f64(sinc.get_unchecked(idx + 2)));
        acc2 = vfmaq_f64(acc2, w2, vld1q_f64(sinc.get_unchecked(idx + 4)));
        acc3 = vfmaq_f64(acc3, w3, vld1q_f64(sinc.get_unchecked(idx + 6)));
        acc4 = vfmaq_f64(acc4, w4, vld1q_f64(sinc.get_unchecked(idx + 8)));
        acc5 = vfmaq_f64(acc5, w5, vld1q_f64(sinc.get_unchecked(idx + 10)));
        acc6 = vfmaq_f64(acc6, w6, vld1q_f64(sinc.get_unchecked(idx + 12)));
        acc7 = vfmaq_f64(acc7, w7, vld1q_f64(sinc.get_unchecked(idx + 14)));
        idx += 16;
    }
    for _ in 0..(length % 16) / 2 {
        acc0 = vfmaq_f64(acc0, vld1q_f64(wave_cut.get_unchecked(idx)), vld1q_f64(sinc.get_unchecked(idx)));
        idx += 2;
    }
    let s01 = vaddq_f64(acc0, acc1);
    let s23 = vaddq_f64(acc2, acc3);
    let s45 = vaddq_f64(acc4, acc5);
    let s67 = vaddq_f64(acc6, acc7);
    let packedsum = vaddq_f64(vaddq_f64(s01, s23), vaddq_f64(s45, s67));
    let mut values = [0.0f64, 0.0f64];
    vst1q_f64(values.as_mut_ptr(), packedsum);
    values[0] + values[1]
}

/// Trait governing what can be done with an NeonSample.
pub trait NeonSample: Sized + Send {
    /// Compute the dot product of `wave[index..]` with `sinc` using NEON instructions.
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

    /// Compute `out[..length] += scale * input[..length]` using NEON instructions.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `out[..length]` and `input[..length]` are valid,
    /// and that `length` is a multiple of 8.
    unsafe fn saxpy_unsafe(out: &mut [Self], scale: Self, input: &[Self], length: usize);
}

impl NeonSample for f32 {
    #[target_feature(enable = "neon")]
    unsafe fn saxpy_unsafe(out: &mut [f32], scale: f32, input: &[f32], length: usize) {
        let scale_vec = vmovq_n_f32(scale);
        let mut idx = 0;
        for _ in 0..length / 4 {
            let x = vld1q_f32(input.get_unchecked(idx));
            let y = vld1q_f32(out.get_unchecked(idx));
            vst1q_f32(out.get_unchecked_mut(idx) as *mut f32, vfmaq_f32(y, scale_vec, x));
            idx += 4;
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn get_sinc_dot_product_unsafe(
        wave: &[f32],
        index: usize,
        sinc: &[f32],
        length: usize,
    ) -> f32 {
        dot_neon_f32_dyn(wave, index, sinc, length)
    }
}

impl NeonSample for f64 {
    #[target_feature(enable = "neon")]
    unsafe fn saxpy_unsafe(out: &mut [f64], scale: f64, input: &[f64], length: usize) {
        let scale_vec = vmovq_n_f64(scale);
        let mut idx = 0;
        for _ in 0..length / 2 {
            let x = vld1q_f64(input.get_unchecked(idx));
            let y = vld1q_f64(out.get_unchecked(idx));
            vst1q_f64(out.get_unchecked_mut(idx) as *mut f64, vfmaq_f64(y, scale_vec, x));
            idx += 2;
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn get_sinc_dot_product_unsafe(
        wave: &[f64],
        index: usize,
        sinc: &[f64],
        length: usize,
    ) -> f64 {
        dot_neon_f64_dyn(wave, index, sinc, length)
    }
}

/// A NEON accelerated interpolator.
#[cfg_attr(feature = "bench_asyncro", visibility::make(pub))]
pub(crate) struct NeonInterpolator<T>
where
    T: NeonSample,
{
    sincs: Vec<AlignedBuf<T>>,
    length: usize,
    nbr_sincs: usize,
}

impl<T> SincInterpolator<T> for NeonInterpolator<T>
where
    T: NeonSample,
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
                let ptr = self.sincs.get_unchecked(subindex).as_ptr();
                core::arch::asm!(
                    "prfm pldl1keep, [{ptr}]",
                    ptr = in(reg) ptr,
                    options(nostack, readonly, preserves_flags)
                );
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
                T::saxpy_unsafe(
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

impl<T> NeonInterpolator<T>
where
    T: NeonSample + Sample,
{
    /// Create a new NeonInterpolator.
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

// Suppress dead_code warnings for float32x4_t/float64x2_t used only in
// target_feature-gated functions.
#[allow(dead_code)]
const _: () = {
    let _ = core::mem::size_of::<float32x4_t>();
    let _ = core::mem::size_of::<float64x2_t>();
};

#[cfg(test)]
mod tests {
    use crate::sinc::make_sincs;
    use crate::sinc_interpolator::sinc_interpolator_neon::NeonInterpolator;
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
    fn test_neon_interpolator_64() {
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
            NeonInterpolator::<f64>::new(sinc_len, oversampling_factor, f_cutoff, window).unwrap();
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &sincs[123]);
        assert!((value - check).abs() < 1.0e-9);
    }

    #[test]
    fn test_neon_interpolator_32() {
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
            NeonInterpolator::<f32>::new(sinc_len, oversampling_factor, f_cutoff, window).unwrap();
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &sincs[123]);
        assert!((value - check).abs() < 1.0e-5);
    }
}
