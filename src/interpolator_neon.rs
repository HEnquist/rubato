use crate::windows::WindowFunction;

use crate::sinc::make_sincs;

use core::arch::aarch64::{float32x4_t, float64x2_t};
use core::arch::aarch64::{vaddq_f32, vmulq_f32};
use core::arch::aarch64::{vaddq_f64, vmulq_f64};
use packed_simd_2::{f32x4, f64x2};
use std::marker::PhantomData;

use crate::asynchro::SincInterpolator;

/// A SSE accelerated interpolator
pub struct NeonInterpolator<T> {
    sincs_s: Option<Vec<Vec<float32x4_t>>>,
    sincs_d: Option<Vec<Vec<float64x2_t>>>,
    length: usize,
    nbr_sincs: usize,
    phantom: PhantomData<T>,
}

impl SincInterpolator<f32> for NeonInterpolator<f32> {
    /// Calculate the scalar produt of an input wave and the selected sinc filter
    fn get_sinc_interpolated(&self, wave: &[f32], index: usize, subindex: usize) -> f32 {
        assert!((index + self.length) < wave.len());
        assert!(subindex < self.nbr_sincs);
        unsafe { self.get_sinc_interpolated_unsafe(wave, index, subindex) }
    }

    fn len(&self) -> usize {
        self.length
    }

    fn nbr_sincs(&self) -> usize {
        self.nbr_sincs
    }
}

impl SincInterpolator<f64> for NeonInterpolator<f64> {
    /// Calculate the scalar produt of an input wave and the selected sinc filter
    fn get_sinc_interpolated(&self, wave: &[f64], index: usize, subindex: usize) -> f64 {
        assert!((index + self.length) < wave.len());
        assert!(subindex < self.nbr_sincs);
        unsafe { self.get_sinc_interpolated_unsafe(wave, index, subindex) }
    }

    fn len(&self) -> usize {
        self.length
    }

    fn nbr_sincs(&self) -> usize {
        self.nbr_sincs
    }
}

impl NeonInterpolator<f32> {
    /// Create a new NeonInterpolator
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
            is_aarch64_feature_detected!("neon"),
            "CPU does not have the required Neon support!"
        );
        assert!(sinc_len % 8 == 0, "Sinc length must be a multiple of 8.");
        let sincs = make_sincs(sinc_len, oversampling_factor, f_cutoff, window);
        let sincs = Self::pack_sincs_single(sincs);
        Self {
            sincs_s: Some(sincs),
            sincs_d: None,
            length: sinc_len,
            nbr_sincs: oversampling_factor,
            phantom: PhantomData,
        }
    }

    fn pack_sincs_single(sincs: Vec<Vec<f32>>) -> Vec<Vec<float32x4_t>> {
        let mut packed_sincs = Vec::new();
        for sinc in sincs.iter() {
            let mut packed = Vec::new();
            for elements in sinc.chunks(4) {
                unsafe {
                    let elem = f32x4::from_slice_unaligned(elements);
                    let packed_elems = std::mem::transmute(elem);
                    packed.push(packed_elems);
                }
            }
            packed_sincs.push(packed);
        }
        packed_sincs
    }

    #[target_feature(enable = "neon")]
    unsafe fn get_sinc_interpolated_unsafe(
        &self,
        wave: &[f32],
        index: usize,
        subindex: usize,
    ) -> f32 {
        let sinc = &self.sincs_s.as_ref().unwrap().get_unchecked(subindex);
        let wave_cut = &wave[index..(index + self.length)];
        let mut acc0 = std::mem::transmute::<f32x4, float32x4_t>(f32x4::new(0.0, 0.0, 0.0, 0.0));
        let mut acc1 = std::mem::transmute::<f32x4, float32x4_t>(f32x4::new(0.0, 0.0, 0.0, 0.0));
        let mut w_idx = 0;
        let mut s_idx = 0;
        for _ in 0..wave_cut.len() / 8 {
            let w0 = std::mem::transmute(f32x4::from_slice_unaligned(
                wave_cut.get_unchecked(w_idx..w_idx + 4),
            ));
            let w1 = std::mem::transmute(f32x4::from_slice_unaligned(
                wave_cut.get_unchecked(w_idx + 4..w_idx + 8),
            ));
            let s0 = vmulq_f32(w0, *sinc.get_unchecked(s_idx));
            let s1 = vmulq_f32(w1, *sinc.get_unchecked(s_idx + 1));
            acc0 = vaddq_f32(acc0, s0);
            acc1 = vaddq_f32(acc1, s1);
            w_idx += 8;
            s_idx += 2;
        }
        let packedsum = vaddq_f32(acc0, acc1);
        let array = std::mem::transmute::<float32x4_t, [f32; 4]>(packedsum);
        array[0] + array[1] + array[2] + array[3]
    }
}

impl NeonInterpolator<f64> {
    /// Create a new NeonInterpolator
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
            is_aarch64_feature_detected!("neon"),
            "CPU does not have the required Neon support!"
        );
        assert!(sinc_len % 8 == 0, "Sinc length must be a multiple of 8.");
        let sincs = make_sincs(sinc_len, oversampling_factor, f_cutoff, window);
        let sincs = Self::pack_sincs_double(sincs);
        Self {
            sincs_d: Some(sincs),
            sincs_s: None,
            length: sinc_len,
            nbr_sincs: oversampling_factor,
            phantom: PhantomData,
        }
    }

    fn pack_sincs_double(sincs: Vec<Vec<f64>>) -> Vec<Vec<float64x2_t>> {
        let mut packed_sincs = Vec::new();
        for sinc in sincs.iter() {
            let mut packed = Vec::new();
            for elements in sinc.chunks(2) {
                unsafe {
                    let elem = f64x2::from_slice_unaligned(elements);
                    let packed_elems = std::mem::transmute(elem);
                    packed.push(packed_elems);
                }
            }
            packed_sincs.push(packed);
        }
        packed_sincs
    }

    #[target_feature(enable = "neon")]
    unsafe fn get_sinc_interpolated_unsafe(
        &self,
        wave: &[f64],
        index: usize,
        subindex: usize,
    ) -> f64 {
        let sinc = &self.sincs_d.as_ref().unwrap().get_unchecked(subindex);
        let wave_cut = &wave[index..(index + self.length)];
        let mut acc0 = std::mem::transmute::<f64x2, float64x2_t>(f64x2::new(0.0, 0.0));
        let mut acc1 = std::mem::transmute::<f64x2, float64x2_t>(f64x2::new(0.0, 0.0));
        let mut acc2 = std::mem::transmute::<f64x2, float64x2_t>(f64x2::new(0.0, 0.0));
        let mut acc3 = std::mem::transmute::<f64x2, float64x2_t>(f64x2::new(0.0, 0.0));
        let mut w_idx = 0;
        let mut s_idx = 0;
        for _ in 0..wave_cut.len() / 8 {
            let w0 = std::mem::transmute(f64x2::from_slice_unaligned(
                wave_cut.get_unchecked(w_idx..w_idx + 2),
            ));
            let w1 = std::mem::transmute(f64x2::from_slice_unaligned(
                wave_cut.get_unchecked(w_idx + 2..w_idx + 4),
            ));
            let w2 = std::mem::transmute(f64x2::from_slice_unaligned(
                wave_cut.get_unchecked(w_idx + 4..w_idx + 6),
            ));
            let w3 = std::mem::transmute(f64x2::from_slice_unaligned(
                wave_cut.get_unchecked(w_idx + 6..w_idx + 8),
            ));
            let s0 = vmulq_f64(w0, *sinc.get_unchecked(s_idx));
            let s1 = vmulq_f64(w1, *sinc.get_unchecked(s_idx + 1));
            let s2 = vmulq_f64(w2, *sinc.get_unchecked(s_idx + 2));
            let s3 = vmulq_f64(w3, *sinc.get_unchecked(s_idx + 3));
            acc0 = vaddq_f64(acc0, s0);
            acc1 = vaddq_f64(acc1, s1);
            acc2 = vaddq_f64(acc2, s2);
            acc3 = vaddq_f64(acc3, s3);
            w_idx += 8;
            s_idx += 4;
        }
        let mut packedsum0 = vaddq_f64(acc0, acc1);
        let packedsum1 = vaddq_f64(acc2, acc3);
        packedsum0 = vaddq_f64(packedsum0, packedsum1);
        let array = std::mem::transmute::<float64x2_t, [f64; 2]>(packedsum0);
        array[0] + array[1]
    }
}

#[cfg(test)]
mod tests {
    use crate::asynchro::SincInterpolator;
    use crate::interpolator_neon::NeonInterpolator;
    use crate::sinc::make_sincs;
    use crate::WindowFunction;
    use num_traits::Float;
    use rand::Rng;

    fn get_sinc_interpolated<T: Float>(wave: &[T], index: usize, sinc: &[T]) -> T {
        let wave_cut = &wave[index..(index + sinc.len())];
        wave_cut
            .iter()
            .zip(sinc.iter())
            .fold(T::zero(), |acc, (x, y)| acc + *x * *y)
    }

    #[test]
    fn test_neon_interpolator_64() {
        let mut rng = rand::thread_rng();
        let mut wave = Vec::new();
        for _ in 0..2048 {
            wave.push(rng.gen::<f64>());
        }
        let sinc_len = 256;
        let f_cutoff = 0.9473371669037001;
        let oversampling_factor = 256;
        let window = WindowFunction::BlackmanHarris2;
        let sincs = make_sincs::<f64>(sinc_len, oversampling_factor, f_cutoff, window);
        let interpolator =
            NeonInterpolator::<f64>::new(sinc_len, oversampling_factor, f_cutoff, window);
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &sincs[123]);
        assert!((value - check).abs() < 1.0e-9);
    }

    #[test]
    fn test_neon_interpolator_32() {
        let mut rng = rand::thread_rng();
        let mut wave = Vec::new();
        for _ in 0..2048 {
            wave.push(rng.gen::<f32>());
        }
        let sinc_len = 256;
        let f_cutoff = 0.9473371669037001;
        let oversampling_factor = 256;
        let window = WindowFunction::BlackmanHarris2;
        let sincs = make_sincs::<f32>(sinc_len, oversampling_factor, f_cutoff, window);
        let interpolator =
            NeonInterpolator::<f32>::new(sinc_len, oversampling_factor, f_cutoff, window);
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &sincs[123]);
        assert!((value - check).abs() < 1.0e-6);
    }
}
