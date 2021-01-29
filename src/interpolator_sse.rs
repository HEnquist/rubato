use crate::windows::WindowFunction;

use crate::sinc::make_sincs;
use core::arch::x86_64::{__m128, __m128d};
use core::arch::x86_64::{_mm_add_pd, _mm_hadd_pd, _mm_loadu_pd, _mm_mul_pd, _mm_setzero_pd};
use core::arch::x86_64::{_mm_add_ps, _mm_hadd_ps, _mm_loadu_ps, _mm_mul_ps, _mm_setzero_ps};
use std::marker::PhantomData;

use crate::asynchro::SincInterpolator;

/// A SSE accelerated interpolator
pub struct SseInterpolator<T> {
    sincs_s: Option<Vec<Vec<__m128>>>,
    sincs_d: Option<Vec<Vec<__m128d>>>,
    length: usize,
    nbr_sincs: usize,
    phantom: PhantomData<T>,
}

impl SincInterpolator<f32> for SseInterpolator<f32> {
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

impl SincInterpolator<f64> for SseInterpolator<f64> {
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

impl SseInterpolator<f32> {
    /// Create a new SseInterpolator
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
            is_x86_feature_detected!("sse3"),
            "CPU does not have the required SSE3 support!"
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

    fn pack_sincs_single(sincs: Vec<Vec<f32>>) -> Vec<Vec<__m128>> {
        let mut packed_sincs = Vec::new();
        for sinc in sincs.iter() {
            let mut packed = Vec::new();
            for elements in sinc.chunks(4) {
                unsafe {
                    let packed_elems = _mm_loadu_ps(&elements[0]);
                    packed.push(packed_elems);
                }
            }
            packed_sincs.push(packed);
        }
        packed_sincs
    }

    #[target_feature(enable = "sse3")]
    unsafe fn get_sinc_interpolated_unsafe(
        &self,
        wave: &[f32],
        index: usize,
        subindex: usize,
    ) -> f32 {
        let sinc = &self.sincs_s.as_ref().unwrap().get_unchecked(subindex);
        let wave_cut = &wave[index..(index + self.length)];
        let mut acc0 = _mm_setzero_ps();
        let mut acc1 = _mm_setzero_ps();
        let mut w_idx = 0;
        let mut s_idx = 0;
        for _ in 0..wave_cut.len() / 8 {
            let w0 = _mm_loadu_ps(wave_cut.get_unchecked(w_idx));
            let w1 = _mm_loadu_ps(wave_cut.get_unchecked(w_idx + 4));
            let s0 = _mm_mul_ps(w0, *sinc.get_unchecked(s_idx));
            let s1 = _mm_mul_ps(w1, *sinc.get_unchecked(s_idx + 1));
            acc0 = _mm_add_ps(acc0, s0);
            acc1 = _mm_add_ps(acc1, s1);
            w_idx += 8;
            s_idx += 2;
        }
        let mut packedsum = _mm_hadd_ps(acc0, acc1);
        packedsum = _mm_hadd_ps(packedsum, packedsum);
        let array = std::mem::transmute::<__m128, [f32; 4]>(packedsum);
        array[0] + array[1]
    }
}

impl SseInterpolator<f64> {
    /// Create a new SseInterpolator
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
            is_x86_feature_detected!("sse3"),
            "CPU does not have the required SSE3 support!"
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

    fn pack_sincs_double(sincs: Vec<Vec<f64>>) -> Vec<Vec<__m128d>> {
        let mut packed_sincs = Vec::new();
        for sinc in sincs.iter() {
            let mut packed = Vec::new();
            for elements in sinc.chunks(2) {
                unsafe {
                    let packed_elems = _mm_loadu_pd(&elements[0]);
                    packed.push(packed_elems);
                }
            }
            packed_sincs.push(packed);
        }
        packed_sincs
    }

    #[target_feature(enable = "sse3")]
    unsafe fn get_sinc_interpolated_unsafe(
        &self,
        wave: &[f64],
        index: usize,
        subindex: usize,
    ) -> f64 {
        let sinc = &self.sincs_d.as_ref().unwrap().get_unchecked(subindex);
        let wave_cut = &wave[index..(index + self.length)];
        let mut acc0 = _mm_setzero_pd();
        let mut acc1 = _mm_setzero_pd();
        let mut acc2 = _mm_setzero_pd();
        let mut acc3 = _mm_setzero_pd();
        let mut w_idx = 0;
        let mut s_idx = 0;
        for _ in 0..wave_cut.len() / 8 {
            let w0 = _mm_loadu_pd(wave_cut.get_unchecked(w_idx));
            let w1 = _mm_loadu_pd(wave_cut.get_unchecked(w_idx + 2));
            let w2 = _mm_loadu_pd(wave_cut.get_unchecked(w_idx + 4));
            let w3 = _mm_loadu_pd(wave_cut.get_unchecked(w_idx + 6));
            let s0 = _mm_mul_pd(w0, *sinc.get_unchecked(s_idx));
            let s1 = _mm_mul_pd(w1, *sinc.get_unchecked(s_idx + 1));
            let s2 = _mm_mul_pd(w2, *sinc.get_unchecked(s_idx + 2));
            let s3 = _mm_mul_pd(w3, *sinc.get_unchecked(s_idx + 3));
            acc0 = _mm_add_pd(acc0, s0);
            acc1 = _mm_add_pd(acc1, s1);
            acc2 = _mm_add_pd(acc2, s2);
            acc3 = _mm_add_pd(acc3, s3);
            w_idx += 8;
            s_idx += 4;
        }
        let mut packedsum0 = _mm_hadd_pd(acc0, acc1);
        let packedsum1 = _mm_hadd_pd(acc2, acc3);
        packedsum0 = _mm_hadd_pd(packedsum0, packedsum1);
        let array = std::mem::transmute::<__m128d, [f64; 2]>(packedsum0);
        array[0] + array[1]
    }
}

#[cfg(test)]
mod tests {
    use crate::asynchro::SincInterpolator;
    use crate::interpolator_sse::SseInterpolator;
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
    fn test_sse_interpolator_64() {
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
            SseInterpolator::<f64>::new(sinc_len, oversampling_factor, f_cutoff, window);
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &sincs[123]);
        assert!((value - check).abs() < 1.0e-9);
    }

    #[test]
    fn test_sse_interpolator_32() {
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
            SseInterpolator::<f32>::new(sinc_len, oversampling_factor, f_cutoff, window);
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &sincs[123]);
        assert!((value - check).abs() < 1.0e-6);
    }
}
