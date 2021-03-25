use crate::sinc::make_sincs;
use crate::windows::WindowFunction;
use core::arch::x86_64::{
    __m128, __m128d, __m256, __m256d, _mm256_castpd256_pd128, _mm256_castps256_ps128,
    _mm256_extractf128_pd, _mm256_extractf128_ps,
};
use core::arch::x86_64::{
    _mm256_add_pd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm_add_pd,
};
use core::arch::x86_64::{
    _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm_add_ps, _mm_hadd_ps,
};

use crate::asynchro::SincInterpolator;

/// Trait governing what can be done with an AvxSample.
pub trait AvxSample: Sized + num_traits::Float {
    type Sinc;

    unsafe fn pack_sincs(sincs: Vec<Vec<Self>>) -> Vec<Vec<Self::Sinc>>;

    unsafe fn get_sinc_interpolated_unsafe(
        wave: &[Self],
        index: usize,
        subindex: usize,
        sincs: &Vec<Vec<Self::Sinc>>,
        length: usize,
    ) -> Self;
}

impl AvxSample for f32 {
    type Sinc = __m256;

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn pack_sincs(sincs: Vec<Vec<Self>>) -> Vec<Vec<Self::Sinc>> {
        let mut packed_sincs = Vec::new();
        for sinc in sincs.iter() {
            let mut packed = Vec::new();
            for elements in sinc.chunks(8) {
                let packed_elems = _mm256_loadu_ps(&elements[0]);
                packed.push(packed_elems);
            }
            packed_sincs.push(packed);
        }
        packed_sincs
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn get_sinc_interpolated_unsafe(
        wave: &[f32],
        index: usize,
        subindex: usize,
        sincs: &Vec<Vec<Self::Sinc>>,
        length: usize,
    ) -> f32 {
        let sinc = sincs.get_unchecked(subindex);
        let wave_cut = &wave[index..(index + length)];
        let mut acc = _mm256_setzero_ps();
        let mut w_idx = 0;
        for s_idx in 0..length / 8 {
            let w = _mm256_loadu_ps(wave_cut.get_unchecked(w_idx));
            acc = _mm256_fmadd_ps(w, *sinc.get_unchecked(s_idx), acc);
            w_idx += 8;
        }
        let acc_high = _mm256_extractf128_ps(acc, 1);
        let mut acc_low = _mm_add_ps(acc_high, _mm256_castps256_ps128(acc));
        acc_low = _mm_hadd_ps(acc_low, acc_low);
        let array = std::mem::transmute::<__m128, [f32; 4]>(acc_low);
        array[0] + array[1]
    }
}

impl AvxSample for f64 {
    type Sinc = __m256d;

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn pack_sincs(sincs: Vec<Vec<f64>>) -> Vec<Vec<Self::Sinc>> {
        let mut packed_sincs = Vec::new();
        for sinc in sincs.iter() {
            let mut packed = Vec::new();
            for elements in sinc.chunks(4) {
                let packed_elems = _mm256_loadu_pd(&elements[0]);
                packed.push(packed_elems);
            }
            packed_sincs.push(packed);
        }
        packed_sincs
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn get_sinc_interpolated_unsafe(
        wave: &[f64],
        index: usize,
        subindex: usize,
        sincs: &Vec<Vec<Self::Sinc>>,
        length: usize,
    ) -> f64 {
        let sinc = sincs.get_unchecked(subindex);
        let wave_cut = &wave[index..(index + length)];
        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();
        let mut w_idx = 0;
        let mut s_idx = 0;
        for _ in 0..wave_cut.len() / 8 {
            let w0 = _mm256_loadu_pd(wave_cut.get_unchecked(w_idx));
            let w1 = _mm256_loadu_pd(wave_cut.get_unchecked(w_idx + 4));
            acc0 = _mm256_fmadd_pd(w0, *sinc.get_unchecked(s_idx), acc0);
            acc1 = _mm256_fmadd_pd(w1, *sinc.get_unchecked(s_idx + 1), acc1);
            w_idx += 8;
            s_idx += 2;
        }
        let acc_all = _mm256_add_pd(acc0, acc1);
        let acc_high = _mm256_extractf128_pd(acc_all, 1);
        let acc = _mm_add_pd(acc_high, _mm256_castpd256_pd128(acc_all));
        let array = std::mem::transmute::<__m128d, [f64; 2]>(acc);
        array[0] + array[1]
    }
}

/// An AVX accelerated interpolator
pub struct AvxInterpolator<T> where T: AvxSample {
    sincs: Vec<Vec<T::Sinc>>,
    length: usize,
    nbr_sincs: usize,
}

impl<T> SincInterpolator<T> for AvxInterpolator<T> where T: AvxSample {
    /// Calculate the scalar produt of an input wave and the selected sinc filter
    fn get_sinc_interpolated(&self, wave: &[T], index: usize, subindex: usize) -> T {
        assert!((index + self.length) < wave.len());
        assert!(subindex < self.nbr_sincs);
        unsafe { T::get_sinc_interpolated_unsafe(wave, index, subindex, &self.sincs, self.length) }
    }

    fn len(&self) -> usize {
        self.length
    }

    fn nbr_sincs(&self) -> usize {
        self.nbr_sincs
    }
}

impl<T> AvxInterpolator<T> where T: AvxSample {
    /// Create a new AvxInterpolator
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
            is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma"),
            "CPU does not have the required AVX and FMA support!"
        );
        assert!(sinc_len % 8 == 0, "Sinc length must be a multiple of 8.");
        let sincs = make_sincs(sinc_len, oversampling_factor, f_cutoff, window);
        let sincs = unsafe { T::pack_sincs(sincs) };
        Self {
            sincs,
            length: sinc_len,
            nbr_sincs: oversampling_factor,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::asynchro::SincInterpolator;
    use crate::interpolator_avx::AvxInterpolator;
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
    fn test_avx_interpolator_64() {
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
            AvxInterpolator::<f64>::new(sinc_len, oversampling_factor, f_cutoff, window);
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &sincs[123]);
        assert!((value - check).abs() < 1.0e-9);
    }

    #[test]
    fn test_avx_interpolator_32() {
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
            AvxInterpolator::<f32>::new(sinc_len, oversampling_factor, f_cutoff, window);
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &sincs[123]);
        assert!((value - check).abs() < 1.0e-6);
    }
}
