use crate::asynchro::SincInterpolator;
use crate::sinc::make_sincs;
use crate::windows::WindowFunction;
use core::arch::aarch64::{float32x4_t, float64x2_t};
use core::arch::aarch64::{vaddq_f32, vmulq_f32};
use core::arch::aarch64::{vaddq_f64, vmulq_f64};
use packed_simd_2::{f32x4, f64x2};
use crate::error::{MissingCpuFeature, CpuFeature};
use crate::Sample;

/// Collection of cpu features required for this interpolator.
static FEATURES: &[CpuFeature] = &[CpuFeature::Neon];

/// Trait governing what can be done with an NeonSample.
pub trait NeonSample: Sized {
    type Sinc;

    /// Pack sincs into a vector.
    /// 
    /// # Safety
    /// 
    /// This is unsafe because it uses target_enable dispatching. There are no
    /// special requirements from the caller.
    unsafe fn pack_sincs(sincs: Vec<Vec<Self>>) -> Vec<Vec<Self::Sinc>>;

    /// Interpolate a sinc sample.
    /// 
    /// # Safety
    /// 
    /// The caller must ensure that the various indexes are not out of bounds
    /// in the collection of sincs.
    unsafe fn get_sinc_interpolated_unsafe(
        wave: &[Self],
        index: usize,
        subindex: usize,
        sincs: &[Vec<Self::Sinc>],
        length: usize,
    ) -> Self;
}

impl NeonSample for f32 {
    type Sinc = float32x4_t;

    #[target_feature(enable = "neon")]
    unsafe fn pack_sincs(sincs: Vec<Vec<Self>>) -> Vec<Vec<Self::Sinc>> {
        let mut packed_sincs = Vec::new();
        for sinc in sincs.iter() {
            let mut packed = Vec::new();
            for elements in sinc.chunks(4) {
                let elem = f32x4::from_slice_unaligned(elements);
                let packed_elems = std::mem::transmute(elem);
                packed.push(packed_elems);
            }
            packed_sincs.push(packed);
        }
        packed_sincs
    }

    #[target_feature(enable = "neon")]
    unsafe fn get_sinc_interpolated_unsafe(
        wave: &[f32],
        index: usize,
        subindex: usize,
        sincs: &[Vec<Self::Sinc>],
        length: usize,
    ) -> f32 {
        let sinc = sincs.get_unchecked(subindex);
        let wave_cut = &wave[index..(index + length)];
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

impl NeonSample for f64 {
    type Sinc = float64x2_t;

    #[target_feature(enable = "neon")]
    unsafe fn pack_sincs(sincs: Vec<Vec<f64>>) -> Vec<Vec<Self::Sinc>> {
        let mut packed_sincs = Vec::new();
        for sinc in sincs.iter() {
            let mut packed = Vec::new();
            for elements in sinc.chunks(2) {
                let elem = f64x2::from_slice_unaligned(elements);
                let packed_elems = std::mem::transmute(elem);
                packed.push(packed_elems);
            }
            packed_sincs.push(packed);
        }
        packed_sincs
    }

    #[target_feature(enable = "neon")]
    unsafe fn get_sinc_interpolated_unsafe(
        wave: &[f64],
        index: usize,
        subindex: usize,
        sincs: &[Vec<Self::Sinc>],
        length: usize,
    ) -> f64 {
        let sinc = sincs.get_unchecked(subindex);
        let wave_cut = &wave[index..(index + length)];
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

/// A SSE accelerated interpolator
pub struct NeonInterpolator<T> where T: NeonSample {
    sincs: Vec<Vec<T::Sinc>>,
    length: usize,
    nbr_sincs: usize,
}

impl<T> SincInterpolator<T> for NeonInterpolator<T> where T: Sample {
    /// Calculate the scalar produt of an input wave and the selected sinc filter
    fn get_sinc_interpolated(&self, wave: &[T], index: usize, subindex: usize) -> T {
        assert!((index + self.length) < wave.len(), "Tried to interpolate for index {}, max for the given input is {}", index, wave.len()-self.length-1);
        assert!(subindex < self.nbr_sincs, "Tried to use sinc subindex {}, max is {}", subindex, self.nbr_sincs-1);
        unsafe { T::get_sinc_interpolated_unsafe(wave, index, subindex, &self.sincs, self.length) }
    }

    fn len(&self) -> usize {
        self.length
    }

    fn nbr_sincs(&self) -> usize {
        self.nbr_sincs
    }
}

impl<T> NeonInterpolator<T> where T: Sample {
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
    ) -> Result<Self, MissingCpuFeature> {
        if let Some(feature) = FEATURES.iter().find(|f| !f.is_detected()) {
            return Err(MissingCpuFeature(*feature));
        }

        assert!(sinc_len % 8 == 0, "Sinc length must be a multiple of 8.");
        let sincs = make_sincs(sinc_len, oversampling_factor, f_cutoff, window);
        let sincs = unsafe { <T as NeonSample>::pack_sincs(sincs) };

        Ok(Self {
            sincs,
            length: sinc_len,
            nbr_sincs: oversampling_factor,
        })
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
            NeonInterpolator::<f64>::new(sinc_len, oversampling_factor, f_cutoff, window).unwrap();
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
            NeonInterpolator::<f32>::new(sinc_len, oversampling_factor, f_cutoff, window).unwrap();
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &sincs[123]);
        assert!((value - check).abs() < 1.0e-6);
    }
}
