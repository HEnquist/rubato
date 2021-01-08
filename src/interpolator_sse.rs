use crate::windows::WindowFunction;

//use crate::interpolation::*;
//use crate::{InterpolationParameters, InterpolationType};
use crate::sinc::make_sincs;
//use num_traits::Float;
//use std::error;
//use std::fmt;
use core::arch::x86_64::{__m128d, __m128};
use core::arch::x86_64::{_mm_loadu_ps, _mm_setzero_ps, _mm_add_ps, _mm_hadd_ps, _mm_mul_ps};
use core::arch::x86_64::{_mm_loadu_pd, _mm_setzero_pd, _mm_add_pd, _mm_hadd_pd, _mm_mul_pd};
//use std::any::TypeId;
use std::marker::PhantomData;

//type Res<T> = Result<T, Box<dyn error::Error>>;

//use crate::Resampler;
//use crate::ResamplerError;
use crate::asynchro::SincInterpolator;

/// A plain scalar interpolator 
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
        unsafe {
            let sinc = &self.sincs_s.as_ref().unwrap().get_unchecked(subindex);
            let wave_cut = &wave[index..(index + self.length)];
            let mut acc0 = _mm_setzero_ps();
            let mut acc1 = _mm_setzero_ps();
            let mut w_idx = 0;
            let mut s_idx = 0;
            for _ in 0..wave_cut.len()/8 {
                let w0 = _mm_loadu_ps(wave_cut.get_unchecked(w_idx));
                let w1 = _mm_loadu_ps(wave_cut.get_unchecked(w_idx + 4));
                let s0 = _mm_mul_ps(w0, *sinc.get_unchecked(s_idx));
                let s1 = _mm_mul_ps(w1, *sinc.get_unchecked(s_idx+1));
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
        unsafe {
            let sinc = &self.sincs_d.as_ref().unwrap().get_unchecked(subindex);
            let wave_cut = &wave[index..(index + self.length)];
            let mut acc0 = _mm_setzero_pd();
            let mut acc1 = _mm_setzero_pd();
            let mut acc2 = _mm_setzero_pd();
            let mut acc3 = _mm_setzero_pd();
            let mut w_idx = 0;
            let mut s_idx = 0;
            for _ in 0..wave_cut.len()/8 {
                let w0 = _mm_loadu_pd(wave_cut.get_unchecked(w_idx));
                let w1 = _mm_loadu_pd(wave_cut.get_unchecked(w_idx + 2));
                let w2 = _mm_loadu_pd(wave_cut.get_unchecked(w_idx + 4));
                let w3 = _mm_loadu_pd(wave_cut.get_unchecked(w_idx + 6));
                let s0 = _mm_mul_pd(w0, *sinc.get_unchecked(s_idx));
                let s1 = _mm_mul_pd(w1, *sinc.get_unchecked(s_idx+1));
                let s2 = _mm_mul_pd(w2, *sinc.get_unchecked(s_idx+2));
                let s3 = _mm_mul_pd(w3, *sinc.get_unchecked(s_idx+3));
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

    fn len(&self) -> usize {
        self.length
    }

    fn nbr_sincs(&self) -> usize {
        self.nbr_sincs
    }
}

impl SseInterpolator<f32> {
    /// Create a new ScalarInterpolator
    ///
    /// Parameters are:
    /// - `resample_ratio`: Ratio between output and input sample rates.
    /// - `parameters`: Parameters for interpolation, see `InterpolationParameters`
    /// - `chunk_size`: size of input data in frames
    /// - `nbr_channels`: number of channels in input/output
    pub fn new(
        sinc_len: usize,
        oversampling_factor: usize,
        f_cutoff: f32,
        window: WindowFunction,
    ) -> Self {
        assert!(sinc_len%8 == 0);
        let sincs = make_sincs(
            sinc_len,
            oversampling_factor,
            f_cutoff,
            window,
        );
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
}

impl SseInterpolator<f64> {
    /// Create a new ScalarInterpolator
    ///
    /// Parameters are:
    /// - `resample_ratio`: Ratio between output and input sample rates.
    /// - `parameters`: Parameters for interpolation, see `InterpolationParameters`
    /// - `chunk_size`: size of input data in frames
    /// - `nbr_channels`: number of channels in input/output
    pub fn new(
        sinc_len: usize,
        oversampling_factor: usize,
        f_cutoff: f32,
        window: WindowFunction,
    ) -> Self {
        assert!(sinc_len%8 == 0);
        let sincs = make_sincs(
            sinc_len,
            oversampling_factor,
            f_cutoff,
            window,
        );
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
}

