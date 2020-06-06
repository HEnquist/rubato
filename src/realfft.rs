//use crate::windows::WindowFunction;

//use crate::sinc::make_sincs;
//use num_traits::Float;
//use num::integer;
use std::error;
//use std::fmt;

//use rustfft::algorithm::Radix4;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;
//use rustfft::FFT;

type Res<T> = Result<T, Box<dyn error::Error>>;

use crate::ResamplerError;

//use crate::Resampler;

pub struct RealToComplex<T> {
    sin: Vec<T>,
    cos: Vec<T>,
    length: usize,
    fft: std::sync::Arc<dyn rustfft::FFT<T>>,
    buffer_in: Vec<Complex<T>>,
    buffer_out: Vec<Complex<T>>,
}

pub struct ComplexToReal<T> {
    sin: Vec<T>,
    cos: Vec<T>,
    length: usize,
    fft: std::sync::Arc<dyn rustfft::FFT<T>>,
    buffer_in: Vec<Complex<T>>,
    buffer_out: Vec<Complex<T>>,
}

macro_rules! impl_r2c {
    ($ft:ty) => {
        impl RealToComplex<$ft> {
            pub fn new(length: usize) -> Self {
                let buffer_in = vec![Complex::zero(); length / 2];
                let buffer_out = vec![Complex::zero(); length / 2 + 1];
                let mut sin = Vec::with_capacity(length / 2);
                let mut cos = Vec::with_capacity(length / 2);
                let pi = std::f64::consts::PI as $ft;
                for k in 0..length / 2 {
                    sin.push((k as $ft * pi / (length / 2) as $ft).sin());
                    cos.push((k as $ft * pi / (length / 2) as $ft).cos());
                }
                let mut fft_planner = FFTplanner::<$ft>::new(false);
                let fft = fft_planner.plan_fft(length / 2);
                RealToComplex {
                    sin,
                    cos,
                    length,
                    fft,
                    buffer_in,
                    buffer_out,
                }
            }

            pub fn process(&mut self, input: &[$ft], output: &mut [Complex<$ft>]) -> Res<()> {
                if input.len() != self.length {
                    return Err(Box::new(ResamplerError::new(
                        format!(
                            "Wrong length of input, expected {}, got {}",
                            self.length,
                            input.len()
                        )
                        .as_str(),
                    )));
                }
                if output.len() != (self.length / 2 + 1) {
                    return Err(Box::new(ResamplerError::new(
                        format!(
                            "Wrong length of output, expected {}, got {}",
                            self.length / 2 + 1,
                            input.len()
                        )
                        .as_str(),
                    )));
                }
                let fftlen = self.length / 2;
                for (val, buf) in input.chunks(2).take(fftlen).zip(self.buffer_in.iter_mut()) {
                    *buf = Complex::new(val[0], val[1]);
                }

                // FFT and store result in buffer_out
                self.fft
                    .process(&mut self.buffer_in, &mut self.buffer_out[0..fftlen]);

                self.buffer_out[fftlen] = self.buffer_out[0];

                for k in 0..fftlen {
                    let xr = 0.5
                        * ((self.buffer_out[k].re + self.buffer_out[fftlen - k].re)
                            + self.cos[k]
                                * (self.buffer_out[k].im + self.buffer_out[fftlen - k].im)
                            - self.sin[k]
                                * (self.buffer_out[k].re - self.buffer_out[fftlen - k].re));
                    let xi = 0.5
                        * ((self.buffer_out[k].im - self.buffer_out[fftlen - k].im)
                            - self.sin[k]
                                * (self.buffer_out[k].im + self.buffer_out[fftlen - k].im)
                            - self.cos[k]
                                * (self.buffer_out[k].re - self.buffer_out[fftlen - k].re));
                    output[k] = Complex::new(xr, xi);
                }
                output[fftlen] = Complex::new(self.buffer_out[0].re - self.buffer_out[0].im, 0.0);
                Ok(())
            }
        }
    };
}
impl_r2c!(f64);
impl_r2c!(f32);

macro_rules! impl_c2r {
    ($ft:ty) => {
        impl ComplexToReal<$ft> {
            pub fn new(length: usize) -> Self {
                let buffer_in = vec![Complex::zero(); length / 2];
                let buffer_out = vec![Complex::zero(); length / 2];
                let mut sin = Vec::with_capacity(length / 2);
                let mut cos = Vec::with_capacity(length / 2);
                let pi = std::f64::consts::PI as $ft;
                for k in 0..length / 2 {
                    sin.push((k as $ft * pi / (length / 2) as $ft).sin());
                    cos.push((k as $ft * pi / (length / 2) as $ft).cos());
                }
                let mut fft_planner = FFTplanner::<$ft>::new(true);
                let fft = fft_planner.plan_fft(length / 2);
                ComplexToReal {
                    sin,
                    cos,
                    length,
                    fft,
                    buffer_in,
                    buffer_out,
                }
            }

            pub fn process(&mut self, input: &[Complex<$ft>], output: &mut [$ft]) -> Res<()> {
                if input.len() != (self.length / 2 + 1) {
                    return Err(Box::new(ResamplerError::new(
                        format!(
                            "Wrong length of input, expected {}, got {}",
                            self.length / 2 + 1,
                            input.len()
                        )
                        .as_str(),
                    )));
                }
                if output.len() != self.length {
                    return Err(Box::new(ResamplerError::new(
                        format!(
                            "Wrong length of output, expected {}, got {}",
                            self.length,
                            input.len()
                        )
                        .as_str(),
                    )));
                }
                let fftlen = self.length / 2;

                for k in 0..fftlen {
                    let xr = 0.5
                        * ((input[k].re + input[fftlen - k].re)
                            - self.cos[k] * (input[k].im + input[fftlen - k].im)
                            - self.sin[k] * (input[k].re - input[fftlen - k].re));
                    let xi = 0.5
                        * ((input[k].im - input[fftlen - k].im)
                            + self.cos[k] * (input[k].re - input[fftlen - k].re)
                            - self.sin[k] * (input[k].im + input[fftlen - k].im));
                    self.buffer_in[k] = Complex::new(xr, xi);
                }

                // FFT and store result in buffer_out
                self.fft.process(&mut self.buffer_in, &mut self.buffer_out);

                for (val, out) in self.buffer_out.iter().zip(output.chunks_mut(2)) {
                    out[0] = val.re;
                    out[1] = val.im;
                }
                Ok(())
            }
        }
    };
}
impl_c2r!(f64);
impl_c2r!(f32);
