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

#[cfg(test)]
mod tests {
    use crate::realfft::{ComplexToReal, RealToComplex};
    use rustfft::num_complex::Complex;
    use rustfft::num_traits::Zero;
    use rustfft::FFTplanner;

    fn compare_complex(a: &[Complex<f64>], b: &[Complex<f64>], tol: f64) -> bool {
        a.iter().zip(b.iter()).fold(true, |eq, (val_a, val_b)| {
            eq && (val_a.re - val_b.re).abs() < tol && (val_a.im - val_b.im).abs() < tol
        })
    }

    fn compare_f64(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.iter()
            .zip(b.iter())
            .fold(true, |eq, (val_a, val_b)| eq && (val_a - val_b).abs() < tol)
    }

    // Compare RealToComplex with standard FFT
    #[test]
    fn real_to_complex() {
        let mut indata = vec![0.0f64; 256];
        indata[0] = 1.0;
        indata[3] = 0.5;
        let mut indata_c = indata
            .iter()
            .map(|val| Complex::from(val))
            .collect::<Vec<Complex<f64>>>();
        let mut fft_planner = FFTplanner::<f64>::new(false);
        let fft = fft_planner.plan_fft(256);

        let mut r2c = RealToComplex::<f64>::new(256);
        let mut out_a: Vec<Complex<f64>> = vec![Complex::zero(); 129];
        let mut out_b: Vec<Complex<f64>> = vec![Complex::zero(); 256];

        fft.process(&mut indata_c, &mut out_b);
        r2c.process(&indata, &mut out_a).unwrap();
        assert!(compare_complex(&out_a[0..129], &out_b[0..129], 1.0e-9));
    }

    // Compare ComplexToReal with standard iFFT
    #[test]
    fn complex_to_real() {
        let mut indata = vec![Complex::<f64>::zero(); 256];
        indata[0] = Complex::new(1.0, 0.0);
        indata[1] = Complex::new(1.0, 0.4);
        indata[255] = Complex::new(1.0, -0.4);
        indata[3] = Complex::new(0.3, 0.2);
        indata[253] = Complex::new(0.3, -0.2);

        let mut fft_planner = FFTplanner::<f64>::new(true);
        let fft = fft_planner.plan_fft(256);

        let mut c2r = ComplexToReal::<f64>::new(256);
        let mut out_a: Vec<f64> = vec![0.0; 256];
        let mut out_b: Vec<Complex<f64>> = vec![Complex::zero(); 256];

        c2r.process(&indata[0..129], &mut out_a).unwrap();
        fft.process(&mut indata, &mut out_b);

        let out_b_r = out_b.iter().map(|val| 0.5 * val.re).collect::<Vec<f64>>();
        assert!(compare_f64(&out_a, &out_b_r, 1.0e-9));
    }
}
