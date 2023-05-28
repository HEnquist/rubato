use criterion::{black_box, criterion_group, criterion_main, Criterion};
extern crate rubato;

use rubato::sinc_interpolator::ScalarInterpolator;

#[cfg(target_arch = "x86_64")]
use rubato::sinc_interpolator::sinc_interpolator_avx::AvxInterpolator;
#[cfg(target_arch = "aarch64")]
use rubato::sinc_interpolator::sinc_interpolator_neon::NeonInterpolator;
#[cfg(target_arch = "x86_64")]
use rubato::sinc_interpolator::sinc_interpolator_sse::SseInterpolator;

use rubato::{
    FastFixedIn, FftFixedIn, PolynomialDegree, Resampler, SincFixedIn, SincInterpolationType,
    WindowFunction,
};

fn bench_fftfixedin(c: &mut Criterion) {
    let chunksize = 1024;
    let mut resampler = FftFixedIn::<f64>::new(44100, 192000, 1024, 2, 1).unwrap();
    let waveform = vec![vec![0.0 as f64; chunksize]; 1];
    c.bench_function("FftFixedIn f64", |b| {
        b.iter(|| resampler.process(black_box(&waveform), None).unwrap())
    });
}

fn bench_fftfixedin_32(c: &mut Criterion) {
    let chunksize = 1024;
    let mut resampler = FftFixedIn::<f32>::new(44100, 192000, 1024, 2, 1).unwrap();
    let waveform = vec![vec![0.0 as f32; chunksize]; 1];
    c.bench_function("FftFixedIn f32", |b| {
        b.iter(|| resampler.process(black_box(&waveform), None).unwrap())
    });
}

/// Helper to unwrap the constructed interpolator if appropriate.
macro_rules! unwrap_helper {
    (infallible $var:ident) => {
        $var
    };
    ($var:ident) => {
        $var.unwrap()
    };
}

macro_rules! bench_async_resampler {
    ($ft:ty, $it:ident, $ip:expr, $f:ident, $desc:literal $(, $unwrap:tt)?) => {
        fn $f(c: &mut Criterion) {
            let chunksize = 1024;
            let sinc_len = 256;
            let f_cutoff = 0.9473371669037001;
            let oversampling_factor = 256;
            let window = WindowFunction::BlackmanHarris2;
            let resample_ratio = 192000 as f64 / 44100 as f64;
            let interpolation_type = $ip;

            let interpolator = $it::<$ft>::new(
                sinc_len,
                oversampling_factor,
                f_cutoff,
                window,
            );
            let interpolator = unwrap_helper!($($unwrap)* interpolator);
            let interpolator = Box::new(interpolator);
            let mut resampler = SincFixedIn::<$ft>::new_with_interpolator(
                resample_ratio,
                1.1,
                interpolation_type,
                interpolator,
                chunksize,
                1,
            ).unwrap();
            let waveform = vec![vec![0.0 as $ft; chunksize]; 1];
            c.bench_function($desc, |b| b.iter(|| resampler.process(black_box(&waveform), None).unwrap()));
        }
    };
}

bench_async_resampler!(
    f32,
    ScalarInterpolator,
    SincInterpolationType::Cubic,
    bench_scalar_async_cubic_32,
    "scalar async cubic   32",
    infallible
);
bench_async_resampler!(
    f32,
    ScalarInterpolator,
    SincInterpolationType::Linear,
    bench_scalar_async_linear_32,
    "scalar async linear  32",
    infallible
);
bench_async_resampler!(
    f32,
    ScalarInterpolator,
    SincInterpolationType::Nearest,
    bench_scalar_async_nearest_32,
    "scalar async nearest 32",
    infallible
);
bench_async_resampler!(
    f64,
    ScalarInterpolator,
    SincInterpolationType::Cubic,
    bench_scalar_async_cubic_64,
    "scalar async cubic   64",
    infallible
);
bench_async_resampler!(
    f64,
    ScalarInterpolator,
    SincInterpolationType::Linear,
    bench_scalar_async_linear_64,
    "scalar async linear  64",
    infallible
);
bench_async_resampler!(
    f64,
    ScalarInterpolator,
    SincInterpolationType::Nearest,
    bench_scalar_async_nearest_64,
    "scalar async nearest 64",
    infallible
);

#[cfg(target_arch = "x86_64")]
bench_async_resampler!(
    f32,
    SseInterpolator,
    SincInterpolationType::Cubic,
    bench_sse_async_cubic_32,
    "sse async cubic   32"
);
#[cfg(target_arch = "x86_64")]
bench_async_resampler!(
    f32,
    SseInterpolator,
    SincInterpolationType::Linear,
    bench_sse_async_linear_32,
    "sse async linear  32"
);
#[cfg(target_arch = "x86_64")]
bench_async_resampler!(
    f32,
    SseInterpolator,
    SincInterpolationType::Nearest,
    bench_sse_async_nearest_32,
    "sse async nearest 32"
);
#[cfg(target_arch = "x86_64")]
bench_async_resampler!(
    f64,
    SseInterpolator,
    SincInterpolationType::Cubic,
    bench_sse_async_cubic_64,
    "sse async cubic   64"
);
#[cfg(target_arch = "x86_64")]
bench_async_resampler!(
    f64,
    SseInterpolator,
    SincInterpolationType::Linear,
    bench_sse_async_linear_64,
    "sse async linear  64"
);
#[cfg(target_arch = "x86_64")]
bench_async_resampler!(
    f64,
    SseInterpolator,
    SincInterpolationType::Nearest,
    bench_sse_async_nearest_64,
    "sse async nearest 64"
);

#[cfg(target_arch = "x86_64")]
bench_async_resampler!(
    f32,
    AvxInterpolator,
    SincInterpolationType::Cubic,
    bench_avx_async_cubic_32,
    "avx async cubic   32"
);
#[cfg(target_arch = "x86_64")]
bench_async_resampler!(
    f32,
    AvxInterpolator,
    SincInterpolationType::Linear,
    bench_avx_async_linear_32,
    "avx async linear  32"
);
#[cfg(target_arch = "x86_64")]
bench_async_resampler!(
    f32,
    AvxInterpolator,
    SincInterpolationType::Nearest,
    bench_avx_async_nearest_32,
    "avx async nearest 32"
);
#[cfg(target_arch = "x86_64")]
bench_async_resampler!(
    f64,
    AvxInterpolator,
    SincInterpolationType::Cubic,
    bench_avx_async_cubic_64,
    "avx async cubic   64"
);
#[cfg(target_arch = "x86_64")]
bench_async_resampler!(
    f64,
    AvxInterpolator,
    SincInterpolationType::Linear,
    bench_avx_async_linear_64,
    "avx async linear  64"
);
#[cfg(target_arch = "x86_64")]
bench_async_resampler!(
    f64,
    AvxInterpolator,
    SincInterpolationType::Nearest,
    bench_avx_async_nearest_64,
    "avx async nearest 64"
);

#[cfg(target_arch = "aarch64")]
bench_async_resampler!(
    f32,
    NeonInterpolator,
    SincInterpolationType::Cubic,
    bench_neon_async_cubic_32,
    "neon async cubic   32"
);
#[cfg(target_arch = "aarch64")]
bench_async_resampler!(
    f32,
    NeonInterpolator,
    SincInterpolationType::Linear,
    bench_neon_async_linear_32,
    "neon async linear  32"
);
#[cfg(target_arch = "aarch64")]
bench_async_resampler!(
    f32,
    NeonInterpolator,
    SincInterpolationType::Nearest,
    bench_neon_async_nearest_32,
    "neon async nearest 32"
);
#[cfg(target_arch = "aarch64")]
bench_async_resampler!(
    f64,
    NeonInterpolator,
    SincInterpolationType::Cubic,
    bench_neon_async_cubic_64,
    "neon async cubic   64"
);
#[cfg(target_arch = "aarch64")]
bench_async_resampler!(
    f64,
    NeonInterpolator,
    SincInterpolationType::Linear,
    bench_neon_async_linear_64,
    "neon async linear  64"
);
#[cfg(target_arch = "aarch64")]
bench_async_resampler!(
    f64,
    NeonInterpolator,
    SincInterpolationType::Nearest,
    bench_neon_async_nearest_64,
    "neon async nearest 64"
);

macro_rules! bench_fast_async_resampler {
    ($ft:ty, $ip:expr, $f:ident, $desc:literal) => {
        fn $f(c: &mut Criterion) {
            let chunksize = 1024;
            let interpolation_type = $ip;
            let resample_ratio = 192000 as f64 / 44100 as f64;
            let mut resampler =
                FastFixedIn::<$ft>::new(resample_ratio, 1.1, interpolation_type, chunksize, 1)
                    .unwrap();
            let waveform = vec![vec![0.0 as $ft; chunksize]; 1];
            c.bench_function($desc, |b| {
                b.iter(|| resampler.process(black_box(&waveform), None).unwrap())
            });
        }
    };
}

bench_fast_async_resampler!(
    f32,
    PolynomialDegree::Septic,
    bench_fast_async_septic_32,
    "fast async septic  32"
);
bench_fast_async_resampler!(
    f32,
    PolynomialDegree::Quintic,
    bench_fast_async_quintic_32,
    "fast async quintic  32"
);
bench_fast_async_resampler!(
    f32,
    PolynomialDegree::Cubic,
    bench_fast_async_cubic_32,
    "fast async cubic   32"
);
bench_fast_async_resampler!(
    f32,
    PolynomialDegree::Linear,
    bench_fast_async_linear_32,
    "fast async linear  32"
);
bench_fast_async_resampler!(
    f32,
    PolynomialDegree::Nearest,
    bench_fast_async_nearest_32,
    "fast async nearest 32"
);
bench_fast_async_resampler!(
    f64,
    PolynomialDegree::Septic,
    bench_fast_async_septic_64,
    "fast async septic  64"
);
bench_fast_async_resampler!(
    f64,
    PolynomialDegree::Quintic,
    bench_fast_async_quintic_64,
    "fast async quintic  64"
);
bench_fast_async_resampler!(
    f64,
    PolynomialDegree::Cubic,
    bench_fast_async_cubic_64,
    "fast async cubic   64"
);
bench_fast_async_resampler!(
    f64,
    PolynomialDegree::Linear,
    bench_fast_async_linear_64,
    "fast async linear  64"
);
bench_fast_async_resampler!(
    f64,
    PolynomialDegree::Nearest,
    bench_fast_async_nearest_64,
    "fast async nearest 64"
);

#[cfg(target_arch = "x86_64")]
criterion_group!(
    benches,
    bench_fftfixedin,
    bench_fftfixedin_32,
    bench_fast_async_septic_32,
    bench_fast_async_quintic_32,
    bench_fast_async_cubic_32,
    bench_fast_async_linear_32,
    bench_fast_async_nearest_32,
    bench_fast_async_septic_64,
    bench_fast_async_quintic_64,
    bench_fast_async_cubic_64,
    bench_fast_async_linear_64,
    bench_fast_async_nearest_64,
    bench_scalar_async_cubic_32,
    bench_scalar_async_linear_32,
    bench_scalar_async_nearest_32,
    bench_scalar_async_cubic_64,
    bench_scalar_async_linear_64,
    bench_scalar_async_nearest_64,
    bench_sse_async_cubic_32,
    bench_sse_async_linear_32,
    bench_sse_async_nearest_32,
    bench_sse_async_cubic_64,
    bench_sse_async_linear_64,
    bench_sse_async_nearest_64,
    bench_avx_async_cubic_32,
    bench_avx_async_linear_32,
    bench_avx_async_nearest_32,
    bench_avx_async_cubic_64,
    bench_avx_async_linear_64,
    bench_avx_async_nearest_64,
);

#[cfg(target_arch = "aarch64")]
criterion_group!(
    benches,
    bench_fftfixedin,
    bench_fftfixedin_32,
    bench_fast_async_septic_32,
    bench_fast_async_quintic_32,
    bench_fast_async_cubic_32,
    bench_fast_async_linear_32,
    bench_fast_async_nearest_32,
    bench_fast_async_septic_64,
    bench_fast_async_quintic_64,
    bench_fast_async_cubic_64,
    bench_fast_async_linear_64,
    bench_fast_async_nearest_64,
    bench_scalar_async_cubic_32,
    bench_scalar_async_linear_32,
    bench_scalar_async_nearest_32,
    bench_scalar_async_cubic_64,
    bench_scalar_async_linear_64,
    bench_scalar_async_nearest_64,
    bench_neon_async_cubic_32,
    bench_neon_async_linear_32,
    bench_neon_async_nearest_32,
    bench_neon_async_cubic_64,
    bench_neon_async_linear_64,
    bench_neon_async_nearest_64,
);

criterion_main!(benches);
