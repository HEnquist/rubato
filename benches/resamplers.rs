use criterion::{criterion_group, criterion_main, Criterion};
extern crate rubato;

use rubato::ScalarInterpolator;

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
use rubato::interpolator_avx::AvxInterpolator;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
use rubato::interpolator_neon::NeonInterpolator;
#[cfg(target_arch = "x86_64")]
use rubato::interpolator_sse::SseInterpolator;

use rubato::{FftFixedIn, InterpolationType, Resampler, SincFixedIn, WindowFunction};

fn bench_fftfixedin(c: &mut Criterion) {
    let chunksize = 1024;
    let mut resampler = FftFixedIn::<f64>::new(44100, 192000, 1024, 2, 1);
    let waveform = vec![vec![0.0 as f64; chunksize]; 1];
    c.bench_function("FftFixedIn f64", |b| {
        b.iter(|| resampler.process(&waveform).unwrap())
    });
}

fn bench_fftfixedin_32(c: &mut Criterion) {
    let chunksize = 1024;
    let mut resampler = FftFixedIn::<f32>::new(44100, 192000, 1024, 2, 1);
    let waveform = vec![vec![0.0 as f32; chunksize]; 1];
    c.bench_function("FftFixedIn f32", |b| {
        b.iter(|| resampler.process(&waveform).unwrap())
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
                interpolation_type,
                interpolator,
                chunksize,
                1,
            );
            let waveform = vec![vec![0.0 as $ft; chunksize]; 1];
            c.bench_function($desc, |b| b.iter(|| resampler.process(&waveform).unwrap()));
        }
    };
}

bench_async_resampler!(
    f32,
    ScalarInterpolator,
    InterpolationType::Cubic,
    bench_scalar_async_cubic_32,
    "scalar async cubic   32",
    infallible
);
bench_async_resampler!(
    f32,
    ScalarInterpolator,
    InterpolationType::Linear,
    bench_scalar_async_linear_32,
    "scalar async linear  32",
    infallible
);
bench_async_resampler!(
    f32,
    ScalarInterpolator,
    InterpolationType::Nearest,
    bench_scalar_async_nearest_32,
    "scalar async nearest 32",
    infallible
);
bench_async_resampler!(
    f64,
    ScalarInterpolator,
    InterpolationType::Cubic,
    bench_scalar_async_cubic_64,
    "scalar async cubic   64",
    infallible
);
bench_async_resampler!(
    f64,
    ScalarInterpolator,
    InterpolationType::Linear,
    bench_scalar_async_linear_64,
    "scalar async linear  64",
    infallible
);
bench_async_resampler!(
    f64,
    ScalarInterpolator,
    InterpolationType::Nearest,
    bench_scalar_async_nearest_64,
    "scalar async nearest 64",
    infallible
);

#[cfg(target_arch = "x86_64")]
bench_async_resampler!(
    f32,
    SseInterpolator,
    InterpolationType::Cubic,
    bench_sse_async_cubic_32,
    "sse async cubic   32"
);
#[cfg(target_arch = "x86_64")]
bench_async_resampler!(
    f32,
    SseInterpolator,
    InterpolationType::Linear,
    bench_sse_async_linear_32,
    "sse async linear  32"
);
#[cfg(target_arch = "x86_64")]
bench_async_resampler!(
    f32,
    SseInterpolator,
    InterpolationType::Nearest,
    bench_sse_async_nearest_32,
    "sse async nearest 32"
);
#[cfg(target_arch = "x86_64")]
bench_async_resampler!(
    f64,
    SseInterpolator,
    InterpolationType::Cubic,
    bench_sse_async_cubic_64,
    "sse async cubic   64"
);
#[cfg(target_arch = "x86_64")]
bench_async_resampler!(
    f64,
    SseInterpolator,
    InterpolationType::Linear,
    bench_sse_async_linear_64,
    "sse async linear  64"
);
#[cfg(target_arch = "x86_64")]
bench_async_resampler!(
    f64,
    SseInterpolator,
    InterpolationType::Nearest,
    bench_sse_async_nearest_64,
    "sse async nearest 64"
);

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
bench_async_resampler!(
    f32,
    AvxInterpolator,
    InterpolationType::Cubic,
    bench_avx_async_cubic_32,
    "avx async cubic   32"
);
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
bench_async_resampler!(
    f32,
    AvxInterpolator,
    InterpolationType::Linear,
    bench_avx_async_linear_32,
    "avx async linear  32"
);
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
bench_async_resampler!(
    f32,
    AvxInterpolator,
    InterpolationType::Nearest,
    bench_avx_async_nearest_32,
    "avx async nearest 32"
);
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
bench_async_resampler!(
    f64,
    AvxInterpolator,
    InterpolationType::Cubic,
    bench_avx_async_cubic_64,
    "avx async cubic   64"
);
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
bench_async_resampler!(
    f64,
    AvxInterpolator,
    InterpolationType::Linear,
    bench_avx_async_linear_64,
    "avx async linear  64"
);
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
bench_async_resampler!(
    f64,
    AvxInterpolator,
    InterpolationType::Nearest,
    bench_avx_async_nearest_64,
    "avx async nearest 64"
);

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
bench_async_resampler!(
    f32,
    NeonInterpolator,
    InterpolationType::Cubic,
    bench_neon_async_cubic_32,
    "neon async cubic   32"
);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
bench_async_resampler!(
    f32,
    NeonInterpolator,
    InterpolationType::Linear,
    bench_neon_async_linear_32,
    "neon async linear  32"
);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
bench_async_resampler!(
    f32,
    NeonInterpolator,
    InterpolationType::Nearest,
    bench_neon_async_nearest_32,
    "neon async nearest 32"
);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
bench_async_resampler!(
    f64,
    NeonInterpolator,
    InterpolationType::Cubic,
    bench_neon_async_cubic_64,
    "neon async cubic   64"
);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
bench_async_resampler!(
    f64,
    NeonInterpolator,
    InterpolationType::Linear,
    bench_neon_async_linear_64,
    "neon async linear  64"
);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
bench_async_resampler!(
    f64,
    NeonInterpolator,
    InterpolationType::Nearest,
    bench_neon_async_nearest_64,
    "neon async nearest 64"
);

#[cfg(all(target_arch = "x86_64", not(feature = "avx")))]
criterion_group!(
    benches,
    bench_fftfixedin,
    bench_fftfixedin_32,
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
);

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
criterion_group!(
    benches,
    bench_fftfixedin,
    bench_fftfixedin_32,
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

#[cfg(all(target_arch = "aarch64", not(feature = "neon")))]
criterion_group!(
    benches,
    bench_fftfixedin,
    bench_fftfixedin_32,
    bench_scalar_async_cubic_32,
    bench_scalar_async_linear_32,
    bench_scalar_async_nearest_32,
    bench_scalar_async_cubic_64,
    bench_scalar_async_linear_64,
    bench_scalar_async_nearest_64,
);

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
criterion_group!(
    benches,
    bench_fftfixedin,
    bench_fftfixedin_32,
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
