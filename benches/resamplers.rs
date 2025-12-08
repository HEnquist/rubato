#[cfg(feature = "bench_asyncro")]
mod bench_asyncro {
    use criterion::{criterion_group, Criterion};
    use std::hint::black_box;
    extern crate audioadapter;
    extern crate rubato;
    use rubato::{
        Async, FixedAsync, PolynomialDegree, Resampler, SincInterpolationType, WindowFunction,
    };

    use rubato::sinc_interpolator::ScalarInterpolator;

    #[cfg(target_arch = "x86_64")]
    use rubato::sinc_interpolator::sinc_interpolator_avx::AvxInterpolator;
    #[cfg(target_arch = "aarch64")]
    use rubato::sinc_interpolator::sinc_interpolator_neon::NeonInterpolator;
    #[cfg(target_arch = "x86_64")]
    use rubato::sinc_interpolator::sinc_interpolator_sse::SseInterpolator;

    use audioadapter_buffers::owned::InterleavedOwned;

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
            let mut resampler = Async::<$ft>::new_with_sinc_interpolator(
                resample_ratio,
                1.1,
                interpolation_type,
                interpolator,
                chunksize,
                1,
                FixedAsync::Input,
            ).unwrap();
            let buffer_in =  InterleavedOwned::new(0.0, 1, chunksize);
            let mut buffer_out =  InterleavedOwned::new(0.0, 1, resampler.output_frames_max());
            c.bench_function($desc, |b| b.iter(|| resampler.process_into_buffer(black_box(&buffer_in), &mut buffer_out, None).unwrap()));
        }
    };
}

    bench_async_resampler!(
        f32,
        ScalarInterpolator,
        SincInterpolationType::Cubic,
        bench_sinc_async_scalar_cubic_32,
        "sinc async scalar cubic   f32",
        infallible
    );
    bench_async_resampler!(
        f32,
        ScalarInterpolator,
        SincInterpolationType::Linear,
        bench_sinc_async_scalar_linear_32,
        "sinc async scalar linear  f32",
        infallible
    );
    bench_async_resampler!(
        f32,
        ScalarInterpolator,
        SincInterpolationType::Nearest,
        bench_sinc_async_scalar_nearest_32,
        "sinc async scalar nearest f32",
        infallible
    );
    bench_async_resampler!(
        f64,
        ScalarInterpolator,
        SincInterpolationType::Cubic,
        bench_sinc_async_scalar_cubic_64,
        "sinc async scalar cubic   f64",
        infallible
    );
    bench_async_resampler!(
        f64,
        ScalarInterpolator,
        SincInterpolationType::Linear,
        bench_sinc_async_scalar_linear_64,
        "sinc async scalar linear f64",
        infallible
    );
    bench_async_resampler!(
        f64,
        ScalarInterpolator,
        SincInterpolationType::Nearest,
        bench_sinc_async_scalar_nearest_64,
        "sinc async scalar nearest f64",
        infallible
    );

    #[cfg(target_arch = "x86_64")]
    bench_async_resampler!(
        f32,
        SseInterpolator,
        SincInterpolationType::Cubic,
        bench_sinc_async_sse_cubic_32,
        "sinc async sse cubic f32"
    );
    #[cfg(target_arch = "x86_64")]
    bench_async_resampler!(
        f32,
        SseInterpolator,
        SincInterpolationType::Linear,
        bench_sinc_async_sse_linear_32,
        "sinc async sse linear f32"
    );
    #[cfg(target_arch = "x86_64")]
    bench_async_resampler!(
        f32,
        SseInterpolator,
        SincInterpolationType::Nearest,
        bench_sinc_async_sse_nearest_32,
        "sinc async sse nearest f32"
    );
    #[cfg(target_arch = "x86_64")]
    bench_async_resampler!(
        f64,
        SseInterpolator,
        SincInterpolationType::Cubic,
        bench_sinc_async_sse_cubic_64,
        "sinc async sse cubic f64"
    );
    #[cfg(target_arch = "x86_64")]
    bench_async_resampler!(
        f64,
        SseInterpolator,
        SincInterpolationType::Linear,
        bench_sinc_async_sse_linear_64,
        "sinc async sse linear f64"
    );
    #[cfg(target_arch = "x86_64")]
    bench_async_resampler!(
        f64,
        SseInterpolator,
        SincInterpolationType::Nearest,
        bench_sinc_async_sse_nearest_64,
        "sinc async sse nearest f64"
    );

    #[cfg(target_arch = "x86_64")]
    bench_async_resampler!(
        f32,
        AvxInterpolator,
        SincInterpolationType::Cubic,
        bench_sinc_async_avx_cubic_32,
        "sinc async avx cubic f32"
    );
    #[cfg(target_arch = "x86_64")]
    bench_async_resampler!(
        f32,
        AvxInterpolator,
        SincInterpolationType::Linear,
        bench_sinc_async_avx_linear_32,
        "sinc async avx linear f32"
    );
    #[cfg(target_arch = "x86_64")]
    bench_async_resampler!(
        f32,
        AvxInterpolator,
        SincInterpolationType::Nearest,
        bench_sinc_async_avx_nearest_32,
        "sinc async avx nearest f32"
    );
    #[cfg(target_arch = "x86_64")]
    bench_async_resampler!(
        f64,
        AvxInterpolator,
        SincInterpolationType::Cubic,
        bench_sinc_async_avx_cubic_64,
        "sinc async avx cubic f64"
    );
    #[cfg(target_arch = "x86_64")]
    bench_async_resampler!(
        f64,
        AvxInterpolator,
        SincInterpolationType::Linear,
        bench_sinc_async_avx_linear_64,
        "sinc async avx linear f64"
    );
    #[cfg(target_arch = "x86_64")]
    bench_async_resampler!(
        f64,
        AvxInterpolator,
        SincInterpolationType::Nearest,
        bench_sinc_async_avx_nearest_64,
        "sinc async avx nearest f64"
    );

    #[cfg(target_arch = "aarch64")]
    bench_async_resampler!(
        f32,
        NeonInterpolator,
        SincInterpolationType::Cubic,
        bench_sinc_async_neon_cubic_32,
        "sinc async neon cubic f32"
    );
    #[cfg(target_arch = "aarch64")]
    bench_async_resampler!(
        f32,
        NeonInterpolator,
        SincInterpolationType::Linear,
        bench_sinc_async_neon_linear_32,
        "sinc async neon linear f32"
    );
    #[cfg(target_arch = "aarch64")]
    bench_async_resampler!(
        f32,
        NeonInterpolator,
        SincInterpolationType::Nearest,
        bench_sinc_async_neon_nearest_32,
        "sinc async neon nearest f32"
    );
    #[cfg(target_arch = "aarch64")]
    bench_async_resampler!(
        f64,
        NeonInterpolator,
        SincInterpolationType::Cubic,
        bench_sinc_async_neon_cubic_64,
        "sinc async neon cubic f64"
    );
    #[cfg(target_arch = "aarch64")]
    bench_async_resampler!(
        f64,
        NeonInterpolator,
        SincInterpolationType::Linear,
        bench_sinc_async_neon_linear_64,
        "sinc async neon linear f64"
    );
    #[cfg(target_arch = "aarch64")]
    bench_async_resampler!(
        f64,
        NeonInterpolator,
        SincInterpolationType::Nearest,
        bench_sinc_async_neon_nearest_64,
        "sinc async neon nearest f64"
    );

    macro_rules! bench_poly_async_resampler {
        ($ft:ty, $ip:expr, $f:ident, $desc:literal) => {
            fn $f(c: &mut Criterion) {
                let chunksize = 1024;
                let interpolation_type = $ip;
                let resample_ratio = 192000 as f64 / 44100 as f64;
                let mut resampler = Async::<$ft>::new_poly(
                    resample_ratio,
                    1.1,
                    interpolation_type,
                    chunksize,
                    1,
                    FixedAsync::Input,
                )
                .unwrap();
                let buffer_in = InterleavedOwned::new(0.0, 1, chunksize);
                let mut buffer_out = InterleavedOwned::new(0.0, 1, resampler.output_frames_max());
                c.bench_function($desc, |b| {
                    b.iter(|| {
                        resampler
                            .process_into_buffer(black_box(&buffer_in), &mut buffer_out, None)
                            .unwrap()
                    })
                });
            }
        };
    }

    bench_poly_async_resampler!(
        f32,
        PolynomialDegree::Septic,
        bench_poly_async_septic_32,
        "poly async septic f32"
    );
    bench_poly_async_resampler!(
        f32,
        PolynomialDegree::Quintic,
        bench_poly_async_quintic_32,
        "poly async quintic f32"
    );
    bench_poly_async_resampler!(
        f32,
        PolynomialDegree::Cubic,
        bench_poly_async_cubic_32,
        "poly async cubic f32"
    );
    bench_poly_async_resampler!(
        f32,
        PolynomialDegree::Linear,
        bench_poly_async_linear_32,
        "poly async linear f32"
    );
    bench_poly_async_resampler!(
        f32,
        PolynomialDegree::Nearest,
        bench_poly_async_nearest_32,
        "poly async nearest f32"
    );
    bench_poly_async_resampler!(
        f64,
        PolynomialDegree::Septic,
        bench_poly_async_septic_64,
        "poly async septic f64"
    );
    bench_poly_async_resampler!(
        f64,
        PolynomialDegree::Quintic,
        bench_poly_async_quintic_64,
        "poly async quintic f64"
    );
    bench_poly_async_resampler!(
        f64,
        PolynomialDegree::Cubic,
        bench_poly_async_cubic_64,
        "poly async cubic f64"
    );
    bench_poly_async_resampler!(
        f64,
        PolynomialDegree::Linear,
        bench_poly_async_linear_64,
        "poly async linear f64"
    );
    bench_poly_async_resampler!(
        f64,
        PolynomialDegree::Nearest,
        bench_poly_async_nearest_64,
        "poly async nearest f64"
    );

    #[cfg(target_arch = "x86_64")]
    criterion_group!(
        benches,
        bench_poly_async_septic_32,
        bench_poly_async_quintic_32,
        bench_poly_async_cubic_32,
        bench_poly_async_linear_32,
        bench_poly_async_nearest_32,
        bench_poly_async_septic_64,
        bench_poly_async_quintic_64,
        bench_poly_async_cubic_64,
        bench_poly_async_linear_64,
        bench_poly_async_nearest_64,
        bench_sinc_async_scalar_cubic_32,
        bench_sinc_async_scalar_linear_32,
        bench_sinc_async_scalar_nearest_32,
        bench_sinc_async_scalar_cubic_64,
        bench_sinc_async_scalar_linear_64,
        bench_sinc_async_scalar_nearest_64,
        bench_sinc_async_sse_cubic_32,
        bench_sinc_async_sse_linear_32,
        bench_sinc_async_sse_nearest_32,
        bench_sinc_async_sse_cubic_64,
        bench_sinc_async_sse_linear_64,
        bench_sinc_async_sse_nearest_64,
        bench_sinc_async_avx_cubic_32,
        bench_sinc_async_avx_linear_32,
        bench_sinc_async_avx_nearest_32,
        bench_sinc_async_avx_cubic_64,
        bench_sinc_async_avx_linear_64,
        bench_sinc_async_avx_nearest_64,
    );

    #[cfg(target_arch = "aarch64")]
    criterion_group!(
        benches,
        bench_poly_async_septic_32,
        bench_poly_async_quintic_32,
        bench_poly_async_cubic_32,
        bench_poly_async_linear_32,
        bench_poly_async_nearest_32,
        bench_poly_async_septic_64,
        bench_poly_async_quintic_64,
        bench_poly_async_cubic_64,
        bench_poly_async_linear_64,
        bench_poly_async_nearest_64,
        bench_sinc_async_scalar_cubic_32,
        bench_sinc_async_scalar_linear_32,
        bench_sinc_async_scalar_nearest_32,
        bench_sinc_async_scalar_cubic_64,
        bench_sinc_async_scalar_linear_64,
        bench_sinc_async_scalar_nearest_64,
        bench_sinc_async_neon_cubic_32,
        bench_sinc_async_neon_linear_32,
        bench_sinc_async_neon_nearest_32,
        bench_sinc_async_neon_cubic_64,
        bench_sinc_async_neon_linear_64,
        bench_sinc_async_neon_nearest_64,
    );

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    criterion_group!(
        benches,
        bench_poly_async_septic_32,
        bench_poly_async_quintic_32,
        bench_poly_async_cubic_32,
        bench_poly_async_linear_32,
        bench_poly_async_nearest_32,
        bench_poly_async_septic_64,
        bench_poly_async_quintic_64,
        bench_poly_async_cubic_64,
        bench_poly_async_linear_64,
        bench_poly_async_nearest_64,
        bench_sinc_async_scalar_cubic_32,
        bench_sinc_async_scalar_linear_32,
        bench_sinc_async_scalar_nearest_32,
        bench_sinc_async_scalar_cubic_64,
        bench_sinc_async_scalar_linear_64,
        bench_sinc_async_scalar_nearest_64,
    );
}

#[cfg(feature = "fft_resampler")]
mod bench_fft {
    use criterion::{criterion_group, Criterion};
    use std::hint::black_box;
    extern crate audioadapter;
    extern crate rubato;
    use audioadapter_buffers::owned::InterleavedOwned;
    use rubato::Fft;
    use rubato::{FixedSync, Resampler};

    fn bench_fft_64(c: &mut Criterion) {
        let chunksize = 1024;
        let mut resampler = Fft::<f64>::new(44100, 192000, 1024, 2, 1, FixedSync::Input).unwrap();
        let buffer_in = InterleavedOwned::new(0.0, 1, chunksize);
        let mut buffer_out = InterleavedOwned::new(0.0, 1, resampler.output_frames_max());
        c.bench_function("fft sync f64", |b| {
            b.iter(|| {
                resampler
                    .process_into_buffer(black_box(&buffer_in), &mut buffer_out, None)
                    .unwrap()
            })
        });
    }

    fn bench_fft_32(c: &mut Criterion) {
        let chunksize = 1024;
        let mut resampler = Fft::<f32>::new(44100, 192000, 1024, 2, 1, FixedSync::Input).unwrap();
        let buffer_in = InterleavedOwned::new(0.0, 1, chunksize);
        let mut buffer_out = InterleavedOwned::new(0.0, 1, resampler.output_frames_max());
        c.bench_function("fft sync f32", |b| {
            b.iter(|| {
                resampler
                    .process_into_buffer(black_box(&buffer_in), &mut buffer_out, None)
                    .unwrap()
            })
        });
    }

    criterion_group!(benches, bench_fft_64, bench_fft_32);
}

use criterion::criterion_main;
#[cfg(all(feature = "fft_resampler", feature = "bench_asyncro"))]
criterion_main!(bench_asyncro::benches, bench_fft::benches);
#[cfg(all(feature = "bench_asyncro", not(feature = "fft_resampler")))]
criterion_main!(bench_asyncro::benches);
#[cfg(all(feature = "fft_resampler", not(feature = "bench_asyncro")))]
criterion_main!(bench_fft::benches);
#[cfg(not(any(feature = "fft_resampler", feature = "bench_asyncro")))]
fn main() {
    println!("No benchmarks selected, enable at least one of the features 'fft_resampler' or 'bench_asyncro'.");
}
