use criterion::{criterion_group, criterion_main, Criterion};
extern crate rubato;

use rubato::asynchro::ScalarInterpolator;


#[cfg(target_arch = "x86_64")]
use rubato::interpolator_avx::AvxInterpolator;
#[cfg(target_arch = "x86_64")]
use rubato::interpolator_sse::SseInterpolator;
#[cfg(target_arch = "aarch64")]
use rubato::interpolator_neon::NeonInterpolator;

use rubato::{
    FftFixedIn, FftFixedOut, InterpolationParameters, InterpolationType, Resampler, SincFixedIn,
    WindowFunction,
};

fn bench_fftfixedin(c: &mut Criterion) {
    let chunksize = 1024;
    let mut resampler = FftFixedIn::<f64>::new(44100, 192000, 1024, 2, 1);
    let mut waveform = vec![vec![0.0 as f64; chunksize]; 1];
    c.bench_function("FftFixedIn f64", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

fn bench_fftfixedin_32(c: &mut Criterion) {
    let chunksize = 1024;
    let mut resampler = FftFixedIn::<f32>::new(44100, 192000, 1024, 2, 1);
    let mut waveform = vec![vec![0.0 as f32; chunksize]; 1];
    c.bench_function("FftFixedIn f32", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

//fn bench_fftfixedout(c: &mut Criterion) {
//    let mut resampler = FftFixedOut::<f64>::new(44100, 192000, 4096, 2, 1);
//    c.bench_function("FftFixedOut f64", |b| {
//        b.iter(|| {
//            let needed = resampler.nbr_frames_needed();
//            let mut waveform = vec![vec![0.0 as f64; needed]; 1];
//            let _resampled = resampler.process(&mut waveform).unwrap();
//        })
//    });
//}

fn bench_sincfixedin_cubic(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let oversampling_factor = 256;
    let window = WindowFunction::BlackmanHarris2;
    let resample_ratio = 192000 as f64 / 44100 as f64;
    let interpolation_type = InterpolationType::Cubic;

    let interpolator = Box::new(ScalarInterpolator::<f64>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ));
    let mut resampler = SincFixedIn::<f64>::new_with_interpolator(
        resample_ratio,
        interpolation_type,
        interpolator,
        chunksize,
        1,
    );
    let mut waveform = vec![vec![0.0 as f64; chunksize]; 1];
    c.bench_function("SincFixedIn cubic f64", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

fn bench_sincfixedin_linear(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let oversampling_factor = 256;
    let window = WindowFunction::BlackmanHarris2;
    let resample_ratio = 192000 as f64 / 44100 as f64;
    let interpolation_type = InterpolationType::Linear;

    let interpolator = Box::new(ScalarInterpolator::<f64>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ));
    let mut resampler = SincFixedIn::<f64>::new_with_interpolator(
        resample_ratio,
        interpolation_type,
        interpolator,
        chunksize,
        1,
    );
    let mut waveform = vec![vec![0.0 as f64; chunksize]; 1];
    c.bench_function("SincFixedIn linear f64", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

fn bench_sincfixedin_nearest(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let oversampling_factor = 640;
    let window = WindowFunction::BlackmanHarris2;
    let resample_ratio = 192000 as f64 / 44100 as f64;
    let interpolation_type = InterpolationType::Nearest;

    let interpolator = Box::new(ScalarInterpolator::<f64>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ));
    let mut resampler = SincFixedIn::<f64>::new_with_interpolator(
        resample_ratio,
        interpolation_type,
        interpolator,
        chunksize,
        1,
    );
    let mut waveform = vec![vec![0.0 as f64; chunksize]; 1];
    c.bench_function("SincFixedIn nearest f64", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

fn bench_sincfixedin_cubic_32(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let oversampling_factor = 256;
    let window = WindowFunction::BlackmanHarris2;
    let resample_ratio = 192000 as f64 / 44100 as f64;
    let interpolation_type = InterpolationType::Cubic;

    let interpolator = Box::new(ScalarInterpolator::<f32>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ));
    let mut resampler = SincFixedIn::<f32>::new_with_interpolator(
        resample_ratio,
        interpolation_type,
        interpolator,
        chunksize,
        1,
    );
    let mut waveform = vec![vec![0.0 as f32; chunksize]; 1];
    c.bench_function("SincFixedIn cubic f32", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

fn bench_sincfixedin_linear_32(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let oversampling_factor = 256;
    let window = WindowFunction::BlackmanHarris2;
    let resample_ratio = 192000 as f64 / 44100 as f64;
    let interpolation_type = InterpolationType::Linear;

    let interpolator = Box::new(ScalarInterpolator::<f32>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ));
    let mut resampler = SincFixedIn::<f32>::new_with_interpolator(
        resample_ratio,
        interpolation_type,
        interpolator,
        chunksize,
        1,
    );
    let mut waveform = vec![vec![0.0 as f32; chunksize]; 1];
    c.bench_function("SincFixedIn linear f32", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

fn bench_sincfixedin_nearest_32(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let oversampling_factor = 640;
    let window = WindowFunction::BlackmanHarris2;
    let resample_ratio = 192000 as f64 / 44100 as f64;
    let interpolation_type = InterpolationType::Nearest;

    let interpolator = Box::new(ScalarInterpolator::<f32>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ));
    let mut resampler = SincFixedIn::<f32>::new_with_interpolator(
        resample_ratio,
        interpolation_type,
        interpolator,
        chunksize,
        1,
    );
    let mut waveform = vec![vec![0.0 as f32; chunksize]; 1];
    c.bench_function("SincFixedIn nearest f32", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_sse_sincfixedin_cubic(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let oversampling_factor = 256;
    let window = WindowFunction::BlackmanHarris2;
    let resample_ratio = 192000 as f64 / 44100 as f64;
    let interpolation_type = InterpolationType::Cubic;

    let interpolator = Box::new(SseInterpolator::<f64>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ));
    let mut resampler = SincFixedIn::<f64>::new_with_interpolator(
        resample_ratio,
        interpolation_type,
        interpolator,
        chunksize,
        1,
    );
    let mut waveform = vec![vec![0.0 as f64; chunksize]; 1];
    c.bench_function("SSE SincFixedIn cubic f64", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_sse_sincfixedin_linear(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let oversampling_factor = 256;
    let window = WindowFunction::BlackmanHarris2;
    let resample_ratio = 192000 as f64 / 44100 as f64;
    let interpolation_type = InterpolationType::Linear;

    let interpolator = Box::new(SseInterpolator::<f64>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ));
    let mut resampler = SincFixedIn::<f64>::new_with_interpolator(
        resample_ratio,
        interpolation_type,
        interpolator,
        chunksize,
        1,
    );
    let mut waveform = vec![vec![0.0 as f64; chunksize]; 1];
    c.bench_function("SSE SincFixedIn linear f64", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_sse_sincfixedin_nearest(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let oversampling_factor = 640;
    let window = WindowFunction::BlackmanHarris2;
    let resample_ratio = 192000 as f64 / 44100 as f64;
    let interpolation_type = InterpolationType::Nearest;

    let interpolator = Box::new(SseInterpolator::<f64>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ));
    let mut resampler = SincFixedIn::<f64>::new_with_interpolator(
        resample_ratio,
        interpolation_type,
        interpolator,
        chunksize,
        1,
    );
    let mut waveform = vec![vec![0.0 as f64; chunksize]; 1];
    c.bench_function("SSE SincFixedIn nearest f64", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_sse_sincfixedin_cubic_32(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let oversampling_factor = 256;
    let window = WindowFunction::BlackmanHarris2;
    let resample_ratio = 192000 as f64 / 44100 as f64;
    let interpolation_type = InterpolationType::Cubic;

    let interpolator = Box::new(SseInterpolator::<f32>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ));
    let mut resampler = SincFixedIn::<f32>::new_with_interpolator(
        resample_ratio,
        interpolation_type,
        interpolator,
        chunksize,
        1,
    );
    let mut waveform = vec![vec![0.0 as f32; chunksize]; 1];
    c.bench_function("SSE SincFixedIn cubic f32", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_sse_sincfixedin_linear_32(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let oversampling_factor = 256;
    let window = WindowFunction::BlackmanHarris2;
    let resample_ratio = 192000 as f64 / 44100 as f64;
    let interpolation_type = InterpolationType::Linear;

    let interpolator = Box::new(SseInterpolator::<f32>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ));
    let mut resampler = SincFixedIn::<f32>::new_with_interpolator(
        resample_ratio,
        interpolation_type,
        interpolator,
        chunksize,
        1,
    );
    let mut waveform = vec![vec![0.0 as f32; chunksize]; 1];
    c.bench_function("SSE SincFixedIn linear f32", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_sse_sincfixedin_nearest_32(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let oversampling_factor = 640;
    let window = WindowFunction::BlackmanHarris2;
    let resample_ratio = 192000 as f64 / 44100 as f64;
    let interpolation_type = InterpolationType::Nearest;

    let interpolator = Box::new(SseInterpolator::<f32>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ));
    let mut resampler = SincFixedIn::<f32>::new_with_interpolator(
        resample_ratio,
        interpolation_type,
        interpolator,
        chunksize,
        1,
    );
    let mut waveform = vec![vec![0.0 as f32; chunksize]; 1];
    c.bench_function("SSE SincFixedIn nearest f32", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_avx_sincfixedin_cubic(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let oversampling_factor = 256;
    let window = WindowFunction::BlackmanHarris2;
    let resample_ratio = 192000 as f64 / 44100 as f64;
    let interpolation_type = InterpolationType::Cubic;

    let interpolator = Box::new(AvxInterpolator::<f64>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ));
    let mut resampler = SincFixedIn::<f64>::new_with_interpolator(
        resample_ratio,
        interpolation_type,
        interpolator,
        chunksize,
        1,
    );
    let mut waveform = vec![vec![0.0 as f64; chunksize]; 1];
    c.bench_function("AVX SincFixedIn cubic f64", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_avx_sincfixedin_cubic_32(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let oversampling_factor = 256;
    let window = WindowFunction::BlackmanHarris2;
    let resample_ratio = 192000 as f64 / 44100 as f64;
    let interpolation_type = InterpolationType::Cubic;

    let interpolator = Box::new(AvxInterpolator::<f32>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ));
    let mut resampler = SincFixedIn::<f32>::new_with_interpolator(
        resample_ratio,
        interpolation_type,
        interpolator,
        chunksize,
        1,
    );
    let mut waveform = vec![vec![0.0 as f32; chunksize]; 1];
    c.bench_function("AVX SincFixedIn cubic f32", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}


#[cfg(target_arch = "aarch64")]
fn bench_neon_sincfixedin_cubic(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let oversampling_factor = 256;
    let window = WindowFunction::BlackmanHarris2;
    let resample_ratio = 192000 as f64 / 44100 as f64;
    let interpolation_type = InterpolationType::Cubic;

    let interpolator = Box::new(NeonInterpolator::<f64>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ));
    let mut resampler = SincFixedIn::<f64>::new_with_interpolator(
        resample_ratio,
        interpolation_type,
        interpolator,
        chunksize,
        1,
    );
    let mut waveform = vec![vec![0.0 as f64; chunksize]; 1];
    c.bench_function("Neon SincFixedIn cubic f64", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

#[cfg(target_arch = "aarch64")]
fn bench_neon_sincfixedin_cubic_32(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let oversampling_factor = 256;
    let window = WindowFunction::BlackmanHarris2;
    let resample_ratio = 192000 as f64 / 44100 as f64;
    let interpolation_type = InterpolationType::Cubic;

    let interpolator = Box::new(NeonInterpolator::<f32>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ));
    let mut resampler = SincFixedIn::<f32>::new_with_interpolator(
        resample_ratio,
        interpolation_type,
        interpolator,
        chunksize,
        1,
    );
    let mut waveform = vec![vec![0.0 as f32; chunksize]; 1];
    c.bench_function("Neon SincFixedIn cubic f32", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

#[cfg(target_arch = "x86_64")]
criterion_group!(
    benches,
    bench_fftfixedin,
    bench_fftfixedin_32,
    bench_sincfixedin_cubic,
    bench_sse_sincfixedin_cubic,
    bench_avx_sincfixedin_cubic,
    bench_sincfixedin_cubic_32,
    bench_sse_sincfixedin_cubic_32,
    bench_avx_sincfixedin_cubic_32,
    bench_sincfixedin_linear,
    bench_sse_sincfixedin_linear,
    bench_sincfixedin_linear_32,
    bench_sse_sincfixedin_linear_32,
    bench_sincfixedin_nearest,
    bench_sse_sincfixedin_nearest,
    bench_sincfixedin_nearest_32,
    bench_sse_sincfixedin_nearest_32,
);

#[cfg(target_arch = "aarch64")]
criterion_group!(
    benches,
    bench_fftfixedin,
    bench_fftfixedin_32,
    bench_sincfixedin_cubic,
    bench_neon_sincfixedin_cubic,
    bench_sincfixedin_cubic_32,
    bench_neon_sincfixedin_cubic_32,
    bench_sincfixedin_linear,
    bench_sincfixedin_linear_32,
    bench_sincfixedin_nearest,
    bench_sincfixedin_nearest_32,

);

criterion_main!(benches);
