use criterion::{criterion_group, criterion_main, Criterion};
extern crate rubato;

use rubato::{
    FftFixedIn, FftFixedOut, InterpolationParameters, InterpolationType, Resampler, SincFixedIn, SseSincFixedIn,
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
    let params = InterpolationParameters {
        sinc_len,
        f_cutoff,
        interpolation: InterpolationType::Cubic,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f64>::new(192000 as f64 / 44100 as f64, params, chunksize, 1);
    let mut waveform = vec![vec![0.0 as f64; chunksize]; 1];
    c.bench_function("SincFixedIn cubic f64", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

fn bench_sincfixedin_lin(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let params = InterpolationParameters {
        sinc_len,
        f_cutoff,
        interpolation: InterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f64>::new(192000 as f64 / 44100 as f64, params, chunksize, 1);
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
    let params = InterpolationParameters {
        sinc_len,
        f_cutoff,
        interpolation: InterpolationType::Nearest,
        oversampling_factor: 640,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f64>::new(192000 as f64 / 44100 as f64, params, chunksize, 1);
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
    let params = InterpolationParameters {
        sinc_len,
        f_cutoff,
        interpolation: InterpolationType::Cubic,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(192000 as f64 / 44100 as f64, params, chunksize, 1);
    let mut waveform = vec![vec![0.0 as f32; chunksize]; 1];
    c.bench_function("SincFixedIn cubic f32", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

fn bench_sincfixedin_lin_32(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let params = InterpolationParameters {
        sinc_len,
        f_cutoff,
        interpolation: InterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(192000 as f64 / 44100 as f64, params, chunksize, 1);
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
    let params = InterpolationParameters {
        sinc_len,
        f_cutoff,
        interpolation: InterpolationType::Nearest,
        oversampling_factor: 640,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(192000 as f64 / 44100 as f64, params, chunksize, 1);
    let mut waveform = vec![vec![0.0 as f32; chunksize]; 1];
    c.bench_function("SincFixedIn nearest f32", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}


fn bench_ssesincfixedin_cubic(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let params = InterpolationParameters {
        sinc_len,
        f_cutoff,
        interpolation: InterpolationType::Cubic,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SseSincFixedIn::<f64>::new(192000 as f64 / 44100 as f64, params, chunksize, 1);
    let mut waveform = vec![vec![0.0 as f64; chunksize]; 1];
    c.bench_function("SseSincFixedIn cubic f64", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

fn bench_ssesincfixedin_cubic_32(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let params = InterpolationParameters {
        sinc_len,
        f_cutoff,
        interpolation: InterpolationType::Cubic,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SseSincFixedIn::<f32>::new(192000 as f64 / 44100 as f64, params, chunksize, 1);
    let mut waveform = vec![vec![0.0 as f32; chunksize]; 1];
    c.bench_function("SseSincFixedIn cubic f32", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

fn bench_ssesincfixedin_linear(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let params = InterpolationParameters {
        sinc_len,
        f_cutoff,
        interpolation: InterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SseSincFixedIn::<f64>::new(192000 as f64 / 44100 as f64, params, chunksize, 1);
    let mut waveform = vec![vec![0.0 as f64; chunksize]; 1];
    c.bench_function("SseSincFixedIn linear f64", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

fn bench_ssesincfixedin_linear_32(c: &mut Criterion) {
    let chunksize = 1024;
    let sinc_len = 256;
    let f_cutoff = 0.9473371669037001;
    let params = InterpolationParameters {
        sinc_len,
        f_cutoff,
        interpolation: InterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SseSincFixedIn::<f32>::new(192000 as f64 / 44100 as f64, params, chunksize, 1);
    let mut waveform = vec![vec![0.0 as f32; chunksize]; 1];
    c.bench_function("SseSincFixedIn linear f32", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

criterion_group!(
    benches,
    bench_fftfixedin,
    bench_fftfixedin_32,
    bench_sincfixedin_cubic,
    bench_sincfixedin_lin,
    bench_sincfixedin_nearest,
    bench_sincfixedin_cubic_32,
    bench_sincfixedin_lin_32,
    bench_sincfixedin_nearest_32,
    bench_ssesincfixedin_cubic,
    bench_ssesincfixedin_cubic_32,
    bench_ssesincfixedin_linear,
    bench_ssesincfixedin_linear_32,
);

criterion_main!(benches);
