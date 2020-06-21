use criterion::{criterion_group, criterion_main, Criterion};
extern crate rubato;

use rubato::{
    FftFixedIn, FftFixedOut, InterpolationParameters, InterpolationType, Resampler, SincFixedIn,
    WindowFunction,
};

fn bench_fftfixedin(c: &mut Criterion) {
    let chunksize = 1024;
    let mut resampler = FftFixedIn::<f64>::new(44100, 192000, 1024, 2, 1);
    let mut waveform = vec![vec![0.0 as f64; chunksize]; 1];
    c.bench_function("FftFixedIn", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

fn bench_fftfixedout(c: &mut Criterion) {
    let mut resampler = FftFixedOut::<f64>::new(44100, 192000, 4096, 2, 1);
    c.bench_function("FftFixedOut", |b| {
        b.iter(|| {
            let needed = resampler.nbr_frames_needed();
            let mut waveform = vec![vec![0.0 as f64; needed]; 1];
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

fn bench_sincfixedin(c: &mut Criterion) {
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
    c.bench_function("SincFixedIn async", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

fn bench_sincfixedin_sync(c: &mut Criterion) {
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
    c.bench_function("SincFixedIn sync", |b| {
        b.iter(|| {
            let _resampled = resampler.process(&mut waveform).unwrap();
        })
    });
}

criterion_group!(
    benches,
    bench_fftfixedin,
    bench_sincfixedin,
    bench_sincfixedin_sync,
    bench_fftfixedout
);

criterion_main!(benches);
