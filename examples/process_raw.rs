use audioadapter_buffers::direct::InterleavedSlice;
use clap::{Parser, ValueEnum};
use rubato::{
    Async, FixedAsync, Indexing, PolynomialDegree, Resampler, Sample, SincInterpolationParameters,
    SincInterpolationType, WindowFunction,
};
#[cfg(feature = "fft_resampler")]
use rubato::{Fft, FixedSync};
use std::fs::File;
use std::io::prelude::{Read, Write};
use std::io::{BufReader, BufWriter};
use std::time::Instant;

const BYTE_PER_SAMPLE: usize = 8;

// A resampler app that reads a raw file of little-endian 64 bit floats, and writes the output in the same format.
// This is the fixed ratio case. See the `adjust_ratio_f64` example for applying a constant rate offset,
// and `ramp_ratio_f64` for a ratio that changes while processing.
//
// This has a second purpose beyond showing how to drive the resamplers: together with the
// python scripts it is the measurement tool for the resampling quality. Generate a test
// signal, resample it, and analyze the result. That is why it stays on raw 64 bit floats
// instead of reading and writing wav files, which would be friendlier but would defeat the
// measurement. The interesting noise floors sit 200 dB or more below the signal, way past
// what an integer wav format can hold, and quantizing on the way out would measure the file
// format rather than the resampler. Use the `resample_wav` example for real audio files.
//
// The resampling itself runs in either 32 or 64 bit floats, selected with `--precision`.
// The files stay 64 bit either way, so the only thing that changes is the precision the
// resampler works in, which is what makes the two runs comparable. Resample the same clip
// both ways and compare the spectra to see what the lower precision costs in noise and
// distortion. Note that feeding f32 quantizes the input at around -150 dBFS before the
// resampler sees it, which is part of what processing in f32 means.
//
// The file handling is kept to the bare minimum, since it is not what this example is about.
// Errors simply panic, a partial sample at the end of the input is dropped, and nothing is
// done to limit memory use: the entire clip is held in memory, and the input twice over
// while it is decoded. The `read_file` and `write_file` helpers exist to keep the example
// short, and are not meant to be copied into an application.
//
// To use a sinc resampler with fixed input size to resample the file `sine_f64_2ch.raw` from 44.1kHz
// to 192kHz, and assuming the file has two channels, the command is:
// ```
// cargo run --release --example process_raw sine_f64_2ch.raw test.raw 44100 192000 -r SincFixedInput
// ```
// There are two helper python scripts for testing.
//  - `make_sine.py` to generate test files in raw format.
//    Run it with the `-h` flag for instructions.
//  - `analyze_result.py` to analyze the result.
//    This takes four arguments: file name, number of channels, samplerate, and sample format.
//    Example, to analyze the file created above:
//    ```
//    python examples/analyze_result.py test.raw 2 192000 f64
//    ```
//
// Rubato can log what it is doing through the `log` crate, behind the optional
// `log` feature. Enable the feature and set `RUST_LOG` to see it:
// ```
// RUST_LOG=debug cargo run --release --features log --example process_raw ...
// ```

/// Resample a raw file of 64 bit floats between two fixed sample rates.
#[derive(Parser)]
#[command(version)]
struct Options {
    /// Raw file of little-endian 64 bit floats to read.
    input: String,

    /// Raw file to write, in the same format.
    output: String,

    /// Sample rate of the input file, in Hz.
    input_rate: usize,

    /// Sample rate of the output file, in Hz.
    output_rate: usize,

    /// Resampler to use.
    #[arg(short, long, value_enum, ignore_case = true, default_value_t = ResamplerType::SincFixedInput)]
    resampler: ResamplerType,

    /// Number of channels in the file.
    #[arg(short, long, default_value_t = 2)]
    channels: usize,

    /// Floating point precision to run the resampling in. The files are 64 bit either way.
    #[arg(short, long, value_enum, ignore_case = true, default_value_t = Precision::F64)]
    precision: Precision,
}

/// The sample types the resamplers can be instantiated with.
#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Precision {
    /// 32 bit floats.
    #[value(name = "f32")]
    F32,
    /// 64 bit floats.
    #[value(name = "f64")]
    F64,
}

/// The resampler types this example can build.
#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum ResamplerType {
    /// Sinc interpolation, fixed input size.
    #[value(name = "SincFixedInput")]
    SincFixedInput,
    /// Sinc interpolation, fixed output size.
    #[value(name = "SincFixedOutput")]
    SincFixedOutput,
    /// Polynomial interpolation, fixed input size.
    #[value(name = "PolyFixedInput")]
    PolyFixedInput,
    /// Polynomial interpolation, fixed output size.
    #[value(name = "PolyFixedOutput")]
    PolyFixedOutput,
    /// Synchronous FFT, fixed input size.
    #[cfg(feature = "fft_resampler")]
    #[value(name = "FftFixedInput")]
    FftFixedInput,
    /// Synchronous FFT, fixed output size.
    #[cfg(feature = "fft_resampler")]
    #[value(name = "FftFixedOutput")]
    FftFixedOutput,
    /// Synchronous FFT, both sizes fixed.
    #[cfg(feature = "fft_resampler")]
    #[value(name = "FftFixedBoth")]
    FftFixedBoth,
}

/// Helper to read an entire file to memory as f64 values.
///
/// Minimal on purpose, do not copy this into an application. It panics on any io
/// error, silently ignores a partial sample at the end of the file, and holds the
/// contents in memory twice, as bytes and as samples, while decoding.
fn read_file<R: Read>(inbuffer: &mut R) -> Vec<f64> {
    let mut bytes = Vec::new();
    inbuffer.read_to_end(&mut bytes).unwrap();
    bytes
        .chunks_exact(BYTE_PER_SAMPLE)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Helper to write all frames to a file.
///
/// Minimal on purpose, do not copy this into an application. It panics on any io error,
/// and expects the whole clip to already be in memory.
fn write_file<W: Write>(data: &[f64], output: &mut W) {
    for value in data.iter() {
        let bytes = value.to_le_bytes();
        output.write_all(&bytes).unwrap();
    }
}

/// Read the input, resample it in the precision `T`, and write the result.
///
/// The data on disk is 64 bit either way. The conversion to and from `T` happens
/// here, outside the timed section, so the reported time is the resampling only.
fn resample<T>(opts: &Options)
where
    T: Sample + Into<f64>,
{
    let channels = opts.channels;
    let (fs_in, fs_out) = (opts.input_rate, opts.output_rate);
    println!("Opening files: {}, {}", opts.input, opts.output);
    println!("Resampling from {} to {}", fs_in, fs_out);

    println!("Copy input file to buffer");
    let file_in_disk = File::open(&opts.input).expect("Can't open file");
    let mut file_in_reader = BufReader::new(file_in_disk);
    let indata: Vec<T> = read_file(&mut file_in_reader)
        .into_iter()
        .map(T::coerce)
        .collect();
    let nbr_input_frames = indata.len() / channels;

    let f_ratio = fs_out as f64 / fs_in as f64;

    // Create buffer for storing output
    let mut outdata =
        vec![T::coerce(0.0); 2 * channels * (nbr_input_frames as f64 * f_ratio) as usize];

    println!("Creating resampler");
    // Create resampler
    let mut resampler: Box<dyn Resampler<T>> = match opts.resampler {
        ResamplerType::SincFixedInput => {
            let params = SincInterpolationParameters::new(128, WindowFunction::Blackman2)
                .oversampling_factor(256)
                .interpolation(SincInterpolationType::Quadratic);
            Box::new(
                Async::<T>::new_sinc(f_ratio, 1.1, &params, 1024, channels, FixedAsync::Input)
                    .unwrap(),
            )
        }
        ResamplerType::SincFixedOutput => {
            let params = SincInterpolationParameters::new(128, WindowFunction::Blackman2)
                .oversampling_factor(512)
                .interpolation(SincInterpolationType::Cubic);
            Box::new(
                Async::<T>::new_sinc(f_ratio, 1.1, &params, 1024, channels, FixedAsync::Output)
                    .unwrap(),
            )
        }
        ResamplerType::PolyFixedInput => Box::new(
            Async::<T>::new_poly(
                f_ratio,
                1.1,
                PolynomialDegree::Septic,
                1024,
                channels,
                FixedAsync::Input,
            )
            .unwrap(),
        ),
        ResamplerType::PolyFixedOutput => Box::new(
            Async::<T>::new_poly(
                f_ratio,
                1.1,
                PolynomialDegree::Septic,
                1024,
                channels,
                FixedAsync::Output,
            )
            .unwrap(),
        ),
        #[cfg(feature = "fft_resampler")]
        ResamplerType::FftFixedInput => {
            Box::new(Fft::<T>::new(fs_in, fs_out, 1024, channels, FixedSync::Input).unwrap())
        }
        #[cfg(feature = "fft_resampler")]
        ResamplerType::FftFixedOutput => {
            Box::new(Fft::<T>::new(fs_in, fs_out, 1024, channels, FixedSync::Output).unwrap())
        }
        #[cfg(feature = "fft_resampler")]
        ResamplerType::FftFixedBoth => {
            Box::new(Fft::<T>::new(fs_in, fs_out, 1024, channels, FixedSync::Both).unwrap())
        }
    };

    // Prepare
    let mut input_frames_next = resampler.input_frames_next();
    let resampler_delay = resampler.output_delay();

    let input_adapter = InterleavedSlice::new(&indata, channels, nbr_input_frames).unwrap();
    let outdata_capacity = outdata.len() / channels;
    let mut output_adapter =
        InterleavedSlice::new_mut(&mut outdata, channels, outdata_capacity).unwrap();

    println!("Process all full chunks");
    let start = Instant::now();
    let mut indexing = Indexing::new();
    let mut input_frames_left = nbr_input_frames;

    while input_frames_left >= input_frames_next {
        let (nbr_in, nbr_out) = resampler
            .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
            .unwrap();

        indexing.input_offset += nbr_in;
        indexing.output_offset += nbr_out;
        input_frames_left -= nbr_in;
        input_frames_next = resampler.input_frames_next();
    }

    println!("Process a partial chunk with the last frames.");
    indexing.partial_len = Some(input_frames_left);
    let (_nbr_in, _nbr_out) = resampler
        .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
        .unwrap();

    let duration = start.elapsed();
    println!("Resampling took: {:?}", duration);

    let nbr_output_frames = (nbr_input_frames as f32 * fs_out as f32 / fs_in as f32) as usize;
    println!(
        "Processed {} input frames into {} output frames",
        nbr_input_frames, nbr_output_frames
    );

    println!("Write output to file, trimming off the silent frames from both ends.");
    // The resampler delay is silence at the start, and the tail past the expected
    // length is padding from the last partial chunk.
    let first = resampler_delay * channels;
    let last = first + nbr_output_frames * channels;
    let trimmed: Vec<f64> = outdata[first..last].iter().map(|v| (*v).into()).collect();
    let mut file_out_disk = BufWriter::new(File::create(&opts.output).unwrap());
    write_file(&trimmed, &mut file_out_disk);
}

fn main() {
    env_logger::init();

    let opts = Options::parse();
    match opts.precision {
        Precision::F32 => resample::<f32>(&opts),
        Precision::F64 => resample::<f64>(&opts),
    }
}
