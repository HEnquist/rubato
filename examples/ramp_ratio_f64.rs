use audioadapter_buffers::direct::InterleavedSlice;
use clap::{Parser, ValueEnum};
use rubato::{
    Async, FixedAsync, Indexing, PolynomialDegree, Resampler, SincInterpolationParameters,
    SincInterpolationType, WindowFunction,
};
use std::fs::File;
use std::io::prelude::{Read, Write};
use std::io::{BufReader, BufWriter};
use std::time::Instant;

const BYTE_PER_SAMPLE: usize = std::mem::size_of::<f64>();

// A resampler app that reads a raw file of little-endian 64 bit floats, and writes the output in the same format.
// While resampling, it ramps the resampling ratio from 100% to a user-provided value, during a given time
// duration (measured in output time). Unlike the `adjust_ratio_f64` example, which applies one constant offset,
// this one changes the ratio continuously while processing.
//
// The file handling is kept to the bare minimum, since it is not what this example is about.
// Errors simply panic, a partial sample at the end of the input is dropped, and nothing is
// done to limit memory use: the entire clip is held in memory, and the input twice over
// while it is decoded. The `read_file` and `write_file` helpers exist to keep the example
// short, and are not meant to be copied into an application.
//
// To resample the file `sine_f64_2ch.raw` from 44.1kHz to 192kHz, and assuming the file has two channels,
// and that the resampling ratio should be ramped to 150% during 3 seconds, the command is:
// ```
// cargo run --release --example ramp_ratio_f64 sine_f64_2ch.raw test.raw 44100 192000 -r SincFixedOutput -t 150 -d 3
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
// RUST_LOG=debug cargo run --release --features log --example ramp_ratio_f64 ...
// ```

/// Resample a raw file of 64 bit floats while ramping the ratio.
#[derive(Parser)]
#[command(version)]
struct Options {
    /// Raw file of little-endian 64 bit floats to read.
    input: String,

    /// Raw file to write, in the same format.
    output: String,

    /// Sample rate of the input file, in Hz.
    input_rate: usize,

    /// Nominal sample rate of the output file, in Hz. The ramp is applied on top of this.
    output_rate: usize,

    /// Resampler to use. The synchronous FFT resamplers cannot change ratio and are not offered.
    #[arg(short, long, value_enum, ignore_case = true, default_value_t = ResamplerType::SincFixedOutput)]
    resampler: ResamplerType,

    /// Number of channels in the file.
    #[arg(short, long, default_value_t = 2)]
    channels: usize,

    /// Ratio to ramp to, in percent of the nominal ratio.
    #[arg(short, long, default_value_t = 150.0)]
    target: f64,

    /// Ramp duration in seconds, measured in output time.
    #[arg(short, long, default_value_t = 3.0)]
    duration: f64,
}

/// The adjustable resampler types this example can build.
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

fn main() {
    env_logger::init();

    let opts = Options::parse();
    let channels = opts.channels;
    let (fs_in, fs_out) = (opts.input_rate, opts.output_rate);
    let ramp_duration = opts.duration;
    println!("Opening files: {}, {}", opts.input, opts.output);
    println!("Resampling from {} to {}", fs_in, fs_out);

    println!("Copy input file to buffer");
    let file_in_disk = File::open(&opts.input).expect("Can't open file");
    let mut file_in_reader = BufReader::new(file_in_disk);
    let indata = read_file(&mut file_in_reader);
    let nbr_input_frames = indata.len() / channels;

    let f_ratio = fs_out as f64 / fs_in as f64;

    // Create buffer for storing output, size is preliminary and may grow
    let mut outdata =
        Vec::with_capacity(2 * channels * (nbr_input_frames as f64 * f_ratio) as usize);

    println!("Creating resampler");
    let chunksize = 1024;
    let target_ratio = opts.target / 100.0;
    // The maximum relative ratio must cover the ramp, so it is set to the target.
    let mut resampler: Box<dyn Resampler<f64>> = match opts.resampler {
        ResamplerType::SincFixedInput | ResamplerType::SincFixedOutput => {
            // Balanced for ratio changes: a high oversampling factor keeps the
            // interpolation between the sinc tables cheap and accurate.
            let params = SincInterpolationParameters::new(128, WindowFunction::Blackman2)
                .oversampling_factor(2048)
                .interpolation(SincInterpolationType::Linear);
            let fixed = if opts.resampler == ResamplerType::SincFixedInput {
                FixedAsync::Input
            } else {
                FixedAsync::Output
            };
            Box::new(
                Async::<f64>::new_sinc(f_ratio, target_ratio, &params, chunksize, channels, fixed)
                    .unwrap(),
            )
        }
        ResamplerType::PolyFixedInput | ResamplerType::PolyFixedOutput => {
            let fixed = if opts.resampler == ResamplerType::PolyFixedInput {
                FixedAsync::Input
            } else {
                FixedAsync::Output
            };
            Box::new(
                Async::<f64>::new_poly(
                    f_ratio,
                    target_ratio,
                    PolynomialDegree::Cubic,
                    chunksize,
                    channels,
                    fixed,
                )
                .unwrap(),
            )
        }
    };

    let input_adapter = InterleavedSlice::new(&indata, channels, nbr_input_frames).unwrap();
    let mut indexing = Indexing::new();

    println!("Processing...");
    let start = Instant::now();
    let mut output_time = 0.0;
    let mut frames_left = nbr_input_frames;

    // The same loop drives both the fixed input and the fixed output resamplers.
    // Ask how many input frames the next call needs, and advance by the number it consumed.
    while frames_left > resampler.input_frames_next() {
        let frames_out = resampler.output_frames_next();
        let mut output_scratch = vec![0.0; channels * frames_out];
        let mut output_adapter =
            InterleavedSlice::new_mut(&mut output_scratch, channels, frames_out).unwrap();
        let (nbr_in, nbr_out) = resampler
            .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
            .unwrap();

        // Keep only the frames that were actually written. With a fixed input size,
        // the output size varies and can be shorter than the scratch buffer.
        output_scratch.truncate(channels * nbr_out);
        outdata.append(&mut output_scratch);

        frames_left -= nbr_in;
        indexing.input_offset += nbr_in;

        // Ramp the ratio linearly towards the target, as a function of output time.
        output_time += nbr_out as f64 / fs_out as f64;
        if output_time < ramp_duration {
            let rel_time = output_time / ramp_duration;
            let rel_ratio = 1.0 + (target_ratio - 1.0) * rel_time;
            println!("time {}, rel ratio {}", output_time, rel_ratio);
            resampler
                .as_adjustable()
                .expect("the selected resampler type is adjustable")
                .set_resample_ratio_relative(rel_ratio, true)
                .unwrap();
        }
    }

    // Process the frames that are left over, fewer than the resampler asks for.
    // Setting `partial_len` tells it how many of the frames are real, and it inserts
    // silence in place of the rest. Without this the tail of the clip is dropped.
    if frames_left > 0 {
        let frames_out = resampler.output_frames_next();
        let mut output_scratch = vec![0.0; channels * frames_out];
        let mut output_adapter =
            InterleavedSlice::new_mut(&mut output_scratch, channels, frames_out).unwrap();
        indexing.partial_len = Some(frames_left);
        let (_nbr_in, nbr_out) = resampler
            .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
            .unwrap();
        output_scratch.truncate(channels * nbr_out);
        outdata.append(&mut output_scratch);
    }

    let duration = start.elapsed();
    println!("Resampling took: {:?}", duration);

    let mut f_out_disk = BufWriter::new(File::create(&opts.output).unwrap());
    write_file(&outdata, &mut f_out_disk);
}
