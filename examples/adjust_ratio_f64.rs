use audioadapter_buffers::direct::InterleavedSlice;
use clap::{Parser, ValueEnum};
use rubato::{
    Async, FixedAsync, PolynomialDegree, Resampler, SincInterpolationParameters,
    SincInterpolationType, Slip, WindowFunction,
};
use std::fs::File;
use std::io::prelude::{Read, Write};
use std::io::{BufReader, BufWriter};
use std::time::Instant;

const BYTE_PER_SAMPLE: usize = 8;

// A resampler app that reads a raw file of little-endian 64 bit floats, and writes the output in the same format.
// Unlike the `process_raw` example, which converts between two fixed rates, this one uses one of the
// *adjustable* resamplers to apply a small, constant rate offset. This is the clock-drift / rate-matching case:
// the nominal input and output rates are equal (ratio 1:1), and the resampler is nudged by a user-selected
// offset given in parts per million (ppm). A positive offset produces slightly more output frames than input,
// a negative offset slightly fewer.
//
// The file handling is kept to the bare minimum, since it is not what this example is about.
// Errors simply panic, a partial sample at the end of the input is dropped, and nothing is
// done to limit memory use: the entire clip is held in memory, and the input twice over
// while it is decoded. The `read_file` and `write_file` helpers exist to keep the example
// short, and are not meant to be copied into an application.
//
// The offset is applied through the `Resampler::as_adjustable` capability accessor, so the same code drives every
// adjustable resampler type without knowing the concrete type. The synchronous FFT resamplers cannot change
// ratio and are therefore not offered here; use the `process_raw` example for fixed-ratio conversion.
// For a ratio that changes while processing, see the `ramp_ratio_f64` example.
//
// The adjustable resamplers support offsets up to roughly +/- 10%.
// To apply a +50 ppm offset to the two-channel file `sine_f64_2ch.raw` using the Slip resampler:
// ```
// cargo run --release --example adjust_ratio_f64 sine_f64_2ch.raw test.raw -r SlipFixedOutput -o 50
// ```
// There are two helper python scripts for testing.
//  - `make_tone_scale.py` to generate a scale of stepped pure tones in this format, which makes
//    the corrections easy to hear. It also prints the ppm offset to use for a given correction
//    rate. Run it with the `-h` flag for instructions.
//  - `analyze_result.py` to analyze the result.
//    This takes four arguments: file name, number of channels, samplerate, and sample format.
//    Example, to analyze the file created above:
//    ```
//    python examples/analyze_result.py test.raw 2 44100 f64
//    ```
//
// Rubato can log what it is doing through the `log` crate, behind the optional
// `log` feature. Enable the feature and set `RUST_LOG` to see it:
// ```
// RUST_LOG=debug cargo run --release --features log --example adjust_ratio_f64 ...
// ```

/// Apply a small constant rate offset to a raw file of 64 bit floats.
#[derive(Parser)]
#[command(version)]
struct Options {
    /// Raw file of little-endian 64 bit floats to read.
    input: String,

    /// Raw file to write, in the same format.
    output: String,

    /// Resampler to use. Only the adjustable types are offered, since the
    /// synchronous FFT resamplers cannot change ratio.
    #[arg(short, long, value_enum, ignore_case = true, default_value_t = ResamplerType::SlipFixedOutput)]
    resampler: ResamplerType,

    /// Number of channels in the file.
    #[arg(short, long, default_value_t = 2)]
    channels: usize,

    /// Rate offset in parts per million. Positive gives slightly more output
    /// frames than input, negative slightly fewer.
    #[arg(short, long, default_value_t = 50.0, allow_negative_numbers = true)]
    offset: f64,
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
    /// Slip resampler, fixed input size.
    #[value(name = "SlipFixedInput")]
    SlipFixedInput,
    /// Slip resampler, fixed output size.
    #[value(name = "SlipFixedOutput")]
    SlipFixedOutput,
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
    let offset_ppm = opts.offset;
    println!("Opening files: {}, {}", opts.input, opts.output);

    let rel_ratio = 1.0 + offset_ppm / 1_000_000.0;
    println!(
        "Applying a rate offset of {} ppm (relative ratio {})",
        offset_ppm, rel_ratio
    );

    println!("Copy input file to buffer");
    let file_in_disk = File::open(&opts.input).expect("Can't open file");
    let mut file_in_reader = BufReader::new(file_in_disk);
    let indata = read_file(&mut file_in_reader);
    let nbr_input_frames = indata.len() / channels;

    // The nominal ratio is 1:1, so the output has at most `rel_ratio` frames per input frame.
    // The factor of two leaves generous headroom for the resampler's internal buffering.
    let mut outdata = vec![0.0; 2 * channels * (nbr_input_frames as f64 * rel_ratio) as usize];

    println!("Creating resampler");
    // Every branch is built at the nominal ratio of 1.0. The asynchronous resamplers get a maximum
    // relative ratio of 1.1, matching the Slip resampler's built-in +/- 10% range.
    let chunk_size = 1024;
    let mut resampler: Box<dyn Resampler<f64>> = match opts.resampler {
        ResamplerType::SincFixedInput => {
            let params = SincInterpolationParameters::new(128, WindowFunction::Blackman2)
                .oversampling_factor(256)
                .interpolation(SincInterpolationType::Quadratic);
            Box::new(
                Async::<f64>::new_sinc(1.0, 1.1, &params, chunk_size, channels, FixedAsync::Input)
                    .unwrap(),
            )
        }
        ResamplerType::SincFixedOutput => {
            let params = SincInterpolationParameters::new(128, WindowFunction::Blackman2)
                .oversampling_factor(256)
                .interpolation(SincInterpolationType::Quadratic);
            Box::new(
                Async::<f64>::new_sinc(1.0, 1.1, &params, chunk_size, channels, FixedAsync::Output)
                    .unwrap(),
            )
        }
        ResamplerType::PolyFixedInput => Box::new(
            Async::<f64>::new_poly(
                1.0,
                1.1,
                PolynomialDegree::Septic,
                chunk_size,
                channels,
                FixedAsync::Input,
            )
            .unwrap(),
        ),
        ResamplerType::PolyFixedOutput => Box::new(
            Async::<f64>::new_poly(
                1.0,
                1.1,
                PolynomialDegree::Septic,
                chunk_size,
                channels,
                FixedAsync::Output,
            )
            .unwrap(),
        ),
        ResamplerType::SlipFixedInput => {
            Box::new(Slip::<f64>::new(chunk_size, channels, FixedAsync::Input).unwrap())
        }
        ResamplerType::SlipFixedOutput => {
            Box::new(Slip::<f64>::new(chunk_size, channels, FixedAsync::Output).unwrap())
        }
    };

    // Recover the adjust-ratio capability from the trait object and apply the offset once. Since the
    // nominal ratio is 1.0, the relative ratio is also the absolute ratio. `ramp` is false so the new
    // ratio takes effect from the first chunk.
    let adjustable = resampler
        .as_adjustable()
        .expect("the selected resampler type is adjustable");
    if let Err(e) = adjustable.set_resample_ratio_relative(rel_ratio, false) {
        panic!(
            "Could not apply an offset of {} ppm: {}. The adjustable resamplers support roughly +/- 10%.",
            offset_ppm, e
        );
    }

    // Prepare
    let input_adapter = InterleavedSlice::new(&indata, channels, nbr_input_frames).unwrap();
    let outdata_capacity = outdata.len() / channels;
    let mut output_adapter =
        InterleavedSlice::new_mut(&mut outdata, channels, outdata_capacity).unwrap();

    println!("Processing...");
    let start = Instant::now();

    let (nbr_in, nbr_out) = resampler
        .process_all_into_buffer(&input_adapter, &mut output_adapter, nbr_input_frames, None)
        .unwrap();

    let duration = start.elapsed();
    println!("Resampling took: {:?}", duration);

    println!(
        "Processed {} input frames into {} output frames (realized ratio {:.6})",
        nbr_in,
        nbr_out,
        nbr_out as f64 / nbr_in as f64
    );

    println!("Write output to file, trimming off the silent frames from both ends.");
    let mut file_out_disk = BufWriter::new(File::create(&opts.output).unwrap());
    write_file(&outdata[..nbr_out * channels], &mut file_out_disk);
}
