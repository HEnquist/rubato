extern crate rubato;
use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, PolynomialDegree, Resampler, SincInterpolationParameters,
    SincInterpolationType, Slip, WindowFunction,
};
use std::convert::TryInto;
use std::env;
use std::fs::File;
use std::io::prelude::{Read, Seek, Write};
use std::io::{BufReader, BufWriter};
use std::time::Instant;

extern crate env_logger;
extern crate log;
use env_logger::Builder;
use log::LevelFilter;
const BYTE_PER_SAMPLE: usize = 8;

// A resampler app that reads a raw file of little-endian 64 bit floats, and writes the output in the same format.
// Unlike the `process_all_f64` example, which converts between two fixed rates, this one uses one of the
// *adjustable* resamplers to apply a small, constant rate offset. This is the clock-drift / rate-matching case:
// the nominal input and output rates are equal (ratio 1:1), and the resampler is nudged by a user-selected
// offset given in parts per million (ppm). A positive offset produces slightly more output frames than input,
// a negative offset slightly fewer.
//
// The offset is applied through the `Resampler::as_adjustable` capability accessor, so the same code drives every
// adjustable resampler type without knowing the concrete type. The synchronous FFT resamplers cannot change
// ratio and are therefore not offered here; use the `process_all_f64` example for fixed-ratio conversion.
//
// The command line arguments are resampler type, input filename, output filename, number of channels,
// and the rate offset in ppm. The adjustable resamplers support offsets up to roughly +/- 10%.
// To apply a +50 ppm offset to the two-channel file `sine_f64_2ch.raw` using the Slip resampler:
// ```
// cargo run --release --example adjust_ratio_f64 SlipFixedOutput sine_f64_2ch.raw test.raw 2 50
// ```

/// Helper to read an entire file to memory as f64 values
fn read_file<R: Read + Seek>(inbuffer: &mut R) -> Vec<f64> {
    let mut buffer = vec![0u8; BYTE_PER_SAMPLE];
    let mut data = Vec::new();
    loop {
        match inbuffer.read_exact(&mut buffer) {
            Ok(()) => {
                let value = f64::from_le_bytes(buffer.as_slice().try_into().unwrap());
                data.push(value);
            }
            // A clean end of file stops the loop; a partial trailing read means a malformed file.
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => panic!("Error reading input file: {}", e),
        }
    }
    data
}

/// Helper to write all frames to a file
fn write_file<W: Write + Seek>(data: &[f64], output: &mut W, values_to_write: usize) {
    for value in data.iter().take(values_to_write) {
        let bytes = value.to_le_bytes();
        output.write_all(&bytes).unwrap();
    }
}

fn main() {
    // init logger
    let mut builder = Builder::from_default_env();
    builder.filter(None, LevelFilter::Debug).init();

    let resampler_type = env::args().nth(1).expect(
        "Please specify a resampler type, one of:\nSincFixedInput\nSincFixedOutput\nPolyFixedInput\nPolyFixedOutput\nSlipFixedInput\nSlipFixedOutput",
    );

    let file_in = env::args().nth(2).expect("Please specify an input file.");
    let file_out = env::args().nth(3).expect("Please specify an output file.");
    println!("Opening files: {}, {}", file_in, file_out);

    let channels_str = env::args()
        .nth(4)
        .expect("Please specify number of channels");
    let channels = channels_str.parse::<usize>().unwrap();

    let offset_str = env::args()
        .nth(5)
        .expect("Please specify the rate offset in ppm");
    let offset_ppm = offset_str.parse::<f64>().unwrap();
    let rel_ratio = 1.0 + offset_ppm / 1_000_000.0;
    println!(
        "Applying a rate offset of {} ppm (relative ratio {})",
        offset_ppm, rel_ratio
    );

    println!("Copy input file to buffer");
    let file_in_disk = File::open(file_in).expect("Can't open file");
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
    let mut resampler: Box<dyn Resampler<f64>> = match resampler_type.as_str() {
        "SincFixedInput" => {
            let params = SincInterpolationParameters::new(128, WindowFunction::Blackman2)
                .oversampling_factor(256)
                .interpolation(SincInterpolationType::Quadratic);
            Box::new(
                Async::<f64>::new_sinc(1.0, 1.1, &params, chunk_size, channels, FixedAsync::Input)
                    .unwrap(),
            )
        }
        "SincFixedOutput" => {
            let params = SincInterpolationParameters::new(128, WindowFunction::Blackman2)
                .oversampling_factor(256)
                .interpolation(SincInterpolationType::Quadratic);
            Box::new(
                Async::<f64>::new_sinc(1.0, 1.1, &params, chunk_size, channels, FixedAsync::Output)
                    .unwrap(),
            )
        }
        "PolyFixedInput" => Box::new(
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
        "PolyFixedOutput" => Box::new(
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
        "SlipFixedInput" => {
            Box::new(Slip::<f64>::new(chunk_size, channels, FixedAsync::Input).unwrap())
        }
        "SlipFixedOutput" => {
            Box::new(Slip::<f64>::new(chunk_size, channels, FixedAsync::Output).unwrap())
        }
        _ => panic!(
            "Unknown or non-adjustable resampler type {}\nMust be one of SincFixedInput, SincFixedOutput, PolyFixedInput, PolyFixedOutput, SlipFixedInput, SlipFixedOutput",
            resampler_type
        ),
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
    let mut file_out_disk = BufWriter::new(File::create(file_out).unwrap());
    write_file(&outdata, &mut file_out_disk, nbr_out * channels);
}
