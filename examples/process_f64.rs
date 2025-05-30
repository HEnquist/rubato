extern crate rubato;
use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    calculate_cutoff, Async, FixedAsync, Indexing, PolynomialDegree, Resampler,
    SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
#[cfg(feature = "fft_resampler")]
use rubato::{Fft, FixedSync};
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
// The command line arguments are resampler type, input filename, output filename, input samplerate, output samplerate, number of channels
// To use a sinc resampler with fixed input size to resample the file `sine_f64_2ch.raw` from 44.1kHz to 192kHz, and assuming the file has two channels, the command is:
// ```
// cargo run --release --example process_f64 SincFixedInput sine_f64_2ch.raw test.raw 44100 192000 2
// ```
// There are two helper python scripts for testing.
//  - `makesineraw.py` to generate test files in raw format.
//    Run it with the `-h` flag for instructions.
//  - `analyze_result.py` to analyze the result.
//    This takes three arguments: number of channels, samplerate, and sample format.
//    Example, to analyze the file created above:
//    ```
//    python examples/analyze_result.py test.raw 2 192000 f64
//    ```

/// Helper to read an entire file to memory as f64 values
fn read_file<R: Read + Seek>(inbuffer: &mut R) -> Vec<f64> {
    let mut buffer = vec![0u8; BYTE_PER_SAMPLE];
    let mut data = Vec::new();
    loop {
        let bytes_read = inbuffer.read(&mut buffer).unwrap();
        if bytes_read == 0 {
            break;
        }
        let value = f64::from_le_bytes(buffer.as_slice().try_into().unwrap());
        data.push(value);
    }
    data
}

/// Helper to write all frames to a file
fn write_file<W: Write + Seek>(
    data: &[f64],
    output: &mut W,
    values_to_skip: usize,
    values_to_write: usize,
) {
    for value in data.iter().skip(values_to_skip).take(values_to_write) {
        let bytes = value.to_le_bytes();
        output.write_all(&bytes).unwrap();
    }
}

fn main() {
    // init logger
    let mut builder = Builder::from_default_env();
    builder.filter(None, LevelFilter::Debug).init();

    let resampler_type = env::args()
        .nth(1)
        .expect("Please specify a resampler type, one of:\nSincFixedIn\nSincFixedOut\nFastFixedIn\nFastFixedOut\nFftFixedIn\nFftFixedOut\nFftFixedInOut");

    let file_in = env::args().nth(2).expect("Please specify an input file.");
    let file_out = env::args().nth(3).expect("Please specify an output file.");
    println!("Opening files: {}, {}", file_in, file_out);

    let fs_in_str = env::args()
        .nth(4)
        .expect("Please specify an input sample rate");
    let fs_out_str = env::args()
        .nth(5)
        .expect("Please specify an output sample rate");
    let fs_in = fs_in_str.parse::<usize>().unwrap();
    let fs_out = fs_out_str.parse::<usize>().unwrap();
    println!("Resampling from {} to {}", fs_in, fs_out);

    let channels_str = env::args()
        .nth(6)
        .expect("Please specify number of channels");
    let channels = channels_str.parse::<usize>().unwrap();

    println!("Copy input file to buffer");
    let file_in_disk = File::open(file_in).expect("Can't open file");
    let mut file_in_reader = BufReader::new(file_in_disk);
    let indata = read_file(&mut file_in_reader);
    let nbr_input_frames = indata.len() / channels;

    let f_ratio = fs_out as f64 / fs_in as f64;

    // Create buffer for storing output
    let mut outdata = vec![0.0; 2 * channels * (nbr_input_frames as f64 * f_ratio) as usize];

    println!("Creating resampler");
    // Create resampler
    let mut resampler: Box<dyn Resampler<f64>> = match resampler_type.as_str() {
        "SincFixedInput" => {
            let sinc_len = 128;
            let oversampling_factor = 256;
            let interpolation = SincInterpolationType::Quadratic;
            let window = WindowFunction::Blackman2;

            let f_cutoff = calculate_cutoff(sinc_len, window);
            let params = SincInterpolationParameters {
                sinc_len,
                f_cutoff,
                interpolation,
                oversampling_factor,
                window,
            };
            Box::new(Async::<f64>::new_sinc(f_ratio, 1.1, params, 1024, channels, FixedAsync::Input).unwrap())
        }
        "SincFixedOutput" => {
            let sinc_len = 128;
            let oversampling_factor = 512;
            let interpolation = SincInterpolationType::Cubic;
            let window = WindowFunction::Blackman2;

            let f_cutoff = calculate_cutoff(sinc_len, window);
            let params = SincInterpolationParameters {
                sinc_len,
                f_cutoff,
                interpolation,
                oversampling_factor,
                window,
            };
            Box::new(Async::<f64>::new_sinc(f_ratio, 1.1, params, 1024, channels, FixedAsync::Output).unwrap())
        }
        "PolyFixedInput" => {
            Box::new(Async::<f64>::new_poly(f_ratio, 1.1, PolynomialDegree::Septic, 1024, channels, FixedAsync::Input).unwrap())
        }
        "PolyFixedOutput" => {
            Box::new(Async::<f64>::new_poly(f_ratio, 1.1, PolynomialDegree::Septic, 1024, channels, FixedAsync::Output).unwrap())
        }
        #[cfg(feature = "fft_resampler")]
        "FftFixedInput" => {
            Box::new(Fft::<f64>::new(fs_in, fs_out, 1024, 2, channels, FixedSync::Input).unwrap())
        }
        #[cfg(feature = "fft_resampler")]
        "FftFixedOutput" => {
            Box::new(Fft::<f64>::new(fs_in, fs_out, 1024, 2, channels, FixedSync::Output).unwrap())
        }
        #[cfg(feature = "fft_resampler")]
        "FftFixedBoth" => {
            Box::new(Fft::<f64>::new(fs_in, fs_out, 1024, 1, channels, FixedSync::Both).unwrap())
        }
        _ => panic!("Unknown resampler type {}\nMust be one of SincFixedInput, SincFixedOutput, PolyFixedInput, PolyFixedOutput, FftFixedInput, FftFixedOutput, FftFixedBoth", resampler_type),
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
    let mut indexing = Indexing {
        input_offset: 0,
        output_offset: 0,
        active_channels_mask: None,
        partial_len: None,
    };
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
    let mut file_out_disk = BufWriter::new(File::create(file_out).unwrap());
    write_file(
        &outdata,
        &mut file_out_disk,
        resampler_delay * channels,
        nbr_output_frames * channels,
    );
}
