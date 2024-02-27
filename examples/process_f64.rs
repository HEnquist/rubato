extern crate rubato;
use rubato::{
    calculate_cutoff, implement_resampler, FastFixedIn, FastFixedOut, PolynomialDegree,
    SincFixedIn, SincFixedOut, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
#[cfg(feature = "fft_resampler")]
use rubato::{FftFixedIn, FftFixedInOut, FftFixedOut};
use std::convert::TryInto;
use std::env;
use std::fs::File;
use std::io::prelude::{Read, Seek, Write};
use std::io::BufReader;
use std::time::Instant;

extern crate env_logger;
extern crate log;
use env_logger::Builder;
use log::LevelFilter;
const BYTE_PER_SAMPLE: usize = 8;

// A resampler app that reads a raw file of little-endian 64 bit floats, and writes the output in the same format.
// The command line arguments are resampler type, input filename, output filename, input samplerate, output samplerate, number of channels
// To use a SincFixedIn resampler to resample the file `sine_f64_2ch.raw` from 44.1kHz to 192kHz, and assuming the file has two channels, the command is:
// ```
// cargo run --release --example process_f64 SincFixedIn sine_f64_2ch.raw test.raw 44100 192000 2
// ```
// There are two helper python scripts for testing. `makesineraw.py` simply writes a stereo file
// with a 1 second long 1kHz tone (at 44.1kHz). This script takes no aruments. Modify as needed to create other test files.
// To analyze the result, use the `analyze_result.py` script. This takes three arguments: number of channels, samplerate, and number of bits per sample (32 or 64).
// Example, to analyze the file created above:
// ```
// python examples/analyze_result.py test.raw 2 192000 64
// ```

// Implement an object safe resampler with the input and output types needed in this example.
implement_resampler!(SliceResampler, &[&[T]], &mut [Vec<T>]);

/// Helper to read an entire file to memory
fn read_file<R: Read + Seek>(inbuffer: &mut R, channels: usize) -> Vec<Vec<f64>> {
    let mut buffer = vec![0u8; BYTE_PER_SAMPLE];
    let mut wfs = Vec::with_capacity(channels);
    for _chan in 0..channels {
        wfs.push(Vec::new());
    }
    'outer: loop {
        for wf in wfs.iter_mut() {
            let bytes_read = inbuffer.read(&mut buffer).unwrap();
            if bytes_read == 0 {
                break 'outer;
            }
            let value = f64::from_le_bytes(buffer.as_slice().try_into().unwrap());
            //idx += 8;
            wf.push(value);
        }
    }
    wfs
}

/// Helper to write all frames to a file
fn write_frames<W: Write + Seek>(
    waves: Vec<Vec<f64>>,
    output: &mut W,
    frames_to_skip: usize,
    frames_to_write: usize,
) {
    let channels = waves.len();
    let end = (frames_to_skip + frames_to_write).min(waves[0].len() - 1);
    for frame in frames_to_skip..end {
        for wave in waves.iter().take(channels) {
            let value64 = wave[frame];
            let bytes = value64.to_le_bytes();
            output.write_all(&bytes).unwrap();
        }
    }
}

fn append_frames(buffers: &mut [Vec<f64>], additional: &[Vec<f64>], nbr_frames: usize) {
    buffers
        .iter_mut()
        .zip(additional.iter())
        .for_each(|(b, a)| b.extend_from_slice(&a[..nbr_frames]));
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
    let indata = read_file(&mut file_in_reader, channels);
    let nbr_input_frames = indata[0].len();

    // Create buffer for storing output
    let mut outdata = vec![
        Vec::with_capacity(
            2 * (nbr_input_frames as f32 * fs_out as f32 / fs_in as f32) as usize
        );
        channels
    ];

    let f_ratio = fs_out as f64 / fs_in as f64;

    // Create resampler
    let mut resampler: Box<dyn SliceResampler<f64>> = match resampler_type.as_str() {
        "SincFixedIn" => {
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
            Box::new(SincFixedIn::<f64>::new(f_ratio, 1.1, params, 1024, channels).unwrap())
        }
        "SincFixedOut" => {
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
            Box::new(SincFixedOut::<f64>::new(f_ratio, 1.1, params, 1024, channels).unwrap())
        }
        "FastFixedIn" => {
            Box::new(FastFixedIn::<f64>::new(f_ratio, 1.1, PolynomialDegree::Septic, 1024, channels).unwrap())
        }
        "FastFixedOut" => {
            Box::new(FastFixedOut::<f64>::new(f_ratio, 1.1, PolynomialDegree::Septic, 1024, channels).unwrap())
        }
        #[cfg(feature = "fft_resampler")]
        "FftFixedIn" => {
            Box::new(FftFixedIn::<f64>::new(fs_in, fs_out, 1024, 2, channels).unwrap())
        }
        #[cfg(feature = "fft_resampler")]
        "FftFixedOut" => {
            Box::new(FftFixedOut::<f64>::new(fs_in, fs_out, 1024, 2, channels).unwrap())
        }
        #[cfg(feature = "fft_resampler")]
        "FftFixedInOut" => {
            Box::new(FftFixedInOut::<f64>::new(fs_in, fs_out, 1024, channels).unwrap())
        }
        _ => panic!("Unknown resampler type {}\nMust be one of SincFixedIn, SincFixedOut, FastFixedIn, FastFixedOut, FftFixedIn, FftFixedOut, FftFixedInOut", resampler_type),
    };

    // Prepare
    let mut input_frames_next = resampler.input_frames_next();
    let resampler_delay = resampler.output_delay();
    let mut outbuffer = vec![vec![0.0f64; resampler.output_frames_max()]; channels];
    let mut indata_slices: Vec<&[f64]> = indata.iter().map(|v| &v[..]).collect();

    // Process all full chunks
    let start = Instant::now();

    while indata_slices[0].len() >= input_frames_next {
        let (nbr_in, nbr_out) = resampler
            .process_into_buffer(&indata_slices, &mut outbuffer, None)
            .unwrap();
        for chan in indata_slices.iter_mut() {
            *chan = &chan[nbr_in..];
        }
        append_frames(&mut outdata, &outbuffer, nbr_out);
        input_frames_next = resampler.input_frames_next();
    }

    // Process a partial chunk with the last frames.
    if !indata_slices[0].is_empty() {
        let (_nbr_in, nbr_out) = resampler
            .process_partial_into_buffer(Some(&indata_slices), &mut outbuffer, None)
            .unwrap();
        append_frames(&mut outdata, &outbuffer, nbr_out);
    }

    let duration = start.elapsed();
    println!("Resampling took: {:?}", duration);

    let nbr_output_frames = (nbr_input_frames as f32 * fs_out as f32 / fs_in as f32) as usize;
    println!(
        "Processed {} input frames into {} output frames",
        nbr_input_frames, nbr_output_frames
    );

    // Write output to file, trimming off the silent frames from both ends.
    let mut file_out_disk = File::create(file_out).unwrap();
    write_frames(
        outdata,
        &mut file_out_disk,
        resampler_delay,
        nbr_output_frames,
    );
}
