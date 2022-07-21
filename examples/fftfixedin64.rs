extern crate rubato;
use rubato::{FftFixedIn, Resampler};
use std::convert::TryInto;
use std::env;
use std::fs::File;
use std::io::prelude::{Read, Seek, Write};
use std::io::Cursor;
use std::time::Instant;

extern crate env_logger;
extern crate log;
use env_logger::Builder;
use log::LevelFilter;

///! A resampler app that reads a raw file of little-endian 64 bit floats, and writes the output in the same format.
///! The command line arguments are input filename, output filename, input samplerate, output samplerate, number of channels
///! To resample the file `sine_f64_2ch.raw` from 44.1kHz to 192kHz, and assuming the file has two channels, the command is:
///! ```
///! cargo run --release --example fftfixedin64 sine_f64_2ch.raw test.raw 44100 192000 2
///! ```
///! There are two helper python scripts for testing. `makesineraw.py` simply writes a stereo file
///! with a 1 second long 1kHz tone (at 44.1kHz). This script takes no aruments. Modify as needed to create other test files.
///! To analyze the result, use the `analyze_result.py` script. This takes three arguments: number of channels, samplerate, and number of bits per sample (32 or 64).
///! Example, to analyze the file created above:
///! ```
///! python examples/analyze_result.py test.raw 2 192000 64
///! ```

/// Helper to read frames from a buffer
fn read_frames<R: Read + Seek>(inbuffer: &mut R, nbr: usize, channels: usize) -> Vec<Vec<f64>> {
    let mut buffer = vec![0u8; 8];
    let mut wfs = Vec::with_capacity(channels);
    for _chan in 0..channels {
        wfs.push(Vec::with_capacity(nbr));
    }
    let mut value: f64;
    for _frame in 0..nbr {
        for wf in wfs.iter_mut().take(channels) {
            inbuffer.read(&mut buffer).unwrap();
            value = f64::from_le_bytes(buffer.as_slice().try_into().unwrap()) as f64;
            //idx += 8;
            wf.push(value);
        }
    }
    wfs
}

/// Helper to write frames to a buffer
fn write_frames<W: Write + Seek>(waves: Vec<Vec<f64>>, outbuffer: &mut W, channels: usize) {
    let nbr = waves[0].len();
    for frame in 0..nbr {
        for chan in 0..channels {
            let value64 = waves[chan][frame];
            let bytes = value64.to_le_bytes();
            outbuffer.write(&bytes).unwrap();
        }
    }
}

fn main() {
    // init logger
    let mut builder = Builder::from_default_env();
    builder.filter(None, LevelFilter::Debug).init();

    let file_in = env::args().nth(1).expect("Please specify an input file.");
    let file_out = env::args().nth(2).expect("Please specify an output file.");
    println!("Opening files: {}, {}", file_in, file_out);

    let fs_in_str = env::args()
        .nth(3)
        .expect("Please specify an input sample rate");
    let fs_out_str = env::args()
        .nth(4)
        .expect("Please specify an output sample rate");
    let fs_in = fs_in_str.parse::<usize>().unwrap();
    let fs_out = fs_out_str.parse::<usize>().unwrap();
    println!("Resampling from {} to {}", fs_in, fs_out);

    let channels_str = env::args()
        .nth(5)
        .expect("Please specify number of channels");
    let channels = channels_str.parse::<usize>().unwrap();

    //open files
    let mut f_in_disk = File::open(file_in).expect("Can't open file");
    let mut f_in_ram: Vec<u8> = vec![];

    println!("Copy input file to buffer");
    std::io::copy(&mut f_in_disk, &mut f_in_ram).unwrap();

    let file_size = f_in_ram.len();
    let mut f_out_ram: Vec<u8> =
        Vec::with_capacity((file_size as f32 * fs_out as f32 / fs_in as f32) as usize);

    let mut f_in = Cursor::new(&f_in_ram);
    let mut f_out = Cursor::new(&mut f_out_ram);

    let mut resampler = FftFixedIn::<f64>::new(fs_in, fs_out, 1024, 2, channels).unwrap();
    let chunksize = resampler.input_frames_next();

    let frame_bytes = 8 * channels;
    let chunksize_bytes = frame_bytes * chunksize;
    let num_chunks = f_in_ram.len() / chunksize_bytes;
    let rest_frames = (f_in_ram.len() % chunksize_bytes) / frame_bytes;

    let start = Instant::now();
    for _chunk in 0..num_chunks {
        let waves = read_frames(&mut f_in, chunksize, channels);
        let waves_out = resampler.process(&waves, None).unwrap();
        write_frames(waves_out, &mut f_out, channels);
    }
    // Process a partial chunk with the last frames.
    if rest_frames > 0 {
        let waves = read_frames(&mut f_in, rest_frames, channels);
        let waves_out = resampler.process_partial(Some(&waves), None).unwrap();
        write_frames(waves_out, &mut f_out, channels);
    }
    // Flush once to ensure we get all delayed samples.
    let waves_out = resampler.process_partial::<Vec<f64>>(None, None).unwrap();
    write_frames(waves_out, &mut f_out, channels);

    let duration = start.elapsed();

    println!("Resampling took: {:?}", duration);

    let mut f_out_disk = File::create(file_out).unwrap();
    f_out.seek(std::io::SeekFrom::Start(0)).unwrap();
    std::io::copy(&mut f_out, &mut f_out_disk).unwrap();
}
