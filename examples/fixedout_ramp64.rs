extern crate rubato;
use rubato::{
    calculate_cutoff, Resampler, SincFixedOut, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
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
///! While resampling, it ramps the resampling ratio from 100% to a user-provided value, during a given time duration (measured in output time).
///! This version takes a varying number of input samples per chunk, and outputs a fixed number of samples.
///! The command line arguments are input filename, output filename, input samplerate, output samplerate,
///! number of channels, final relative ratio in percent, and ramp duration in seconds.
///! To resample the file `sine_f64_2ch.raw` from 44.1kHz to 192kHz, and assuming the file has two channels,
///  and that the resampling ratio should be ramped to 150% during 3 seconds, the command is:
///! ```
///! cargo run --release --example fixedout_ramp64 sine_f64_2ch.raw test.raw 44100 192000 2 150 3
///! ```
///! There are two helper python scripts for testing. `makesineraw.py` simply writes a stereo file
///! with a 1 second long 1kHz tone (at 44.1kHz). This script takes no aruments. Modify as needed to create other test files.
///! To analyze the result, use the `analyze_result.py` script. This takes three arguments: number of channels, samplerate, and number of bits per sample (32 or 64).
///! Example, to analyze the file created above:
///! ```
///! python examples/analyze_result.py test.raw 2 192000 64
///! ```

fn read_frames<R: Read + Seek>(inbuffer: &mut R, nbr: usize, channels: usize) -> Vec<Vec<f64>> {
    let mut buffer = vec![0u8; 8];
    let mut wfs = Vec::with_capacity(channels);
    for _chan in 0..channels {
        wfs.push(Vec::with_capacity(nbr));
    }
    let mut value: f64;
    for _frame in 0..nbr {
        for wf in wfs.iter_mut().take(channels) {
            if inbuffer.read(&mut buffer).unwrap() < 8 {
                return wfs;
            }
            value = f64::from_le_bytes(buffer.as_slice().try_into().unwrap());
            //idx += 8;
            wf.push(value);
        }
    }
    wfs
}

fn write_frames<W: Write + Seek>(waves: Vec<Vec<f64>>, outbuffer: &mut W, channels: usize) {
    let nbr = waves[0].len();
    for frame in 0..nbr {
        for wave in waves.iter().take(channels) {
            let value64 = wave[frame];
            let bytes = value64.to_le_bytes();
            outbuffer.write_all(&bytes).unwrap();
        }
    }
}

fn main() {
    // init logger
    let mut builder = Builder::from_default_env();
    builder.filter(None, LevelFilter::Trace).init();

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

    let ratio_str = env::args()
        .nth(6)
        .expect("Please specify final resampling ratio in percent");
    let final_ratio = ratio_str.parse::<f64>().unwrap();

    let duration_str = env::args()
        .nth(7)
        .expect("Please specify ramp time in seconds");
    let duration = duration_str.parse::<f64>().unwrap();

    //open files
    let mut f_in_disk = File::open(file_in).expect("Can't open file");
    let mut f_in_ram: Vec<u8> = vec![];
    let mut f_out_ram: Vec<u8> = vec![];

    println!("Copy input file to buffer");
    std::io::copy(&mut f_in_disk, &mut f_in_ram).unwrap();

    let mut f_in = Cursor::new(&f_in_ram);
    let mut f_out = Cursor::new(&mut f_out_ram);

    let f_ratio = fs_out as f64 / fs_in as f64;

    // Balanced for async, see the fixedin64 example for more config examples
    let sinc_len = 128;
    let oversampling_factor = 2048;
    let interpolation = SincInterpolationType::Linear;
    let window = WindowFunction::Blackman2;

    let f_cutoff = calculate_cutoff(sinc_len, window);
    let params = SincInterpolationParameters {
        sinc_len,
        f_cutoff,
        interpolation,
        oversampling_factor,
        window,
    };

    let chunksize = 1024;
    let target_ratio = final_ratio / 100.0;
    let mut resampler =
        SincFixedOut::<f64>::new(f_ratio, target_ratio, params, chunksize, channels).unwrap();

    let start = Instant::now();
    let mut output_time = 0.0;
    loop {
        let nbr_frames = resampler.input_frames_next();
        let waves = read_frames(&mut f_in, nbr_frames, channels);
        if waves[0].len() < nbr_frames {
            break;
        }
        let waves_out = resampler.process(&waves, None).unwrap();
        write_frames(waves_out, &mut f_out, channels);
        output_time += chunksize as f64 / fs_out as f64;
        if output_time < duration {
            let rel_time = output_time / duration;
            let rel_ratio = 1.0 + (target_ratio - 1.0) * rel_time;
            println!("time {}, rel ratio {}", output_time, rel_ratio);
            resampler
                .set_resample_ratio_relative(rel_ratio, true)
                .unwrap();
        }
    }

    let duration = start.elapsed();

    println!("Resampling took: {:?}", duration);

    let mut f_out_disk = File::create(file_out).unwrap();
    f_out.seek(std::io::SeekFrom::Start(0)).unwrap();
    std::io::copy(&mut f_out, &mut f_out_disk).unwrap();
}
