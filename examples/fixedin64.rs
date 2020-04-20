extern crate camillaresampler;
use camillaresampler::{Interpolation, ResamplerFixedIn};
use std::convert::TryInto;
use std::env;
use std::fs::File;
use std::io::prelude::{Read, Seek, Write};
use std::io::Cursor;
use std::time::Instant;

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
    let mut f_out_ram: Vec<u8> = vec![];

    println!("Copy input file to buffer");
    std::io::copy(&mut f_in_disk, &mut f_in_ram).unwrap();

    let mut f_in = Cursor::new(&f_in_ram);
    let mut f_out = Cursor::new(&mut f_out_ram);

    // Best quality for async
    let mut resampler = ResamplerFixedIn::<f64>::new(fs_out as f32 / fs_in as f32, 64, 0.95, 128, Interpolation::Cubic, 1024, channels);

    // Fast and good for doubling 44100 -> 88200 etc
    //let mut resampler = ResamplerFixedIn::<f64>::new(fs_out as f32 / fs_in as f32, 64, 0.95, 4, Interpolation::Nearest, 1024, channels);

    // Fast and good for  44100 -> 48000
    //let mut resampler = ResamplerFixedIn::<f64>::new(
    //    fs_out as f32 / fs_in as f32,
    //    64,
    //    0.95,
    //    160,
    //    Interpolation::Nearest,
    //    1024,
    //    channels,
    //);
//
    let num_chunks = f_in_ram.len() / (8 * channels * 1024);
    let start = Instant::now();
    for _chunk in 0..num_chunks {
        let waves = read_frames(&mut f_in, 1024, 2);
        let waves_out = resampler.resample_chunk(waves);
        write_frames(waves_out, &mut f_out, 2);
    }

    let duration = start.elapsed();

    println!("Resampling took: {:?}", duration);

    let mut f_out_disk = File::create(file_out).unwrap();
    f_out.seek(std::io::SeekFrom::Start(0)).unwrap();
    std::io::copy(&mut f_out, &mut f_out_disk).unwrap();
}
