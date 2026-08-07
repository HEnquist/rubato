//! A minimal wav resampling command line tool.
//!
//! Reads a wav file, resamples it to a new sample rate with the FFT resampler,
//! and writes the result to a new wav file. The sample format of the output is
//! chosen freely, independent of the format of the input file.
//!
//! Compared to the `process_*` examples this one is deliberately small. The
//! [waveadapter](https://crates.io/crates/waveadapter) crate handles the wav
//! files, and [Resampler::process_all] resamples the whole clip in a single
//! call, taking care of the chunk loop and trimming the resampler delay.
//!
//! Run it with:
//! ```sh
//! cargo run --release --example resample_wav -- input.wav output.wav 48000
//! cargo run --release --example resample_wav -- input.wav output.wav 96000 --format I24_3 --chunk 2048
//! cargo run --release --example resample_wav -- --help
//! ```

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::time::Instant;

use audioadapter::stats::AdapterStats;
use audioadapter::{Adapter, AdapterMut};
use clap::{Parser, ValueEnum};
use rubato::{Fft, FixedSync, Resampler, WindowFunction};
use waveadapter::{SampleFormat, WavReader, WavSpec, WavWriter};

/// Resample a wav file with the rubato FFT resampler.
#[derive(Parser)]
#[command(version)]
struct Options {
    /// Wav file to read.
    input: String,

    /// Wav file to write.
    output: String,

    /// Sample rate of the output file, in Hz.
    output_rate: usize,

    /// Sample format of the output file [default: same as the input file]
    #[arg(short, long, value_enum, ignore_case = true)]
    format: Option<Format>,

    /// Gain in dB to apply to the resampled audio. Use a small negative value
    /// to add headroom, since resampling can overshoot the peak level of the
    /// input and clip in the integer output formats.
    #[arg(short, long, default_value_t = 0.0, allow_negative_numbers = true)]
    gain: f64,

    /// Resampler chunk size in frames. A smaller value gives a lower delay, at
    /// the cost of a lower cutoff frequency of the anti-aliasing filter.
    #[arg(short, long, default_value_t = 1024)]
    chunk: usize,

    /// Anti-aliasing window function.
    #[arg(short, long, value_enum, ignore_case = true, default_value_t = Window::BlackmanHarris2)]
    window: Window,
}

/// The sample formats that waveadapter can write.
///
/// The names are spelled like the [SampleFormat] variants they map to, so that
/// the values this tool accepts match the ones it prints.
#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Format {
    /// Unsigned 8 bit integer.
    #[value(name = "U8")]
    U8,
    /// Signed 16 bit integer.
    #[value(name = "I16")]
    I16,
    /// Signed 24 bit integer, packed in 3 bytes.
    #[value(name = "I24_3")]
    I24_3,
    /// Signed 24 bit integer, left justified in 4 bytes.
    #[value(name = "I24_4")]
    I24_4,
    /// Signed 32 bit integer.
    #[value(name = "I32")]
    I32,
    /// 32 bit float.
    #[value(name = "F32")]
    F32,
    /// 64 bit float.
    #[value(name = "F64")]
    F64,
}

impl From<Format> for SampleFormat {
    fn from(format: Format) -> Self {
        match format {
            Format::U8 => SampleFormat::U8,
            Format::I16 => SampleFormat::I16,
            Format::I24_3 => SampleFormat::I24_3,
            Format::I24_4 => SampleFormat::I24_4,
            Format::I32 => SampleFormat::I32,
            Format::F32 => SampleFormat::F32,
            Format::F64 => SampleFormat::F64,
        }
    }
}

/// The anti-aliasing window functions the FFT resampler accepts.
///
/// Spelled like the [WindowFunction] variants they map to, for the same reason
/// as [Format].
#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Window {
    #[value(name = "Blackman")]
    Blackman,
    #[value(name = "Blackman2")]
    Blackman2,
    #[value(name = "BlackmanHarris")]
    BlackmanHarris,
    #[value(name = "BlackmanHarris2")]
    BlackmanHarris2,
    #[value(name = "Hann")]
    Hann,
    #[value(name = "Hann2")]
    Hann2,
}

impl From<Window> for WindowFunction {
    fn from(window: Window) -> Self {
        match window {
            Window::Blackman => WindowFunction::Blackman,
            Window::Blackman2 => WindowFunction::Blackman2,
            Window::BlackmanHarris => WindowFunction::BlackmanHarris,
            Window::BlackmanHarris2 => WindowFunction::BlackmanHarris2,
            Window::Hann => WindowFunction::Hann,
            Window::Hann2 => WindowFunction::Hann2,
        }
    }
}

fn run(opts: Options) -> Result<(), Box<dyn std::error::Error>> {
    // Read the whole input file into an interleaved buffer of f64 samples.
    // The reader converts from whatever format the file stores.
    let mut reader = WavReader::new(BufReader::new(File::open(&opts.input)?))?;
    let channels = reader.channels();
    let rate_in = reader.sample_rate();
    // A format the float path cannot decode, A-law for example, cannot be
    // resampled here. Say so up front instead of failing inside the read.
    let format_in = reader.sample_format().ok_or_else(|| {
        format!(
            "the input file cannot be decoded, format code 0x{:04X} with {} bits per sample",
            reader.params().format_code,
            reader.params().bits_per_sample
        )
    })?;
    println!(
        "Input:  {}, {} ch, {} Hz, {:?}, {} frames",
        opts.input,
        channels,
        rate_in,
        format_in,
        reader.frames()
    );
    let input = reader.read_all_to_float::<f64>()?;

    // Write the same sample format as the input file unless told otherwise.
    let format_out = opts.format.map(SampleFormat::from).unwrap_or(format_in);

    // One sub chunk per chunk, so each chunk is a single FFT block. The requested
    // chunk size is only a starting point: it is rounded up to a block size that is
    // valid for the sample rate pair, so 1024 frames becomes 1029 for 44.1k to 48k.
    let window = WindowFunction::from(opts.window);
    let mut resampler = Fft::<f64>::new_custom(
        rate_in,
        opts.output_rate,
        opts.chunk,
        1,
        channels,
        window,
        FixedSync::Both,
    )?;

    // Report the block sizes the resampler settled on, not the requested chunk size.
    // The cutoff is relative to the input Nyquist frequency, so scale it by half
    // the input rate to report it in Hz.
    println!(
        "Config: chunks of {} -> {} frames, {:?} window, cutoff {:.0} Hz",
        resampler.fft_size_in(),
        resampler.fft_size_out(),
        window,
        resampler.cutoff() as f64 * rate_in as f64 / 2.0
    );

    // Resample the entire clip in one call. This runs the chunk loop, trims the
    // startup delay, and returns a buffer holding exactly the resampled frames.
    let start = Instant::now();
    let mut output = resampler.process_all(&input, input.frames(), None)?;
    println!(
        "Resampled {} frames to {} frames in {:?}",
        input.frames(),
        output.frames(),
        start.elapsed()
    );

    // Scale the resampled audio. Doing this after resampling is what matters
    // for clipping, since the resampled peak can sit above the input peak.
    if opts.gain != 0.0 {
        let scale = 10.0f64.powf(opts.gain / 20.0);
        for chan in 0..output.channels() {
            for frame in 0..output.frames() {
                let value = output.read_sample(chan, frame).unwrap() * scale;
                output.write_sample(chan, frame, &value);
            }
        }
    }

    // Report the peak, to make it easy to pick a gain that avoids clipping.
    let peak =
        (0..output.channels()).fold(0.0f64, |peak, chan| peak.max(output.channel_peak(chan)));
    println!(
        "Peak level after gain: {:.2} dBFS",
        20.0 * peak.max(1e-12).log10()
    );

    let spec = WavSpec::new(channels, opts.output_rate, format_out);
    let mut writer = WavWriter::new(BufWriter::new(File::create(&opts.output)?), spec)?;
    let clipped = writer.write_float_buffer(&output)?;
    writer.finalize()?;
    println!(
        "Output: {}, {} ch, {} Hz, {:?}, {} frames, {} clipped samples",
        opts.output,
        channels,
        opts.output_rate,
        format_out,
        output.frames(),
        clipped
    );
    Ok(())
}

fn main() {
    if let Err(err) = run(Options::parse()) {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}
