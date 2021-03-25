use std::error;
use std::fmt;

/// An identifier for a cpu feature.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub enum CpuFeature {
    /// x86 sse3 cpu feature.
    Sse3,
    /// x86_64 avx cpu feature.
    Avx,
    /// the fma cpu feature.
    Fma,
    /// aarc64 neon cpu feature.
    Neon,
}

impl fmt::Display for CpuFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CpuFeature::Sse3 => {
                write!(f, "sse3")
            }
            CpuFeature::Avx => {
                write!(f, "avx")
            }
            CpuFeature::Fma => {
                write!(f, "fmx")
            }
            CpuFeature::Neon => {
                write!(f, "neon")
            }
        }
    }
}

/// Error raised when trying to use a CPU feature which is not supported.
#[derive(Debug, Clone, Copy)]
pub struct MissingCpuFeatures(pub(crate) &'static [CpuFeature]);

impl fmt::Display for MissingCpuFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Missing cpu features: ")?;

        let mut it = self.0.iter();
        let last = it.next_back();

        for feature in it {
            write!(f, "{}, ", feature)?;
        }

        if let Some(feature) = last {
            write!(f, "{}", feature)?;
        }

        Ok(())
    }
}

impl error::Error for MissingCpuFeatures {}

/// The error type used by `rubato`.
#[derive(Debug)]
pub enum ResampleError {
    /// Error raised when Resample::set_resample_ratio is called with a ratio
    /// that deviates for more than 10% of the original.
    BadRatioUpdate,
    /// Error raised when the number of channels doesn't match expected.
    WrongNumberOfChannels { expected: usize, actual: usize },
    /// Error raised when the number of frames in a single channel doesn't match
    /// the expected.
    WrongNumberOfFrames {
        channel: usize,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for ResampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadRatioUpdate => {
                write!(f, "New resample ratio is too far off from original")
            }
            Self::WrongNumberOfChannels { expected, actual } => {
                write!(
                    f,
                    "Wrong number of channels {} in input, expected {}",
                    actual, expected
                )
            }
            Self::WrongNumberOfFrames {
                channel,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Wrong number of frames {} in input channel {}, expected {}",
                    actual, channel, expected
                )
            }
        }
    }
}

impl error::Error for ResampleError {}

/// A result alias for the error type used by `rubato`.
pub type ResampleResult<T> = ::std::result::Result<T, ResampleError>;
