use std::error;
use std::fmt;

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
