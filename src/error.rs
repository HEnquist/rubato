use std::error;
use std::fmt;

/// The error type used by `rubato`.
#[derive(Debug)]
pub enum Error {
    /// Error raised when Resample::set_resample_ratio is called with a ratio
    /// that deviates for more than 10% of the original.
    BadResampleRatioUpdate,
    /// Error raised when trying to adjust a synchronous resampler.
    SyncNotAdjustable,
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

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadResampleRatioUpdate => {
                write!(f, "New resample ratio is too far off from original")
            }
            Self::SyncNotAdjustable { .. } => {
                write!(f, "Not possible to adjust a synchronous resampler")
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

impl error::Error for Error {}

/// A result alias for the error type used by `rubato`.
pub type Result<T, E = Error> = ::std::result::Result<T, E>;
