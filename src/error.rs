use std::error;
use std::fmt;

/// An identifier for a cpu feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuFeature {
    /// x86 sse3 cpu feature.
    #[cfg(target_arch = "x86_64")]
    Sse3,
    /// x86_64 avx cpu feature.
    #[cfg(target_arch = "x86_64")]
    Avx,
    /// the fma cpu feature.
    #[cfg(target_arch = "x86_64")]
    Fma,
    /// aarc64 neon cpu feature.
    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    Neon,
}

impl CpuFeature {
    /// Test if the given CPU feature is detected.
    pub fn is_detected(&self) -> bool {
        match *self {
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Sse3 => {
                is_x86_feature_detected!("sse3")
            }
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Avx => {
                is_x86_feature_detected!("avx")
            }
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Fma => {
                is_x86_feature_detected!("fma")
            }
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            CpuFeature::Neon => {
                is_aarch64_feature_detected!("neon")
            }
        }
    }
}

impl fmt::Display for CpuFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Sse3 => {
                write!(f, "sse3")
            }
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Avx => {
                write!(f, "avx")
            }
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Fma => {
                write!(f, "fmx")
            }
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            CpuFeature::Neon => {
                write!(f, "neon")
            }
        }
    }
}

/// Error raised when trying to use a CPU feature which is not supported.
#[derive(Debug, Clone, Copy)]
pub struct MissingCpuFeature(pub(crate) CpuFeature);

impl fmt::Display for MissingCpuFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Missing CPU feature `{}`", self.0)
    }
}

impl error::Error for MissingCpuFeature {}

/// The error type used by `rubato`.
#[derive(Debug)]
pub enum ResampleError {
    /// Error raised when Resample::set_resample_ratio is called with a ratio
    /// that deviates for more than 10% of the original.
    BadRatioUpdate,
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

impl fmt::Display for ResampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadRatioUpdate => {
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

impl error::Error for ResampleError {}

/// A result alias for the error type used by `rubato`.
pub type ResampleResult<T> = ::std::result::Result<T, ResampleError>;
