use std::error;
use std::fmt;

/// The error type used by `rubato`.
pub type Error = Box<dyn error::Error + Send + Sync + 'static>;

/// A result alias for the error type used by `rubato`.
pub type Result<T, E = Error> = ::std::result::Result<T, E>;

/// Custom error returned by resamplers.
#[derive(Debug)]
pub struct ResamplerError {
    desc: Box<str>,
}

impl fmt::Display for ResamplerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.desc)
    }
}

impl error::Error for ResamplerError {
    fn description(&self) -> &str {
        &self.desc
    }
}

impl ResamplerError {
    pub fn new(desc: &str) -> Self {
        ResamplerError { desc: desc.into() }
    }
}
