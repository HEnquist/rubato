use crate::sinc_interpolator::{AvxSample, NeonSample, SseSample};

#[cfg(feature = "fft_resampler")]
use realfft::FftNum;

#[cfg(not(feature = "fft_resampler"))]
use num_traits::{FromPrimitive, Signed};
#[cfg(not(feature = "fft_resampler"))]
use std::fmt::Debug;

#[cfg(not(feature = "fft_resampler"))]
pub trait FftNum: Copy + FromPrimitive + Signed + Sync + Send + Debug + 'static {}

#[cfg(not(feature = "fft_resampler"))]
impl<T> FftNum for T where T: Copy + FromPrimitive + Signed + Sync + Send + Debug + 'static {}

/// The trait governing a single sample.
///
/// There are two types which implements this trait so far:
/// * [f32]
/// * [f64]
pub trait Sample
where
    Self: Copy
        + CoerceFrom<usize>
        + CoerceFrom<f64>
        + CoerceFrom<f32>
        + FftNum
        + std::ops::Mul
        + std::ops::Div
        + std::ops::Add
        + std::ops::Sub
        + std::ops::MulAssign
        + std::ops::RemAssign
        + std::ops::DivAssign
        + std::ops::SubAssign
        + std::ops::AddAssign
        + AvxSample
        + SseSample
        + NeonSample
        + Send,
{
    const PI: Self;

    /// Calculate the sine of `self`.
    fn sin(self) -> Self;

    /// Calculate the cosine of `self`.
    fn cos(self) -> Self;

    /// Coerce `value` into the current type.
    ///
    /// Coercions are governed through the private `CoerceFrom` trait.
    fn coerce<T>(value: T) -> Self
    where
        Self: CoerceFrom<T>,
    {
        Self::coerce_from(value)
    }
}

impl Sample for f32 {
    const PI: Self = std::f32::consts::PI;

    fn sin(self) -> Self {
        f32::sin(self)
    }

    fn cos(self) -> Self {
        f32::cos(self)
    }
}

impl Sample for f64 {
    const PI: Self = std::f64::consts::PI;

    fn sin(self) -> Self {
        f64::sin(self)
    }

    fn cos(self) -> Self {
        f64::cos(self)
    }
}

/// The trait used to coerce a value infallibly from one type to another.
///
/// This is similar to doing `value as T` where `T` is a floating point type.
/// Loss of precision may happen during coercions if the coerced from value
/// doesn't fit fully within the target type.
pub trait CoerceFrom<T> {
    /// Perform a coercion from `value` into the current type.
    fn coerce_from(value: T) -> Self;
}

impl CoerceFrom<usize> for f32 {
    fn coerce_from(value: usize) -> Self {
        value as f32
    }
}

impl CoerceFrom<usize> for f64 {
    fn coerce_from(value: usize) -> Self {
        value as f64
    }
}

impl CoerceFrom<f64> for f32 {
    fn coerce_from(value: f64) -> Self {
        value as f32
    }
}

impl CoerceFrom<f64> for f64 {
    fn coerce_from(value: f64) -> Self {
        value
    }
}

impl CoerceFrom<f32> for f32 {
    fn coerce_from(value: f32) -> Self {
        value
    }
}

impl CoerceFrom<f32> for f64 {
    fn coerce_from(value: f32) -> Self {
        value as f64
    }
}
