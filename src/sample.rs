use crate::{AvxSample, NeonSample, SseSample};

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
        + num_traits::Float
        + realfft::FftNum
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
        + NeonSample,
    num_complex::Complex<Self>: for<'a> std::ops::MulAssign<&'a num_complex::Complex<Self>>,
{
    /// The midpoint of the sample.
    const MID: Self;
    const ONE: Self;
    const HALF: Self;
    const SIX: Self;
    const THREE: Self;

    fn coerce<T>(value: T) -> Self
    where
        Self: CoerceFrom<T>,
    {
        Self::coerce_from(value)
    }
}

impl Sample for f32 {
    const MID: f32 = 0.0;
    const ONE: f32 = 1.0;
    const HALF: f32 = 0.5;
    const SIX: f32 = 6.0;
    const THREE: f32 = 3.0;
}

impl Sample for f64 {
    const MID: f64 = 0.0;
    const ONE: f64 = 1.0;
    const HALF: f64 = 0.5;
    const SIX: f64 = 6.0;
    const THREE: f64 = 3.0;
}

/// The trait used to coerce a value infallibly from one type to another.
///
/// This is similar to doing `value as T` where `T` is a floating point type.
/// So some loss of precision can happen.
pub trait CoerceFrom<T> {
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
        value as f64
    }
}
