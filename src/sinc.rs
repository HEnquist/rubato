use crate::windows::{make_window, WindowFunction};
use crate::Sample;

/// Helper function: sinc(x) = sin(pi*x)/(pi*x).
pub fn sinc<T>(value: T) -> T
where
    T: Sample,
{
    if value == T::zero() {
        T::one()
    } else {
        (value * T::PI).sin() / (value * T::PI)
    }
}

/// Helper function. Make a set of windowed sincs.
pub fn make_sincs<T>(
    npoints: usize,
    factor: usize,
    f_cutoff: f32,
    windowfunc: WindowFunction,
) -> Vec<Vec<T>>
where
    T: Sample,
{
    let totpoints = npoints * factor;
    let mut y = Vec::with_capacity(totpoints);
    let window = make_window::<T>(totpoints, windowfunc);
    let mut sum = T::zero();
    for (x, w) in window.iter().enumerate().take(totpoints) {
        let val = *w
            * sinc(
                (T::coerce(x) - T::coerce(totpoints / 2)) * T::coerce(f_cutoff) / T::coerce(factor),
            );
        sum += val;
        y.push(val);
    }
    sum /= T::coerce(factor);
    debug!(
        "Generate sincs, length: {}, oversampling: {}, normalized by: {:?}",
        npoints, factor, sum
    );
    let mut sincs = vec![vec![T::zero(); npoints]; factor];
    for p in 0..npoints {
        for n in 0..factor {
            sincs[factor - n - 1][p] = y[factor * p + n] / sum;
        }
    }
    sincs
}

#[cfg(test)]
mod tests {
    use crate::sinc::make_sincs;
    use crate::WindowFunction;

    #[test]
    fn sincs() {
        let sincs = make_sincs::<f64>(32, 8, 0.9, WindowFunction::Blackman);
        assert!((sincs[7][16] - 1.0).abs() < 0.2);
        let sum: f64 = sincs.iter().map(|v| v.iter().sum::<f64>()).sum();
        assert!((sum - 8.0).abs() < 0.00001);
    }
}
