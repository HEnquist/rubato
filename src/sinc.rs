use crate::windows::{make_window, WindowFunction};
use num::traits::Float;

/// Helper function: sinc(x) = sin(pi*x)/(pi*x)
pub fn sinc<T: Float>(value: T) -> T {
    let pi = T::from(std::f64::consts::PI).unwrap();
    if value == T::zero() {
        T::from(1.0).unwrap()
    } else {
        (T::from(value).unwrap() * pi).sin() / (T::from(value).unwrap() * pi)
    }
}

/// Helper function. Make a set of windowed sincs.  
pub fn make_sincs<T: Float>(
    npoints: usize,
    factor: usize,
    f_cutoff: f32,
    windowfunc: WindowFunction,
) -> Vec<Vec<T>> {
    let totpoints = (npoints * factor) as isize;
    let mut y = Vec::with_capacity(totpoints as usize);
    let window = make_window::<T>(totpoints as usize, windowfunc);
    let mut sum = T::zero();
    for x in 0..totpoints {
        let val = window[x as usize]
            * sinc(
                T::from(x - totpoints / 2).unwrap() * T::from(f_cutoff).unwrap()
                    / T::from(factor).unwrap(),
            );
        sum = sum + val;
        y.push(val);
    }
    sum = sum / T::from(factor).unwrap();
    debug!(
        "Generate sincs, length: {}, oversampling: {}, normalized by: {:?}",
        npoints,
        factor,
        sum.to_f64()
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
