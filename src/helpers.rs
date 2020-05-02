use crate::WindowFunction;
use num::traits::Float;

/// Calculate the scalar produt of an input wave and the selected sinc filter
pub fn get_sinc_interpolated<T: Float>(
    wave: &[T],
    sincs: &[Vec<T>],
    index: usize,
    subindex: usize,
) -> T {
    wave.iter()
        .skip(index)
        .take(sincs[subindex].len())
        .zip(sincs[subindex].iter())
        .fold(T::zero(), |acc, (x, y)| acc.add(*x * *y))
}

/// Helper function. Standard Blackman-Harris window
pub fn blackman_harris<T: Float>(npoints: usize) -> Vec<T> {
    let mut window = vec![T::zero(); npoints];
    let pi2 = T::from(2.0 * std::f64::consts::PI).unwrap();
    let pi4 = T::from(4.0 * std::f64::consts::PI).unwrap();
    let pi6 = T::from(6.0 * std::f64::consts::PI).unwrap();
    let np_f = T::from(npoints).unwrap();
    let a = T::from(0.35875).unwrap();
    let b = T::from(0.48829).unwrap();
    let c = T::from(0.14128).unwrap();
    let d = T::from(0.01168).unwrap();
    for (x, item) in window.iter_mut().enumerate() {
        let x_float = T::from(x).unwrap();
        *item = a - b * (pi2 * x_float / np_f).cos() + c * (pi4 * x_float / np_f).cos()
            - d * (pi6 * x_float / np_f).cos();
    }
    window
}

/// Helper function. Standard Blackman window
pub fn blackman<T: Float>(npoints: usize) -> Vec<T> {
    let mut window = vec![T::zero(); npoints];
    let pi2 = T::from(2.0 * std::f64::consts::PI).unwrap();
    let pi4 = T::from(4.0 * std::f64::consts::PI).unwrap();
    let np_f = T::from(npoints).unwrap();
    let a = T::from(0.42).unwrap();
    let b = T::from(0.5).unwrap();
    let c = T::from(0.08).unwrap();
    for (x, item) in window.iter_mut().enumerate() {
        let x_float = T::from(x).unwrap();
        *item = a - b * (pi2 * x_float / np_f).cos() + c * (pi4 * x_float / np_f).cos();
    }
    window
}

/// Helper function. Standard Hann window
pub fn hann<T: Float>(npoints: usize) -> Vec<T> {
    let mut window = vec![T::zero(); npoints];
    let pi2 = T::from(2.0 * std::f64::consts::PI).unwrap();
    let np_f = T::from(npoints).unwrap();
    let a = T::from(0.5).unwrap();
    for (x, item) in window.iter_mut().enumerate() {
        let x_float = T::from(x).unwrap();
        *item = a - a * (pi2 * x_float / np_f).cos();
    }
    window
}

/// Helper function: sinc(x) = sin(pi*x)/(pi*x)
pub fn sinc<T: Float>(value: T) -> T {
    let pi = T::from(std::f64::consts::PI).unwrap();
    if value == T::zero() {
        T::from(1.0).unwrap()
    } else {
        (T::from(value).unwrap() * pi).sin() / (T::from(value).unwrap() * pi)
    }
}

fn make_window<T: Float>(npoints: usize, windowfunc: WindowFunction) -> Vec<T> {
    let mut window = match windowfunc {
        WindowFunction::BlackmanHarris | WindowFunction::BlackmanHarris2 => {
            blackman_harris::<T>(npoints)
        }
        WindowFunction::Blackman | WindowFunction::Blackman2 => blackman::<T>(npoints),
        WindowFunction::Hann | WindowFunction::Hann2 => hann::<T>(npoints),
    };
    match windowfunc {
        WindowFunction::Blackman2 | WindowFunction::BlackmanHarris2 | WindowFunction::Hann2 => {
            window.iter_mut().for_each(|y| *y = *y * *y);
        }
        _ => {}
    };
    window
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
    println!("sum {:?}", sum.to_f64());
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
    use crate::helpers::blackman;
    use crate::helpers::blackman_harris;
    use crate::helpers::hann;
    use crate::helpers::make_sincs;
    use crate::helpers::make_window;
    use crate::WindowFunction;

    #[test]
    fn sincs() {
        let sincs = make_sincs::<f64>(32, 8, 1.0, WindowFunction::Blackman);
        println!("{:?}", sincs);
        assert!((sincs[7][16] - 1.0).abs() < 0.001);
        let sum: f64 = sincs.iter().map(|v| v.iter().sum::<f64>()).sum();
        assert!((sum - 8.0).abs() < 0.00001);
    }

    #[test]
    fn test_blackman_harris() {
        let wnd = blackman_harris::<f64>(16);
        assert!((wnd[8] - 1.0).abs() < 0.000001);
        assert!(wnd[0] < 0.001);
        assert!(wnd[15] < 0.1);
    }

    #[test]
    fn test_blackman() {
        let wnd = blackman::<f64>(16);
        assert!((wnd[8] - 1.0).abs() < 0.000001);
        assert!(wnd[0] < 0.000001);
        assert!(wnd[15] < 0.1);
    }

    #[test]
    fn test_blackman2() {
        let wnd = make_window::<f64>(16, WindowFunction::Blackman);
        let wnd2 = make_window::<f64>(16, WindowFunction::Blackman2);
        assert!((wnd[1] * wnd[1] - wnd2[1]).abs() < 0.000001);
        assert!((wnd[4] * wnd[4] - wnd2[4]).abs() < 0.000001);
        assert!((wnd[7] * wnd[7] - wnd2[7]).abs() < 0.000001);
        assert!(wnd2[1] > 0.000001);
        assert!(wnd2[4] > 0.000001);
        assert!(wnd2[7] > 0.000001);
    }

    #[test]
    fn test_hann() {
        let wnd = hann::<f64>(16);
        assert!((wnd[8] - 1.0).abs() < 0.000001);
        assert!(wnd[0] < 0.000001);
        assert!(wnd[15] < 0.1);
    }
}
