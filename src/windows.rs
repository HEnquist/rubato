use num::traits::Float;

/// Different window functions that can be used to window the sinc function.
#[derive(Debug)]
pub enum WindowFunction {
    /// Blackman. Intermediate rolloff and intermediate attenuation.
    Blackman,
    /// Squared Blackman. Slower rolloff but better attenuation than Blackman.
    Blackman2,
    /// Blackman-Harris. Slow rolloff but good attenuation.
    BlackmanHarris,
    /// Squared Blackman-Harris. Slower rolloff but better attenuation than Blackman-Harris.
    BlackmanHarris2,
    /// Hann, fast rolloff but not very high attenuation
    Hann,
    /// Squared Hann, slower rolloff and higher attenuation than simple Hann
    Hann2,
}

/// Helper function. Standard Blackman-Harris window
pub fn blackman_harris<T: Float>(npoints: usize) -> Vec<T> {
    trace!("Making a BlackmanHarris windows with {} points", npoints);
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
    trace!("Making a Blackman windows with {} points", npoints);
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

/// Standard Hann window
pub fn hann<T: Float>(npoints: usize) -> Vec<T> {
    trace!("Making a Hann windows with {} points", npoints);
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

/// Make the selected window function
pub fn make_window<T: Float>(npoints: usize, windowfunc: WindowFunction) -> Vec<T> {
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

#[cfg(test)]
mod tests {
    use crate::windows::blackman;
    use crate::windows::blackman_harris;
    use crate::windows::hann;
    use crate::windows::make_window;
    use crate::windows::WindowFunction;

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
