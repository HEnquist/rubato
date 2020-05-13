use num::traits::Float;

/// Perform cubic polynomial interpolation to get value at x.
/// Input points are assumed to be at x = -1, 0, 1, 2
pub fn interp_cubic<T: Float>(x: T, yvals: &[T]) -> T {
    let a0 = yvals[1];
    let a1 = -T::from(1.0 / 3.0).unwrap() * yvals[0] - T::from(0.5).unwrap() * yvals[1] + yvals[2]
        - T::from(1.0 / 6.0).unwrap() * yvals[3];
    let a2 = T::from(1.0 / 2.0).unwrap() * (yvals[0] + yvals[2]) - yvals[1];
    let a3 = T::from(1.0 / 2.0).unwrap() * (yvals[1] - yvals[2])
        + T::from(1.0 / 6.0).unwrap() * (yvals[3] - yvals[0]);
    a0 + a1 * x + a2 * x.powi(2) + a3 * x.powi(3)
}

/// Linear interpolation between two points at x=0 and x=1
pub fn interp_lin<T: Float>(x: T, yvals: &[T]) -> T {
    (T::one() - x) * yvals[0] + x * yvals[1]
}

/// Get the two nearest time points for time t in format (index, subindex)
pub fn get_nearest_times_2<T: Float>(t: T, factor: isize, points: &mut [(isize, isize)]) {
    let mut index = t.floor().to_isize().unwrap();
    let mut subindex = ((t - t.floor()) * T::from(factor).unwrap())
        .floor()
        .to_isize()
        .unwrap();
    points[0] = (index, subindex);
    subindex += 1;
    if subindex >= factor {
        subindex -= factor;
        index += 1;
    }
    points[1] = (index, subindex);
}

/// Get the four nearest time points for time t in format (index, subindex).
pub fn get_nearest_times_4<T: Float>(t: T, factor: isize, points: &mut [(isize, isize)]) {
    let start = t.floor().to_isize().unwrap();
    let frac = ((t - t.floor()) * T::from(factor).unwrap())
        .floor()
        .to_isize()
        .unwrap();
    let mut index;
    let mut subindex;
    for (idx, sub) in (-1..3).enumerate() {
        index = start;
        subindex = frac + sub;
        if subindex < 0 {
            subindex += factor;
            index -= 1;
        } else if subindex >= factor {
            subindex -= factor;
            index += 1;
        }
        points[idx] = (index, subindex);
    }
}

/// Get the nearest time point for time t in format (index, subindex).
pub fn get_nearest_time<T: Float>(t: T, factor: isize) -> (isize, isize) {
    let mut index = t.floor().to_isize().unwrap();
    let mut subindex = ((t - t.floor()) * T::from(factor).unwrap())
        .round()
        .to_isize()
        .unwrap();
    if subindex >= factor {
        subindex -= factor;
        index += 1;
    }
    (index, subindex)
}

#[cfg(test)]
mod tests {
    use crate::interpolation::get_nearest_time;
    use crate::interpolation::get_nearest_times_2;
    use crate::interpolation::get_nearest_times_4;
    use crate::interpolation::interp_cubic;
    use crate::interpolation::interp_lin;
    #[test]
    fn int_cubic() {
        let yvals = vec![0.0f64, 2.0f64, 4.0f64, 6.0f64];
        let interp = interp_cubic(0.5f64, &yvals);
        assert_eq!(interp, 3.0f64);
    }

    #[test]
    fn int_lin() {
        let yvals = vec![1.0f64, 5.0f64];
        let interp = interp_lin(0.25f64, &yvals);
        assert_eq!(interp, 2.0f64);
    }

    #[test]
    fn get_nearest_2() {
        let t = 5.9f64;
        let mut times = vec![(0isize, 0isize); 2];
        get_nearest_times_2(t, 8, &mut times);
        assert_eq!(times[0], (5, 7));
        assert_eq!(times[1], (6, 0));
    }

    #[test]
    fn get_nearest_4() {
        let t = 5.9f64;
        let mut times = vec![(0isize, 0isize); 4];
        get_nearest_times_4(t, 8, &mut times);
        assert_eq!(times[0], (5, 6));
        assert_eq!(times[1], (5, 7));
        assert_eq!(times[2], (6, 0));
        assert_eq!(times[3], (6, 1));
    }

    #[test]
    fn get_nearest_4_neg() {
        let t = -5.999f64;
        let mut times = vec![(0isize, 0isize); 4];
        get_nearest_times_4(t, 8, &mut times);
        assert_eq!(times[0], (-7, 7));
        assert_eq!(times[1], (-6, 0));
        assert_eq!(times[2], (-6, 1));
        assert_eq!(times[3], (-6, 2));
    }

    #[test]
    fn get_nearest_4_zero() {
        let t = -0.00001f64;
        let mut times = vec![(0isize, 0isize); 4];
        get_nearest_times_4(t, 8, &mut times);
        assert_eq!(times[0], (-1, 6));
        assert_eq!(times[1], (-1, 7));
        assert_eq!(times[2], (0, 0));
        assert_eq!(times[3], (0, 1));
    }

    #[test]
    fn get_nearest_single() {
        let t = 5.5f64;
        let time = get_nearest_time(t, 8);
        assert_eq!(time, (5, 4));
    }
}
