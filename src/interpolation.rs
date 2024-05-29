/// Get the two nearest time points for time t in format (index, subindex).
pub fn get_nearest_times_2(t: f64, factor: isize, points: &mut [(isize, isize); 2]) {
    let mut index = t.floor() as isize;
    let mut subindex = ((t - t.floor()) * (factor as f64)).floor() as isize;
    points[0] = (index, subindex);
    subindex += 1;
    if subindex >= factor {
        subindex -= factor;
        index += 1;
    }
    points[1] = (index, subindex);
}

/// Get the three nearest time points for time t in format (index, subindex).
pub fn get_nearest_times_3(t: f64, factor: isize, points: &mut [(isize, isize); 3]) {
    let start = t.floor() as isize;
    let frac = ((t - t.floor()) * (factor as f64)).floor() as isize;
    let mut index;
    let mut subindex;
    for (idx, sub) in (0..3).enumerate() {
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

/// Get the four nearest time points for time t in format (index, subindex).
pub fn get_nearest_times_4(t: f64, factor: isize, points: &mut [(isize, isize); 4]) {
    let start = t.floor() as isize;
    let frac = ((t - t.floor()) * (factor as f64)).floor() as isize;
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
pub fn get_nearest_time(t: f64, factor: isize) -> (isize, isize) {
    let mut index = t.floor() as isize;
    let mut subindex = ((t - t.floor()) * (factor as f64)).round() as isize;
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
    use crate::interpolation::get_nearest_times_3;
    use crate::interpolation::get_nearest_times_4;
    use test_log::test;

    #[test]
    fn get_nearest_2() {
        let t = 5.9f64;
        let mut times = [(0isize, 0isize); 2];
        get_nearest_times_2(t, 8, &mut times);
        assert_eq!(times[0], (5, 7));
        assert_eq!(times[1], (6, 0));
    }

    #[test]
    fn get_nearest_4() {
        let t = 5.9f64;
        let mut times = [(0isize, 0isize); 4];
        get_nearest_times_4(t, 8, &mut times);
        assert_eq!(times[0], (5, 6));
        assert_eq!(times[1], (5, 7));
        assert_eq!(times[2], (6, 0));
        assert_eq!(times[3], (6, 1));
    }

    #[test]
    fn get_nearest_3() {
        let t = 5.9f64;
        let mut times = [(0isize, 0isize); 3];
        get_nearest_times_3(t, 8, &mut times);
        assert_eq!(times[0], (5, 7));
        assert_eq!(times[1], (6, 0));
        assert_eq!(times[2], (6, 1));
    }

    #[test]
    fn get_nearest_4_neg() {
        let t = -5.999f64;
        let mut times = [(0isize, 0isize); 4];
        get_nearest_times_4(t, 8, &mut times);
        assert_eq!(times[0], (-7, 7));
        assert_eq!(times[1], (-6, 0));
        assert_eq!(times[2], (-6, 1));
        assert_eq!(times[3], (-6, 2));
    }

    #[test]
    fn get_nearest_4_zero() {
        let t = -0.00001f64;
        let mut times = [(0isize, 0isize); 4];
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
