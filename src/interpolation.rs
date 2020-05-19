///// Calculate the scalar produt of an input wave and the selected sinc filter
//pub fn get_sinc_interpolated_f32(
//    wave: &[f32],
//    sincs: &[Vec<f32>],
//    index: usize,
//    subindex: usize,
//) -> f32 {
//    let wave_cut = &wave[index..(index + sincs[subindex].len())];
//    wave_cut
//        .chunks(8)
//        .zip(sincs[subindex].chunks(8))
//        .fold([0.0f32; 8], |acc, (x, y)| {
//            [
//                acc[0] + x[0] * y[0],
//                acc[1] + x[1] * y[1],
//                acc[2] + x[2] * y[2],
//                acc[3] + x[3] * y[3],
//                acc[4] + x[4] * y[4],
//                acc[5] + x[5] * y[5],
//                acc[6] + x[6] * y[6],
//                acc[7] + x[7] * y[7],
//            ]
//        })
//        .iter()
//        .sum()
//}
macro_rules! make_sinc_interp {
    ($t:ty, $name:ident) => {
        /// Calculate the scalar produt of an input wave and the selected sinc filter
        pub fn $name (
            wave: &[$t],
            sincs: &[Vec<$t>],
            index: usize,
            subindex: usize,
        ) -> $t {
            let wave_cut = &wave[index..(index + sincs[subindex].len())];
            wave_cut
                .chunks(8)
                .zip(sincs[subindex].chunks(8))
                .fold([0.0; 8], |acc, (x, y)| {
                    [
                        acc[0] + x[0] * y[0],
                        acc[1] + x[1] * y[1],
                        acc[2] + x[2] * y[2],
                        acc[3] + x[3] * y[3],
                        acc[4] + x[4] * y[4],
                        acc[5] + x[5] * y[5],
                        acc[6] + x[6] * y[6],
                        acc[7] + x[7] * y[7],
                    ]
                })
                .iter()
                .sum()
        }
    }
}

make_sinc_interp!(f32, get_sinc_interpolated_f32);
make_sinc_interp!(f64, get_sinc_interpolated_f64);

/// Calculate the scalar produt of an input wave and the selected sinc filter
//pub fn get_sinc_interpolated_f64(
//    wave: &[f64],
//    sincs: &[Vec<f64>],
//    index: usize,
//    subindex: usize,
//) -> f64 {
//    let wave_cut = &wave[index..(index + sincs[subindex].len())];
//    wave_cut
//        .chunks(8)
//        .zip(sincs[subindex].chunks(8))
//        .fold([0.0f64; 8], |acc, (x, y)| {
//            [
//                acc[0] + x[0] * y[0],
//                acc[1] + x[1] * y[1],
//                acc[2] + x[2] * y[2],
//                acc[3] + x[3] * y[3],
//                acc[4] + x[4] * y[4],
//                acc[5] + x[5] * y[5],
//                acc[6] + x[6] * y[6],
//                acc[7] + x[7] * y[7],
//            ]
//        })
//        .iter()
//        .sum()
//}

/// Perform cubic polynomial interpolation to get value at x.
/// Input points are assumed to be at x = -1, 0, 1, 2
pub fn interp_cubic_f32(x: f32, yvals: &[f32]) -> f32 {
    let a0 = yvals[1];
    let a1 = -(1.0 / 3.0) * yvals[0] - 0.5 * yvals[1] + yvals[2] - (1.0 / 6.0) * yvals[3];
    let a2 = 0.5 * (yvals[0] + yvals[2]) - yvals[1];
    let a3 = 0.5 * (yvals[1] - yvals[2]) + (1.0 / 6.0) * (yvals[3] - yvals[0]);
    a0 + a1 * x + a2 * x.powi(2) + a3 * x.powi(3)
}

/// Perform cubic polynomial interpolation to get value at x.
/// Input points are assumed to be at x = -1, 0, 1, 2
pub fn interp_cubic_f64(x: f64, yvals: &[f64]) -> f64 {
    let a0 = yvals[1];
    let a1 = -(1.0 / 3.0) * yvals[0] - 0.5 * yvals[1] + yvals[2] - (1.0 / 6.0) * yvals[3];
    let a2 = 0.5 * (yvals[0] + yvals[2]) - yvals[1];
    let a3 = 0.5 * (yvals[1] - yvals[2]) + (1.0 / 6.0) * (yvals[3] - yvals[0]);
    a0 + a1 * x + a2 * x.powi(2) + a3 * x.powi(3)
}

/// Linear interpolation between two points at x=0 and x=1
pub fn interp_lin_f32(x: f32, yvals: &[f32]) -> f32 {
    (1.0 - x) * yvals[0] + x * yvals[1]
}

/// Linear interpolation between two points at x=0 and x=1
pub fn interp_lin_f64(x: f64, yvals: &[f64]) -> f64 {
    (1.0 - x) * yvals[0] + x * yvals[1]
}

/// Get the two nearest time points for time t in format (index, subindex)
pub fn get_nearest_times_2(t: f64, factor: isize, points: &mut [(isize, isize)]) {
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

/// Get the four nearest time points for time t in format (index, subindex).
pub fn get_nearest_times_4(t: f64, factor: isize, points: &mut [(isize, isize)]) {
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
    use crate::interpolation::get_nearest_times_4;
    use crate::interpolation::interp_cubic_f32;
    use crate::interpolation::interp_cubic_f64;
    use crate::interpolation::interp_lin_f32;
    use crate::interpolation::interp_lin_f64;
    #[test]
    fn int_cubic() {
        let yvals = vec![0.0f64, 2.0f64, 4.0f64, 6.0f64];
        let interp = interp_cubic_f64(0.5f64, &yvals);
        assert_eq!(interp, 3.0f64);
    }

    #[test]
    fn int_lin_32() {
        let yvals = vec![1.0f32, 5.0f32];
        let interp = interp_lin_f32(0.25f32, &yvals);
        assert_eq!(interp, 2.0f32);
    }

    #[test]
    fn int_cubic_32() {
        let yvals = vec![0.0f32, 2.0f32, 4.0f32, 6.0f32];
        let interp = interp_cubic_f32(0.5f32, &yvals);
        assert_eq!(interp, 3.0f32);
    }

    #[test]
    fn int_lin() {
        let yvals = vec![1.0f64, 5.0f64];
        let interp = interp_lin_f64(0.25f64, &yvals);
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
