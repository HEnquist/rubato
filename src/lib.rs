
use num::traits::Float;
//use num::traits::NumCast;
//type Float = f64;

pub struct ResamplerFixedIn<T> {
    upsample_factor: usize,
    last_index: T,
    sinc_len: usize,
    sincs: Vec<Vec<T>>,
    prev: Vec<Vec<T>>,
}

pub struct ResamplerFixedOut<T> {
    upsample_factor: usize,
    last_index: T,
    sinc_len: usize,
    sincs: Vec<Vec<T>>,
    prev: Vec<Vec<T>>,
}

fn blackman_harris<T: Float>(npoints: usize) -> Vec<T> {
    // blackman-harris window
    let mut window = vec![T::zero(); npoints];
    let pi2 = T::from(2.0*std::f64::consts::PI).unwrap();
    let pi4 = T::from(4.0*std::f64::consts::PI).unwrap();
    let pi6 = T::from(6.0*std::f64::consts::PI).unwrap();
    let np_f = T::from(npoints).unwrap();
    let a = T::from(0.35875).unwrap();
    let b = T::from(0.48829).unwrap();
    let c = T::from(0.14128).unwrap();
    let d = T::from(0.01168).unwrap();
    for x in 0..npoints {
        let x_float = T::from(x).unwrap();
        window[x] = a - b*(pi2*x_float/np_f).cos() + c*(pi4*x_float/np_f).cos() - d*(pi6*x_float/np_f).cos();
    }
    window
}

fn sinc<T:Float>(value: T) -> T {
    let pi = T::from(std::f64::consts::PI).unwrap();
    if value == T::zero() {
        T::from(1.0).unwrap()
    }
    else {
        (T::from(value).unwrap() * pi).sin()/(T::from(value).unwrap() * pi)
    }
}

fn make_sincs<T:Float>(npoints: usize, factor: usize, f_cutoff: T) -> Vec<Vec<T>> {
    let totpoints = (npoints*factor) as isize;
    let mut y = Vec::with_capacity(totpoints as usize);
    let window = blackman_harris::<T>(totpoints as usize);
    for x in 0..totpoints {
        let val = window[x as usize]*window[x as usize]*sinc(T::from(x-totpoints/2).unwrap() * f_cutoff/T::from(factor).unwrap());
        y.push(val);
    }
    //println!("{:?}",y);
    let mut sincs = vec![vec![T::zero();npoints]; factor];
    for p in 0..npoints {
        for n in 0..factor {
            sincs[factor-n-1][p]=y[factor*p+n];
        }
    }
    sincs
}

fn cj<T: Float>(x:T, j: isize) -> T {
    //helper for Lagrange polynomials
    //xvals = [m for m in range(4) if m!=j]
    let mut out = T::from(1.0).unwrap();
    let xj = T::from(j).unwrap();
    for xm in (0..4).filter(|xx| *xx != j).map(|xx| T::from(xx).unwrap()) {
        out = out*(x-xm)/(xj-xm);
    }
    out
}

fn interp_cubic<T: Float>(x: T, yvals: Vec<T>) -> T {
    //fit a cubic polynimial to four points (at x=0..3), return interpolated at x
    let mut val = T::zero();
    for (j, y) in yvals.iter().enumerate() {
        val = val + *y*cj(x,j as isize);
    }
    val
}


fn interp_lin<T: Float>(x: T, yvals: Vec<T>) -> T {
    //linear interpolation
    (T::from(1.0).unwrap()-x)*yvals[0] + x*yvals[1]
}


fn get_sinc_interpolated<T: Float>(wave: &[T], sincs: Vec<Vec<T>>, index: usize, subindex: usize) -> T {
    // get the sinc-interpolated point at index:subindex
    //let ycut = wave[index..(index+sincs[subindex].len())];
    let mut ynew = T::zero(); 
    for (s, y) in wave.iter().skip(index).take(sincs[subindex].len()).zip(sincs[subindex].iter()) {
        ynew = ynew + *s * *y;
    }
    ynew
}

fn get_nearest_times_2<T:Float>(t: T, factor: usize) -> Vec<(usize, usize)> {
    // Get nearest sample time points, as index:subindex
    let start = (t*T::from(factor).unwrap()).floor().to_usize().unwrap();
    let times_ups = vec![(start/factor, start%factor), ((start+1)/factor, (start+1)%factor)];
    times_ups
}

fn get_nearest_times_4<T:Float>(t: T, factor: usize) -> Vec<(usize, usize)> {
    // Get nearest sample time points, as index:subindex
    let start = (t*T::from(factor).unwrap()).floor().to_usize().unwrap();
    let times_ups = vec![((start-1)/factor, (start-1)%factor),
                         (start/factor, start%factor), 
                         ((start+1)/factor, (start+1)%factor),
                         ((start+2)/factor, (start+2)%factor)];
    times_ups
}

fn get_nearest_time<T:Float>(t: T, factor: usize) -> (usize, usize) {
    // Get nearest sample time points, as index:subindex
    let point = (t*T::from(factor).unwrap()).round().to_usize().unwrap();
    (point/factor, point%factor)
}
//def get_nearest_time(t, factor):
//    # Get nearest sample time points, in upsampled sample number
//    t = np.round(t*factor)
//    return t


//impl<T> ResamplerFixedOut<T> {
//    fn make_sincs
//}

#[cfg(test)]
mod tests {
    use crate::make_sincs;
    use crate::blackman_harris;
    use crate::interp_cubic;
    use crate::interp_lin;
    use crate::get_nearest_times_2;
    use crate::get_nearest_times_4;
    use crate::get_nearest_time;

    #[test]
    fn sincs() {
        let sincs = make_sincs::<f64>(16,4,1.0);
        println!("{:?}", sincs);
        assert_eq!(sincs[3][8], 1.0);
    }

    #[test]
    fn blackman() {
        let wnd = blackman_harris::<f64>(16);
        assert_eq!(wnd[8], 1.0);
        assert!(wnd[0]<0.001);
        assert!(wnd[15]<0.01);
    }

    #[test]
    fn int_cubic() {
        let yvals = vec![0.0f64, 2.0f64, 4.0f64, 6.0f64];
        let interp = interp_cubic(1.5f64, yvals);
        assert_eq!(interp, 3.0f64);
    }


    #[test]
    fn int_lin() {
        let yvals = vec![1.0f64, 5.0f64];
        let interp = interp_lin(0.25f64, yvals);
        assert_eq!(interp, 2.0f64);
    }

    #[test]
    fn get_nearest_2() {
        let t = 5.9f64;
        let times = get_nearest_times_2(t, 8);
        assert_eq!(times[0], (5, 7));
        assert_eq!(times[1], (6, 0));
    }

    #[test]
    fn get_nearest_4() {
        let t = 5.9f64;
        let times = get_nearest_times_4(t, 8);
        assert_eq!(times[0], (5, 6));
        assert_eq!(times[1], (5, 7));
        assert_eq!(times[2], (6, 0));
        assert_eq!(times[3], (6, 1));
    }

    #[test]
    fn get_nearest_single() {
        let t = 5.5f64;
        let time = get_nearest_time(t, 8);
        assert_eq!(time, (5, 4));
    }
}
