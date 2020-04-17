
use num::traits::Float;
//use num::traits::NumCast;
//type Float = f64;

pub struct ResamplerFixedIn<T: Float> {
    nbr_channels: usize,
    chunk_size: usize,
    upsample_factor: usize,
    last_index: f64,
    resample_ratio: f32,
    sinc_len: usize,
    sincs: Vec<Vec<T>>,
    buffer: Vec<Vec<T>>,
}

pub struct ResamplerFixedOut<T: Float> {
    nbr_channels: usize,
    chunk_size: usize,
    upsample_factor: usize,
    last_index: f64,
    sinc_len: usize,
    sincs: Vec<Vec<T>>,
    prev: Vec<Vec<T>>,
}

impl<T: Float> ResamplerFixedIn<T> {
    pub fn new(rate_in: usize, rate_out: usize, sinc_len: usize, f_cutoff: f32, upsample_factor: usize, chunk_size: usize, nbr_channels: usize) -> Self {
        let resample_ratio = rate_out as f32 / rate_in as f32;
        let sinc_cutoff = if rate_out >= rate_in {
            f_cutoff
        }
        else {
            f_cutoff*resample_ratio
        };
        let sincs = make_sincs(sinc_len, upsample_factor, sinc_cutoff);
        let buffer = vec![vec![T::zero();2*chunk_size]; nbr_channels];
        ResamplerFixedIn {
            nbr_channels,
            chunk_size,
            upsample_factor,
            last_index: -(sinc_len as f64),
            resample_ratio,
            sinc_len,
            sincs,
            buffer,
        }

    }



    pub fn resample_chunk_cubic(&mut self, wave_in: Vec<Vec<T>>) -> Vec<Vec<T>> {
//        if chunk == (nchunks-1):
//            curr=wave[chunk*chunksize:]
//            end_idx = len(curr) - sinclen/2
//            end = np.zeros(2*sinclen)
//            wave_long = np.concatenate((prev, curr, end))
//        else:
        let end_idx = self.chunk_size - (self.sinc_len + 1);
        //let curr = wave_in;
        for idx in 0..self.chunk_size {
            for chan in 0..self.nbr_channels {
                self.buffer[chan][idx] = self.buffer[chan][idx+self.chunk_size];
                self.buffer[chan][idx+self.chunk_size] = wave_in[chan][idx];
            }
        }
        let mut idx = self.last_index;
        let t_ratio = 1.0/self.resample_ratio as f64;

        let mut wave_out = vec![Vec::new();self.nbr_channels];
        let mut points = vec![T::zero();4];
        //println!("start loop");
        while idx<end_idx as f64 {
            idx = idx + t_ratio;
            //println!("idx {}", idx);
            let nearest = get_nearest_times_4(idx, self.upsample_factor as isize);
            //println!("nearest {:?}", nearest);
            let frac = T::from((idx*self.upsample_factor as f64).fract() + 1.0).unwrap();
            for chan in 0..self.nbr_channels {
                for p in 0..4 {
                    //println!("get {}",nearest[p].0+self.chunk_size as isize);
                    points[p] = get_sinc_interpolated(&self.buffer[chan], &self.sincs, (nearest[p].0+self.chunk_size as isize) as usize, nearest[p].1 as usize);
                }
                wave_out[chan].push(interp_cubic(frac, &points));
            }
        }
        wave_out
    }
}
            
//                points[p] = get_upsampled(wave_long, sincs, nearest[p]+factor*chunksize)
//            out[n] = interp(frac+1, points)
//            n+=1
//        prev = curr
//        last_idx = idx-chunksize
//def resample_cubic_fixedin(fs, wave, fs_new):
//    #resample original (time, wave) at sample times time_new
//    factor = 256
//    sinclen = 64
//    sincs = make_sincs(sinclen, factor)
//    tdiff = 1/fs
//    tdiff_new = 1/fs_new
//    t_ratio = tdiff_new/tdiff
//    out = np.zeros(int(np.ceil(len(wave)*fs_new/fs)))
//    points = np.zeros(4)
//    start = time.time()
//    chunksize = 1024
//    nchunks = int(np.ceil(len(wave)/chunksize))
//    prev = np.zeros(chunksize)
//    last_idx = -(sinclen/2)
//    n = 0
//    for chunk in range(nchunks):
//        if chunk == (nchunks-1):
//            curr=wave[chunk*chunksize:]
//            end_idx = len(curr) - sinclen/2
//            end = np.zeros(2*sinclen)
//            wave_long = np.concatenate((prev, curr, end))
//        else:
//            end_idx = chunksize - (sinclen + 1)
//            curr=wave[chunk*chunksize:(chunk+1)*chunksize]
//            wave_long = np.concatenate((prev, curr))
//        idx = last_idx
//        while idx<end_idx:
//            
//            idx = idx + t_ratio
//            print("chunk: {}, idx: {}".format(chunk, idx))
//            nearest = get_nearest_times_4(idx, factor)
//            frac = (idx*factor)%1
//            for p in range(4):
//                points[p] = get_upsampled(wave_long, sincs, nearest[p]+factor*chunksize)
//            out[n] = interp(frac+1, points)
//            n+=1
//        prev = curr
//        last_idx = idx-chunksize
//    print("took {}".format(time.time()-start))
//    return out





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

fn make_sincs<T:Float>(npoints: usize, factor: usize, f_cutoff: f32) -> Vec<Vec<T>> {
    let totpoints = (npoints*factor) as isize;
    let mut y = Vec::with_capacity(totpoints as usize);
    let window = blackman_harris::<T>(totpoints as usize);
    for x in 0..totpoints {
        let val = window[x as usize]*window[x as usize]*sinc(T::from(x-totpoints/2).unwrap() * T::from(f_cutoff).unwrap()/T::from(factor).unwrap());
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

fn interp_cubic<T: Float>(x: T, yvals: &Vec<T>) -> T {
    //fit a cubic polynimial to four points (at x=0..3), return interpolated at x
    let mut val = T::zero();
    for (j, y) in yvals.iter().enumerate() {
        val = val + *y*cj(x,j as isize);
    }
    val
}


fn interp_lin<T: Float>(x: T, yvals: &Vec<T>) -> T {
    //linear interpolation
    (T::from(1.0).unwrap()-x)*yvals[0] + x*yvals[1]
}


fn get_sinc_interpolated<T: Float>(wave: &[T], sincs: &Vec<Vec<T>>, index: usize, subindex: usize) -> T {
    // get the sinc-interpolated point at index:subindex
    //let ycut = wave[index..(index+sincs[subindex].len())];
    let mut ynew = T::zero(); 
    for (s, y) in wave.iter().skip(index).take(sincs[subindex].len()).zip(sincs[subindex].iter()) {
        ynew = ynew + *s * *y;
    }
    ynew
}

fn get_nearest_times_2<T:Float>(t: T, factor: isize) -> Vec<(isize, isize)> {
    // Get nearest sample time points, as index:subindex
    let start = (t*T::from(factor).unwrap()).floor().to_isize().unwrap();
    let times_ups = vec![(start/factor, start%factor), ((start+1)/factor, (start+1)%factor)];
    times_ups
}

fn get_nearest_times_4<T:Float>(t: T, factor: isize) -> Vec<(isize, isize)> {
    // Get nearest sample time points, as index:subindex
    let start = t.floor().to_isize().unwrap();
    let frac = if start >= 0 {
        (t.fract()*T::from(factor).unwrap()).floor().to_isize().unwrap()
    }
    else {
        factor + (t.fract()*T::from(factor).unwrap()).floor().to_isize().unwrap()
    };
    let mut times = Vec::new();
    for sub in -1..3 {
        let mut index = start;
        let mut subindex = frac+sub;
        if subindex < 0 {
            subindex += factor;
            index -=1; 
        }
        else if subindex >= factor {
            subindex -= factor;
            index += 1;
        }
        times.push((index, subindex));
    }
    //let times_ups = vec![((start-1)/factor, (start-1)%factor),
    //                     (start/factor, start%factor), 
    //                     ((start+1)/factor, (start+1)%factor),
    //                     ((start+2)/factor, (start+2)%factor)];
    //times_ups
    times
}

fn get_nearest_time<T:Float>(t: T, factor: isize) -> (isize, isize) {
    // Get nearest sample time points, as index:subindex
    let point = (t*T::from(factor).unwrap()).round().to_isize().unwrap();
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
    use crate::ResamplerFixedIn;

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
        let interp = interp_cubic(1.5f64, &yvals);
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
    fn get_nearest_4_neg() {
        let t = -5.9f64;
        let times = get_nearest_times_4(t, 8);
        assert_eq!(times[0], (-7, 7));
        assert_eq!(times[1], (-6, 0));
        assert_eq!(times[2], (-6, 1));
        assert_eq!(times[3], (-6, 2));
    }

    #[test]
    fn get_nearest_single() {
        let t = 5.5f64;
        let time = get_nearest_time(t, 8);
        assert_eq!(time, (5, 4));
    }

    #[test]
    fn make_resampler_fi() {
        let resampler = ResamplerFixedIn::<f64>::new(10000, 12000, 64, 0.95, 16, 1000, 2);
        let waves = vec![vec![0.0f64; 1024]; 2];
        let out = resampler.resample_chunk_cubic(waves);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1199);
    }
}
