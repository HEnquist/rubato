CamillaResampler
---
---

CamillaResampler is a library for asynchronous resamplig of audio data. The ratio between output and input sample rates can be any number.

Audio data is processed in chunks. Two implementations are provided, one that accepts a fixed size input chunk and returns a variable length result, and one that instead provides fixed length output chunks if it's provided with enough data.


Fixed input size
---
Create a SincFixedIn:
```
let mut resampler = SincFixedIn::<f64>::new(fs_out as f32 / fs_in as f32, 64, 0.95, 128, Interpolation::Cubic, 1024, 2);
let waves_out = resampler.resample_chunk(waves_in);
```

The arguments for creating a SincFixed in are:
```
SincFixedIn::new(resample_ratio: f32, sinc_len: usize, f_cutoff: f32, upsample_factor: usize, interpolation: Interpolation, chunk_size: usize, nbr_channels: usize)
   
```
- resample_ratio: Ratio, rate_out/rate_in
- sinc_len: Length of the sinc interpolation filter. Higher values can allow a higher cut-off frequency leading to less high frequency roll-off at the expense of higher cpu usage. 64 is a good starting point.
- f_cutoff: Relative cutoff frequency of the sinc interpolation filter (relative to the lowest one of fs_in/2 or fs_out/2). Start at 0.95, and increase if needed. 
- upsample_factor: The number of intermediate points go use for interpolation. Higher values use more memory for storing the sinc filters. Only the points actually needed are calculated dusing processing so a larger number does not directly lead to higher cpu usage. But keeping it down helps in keeping the sincs in the cpu cache. Start at 128.
- interpolation: Interpolation technique. Valid choices are Cubic, Linear and Nearest. See explanation further down.
- chunk_size: Input chunk size in frames.
- nbr_channels: Number of audio channels. 


Interpolation techiques
---
For asynchronous interpolation where the ratio between inut and output sample rates can be any number, it's not possible to pre-calculate all the needed interpolation filters. Instead the have to be computed as needed, which is very expensive in terms of cpu time. Instead it's better to combine the sinc filters with some other interpolation technique. Then sinc filters are used to provide a fixed number of interpolated points between input samples, and then the new value is calculated by interpolation between those points.

Linear interpolation
--

With linear interpolation the new sample value is calculated by linear interpolation between the two nearest points. This requires two intermediate points to be calcuated using sinc interpolation, and te output is a weighted average of these two. This is relatively fast, but needs a large number of intermediate points to push the resampling artefacts below the noise floor.

Cubic interpolation
--
For cubic interpolation, the four nearest intermediate points are calculated using sinc interpolation. Then a cubic polynomial is fitted to these points, and is then used to calculate the new smple value. The computation time as about twice the one for linear interpolation, but it requires much fewer intermediate points for a good result.

Nearest point
--
The Nearest mode doesn't do any interpolation, but simply picks the nearest intermediate point. This is useful when the nearest point is actually the correct one, for example when upsampling by a factor 2, like 48kHz->96kHz. Then setting the umsample_factor to 2, and using Nearest mode, no unneccesary computations are performed and the result is the same as for synchronous resampling. This also works for other ration that can be expressed by a fraction. For 44.1kHz -> 48 kHz, setting upsample_factor to 160 gives the desired result (since 48kHz = 160/147 * 44.1kHz)
