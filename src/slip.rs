use audioadapter::{Adapter, AdapterMut};
use std::fmt;

use crate::asynchro::FixedAsync;
use crate::error::{ResampleError, ResampleResult, ResamplerConstructionError};
use crate::{get_offsets, get_partial_len, update_mask, Indexing};
use crate::{validate_buffers, Adjustable, Resampler, Resizable, Sample};

/// Length in frames of the crossfade that hides each slip.
///
/// During the crossfade the signal is blended with a one-sample-shifted copy of itself, which acts
/// as a mild comb filter (a gentle notch up near Nyquist), so the fade is kept short to keep that
/// coloration brief. Going much shorter (one or two frames) would approach a hard cut and bring back
/// an audible click, while a longer fade would only spread the coloration over more of the signal.
/// A handful of frames is the sweet spot, and the result is insensitive to the exact value, so it is
/// fixed rather than exposed as a parameter. It also sets how densely slips can be packed, and thus
/// the maximum drift the resampler can absorb, which works out to far more than any realistic clock
/// drift.
const CROSSFADE_LEN: usize = 8;

/// A clutch for matching two almost-equal sample rates, slipping a frame when needed.
///
/// [Slip] keeps an audio stream that is clocked at one rate playing out at a slightly different
/// rate by occasionally inserting or dropping a single frame, like a slipping clutch that lets two
/// shafts turn at not quite the same speed. It is meant for compensating small clock differences,
/// such as the drift between a source clock and a sound card whose crystal is a few ppm off.
///
/// # Not a sample rate converter
/// Despite implementing [Resampler], this is **not** a real sample rate converter. It does not
/// interpolate or filter the signal; it just passes samples through and now and then slips one. The
/// further the ratio is from `1.0`, the more often it has to slip, and the audible quality drops
/// accordingly. For anything beyond tracking tiny clock differences, use a proper resampler such as
/// [Async](crate::Async).
///
/// # Trade-offs versus a full async resampler
/// Compared to running a real [Async](crate::Async) resampler to do the same rate matching,
/// slipping has both upsides and downsides.
///
/// Advantages:
/// - **Very low CPU usage.** Most chunks are a plain copy, and a correction is just one short
///   crossfade.
/// - **No delay.** Each chunk is self contained, so there is no filter delay to report or trim.
/// - **No high frequency roll-off.** Nothing is filtered, so the passband is untouched all the way
///   up to Nyquist, unlike the gentle roll-off an anti-aliasing filter introduces.
///
/// Disadvantages:
/// - It is not a true resampler, so each correction is a small local distortion (a brief coloration
///   from the crossfade) rather than a clean, uniform resampling of the whole signal.
/// - It only works for ratios very close to `1.0`, while a real resampler handles any ratio.
///
/// # How it works
/// Most chunks are passed through unchanged. When the accumulated timing error reaches a whole
/// frame, it slips one: an extra frame is inserted when the output rate is higher, or a frame is
/// dropped when it is lower. If the error has built up to several frames, several are slipped in the
/// same chunk. A slip is not a hard cut, which would produce an audible click. Instead a short
/// crossfade blends the signal across each splice, so the only residual artefact is a brief, mild
/// coloration rather than a broadband click. Because the slip is hidden by the crossfade, there is
/// no need to wait for a quiet moment or a zero crossing, and the timing of each correction is fully
/// deterministic.
///
/// Each processed chunk is self contained: it reads the real input frames and writes output that
/// begins at the first input frame and ends at the last one. There is therefore no internal
/// history buffer and no startup delay, and the signal stays continuous across chunk boundaries.
///
/// # Fixed input or output
/// The `fixed` argument works like it does for [Async](crate::Async). With [FixedAsync::Input] the
/// input chunk size is constant and the output size is `chunk_size`, give or take the number of
/// frames slipped in that chunk. With [FixedAsync::Output] the output size is constant and the input
/// size varies instead.
///
/// # Correction rate limit
/// Several frames can be inserted or dropped within a single chunk, as long as their crossfades do
/// not overlap. With the short crossfade used internally this puts the ceiling at about 10% drift,
/// essentially independent of the chunk size and far more than any realistic clock drift. The
/// ratio starts at 1.0 and is adjusted through [Adjustable::set_resample_ratio], which clamps to
/// this range and applies the nearest limit but still returns an error, so a feedback loop runs at
/// the best achievable rate and can tell when it has saturated. The ratio is intended to be adjusted
/// continuously by such a loop (see [Adjustable]) that measures the actual buffer fill and nudges
/// the rate to keep it centered.
///
/// # Example
/// A clock-drift feedback loop feeding a sound card. The card asks for a fixed-size buffer each
/// period ([FixedAsync::Output]) and its clock runs slightly fast, so the input drifts behind the
/// nominal-rate source. A proportional controller steers the ratio from the input buffer fill until
/// the rates match. `Backend` here just mocks the device and source; in a real program it would be
/// your audio callbacks and ring buffers.
/// ```
/// use rubato::{Slip, FixedAsync, Resampler, Adjustable};
/// use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
///
/// // Mock sound card plus source. The card clock runs 100 ppm fast (consumer_ratio = 1.0001).
/// struct Backend {
///     period: usize,       // fixed buffer size the card asks us to fill
///     consumer_ratio: f64, // card clock relative to nominal
///     input_fill: f64,     // input frames buffered, waiting to be consumed
/// }
/// impl Backend {
///     // Block until the card wants the next buffer. While we waited, the source topped up the
///     // input buffer by one card period worth of frames.
///     fn wait_for_device(&mut self) -> Vec<Vec<f64>> {
///         self.input_fill += self.period as f64 / self.consumer_ratio;
///         vec![vec![0.0; self.period]; 2]
///     }
///     // Take `frames` frames out of the input buffer to feed the resampler.
///     fn read_source(&mut self, frames: usize) -> Vec<Vec<f64>> {
///         self.input_fill -= frames as f64;
///         vec![vec![0.0; frames]; 2]
///     }
///     // Hand the filled buffer back to the card to be played.
///     fn play(&mut self, _buffer: Vec<Vec<f64>>) {}
///     // Feedback: how far the input buffer has drifted from its starting fill.
///     fn rate_error(&self) -> f64 {
///         self.input_fill
///     }
/// }
///
/// let mut backend = Backend { period: 512, consumer_ratio: 1.0001, input_fill: 0.0 };
/// let mut slip = Slip::<f64>::new(backend.period, 2, FixedAsync::Output).unwrap();
/// let gain = 1e-5; // proportional controller gain
///
/// for _ in 0..1000 {
///     let mut output_data = backend.wait_for_device();
///     let frames_in = slip.input_frames_next();
///     let input_data = backend.read_source(frames_in);
///
///     let input = SequentialSliceOfVecs::new(&input_data, 2, frames_in).unwrap();
///     let mut output = SequentialSliceOfVecs::new_mut(&mut output_data, 2, backend.period).unwrap();
///     slip.process_into_buffer(&input, &mut output, None).unwrap();
///     backend.play(output_data); // hand the filled buffer to the card
///
///     // A draining input buffer means we are consuming too fast; raise the ratio to take fewer
///     // input frames per output chunk (and vice versa).
///     slip.set_resample_ratio(1.0 - gain * backend.rate_error(), false).unwrap();
/// }
///
/// // The loop converged on the card's actual rate and the input buffer stayed bounded.
/// assert!((slip.resample_ratio() - backend.consumer_ratio).abs() < 1e-3);
/// assert!(backend.rate_error().abs() < 50.0);
/// ```
pub struct Slip<T> {
    nbr_channels: usize,
    chunk_size: usize,
    max_chunk_size: usize,
    max_correction: usize,
    needed_input_size: usize,
    needed_output_size: usize,
    correction: i32,
    drift_acc: f64,
    resample_ratio: f64,
    input_scratch: Vec<T>,
    output_scratch: Vec<T>,
    channel_mask: Vec<bool>,
    fixed: FixedAsync,
}

impl<T> fmt::Debug for Slip<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Slip")
            .field("nbr_channels", &self.nbr_channels)
            .field("chunk_size", &self.chunk_size)
            .field("max_chunk_size", &self.max_chunk_size)
            .field("max_correction", &self.max_correction)
            .field("needed_input_size", &self.needed_input_size)
            .field("needed_output_size", &self.needed_output_size)
            .field("correction", &self.correction)
            .field("drift_acc", &self.drift_acc)
            .field("resample_ratio", &self.resample_ratio)
            .field("channel_mask", &self.channel_mask)
            .field("fixed", &self.fixed)
            .finish()
    }
}

/// Crossfade weights, a smootherstep S-curve going from 0 to 1.
///
/// smootherstep (`6x^5 - 15x^4 + 10x^3`) has zero first and second derivative at both ends, so the
/// fade eases into and out of the surrounding straight-copy regions with no slope kink that would
/// reintroduce a click. The two blended signals differ by only a single sample, so the amplitude
/// complementary (`w` and `1 - w`) weighting preserves the level across the splice. Being a plain
/// polynomial it needs no transcendental, so the whole table is evaluated at compile time.
///
/// See <https://en.wikipedia.org/wiki/Smoothstep#Variations> for the smootherstep polynomial.
const FADE: [f64; CROSSFADE_LEN] = {
    let mut fade = [0.0; CROSSFADE_LEN];
    let mut k = 0;
    while k < CROSSFADE_LEN {
        let x = (k as f64 + 0.5) / CROSSFADE_LEN as f64;
        fade[k] = x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
        k += 1;
    }
    fade
};

/// Largest number of corrections that fit in one chunk without the crossfades overlapping.
///
/// Each correction needs its own crossfade of `crossfade_len` frames plus a one frame gap on
/// either side, so the achievable correction density is one per `crossfade_len + 2` frames.
/// This is what bounds the maximum drift the resampler can absorb, independent of the chunk size.
fn max_corrections(chunk_size: usize, crossfade_len: usize) -> usize {
    chunk_size.saturating_sub(1) / (crossfade_len + 2)
}

/// The timing error, in frames, that one chunk adds to the accumulator.
///
/// It is expressed against the fixed side so the realized ratio is exact in both modes. With a
/// fixed input of `chunk_size` frames the output should be `ratio * chunk_size`, an excess of
/// `(ratio - 1) * chunk_size`. With a fixed output of `chunk_size` frames the input should be
/// `chunk_size / ratio`, so the output runs ahead by `(1 - 1 / ratio) * chunk_size`.
fn drift_per_chunk(ratio: f64, chunk_size: usize, fixed: FixedAsync) -> f64 {
    match fixed {
        FixedAsync::Input => (ratio - 1.0) * chunk_size as f64,
        FixedAsync::Output => (1.0 - 1.0 / ratio) * chunk_size as f64,
    }
}

/// The range of resample ratios the resampler can sustain for a given chunk size and correction
/// capacity. A ratio outside this range drifts faster than the corrections can keep up with.
fn ratio_range(max_correction: usize, chunk_size: usize, fixed: FixedAsync) -> (f64, f64) {
    let f = max_correction as f64 / chunk_size as f64;
    match fixed {
        // (ratio - 1) * chunk_size must stay within +/- max_correction.
        FixedAsync::Input => (1.0 - f, 1.0 + f),
        // (1 - 1 / ratio) * chunk_size must stay within +/- max_correction.
        FixedAsync::Output => (1.0 / (1.0 + f), 1.0 / (1.0 - f)),
    }
}

/// Copy `input` to `output`, inserting or dropping frames along the way as requested.
///
/// `correction` is `0` for a straight copy (`output.len() == input.len()`), positive to insert that
/// many frames (output that much longer than input), or negative to drop that many (output shorter).
/// The splices are spread evenly across the chunk and each one is hidden by crossfading over
/// `CROSSFADE_LEN` frames. The regions never overlap, which the caller guarantees by keeping
/// `correction` within [max_corrections].
fn place_correction<T: Sample>(input: &[T], output: &mut [T], correction: i32) {
    let out_len = output.len();
    let l = CROSSFADE_LEN;
    let n = correction.unsigned_abs() as usize;
    if n == 0 {
        output.copy_from_slice(&input[..out_len]);
        return;
    }
    // A positive correction inserts frames, so the read position falls one frame behind at each
    // splice; a negative one drops frames and the read position runs one frame ahead.
    let step: isize = if correction > 0 { -1 } else { 1 };

    // Spread the n crossfade regions evenly, separated by gaps so they never overlap. There are
    // n + 1 gaps (before, between and after the regions); distribute the spare frames as evenly as
    // possible. The capacity check guarantees at least one frame in every gap.
    let gap_total = out_len - n * l;
    let base_gap = gap_total / (n + 1);
    let extra = gap_total % (n + 1);

    let mut offset: isize = 0;
    let mut pos = 0;
    for r in 0..n {
        let gap = base_gap + if r < extra { 1 } else { 0 };
        let src = (pos as isize + offset) as usize;
        output[pos..pos + gap].copy_from_slice(&input[src..src + gap]);
        pos += gap;
        // Crossfade from the current read offset to the next one.
        for &wf in &FADE {
            let w = T::coerce(wf);
            let i = (pos as isize + offset) as usize;
            let a = input[i];
            let b = input[(i as isize + step) as usize];
            output[pos] = a + w * (b - a);
            pos += 1;
        }
        offset += step;
    }
    // Final gap: copy the remainder at the final read offset.
    let src = (pos as isize + offset) as usize;
    output[pos..].copy_from_slice(&input[src..src + (out_len - pos)]);
}

impl<T> Slip<T>
where
    T: Sample,
{
    /// Create a new [Slip] resampler.
    ///
    /// The resample ratio starts at 1.0 and is meant to be tuned at runtime through
    /// [Adjustable::set_resample_ratio] by a feedback loop; see the [type docs](Slip) for the range
    /// it can sustain.
    ///
    /// Parameters are:
    /// - `chunk_size`: Size of the fixed side (input or output, see `fixed`) in frames. Must be at
    ///   least 18 frames (twice the internal crossfade plus a margin).
    /// - `nbr_channels`: Number of channels in input/output.
    /// - `fixed`: Whether the input or the output chunk size is fixed.
    pub fn new(
        chunk_size: usize,
        nbr_channels: usize,
        fixed: FixedAsync,
    ) -> Result<Self, ResamplerConstructionError> {
        debug!(
            "Create new Slip with fixed {:?}, chunk_size: {}, channels: {}",
            fixed, chunk_size, nbr_channels,
        );

        if chunk_size < 2 * CROSSFADE_LEN + 2 {
            return Err(ResamplerConstructionError::InvalidChunkSize(chunk_size));
        }

        let max_correction = max_corrections(chunk_size, CROSSFADE_LEN);
        // The variable side can be up to `max_correction` frames larger than the fixed side.
        let scratch_len = chunk_size + max_correction;

        let mut resampler = Slip {
            nbr_channels,
            chunk_size,
            max_chunk_size: chunk_size,
            max_correction,
            needed_input_size: chunk_size,
            needed_output_size: chunk_size,
            correction: 0,
            drift_acc: 0.0,
            resample_ratio: 1.0,
            input_scratch: vec![T::zero(); scratch_len],
            output_scratch: vec![T::zero(); scratch_len],
            channel_mask: vec![true; nbr_channels],
            fixed,
        };
        resampler.replan();
        Ok(resampler)
    }

    /// The range of resample ratios the resampler can sustain with the current chunk size.
    fn current_ratio_range(&self) -> (f64, f64) {
        ratio_range(
            max_corrections(self.chunk_size, CROSSFADE_LEN),
            self.chunk_size,
            self.fixed,
        )
    }

    /// Work out the correction and the input/output sizes for the next chunk.
    ///
    /// This is pure with respect to `drift_acc`: it projects the accumulator forward by one chunk
    /// to decide whether a correction is due, but does not commit the change. That keeps it safe to
    /// call again after the ratio or chunk size is adjusted.
    fn replan(&mut self) {
        let projected =
            self.drift_acc + drift_per_chunk(self.resample_ratio, self.chunk_size, self.fixed);
        // Take as many whole frames of correction as the projected error calls for, capped at what
        // fits in this chunk without the crossfades overlapping. Any remainder stays in the
        // accumulator and is applied on a later chunk.
        let cap = max_corrections(self.chunk_size, CROSSFADE_LEN) as i32;
        self.correction = (projected.trunc() as i32).clamp(-cap, cap);
        match self.fixed {
            FixedAsync::Input => {
                self.needed_input_size = self.chunk_size;
                self.needed_output_size =
                    (self.chunk_size as i64 + self.correction as i64) as usize;
            }
            FixedAsync::Output => {
                self.needed_output_size = self.chunk_size;
                self.needed_input_size = (self.chunk_size as i64 - self.correction as i64) as usize;
            }
        }
    }
}

impl<T> Resampler<T> for Slip<T>
where
    T: Sample,
{
    fn process_into_buffer(
        &mut self,
        buffer_in: &dyn Adapter<T>,
        buffer_out: &mut dyn AdapterMut<T>,
        indexing: Option<&Indexing>,
    ) -> ResampleResult<(usize, usize)> {
        update_mask(&indexing, &mut self.channel_mask)?;
        let (input_offset, output_offset) = get_offsets(&indexing);

        let partial_input_len = get_partial_len(&indexing);
        let frames_to_read = if let Some(frames) = partial_input_len {
            frames.min(self.needed_input_size)
        } else {
            self.needed_input_size
        };

        trace!("Start processing, {:?}", self);

        validate_buffers(
            buffer_in,
            buffer_out,
            self.nbr_channels,
            frames_to_read + input_offset,
            self.needed_output_size + output_offset,
        )?;

        let input_len = self.needed_input_size;
        let output_len = self.needed_output_size;

        for (chan, active) in self.channel_mask.iter().enumerate() {
            if !*active {
                continue;
            }
            if self.correction == 0 {
                // No slip this chunk: read straight into the output scratch, skipping the
                // input scratch and the redundant scratch-to-scratch copy.
                buffer_in.copy_from_channel_to_slice(
                    chan,
                    input_offset,
                    &mut self.output_scratch[..output_len],
                );
                // Zero pad if this is a short final chunk.
                if frames_to_read < output_len {
                    for value in self.output_scratch[frames_to_read..output_len].iter_mut() {
                        *value = T::zero();
                    }
                }
            } else {
                buffer_in.copy_from_channel_to_slice(
                    chan,
                    input_offset,
                    &mut self.input_scratch[..input_len],
                );
                // Zero pad if this is a short final chunk.
                if frames_to_read < input_len {
                    for value in self.input_scratch[frames_to_read..input_len].iter_mut() {
                        *value = T::zero();
                    }
                }
                place_correction(
                    &self.input_scratch[..input_len],
                    &mut self.output_scratch[..output_len],
                    self.correction,
                );
            }
            buffer_out.copy_from_slice_to_channel(
                chan,
                output_offset,
                &self.output_scratch[..output_len],
            );
        }

        // Commit the timing error for this chunk and plan the next one.
        self.drift_acc += drift_per_chunk(self.resample_ratio, self.chunk_size, self.fixed);
        self.drift_acc -= self.correction as f64;
        self.replan();

        trace!(
            "Resampling channels {:?}, {} frames in, {} frames out",
            self.channel_mask,
            input_len,
            output_len,
        );
        Ok((input_len, output_len))
    }

    fn output_frames_max(&self) -> usize {
        match self.fixed {
            FixedAsync::Input => self.max_chunk_size + self.max_correction,
            FixedAsync::Output => self.max_chunk_size,
        }
    }

    fn output_frames_next(&self) -> usize {
        self.needed_output_size
    }

    fn output_delay(&self) -> usize {
        0
    }

    fn nbr_channels(&self) -> usize {
        self.nbr_channels
    }

    fn input_frames_max(&self) -> usize {
        match self.fixed {
            FixedAsync::Input => self.max_chunk_size,
            FixedAsync::Output => self.max_chunk_size + self.max_correction,
        }
    }

    fn input_frames_next(&self) -> usize {
        self.needed_input_size
    }

    fn resample_ratio(&self) -> f64 {
        self.resample_ratio
    }

    fn reset(&mut self) {
        self.channel_mask.iter_mut().for_each(|val| *val = true);
        self.drift_acc = 0.0;
        // Back to the nominal rate; the feedback loop re-tunes from there.
        self.resample_ratio = 1.0;
        self.chunk_size = self.max_chunk_size;
        self.replan();
    }

    fn as_adjustable(&mut self) -> Option<&mut dyn Adjustable<T>> {
        Some(self)
    }

    fn as_resizable(&mut self) -> Option<&mut dyn Resizable<T>> {
        Some(self)
    }
}

impl<T> Adjustable<T> for Slip<T>
where
    T: Sample,
{
    fn set_resample_ratio(&mut self, new_ratio: f64, _ramp: bool) -> ResampleResult<()> {
        trace!("Change resample ratio to {}", new_ratio);
        // The ratio is held within the range the corrections can sustain, the same range the
        // constructor enforces. A request outside it is clamped to the nearest limit and applied,
        // so the resampler runs at its best achievable rate, but an error is still returned so a
        // feedback loop can tell that it has saturated. Corrections are discrete single frame
        // events, so there is no per sample ramp to apply and the new ratio takes effect from the
        // next chunk.
        let (min, max) = self.current_ratio_range();
        let in_range = new_ratio >= min && new_ratio <= max;
        self.resample_ratio = if new_ratio > max {
            max
        } else if new_ratio < min || new_ratio.is_nan() {
            min
        } else {
            new_ratio
        };
        self.replan();
        if in_range {
            Ok(())
        } else {
            Err(ResampleError::RatioOutsideRange {
                provided: new_ratio,
                min,
                max,
            })
        }
    }

    fn set_resample_ratio_relative(&mut self, rel_ratio: f64, ramp: bool) -> ResampleResult<()> {
        // The nominal ratio is 1.0, so the relative ratio is also the absolute ratio.
        self.set_resample_ratio(rel_ratio, ramp)
    }
}

impl<T> Resizable<T> for Slip<T>
where
    T: Sample,
{
    fn set_chunk_size(&mut self, chunksize: usize) -> ResampleResult<()> {
        if chunksize > self.max_chunk_size || chunksize == 0 || chunksize < 2 * CROSSFADE_LEN + 2 {
            return Err(ResampleError::InvalidChunkSize {
                max: self.max_chunk_size,
                requested: chunksize,
            });
        }
        self.chunk_size = chunksize;
        self.replan();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Slip, FADE};
    use crate::tests::expected_output_value;
    use crate::FixedAsync;
    use crate::Indexing;
    use crate::{check_input_offset, check_masked, check_output, check_output_offset, check_reset};
    use crate::{Adjustable, Resampler, Resizable};
    use audioadapter_buffers::direct::SequentialSliceOfVecs;
    use test_case::test_matrix;

    /// The compile-time fade table must be a valid amplitude-complementary crossfade: monotonic
    /// from ~0 to ~1, and symmetric so a slip preserves the signal level across the splice.
    #[test]
    fn fade_table_is_valid() {
        let fade = FADE;
        let n = fade.len();
        assert!(fade[0] > 0.0 && fade[0] < 0.02, "should ease in from near 0");
        assert!(
            fade[n - 1] > 0.98 && fade[n - 1] < 1.0,
            "should ease out to near 1"
        );
        for pair in fade.windows(2) {
            assert!(pair[1] > pair[0], "must be strictly increasing");
        }
        // Symmetric: w(k) + w(N-1-k) == 1, so blended levels stay constant.
        for k in 0..n {
            assert!((fade[k] + fade[n - 1 - k] - 1.0).abs() < 1e-12);
        }
    }

    #[test_log::test(test_matrix(
        [50, 1024],
        [1.0, 1.0005, 0.9995],
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn drift_output(chunksize: usize, ratio: f64, fixed: FixedAsync) {
        let mut resampler = Slip::<f64>::new(chunksize, 2, fixed).unwrap();
        resampler.set_resample_ratio(ratio, false).unwrap();
        check_output!(resampler, f64);
    }

    #[test_log::test(test_matrix(
        [50, 1024],
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn drift_reset(chunksize: usize, fixed: FixedAsync) {
        let mut resampler = Slip::<f64>::new(chunksize, 2, fixed).unwrap();
        check_reset!(resampler);
    }

    #[test_log::test(test_matrix(
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn drift_masked(fixed: FixedAsync) {
        let mut resampler = Slip::<f64>::new(1024, 2, fixed).unwrap();
        check_masked!(resampler);
    }

    #[test_log::test(test_matrix(
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn drift_input_offset(fixed: FixedAsync) {
        let mut resampler = Slip::<f64>::new(1024, 2, fixed).unwrap();
        check_input_offset!(resampler);
    }

    #[test_log::test(test_matrix(
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn drift_output_offset(fixed: FixedAsync) {
        let mut resampler = Slip::<f64>::new(1024, 2, fixed).unwrap();
        check_output_offset!(resampler);
    }

    #[test]
    fn rejects_short_chunk() {
        // chunk_size must be at least 2 * CROSSFADE_LEN + 2 (18).
        assert!(Slip::<f64>::new(17, 1, FixedAsync::Input).is_err());
        assert!(Slip::<f64>::new(18, 1, FixedAsync::Input).is_ok());
    }

    /// At the default ratio of 1.0 the output must be a bit-exact copy of the input.
    #[test]
    fn unit_ratio_is_identity() {
        let mut resampler = Slip::<f64>::new(64, 1, FixedAsync::Input).unwrap();
        let input_data: Vec<Vec<f64>> = vec![(0..64).map(|i| (i as f64 * 0.3).sin()).collect()];
        let input = SequentialSliceOfVecs::new(&input_data, 1, 64).unwrap();
        let mut output_data = vec![vec![0.0; 65]; 1];
        let mut output = SequentialSliceOfVecs::new_mut(&mut output_data, 1, 65).unwrap();
        let (frames_in, frames_out) = resampler
            .process_into_buffer(&input, &mut output, None)
            .unwrap();
        assert_eq!((frames_in, frames_out), (64, 64));
        assert_eq!(&output_data[0][..64], &input_data[0][..]);
    }

    /// A constant (DC) signal must stay constant straight through a correction, because the
    /// crossfade only ever blends equal neighbouring samples.
    #[test]
    fn dc_stays_flat_across_correction() {
        let chunk = 64;
        let mut resampler = Slip::<f64>::new(chunk, 1, FixedAsync::Input).unwrap();
        resampler.set_resample_ratio(1.02, false).unwrap();
        let mut corrected = false;
        for _ in 0..10 {
            let out_len = resampler.output_frames_next();
            if out_len != chunk {
                corrected = true;
            }
            let input_data = vec![vec![0.5f64; chunk]];
            let input = SequentialSliceOfVecs::new(&input_data, 1, chunk).unwrap();
            let mut output_data = vec![vec![0.0; out_len]; 1];
            let mut output = SequentialSliceOfVecs::new_mut(&mut output_data, 1, out_len).unwrap();
            resampler
                .process_into_buffer(&input, &mut output, None)
                .unwrap();
            for &v in output_data[0].iter() {
                assert!((v - 0.5).abs() < 1e-12, "DC not preserved: {}", v);
            }
        }
        assert!(corrected, "expected at least one correction in the run");
    }

    /// A ratio far enough from 1.0 forces several corrections inside a single chunk. The realized
    /// ratio must still track, and a strictly increasing ramp must stay monotonic through every
    /// splice (which would break if any crossfade read from the wrong place).
    #[test_log::test(test_matrix(
        [1.01, 0.99],
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn multiple_corrections_per_chunk(ratio: f64, fixed: FixedAsync) {
        let chunk = 1024;
        let mut resampler = Slip::<f64>::new(chunk, 1, fixed).unwrap();
        resampler.set_resample_ratio(ratio, false).unwrap();
        let mut total_in = 0;
        let mut total_out = 0;
        let mut max_delta = 0;
        let mut ramp = 0.0f64;
        for _ in 0..500 {
            let frames_in = resampler.input_frames_next();
            let frames_out = resampler.output_frames_next();
            max_delta = max_delta.max((frames_in as isize - frames_out as isize).unsigned_abs());
            let input_data: Vec<Vec<f64>> = vec![(0..frames_in).map(|i| ramp + i as f64).collect()];
            ramp += frames_in as f64;
            let input = SequentialSliceOfVecs::new(&input_data, 1, frames_in).unwrap();
            let mut output_data = vec![vec![0.0; frames_out]; 1];
            let mut output =
                SequentialSliceOfVecs::new_mut(&mut output_data, 1, frames_out).unwrap();
            let (got_in, got_out) = resampler
                .process_into_buffer(&input, &mut output, None)
                .unwrap();
            total_in += got_in;
            total_out += got_out;
            for w in output_data[0].windows(2) {
                assert!(
                    w[1] >= w[0] - 1e-9,
                    "ramp not monotonic: {} -> {}",
                    w[0],
                    w[1]
                );
            }
        }
        assert!(
            max_delta > 1,
            "expected more than one correction in a chunk, got max delta {}",
            max_delta
        );
        let measured = total_out as f64 / total_in as f64;
        assert!(
            (measured - ratio).abs() < 1e-4,
            "measured ratio {} too far from target {}",
            measured,
            ratio
        );
    }

    /// Over many chunks the realized output/input ratio must track the requested ratio.
    #[test_log::test(test_matrix(
        [1.0003, 0.9997],
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn realized_ratio_tracks_target(ratio: f64, fixed: FixedAsync) {
        let mut resampler = Slip::<f64>::new(1024, 2, fixed).unwrap();
        resampler.set_resample_ratio(ratio, false).unwrap();
        let mut total_in = 0;
        let mut total_out = 0;
        for _ in 0..2000 {
            let frames_in = resampler.input_frames_next();
            let frames_out = resampler.output_frames_next();
            let input_data = vec![vec![0.0f64; frames_in]; 2];
            let input = SequentialSliceOfVecs::new(&input_data, 2, frames_in).unwrap();
            let mut output_data = vec![vec![0.0; frames_out]; 2];
            let mut output =
                SequentialSliceOfVecs::new_mut(&mut output_data, 2, frames_out).unwrap();
            let (got_in, got_out) = resampler
                .process_into_buffer(&input, &mut output, None)
                .unwrap();
            total_in += got_in;
            total_out += got_out;
        }
        let measured = total_out as f64 / total_in as f64;
        assert!(
            (measured - ratio).abs() < 1e-4,
            "measured ratio {} too far from target {}",
            measured,
            ratio
        );
    }

    #[test_log::test(test_matrix(
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn drift_resize(fixed: FixedAsync) {
        let mut resampler = Slip::<f64>::new(1024, 2, fixed).unwrap();
        resampler.set_resample_ratio(1.0005, false).unwrap();
        resampler.set_chunk_size(600).unwrap();
        check_output!(resampler, f64);
    }

    #[test]
    fn set_ratio_respects_range() {
        let mut resampler = Slip::<f64>::new(1024, 2, FixedAsync::Input).unwrap();
        let (min, max) = resampler.current_ratio_range();
        // A ratio within the supported range (about 1 +/- 0.1 here) is accepted and applied.
        assert!(resampler.set_resample_ratio(1.05, false).is_ok());
        assert_eq!(resampler.resample_ratio(), 1.05);
        // A ratio above the range is clamped to the maximum, applied, and still returns an error.
        assert!(resampler.set_resample_ratio(2.0, false).is_err());
        assert_eq!(resampler.resample_ratio(), max);
        // Likewise a ratio below the range (including non-positive ones) clamps to the minimum.
        assert!(resampler.set_resample_ratio(0.5, false).is_err());
        assert_eq!(resampler.resample_ratio(), min);
        assert!(resampler.set_resample_ratio(-1.0, false).is_err());
        assert_eq!(resampler.resample_ratio(), min);
    }
}
