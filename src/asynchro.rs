use audioadapter::{Adapter, AdapterMut};
use std::fmt;
use std::marker::PhantomData;

use crate::asynchro_fast::{InnerPoly, PolynomialDegree};
use crate::asynchro_sinc::{
    make_interpolator, InnerSinc, SincInterpolationParameters, SincInterpolationType,
};
use crate::error::{ResampleError, ResampleResult, ResamplerConstructionError};
use crate::sinc_interpolator::{
    AnyInterpolator, AvxSample, NeonSample, SincInterpolator, SseSample,
};
use crate::{get_offsets, get_partial_len, update_mask, Indexing};
use crate::{validate_buffers, Resampler, Sample};

/// An enum for specifying which side of an asynchronous resampler should be fixed size.
/// This is similar to [FixedSync](crate::FixedSync) that is used for the synchronous resamplers.
/// The difference is asynchronous resamplers must allow one side to vary,
/// and can therefore not support the `Both` option.
#[derive(Debug, Copy, Clone)]
pub enum FixedAsync {
    /// Input size is fixed, output size varies.
    Input,
    /// Output size is fixed, input size varies.
    Output,
}

/// Functions for making the scalar product with a sinc.
pub trait InnerResampler<T>: Send {
    /// Make the scalar product between the waveform starting at `index` and the sinc of `subindex`.
    #[allow(clippy::too_many_arguments)]
    fn process(
        &mut self,
        index: f64,
        nbr_frames: usize,
        channel_mask: &[bool],
        t_ratio: f64,
        t_ratio_increment: f64,
        wave_in: &[Vec<T>],
        wave_out: &mut dyn AdapterMut<'_, T>,
        output_offset: usize,
    ) -> f64;

    /// Get interpolator length.
    fn nbr_points(&self) -> usize;

    /// Get initial value for last_index.
    fn init_last_index(&self) -> f64;
}

/// An asynchronous resampler that uses either polynomial or sinc interpolation.
///
/// The `fixed` argument determines if input of output should be fixed size.
/// When the input size is fixed, the output size varies from call to call,
/// and when output size is fixed, the input size varies.
///
/// The number of frames on the fixed side is determined by the chunk size argument to the constructor.
/// This value can be changed by the `set_chunk_size()` method,
/// to let the resampler process smaller chunks of audio data.
/// Note that the chunk size cannot exceed the value given at creation time.
///
/// When the input size is fixed, the maximum value can be retrieved using the `input_size_max()` method,
/// and `input_frames_next()` gives the current value.
/// When the output size is fixed, the corresponding values are instead provided by the `output_size_max()`
/// and `output_size_next()` methods.
///
/// # Interpolation
/// This resampler can use either polynomial or sinc interpolation.
/// Sinc interpolation gives the best quality, while polynomial interpolation
/// runs significantly faster.
///
/// ## Polynomial
/// The resampling is done by interpolating between the input samples by fitting a polynomial.
/// The polynomial degree can be selected, see [PolynomialDegree] for the available options.
/// Higher polynomial degrees give better quality but run slower.
///
/// Note that no anti-aliasing filter is used.
/// This makes it run considerably faster than the corresponding Sinc resampler, which performs anti-aliasing filtering.
/// The price is that the resampling creates some artefacts in the output, mainly at higher frequencies.
/// Use a Sinc resampler if this can not be tolerated.
///
/// ## Sinc
/// The resampling is done by creating a number of intermediate points (defined by oversampling_factor)
/// by sinc interpolation. The new samples are then calculated by interpolating between these points.
///
/// # Adjusting the resampling ratio
/// The resampling ratio can be freely adjusted within the range specified to the constructor.
/// Adjusting the ratio does not recalculate the sinc functions used by the anti-aliasing filter.
/// This causes no issue when increasing the ratio (which slows down the output).
/// However, when decreasing more than a few percent (or speeding up the output),
/// the filters can no longer suppress all aliasing and this may lead to some artefacts.
///
/// The resampling ratio can be freely adjusted within the range specified to the constructor.
/// Higher maximum ratios require more memory to be allocated by an internal buffer,
/// and increase the maximum length of the variable length input or output buffer.
pub struct Async<T> {
    nbr_channels: usize,
    chunk_size: usize,
    max_chunk_size: usize,
    needed_input_size: usize,
    needed_output_size: usize,
    last_index: f64,
    current_buffer_fill: usize,
    resample_ratio: f64,
    resample_ratio_original: f64,
    target_ratio: f64,
    max_relative_ratio: f64,
    buffer: Vec<Vec<T>>,
    inner_resampler: Box<dyn InnerResampler<T>>,
    channel_mask: Vec<bool>,
    fixed: FixedAsync,
}

impl<T> fmt::Debug for Async<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Fast")
            .field("nbr_channels", &self.nbr_channels)
            .field("chunk_size,", &self.chunk_size)
            .field("max_chunk_size,", &self.max_chunk_size)
            .field("needed_input_size,", &self.needed_input_size)
            .field("needed_output_size,", &self.needed_output_size)
            .field("last_index", &self.last_index)
            .field("current_buffer_fill", &self.current_buffer_fill)
            .field("resample_ratio", &self.resample_ratio)
            .field("resample_ratio_original", &self.resample_ratio_original)
            .field("target_ratio", &self.target_ratio)
            .field("max_relative_ratio", &self.max_relative_ratio)
            .field("buffer[0].len()", &self.buffer[0].len())
            .field("channel_mask", &self.channel_mask)
            .field("fixed", &self.fixed)
            .finish()
    }
}

fn validate_ratios(
    resample_ratio: f64,
    max_resample_ratio_relative: f64,
) -> Result<(), ResamplerConstructionError> {
    if resample_ratio <= 0.0 {
        return Err(ResamplerConstructionError::InvalidRatio(resample_ratio));
    }
    if max_resample_ratio_relative < 1.0 {
        return Err(ResamplerConstructionError::InvalidRelativeRatio(
            max_resample_ratio_relative,
        ));
    }
    Ok(())
}

impl<T> Async<T>
where
    T: Sample,
{
    /// Create a new Async resampler that uses polynomial interpolation.
    ///
    /// Parameters are:
    /// - `resample_ratio`: Starting ratio between output and input sample rates, must be > 0.
    /// - `max_resample_ratio_relative`: Maximum ratio that can be set with [Resampler::set_resample_ratio] relative to `resample_ratio`, must be >= 1.0. The minimum relative ratio is the reciprocal of the maximum. For example, with `max_resample_ratio_relative` of 10.0, the ratio can be set between `resample_ratio * 10.0` and `resample_ratio / 10.0`.
    /// - `interpolation_type`: Degree of polynomial used for interpolation, see [PolynomialDegree].
    /// - `chunk_size`: Size of input data in frames.
    /// - `nbr_channels`: Number of channels in input/output.
    /// - `fixed`: Deciding whether input or output size is fixed.
    pub fn new_poly(
        resample_ratio: f64,
        max_resample_ratio_relative: f64,
        interpolation_type: PolynomialDegree,
        chunk_size: usize,
        nbr_channels: usize,
        fixed: FixedAsync,
    ) -> Result<Self, ResamplerConstructionError> {
        debug!(
            "Create new Fast with fixed {:?}, ratio: {}, chunk_size: {}, channels: {}",
            fixed, resample_ratio, chunk_size, nbr_channels,
        );

        validate_ratios(resample_ratio, max_resample_ratio_relative)?;

        if chunk_size == 0 {
            return Err(ResamplerConstructionError::InvalidChunkSize(chunk_size));
        }

        let channel_mask = vec![true; nbr_channels];

        let interpolator_len = interpolation_type.nbr_points();
        let inner_resampler = InnerPoly {
            interpolation: interpolation_type,
            _phantom: PhantomData,
        };

        let last_index = inner_resampler.init_last_index();
        let needed_input_size = Self::calculate_input_size(
            chunk_size,
            resample_ratio,
            resample_ratio,
            last_index,
            interpolator_len,
            &fixed,
        );
        let needed_output_size = Self::calculate_output_size(
            chunk_size,
            resample_ratio,
            resample_ratio,
            last_index,
            interpolator_len,
            &fixed,
        );

        let buffer_len = Self::calculate_max_input_size(
            chunk_size,
            resample_ratio,
            max_resample_ratio_relative,
            interpolator_len,
            &fixed,
        ) + 2 * interpolator_len;
        let buffer = vec![vec![T::zero(); buffer_len]; nbr_channels];

        Ok(Async {
            nbr_channels,
            chunk_size,
            max_chunk_size: chunk_size,
            needed_input_size,
            needed_output_size,
            current_buffer_fill: needed_input_size,
            last_index,
            resample_ratio,
            resample_ratio_original: resample_ratio,
            target_ratio: resample_ratio,
            max_relative_ratio: max_resample_ratio_relative,
            buffer,
            inner_resampler: Box::new(inner_resampler),
            channel_mask,
            fixed,
        })
    }

    /// Create a new [Async] resampler that uses sinc interpolation.
    ///
    /// Parameters are:
    /// - `resample_ratio`: Starting ratio between output and input sample rates, must be > 0.
    /// - `max_resample_ratio_relative`: Maximum ratio that can be set with [Resampler::set_resample_ratio] relative to `resample_ratio`, must be >= 1.0. The minimum relative ratio is the reciprocal of the maximum. For example, with `max_resample_ratio_relative` of 10.0, the ratio can be set between `resample_ratio * 10.0` and `resample_ratio / 10.0`.
    /// - `parameters`: Parameters for interpolation, see [SincInterpolationParameters].
    /// - `chunk_size`: Size of input data in frames.
    /// - `nbr_channels`: Number of channels in input/output.
    pub fn new_sinc(
        resample_ratio: f64,
        max_resample_ratio_relative: f64,
        parameters: &SincInterpolationParameters,
        chunk_size: usize,
        nbr_channels: usize,
        fixed: FixedAsync,
    ) -> Result<Self, ResamplerConstructionError> {
        debug!(
            "Create new Sinc fixed {:?}, ratio: {}, chunk_size: {}, channels: {}, parameters: {:?}",
            fixed, resample_ratio, chunk_size, nbr_channels, parameters
        );

        let interpolator = make_interpolator(
            parameters.sinc_len,
            resample_ratio,
            parameters.f_cutoff,
            parameters.oversampling_factor,
            parameters.window,
        );

        Self::new_with_sinc_interpolator(
            resample_ratio,
            max_resample_ratio_relative,
            parameters.interpolation,
            interpolator,
            chunk_size,
            nbr_channels,
            fixed,
        )
    }

    /// Create a new Sinc using an existing Interpolator.
    ///
    /// Parameters are:
    /// - `resample_ratio`: Starting ratio between output and input sample rates, must be > 0.
    /// - `max_resample_ratio_relative`: Maximum ratio that can be set with [Resampler::set_resample_ratio] relative to `resample_ratio`, must be >= 1.0. The minimum relative ratio is the reciprocal of the maximum. For example, with `max_resample_ratio_relative` of 10.0, the ratio can be set between `resample_ratio` * 10.0 and `resample_ratio` / 10.0.
    /// - `interpolation_type`: Parameters for interpolation, see `SincInterpolationParameters`.
    /// - `interpolator`: The interpolator to use.
    /// - `chunk_size`: Size of output data in frames.
    /// - `nbr_channels`: Number of channels in input/output.
    #[cfg_attr(feature = "bench_asyncro", visibility::make(pub))]
    fn new_with_sinc_interpolator(
        resample_ratio: f64,
        max_resample_ratio_relative: f64,
        interpolation_type: SincInterpolationType,
        interpolator: AnyInterpolator<T>,
        chunk_size: usize,
        nbr_channels: usize,
        fixed: FixedAsync,
    ) -> Result<Self, ResamplerConstructionError>
    where
        T: AvxSample + SseSample + NeonSample,
    {
        validate_ratios(resample_ratio, max_resample_ratio_relative)?;

        if chunk_size == 0 {
            return Err(ResamplerConstructionError::InvalidChunkSize(chunk_size));
        }

        let interpolator_len = interpolator.nbr_points();

        let inner_resampler = InnerSinc::new(interpolator, interpolation_type);

        let last_index = inner_resampler.init_last_index();
        let needed_input_size = Self::calculate_input_size(
            chunk_size,
            resample_ratio,
            resample_ratio,
            last_index,
            interpolator_len,
            &fixed,
        );
        let needed_output_size = Self::calculate_output_size(
            chunk_size,
            resample_ratio,
            resample_ratio,
            last_index,
            interpolator_len,
            &fixed,
        );

        let buffer_len = Self::calculate_max_input_size(
            chunk_size,
            resample_ratio,
            max_resample_ratio_relative,
            interpolator_len,
            &fixed,
        ) + 2 * interpolator_len;

        let buffer = vec![vec![T::zero(); buffer_len]; nbr_channels];

        let channel_mask = vec![true; nbr_channels];

        Ok(Async {
            nbr_channels,
            chunk_size,
            max_chunk_size: chunk_size,
            needed_input_size,
            needed_output_size,
            last_index,
            current_buffer_fill: needed_input_size,
            resample_ratio,
            resample_ratio_original: resample_ratio,
            target_ratio: resample_ratio,
            max_relative_ratio: max_resample_ratio_relative,
            inner_resampler: Box::new(inner_resampler),
            buffer,
            channel_mask,
            fixed,
        })
    }

    fn calculate_input_size(
        chunk_size: usize,
        resample_ratio: f64,
        target_ratio: f64,
        last_index: f64,
        interpolator_len: usize,
        fixed: &FixedAsync,
    ) -> usize {
        match fixed {
            FixedAsync::Input => chunk_size,
            FixedAsync::Output => (last_index
                + chunk_size as f64 / (0.5 * resample_ratio + 0.5 * target_ratio)
                + interpolator_len as f64)
                .ceil() as usize,
        }
    }

    fn calculate_output_size(
        chunk_size: usize,
        resample_ratio: f64,
        target_ratio: f64,
        last_index: f64,
        interpolator_len: usize,
        fixed: &FixedAsync,
    ) -> usize {
        match fixed {
            FixedAsync::Output => chunk_size,
            FixedAsync::Input => ((chunk_size as f64 - (interpolator_len + 1) as f64 - last_index)
                * (0.5 * resample_ratio + 0.5 * target_ratio))
                .floor() as usize,
        }
    }

    fn calculate_max_input_size(
        chunk_size: usize,
        resample_ratio_original: f64,
        max_relative_ratio: f64,
        interpolator_len: usize,
        fixed: &FixedAsync,
    ) -> usize {
        match fixed {
            FixedAsync::Input => chunk_size,
            FixedAsync::Output => {
                (chunk_size as f64 / resample_ratio_original * max_relative_ratio).ceil() as usize
                    + 2
                    + interpolator_len / 2
            }
        }
    }

    fn calculate_max_output_size(
        chunk_size: usize,
        resample_ratio_original: f64,
        max_relative_ratio: f64,
        fixed: &FixedAsync,
    ) -> usize {
        match fixed {
            FixedAsync::Output => chunk_size,
            FixedAsync::Input => {
                (chunk_size as f64 * resample_ratio_original * max_relative_ratio + 10.0) as usize
            }
        }
    }

    fn update_lengths(&mut self) {
        self.needed_input_size = Async::<T>::calculate_input_size(
            self.chunk_size,
            self.resample_ratio,
            self.target_ratio,
            self.last_index,
            self.inner_resampler.nbr_points(),
            &self.fixed,
        );
        self.needed_output_size = Async::<T>::calculate_output_size(
            self.chunk_size,
            self.resample_ratio,
            self.target_ratio,
            self.last_index,
            self.inner_resampler.nbr_points(),
            &self.fixed,
        );
        trace!(
            "Updated lengths, input: {}, output: {}",
            self.needed_input_size,
            self.needed_output_size
        );
    }
}

impl<T> Resampler<T> for Async<T>
where
    T: Sample,
{
    fn process_into_buffer<'a>(
        &mut self,
        buffer_in: &dyn Adapter<'a, T>,
        buffer_out: &mut dyn AdapterMut<'a, T>,
        indexing: Option<&Indexing>,
    ) -> ResampleResult<(usize, usize)> {
        // read the optional indexing struct
        update_mask(&indexing, &mut self.channel_mask);
        let (input_offset, output_offset) = get_offsets(&indexing);

        // figure out how many frames to read
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
            &self.channel_mask,
            self.nbr_channels,
            frames_to_read + input_offset,
            self.needed_output_size + output_offset,
        )?;

        let interpolator_len = self.inner_resampler.nbr_points();

        let t_ratio = 1.0 / self.resample_ratio;
        let t_ratio_end = 1.0 / self.target_ratio;

        let t_ratio_increment = (t_ratio_end - t_ratio) / self.needed_output_size as f64;

        // Update buffer with new data.
        for buf in self.buffer.iter_mut() {
            buf.copy_within(
                self.current_buffer_fill..self.current_buffer_fill + 2 * interpolator_len,
                0,
            );
        }

        for (chan, active) in self.channel_mask.iter().enumerate() {
            if *active {
                let slice = &mut self.buffer[chan]
                    [2 * interpolator_len..2 * interpolator_len + frames_to_read];
                buffer_in.copy_from_channel_to_slice(chan, input_offset, slice);
                // partial, write zeros to internal buffer
                if frames_to_read < self.needed_input_size {
                    for value in self.buffer[chan][2 * interpolator_len + frames_to_read
                        ..2 * interpolator_len + self.needed_input_size]
                        .iter_mut()
                    {
                        *value = T::zero();
                    }
                }
            }
        }

        self.current_buffer_fill = self.needed_input_size;

        let mut idx = self.last_index;

        // Process
        idx = self.inner_resampler.as_mut().process(
            idx,
            self.needed_output_size,
            &self.channel_mask,
            t_ratio,
            t_ratio_increment,
            &self.buffer,
            buffer_out,
            output_offset,
        );

        // Store last index for next iteration.
        self.last_index = idx - self.needed_input_size as f64;
        self.resample_ratio = self.target_ratio;
        trace!(
            "Resampling channels {:?}, {} frames in, {} frames out",
            self.channel_mask,
            self.needed_input_size,
            self.needed_output_size,
        );
        let input_size = self.needed_input_size;
        let output_size = self.needed_output_size;
        self.update_lengths();
        Ok((input_size, output_size))
    }

    fn output_frames_max(&self) -> usize {
        Async::<T>::calculate_max_output_size(
            self.max_chunk_size,
            self.resample_ratio_original,
            self.max_relative_ratio,
            &self.fixed,
        )
    }

    fn output_frames_next(&self) -> usize {
        self.needed_output_size
    }

    fn output_delay(&self) -> usize {
        (self.inner_resampler.nbr_points() as f64 * self.resample_ratio / 2.0) as usize
    }

    fn nbr_channels(&self) -> usize {
        self.nbr_channels
    }

    fn input_frames_max(&self) -> usize {
        Async::<T>::calculate_max_input_size(
            self.max_chunk_size,
            self.resample_ratio_original,
            self.max_relative_ratio,
            self.inner_resampler.nbr_points(),
            &self.fixed,
        )
    }

    fn input_frames_next(&self) -> usize {
        self.needed_input_size
    }

    fn set_resample_ratio(&mut self, new_ratio: f64, ramp: bool) -> ResampleResult<()> {
        trace!("Change resample ratio to {}", new_ratio);
        if (new_ratio / self.resample_ratio_original >= 1.0 / self.max_relative_ratio)
            && (new_ratio / self.resample_ratio_original <= self.max_relative_ratio)
        {
            if !ramp {
                self.resample_ratio = new_ratio;
            }
            self.target_ratio = new_ratio;
            self.update_lengths();
            Ok(())
        } else {
            Err(ResampleError::RatioOutOfBounds {
                provided: new_ratio,
                original: self.resample_ratio_original,
                max_relative_ratio: self.max_relative_ratio,
            })
        }
    }

    fn resample_ratio(&self) -> f64 {
        self.resample_ratio
    }

    fn set_resample_ratio_relative(&mut self, rel_ratio: f64, ramp: bool) -> ResampleResult<()> {
        let new_ratio = self.resample_ratio_original * rel_ratio;
        self.set_resample_ratio(new_ratio, ramp)
    }

    fn reset(&mut self) {
        self.buffer
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = T::zero()));
        self.channel_mask.iter_mut().for_each(|val| *val = true);
        self.last_index = self.inner_resampler.init_last_index();
        self.resample_ratio = self.resample_ratio_original;
        self.target_ratio = self.resample_ratio_original;
        self.chunk_size = self.max_chunk_size;
        self.update_lengths();
    }

    fn set_chunk_size(&mut self, chunksize: usize) -> ResampleResult<()> {
        if chunksize > self.max_chunk_size || chunksize == 0 {
            return Err(ResampleError::InvalidChunkSize {
                max: self.max_chunk_size,
                requested: chunksize,
            });
        }
        self.chunk_size = chunksize;
        self.update_lengths();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::expected_output_value;
    use crate::Indexing;
    use crate::PolynomialDegree;
    use crate::Resampler;
    use crate::SincInterpolationParameters;
    use crate::SincInterpolationType;
    use crate::WindowFunction;
    use crate::{
        assert_fi_len, assert_fo_len, check_input_offset, check_masked, check_output,
        check_output_offset, check_ratio, check_reset,
    };
    use crate::{Async, FixedAsync};
    use audioadapter_buffers::direct::SequentialSliceOfVecs;
    use test_case::test_matrix;

    fn basic_params() -> SincInterpolationParameters {
        SincInterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        }
    }

    #[test_log::test(test_matrix(
        [1, 100, 1024],
        [0.8, 1.2, 0.125, 8.0],
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn poly_output(chunksize: usize, ratio: f64, fixed: FixedAsync) {
        let mut resampler =
            Async::<f64>::new_poly(ratio, 1.0, PolynomialDegree::Cubic, chunksize, 2, fixed)
                .unwrap();
        check_output!(resampler, f64);
    }

    #[test_log::test(test_matrix(
        [1, 100, 1024],
        [0.8, 1.2, 0.125, 8.0],
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn poly_ratio(chunksize: usize, ratio: f64, fixed: FixedAsync) {
        let mut resampler =
            Async::<f64>::new_poly(ratio, 1.0, PolynomialDegree::Cubic, chunksize, 2, fixed)
                .unwrap();
        check_ratio!(resampler, 100000 / chunksize, 0.001, f64);
    }

    #[test_log::test(test_matrix(
        [1, 100, 1024],
        [0.8, 1.2, 0.125, 8.0],
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn poly_len(chunksize: usize, ratio: f64, fixed: FixedAsync) {
        let resampler =
            Async::<f64>::new_poly(ratio, 1.0, PolynomialDegree::Cubic, chunksize, 2, fixed)
                .unwrap();
        match fixed {
            FixedAsync::Input => {
                assert_fi_len!(resampler, chunksize);
            }
            FixedAsync::Output => {
                assert_fo_len!(resampler, chunksize);
            }
        }
    }

    #[test_log::test(test_matrix(
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn poly_masked(fixed: FixedAsync) {
        let mut resampler =
            Async::<f64>::new_poly(0.75, 1.0, PolynomialDegree::Cubic, 1024, 2, fixed).unwrap();
        check_masked!(resampler);
    }

    #[test_log::test(test_matrix(
        [1, 100, 1024],
        [0.8, 1.2, 0.125, 8.0],
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn poly_reset(chunksize: usize, ratio: f64, fixed: FixedAsync) {
        let mut resampler =
            Async::<f64>::new_poly(ratio, 1.0, PolynomialDegree::Cubic, chunksize, 2, fixed)
                .unwrap();
        check_reset!(resampler);
    }

    #[test_log::test(test_matrix(
        [1, 100, 1024],
        [0.8, 1.2, 0.125, 8.0],
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn sinc_output(chunksize: usize, ratio: f64, fixed: FixedAsync) {
        let params = basic_params();
        let mut resampler =
            Async::<f64>::new_sinc(ratio, 1.0, &params, chunksize, 2, fixed).unwrap();
        check_output!(resampler, f64);
    }

    #[test_log::test(test_matrix(
        [1, 100, 1024],
        [0.8, 1.2, 0.125, 8.0],
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn sinc_ratio(chunksize: usize, ratio: f64, fixed: FixedAsync) {
        let params = basic_params();
        let mut resampler =
            Async::<f64>::new_sinc(ratio, 1.0, &params, chunksize, 2, fixed).unwrap();
        check_ratio!(resampler, 100000 / chunksize, 0.001, f64);
    }

    #[test_log::test(test_matrix(
        [1, 100, 1024],
        [0.8, 1.2, 0.125, 8.0],
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn sinc_len(chunksize: usize, ratio: f64, fixed: FixedAsync) {
        let params = basic_params();
        let resampler = Async::<f64>::new_sinc(ratio, 1.0, &params, chunksize, 2, fixed).unwrap();
        match fixed {
            FixedAsync::Input => {
                assert_fi_len!(resampler, chunksize);
            }
            FixedAsync::Output => {
                assert_fo_len!(resampler, chunksize);
            }
        }
    }

    #[test_log::test(test_matrix(
        [1, 100, 1024],
        [0.8, 1.2, 0.125, 8.0],
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn sinc_reset(chunksize: usize, ratio: f64, fixed: FixedAsync) {
        let params = basic_params();
        let mut resampler =
            Async::<f64>::new_sinc(ratio, 1.0, &params, chunksize, 2, fixed).unwrap();
        check_reset!(resampler);
    }

    #[test_log::test(test_matrix(
        [1, 100, 1024],
        [0.8, 1.2, 0.125, 8.0],
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn async_input_offset(chunksize: usize, ratio: f64, fixed: FixedAsync) {
        let params = basic_params();
        let mut resampler =
            Async::<f64>::new_sinc(ratio, 1.0, &params, chunksize, 2, fixed).unwrap();
        check_input_offset!(resampler);
    }

    #[test_log::test(test_matrix(
        [1, 100, 1024],
        [0.8, 1.2, 0.125, 8.0],
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn async_output_offset(chunksize: usize, ratio: f64, fixed: FixedAsync) {
        let params = basic_params();
        let mut resampler =
            Async::<f64>::new_sinc(ratio, 1.0, &params, chunksize, 2, fixed).unwrap();
        check_output_offset!(resampler);
    }

    #[test_log::test(test_matrix(
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn sinc_masked(fixed: FixedAsync) {
        let params = basic_params();
        let mut resampler = Async::<f64>::new_sinc(0.75, 1.0, &params, 1024, 2, fixed).unwrap();
        check_masked!(resampler);
    }

    #[test_log::test(test_matrix(
        [0.8, 1.2, 0.125, 8.0],
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn async_resize(ratio: f64, fixed: FixedAsync) {
        let params = basic_params();
        let mut resampler = Async::<f64>::new_sinc(ratio, 1.0, &params, 1024, 2, fixed).unwrap();
        resampler.set_chunk_size(600).unwrap();
        check_output!(resampler, f64);
    }

    fn process_and_get_frame_counts(resampler: &mut Async<f64>) -> (usize, usize) {
        let input_frames = resampler.input_frames_next();
        let output_frames_max = resampler.output_frames_max();
        let input_data = vec![vec![0.25; input_frames]; 2];
        let input = SequentialSliceOfVecs::new(&input_data, 2, input_frames).unwrap();

        let mut output_data = vec![vec![0.0; output_frames_max]; 2];
        let mut output =
            SequentialSliceOfVecs::new_mut(&mut output_data, 2, output_frames_max).unwrap();

        let (consumed_frames, produced_frames) = resampler
            .process_into_buffer(&input, &mut output, None)
            .unwrap();
        (consumed_frames, produced_frames)
    }

    fn process_chunks_and_get_total_frame_counts(
        resampler: &mut Async<f64>,
        nbr_chunks: usize,
    ) -> (usize, usize) {
        (0..nbr_chunks).fold((0, 0), |(sum_in, sum_out), _| {
            let (frames_in, frames_out) = process_and_get_frame_counts(resampler);
            (sum_in + frames_in, sum_out + frames_out)
        })
    }

    fn assert_ratio_within_tolerance(
        total_in: usize,
        total_out: usize,
        expected_ratio: f64,
        tolerance: f64,
        label: &str,
    ) {
        let measured_ratio = total_out as f64 / total_in as f64;
        let lower = expected_ratio - tolerance;
        let upper = expected_ratio + tolerance;
        assert!(
            measured_ratio >= lower && measured_ratio <= upper,
            "{} ratio out of range: got {}, expected {} +/- {} (out {}, in {})",
            label,
            measured_ratio,
            expected_ratio,
            tolerance,
            total_out,
            total_in,
        );
    }

    fn check_relative_ratio_changes_frame_ratio(mut resampler: Async<f64>) {
        let nbr_chunks = 8;
        let tolerance = 0.01;

        let (baseline_in, baseline_out) =
            process_chunks_and_get_total_frame_counts(&mut resampler, nbr_chunks);
        assert_ratio_within_tolerance(baseline_in, baseline_out, 1.0, tolerance, "Baseline");

        // Lower ratio speeds up output.
        resampler.set_resample_ratio_relative(0.75, false).unwrap();
        let (lower_in, lower_out) =
            process_chunks_and_get_total_frame_counts(&mut resampler, nbr_chunks);
        assert_ratio_within_tolerance(lower_in, lower_out, 0.75, tolerance, "0.75x");

        // Higher ratio slows down output.
        resampler.set_resample_ratio_relative(1.25, false).unwrap();
        let (higher_in, higher_out) =
            process_chunks_and_get_total_frame_counts(&mut resampler, nbr_chunks);
        assert_ratio_within_tolerance(higher_in, higher_out, 1.25, tolerance, "1.25x");
    }

    #[test_log::test(test_matrix(
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn poly_relative_ratio_changes_frame_ratio(fixed: FixedAsync) {
        let resampler =
            Async::<f64>::new_poly(1.0, 4.0, PolynomialDegree::Cubic, 1024, 2, fixed).unwrap();
        check_relative_ratio_changes_frame_ratio(resampler);
    }

    #[test_log::test(test_matrix(
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn sinc_relative_ratio_changes_frame_ratio(fixed: FixedAsync) {
        let params = basic_params();
        let resampler = Async::<f64>::new_sinc(1.0, 4.0, &params, 1024, 2, fixed).unwrap();
        check_relative_ratio_changes_frame_ratio(resampler);
    }

    /// Run a 1-channel and a 4-channel sinc resampler with identical per-channel input and
    /// compare their outputs. The 1-channel resampler uses the direct path (separate dot
    /// products per nearest point); the 4-channel resampler uses the combined-sinc path
    /// (scaled accumulation into a combined filter, then one dot product per channel). They must agree within floating-
    /// point rounding tolerance.
    fn compare_1ch_4ch_sinc_output(
        interpolation: SincInterpolationType,
        ratio: f64,
        fixed: FixedAsync,
    ) {
        let params = SincInterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let chunk = 256;

        let mut r1 = Async::<f64>::new_sinc(ratio, 1.0, &params, chunk, 1, fixed).unwrap();
        let mut r4 = Async::<f64>::new_sinc(ratio, 1.0, &params, chunk, 4, fixed).unwrap();

        let mut phase = 0.0f64;
        for _ in 0..20 {
            let frames_in = r1.input_frames_next();
            let frames_out = r1.output_frames_next();
            assert_eq!(frames_in, r4.input_frames_next());
            assert_eq!(frames_out, r4.output_frames_next());

            let wave: Vec<f64> = (0..frames_in)
                .map(|i| (phase + i as f64 * 0.1).sin())
                .collect();
            phase += frames_in as f64 * 0.1;

            let in1_data = vec![wave.clone()];
            let in4_data = vec![wave.clone(), wave.clone(), wave.clone(), wave.clone()];
            let input_1ch = SequentialSliceOfVecs::new(&in1_data, 1, frames_in).unwrap();
            let input_4ch = SequentialSliceOfVecs::new(&in4_data, 4, frames_in).unwrap();

            let mut out1_data = vec![vec![0.0f64; frames_out]; 1];
            let mut out4_data = vec![vec![0.0f64; frames_out]; 4];
            let mut out1 = SequentialSliceOfVecs::new_mut(&mut out1_data, 1, frames_out).unwrap();
            let mut out4 = SequentialSliceOfVecs::new_mut(&mut out4_data, 4, frames_out).unwrap();

            r1.process_into_buffer(&input_1ch, &mut out1, None).unwrap();
            r4.process_into_buffer(&input_4ch, &mut out4, None).unwrap();

            for frame in 0..frames_out {
                let expected = out1_data[0][frame];
                // All 4 channels must be exactly equal (identical input, same combined sinc).
                for ch in 1..4 {
                    assert_eq!(
                        out4_data[ch][frame], out4_data[0][frame],
                        "interp={interpolation:?} ratio={ratio} frame={frame}: \
                         ch{ch} differs from ch0 inside 4ch resampler"
                    );
                }
                // 4ch output must agree with 1ch output within floating-point tolerance.
                for (ch, ch_data) in out4_data.iter().enumerate() {
                    let diff = (ch_data[frame] - expected).abs();
                    assert!(
                        diff < 1e-10,
                        "interp={interpolation:?} ratio={ratio} ch={ch} frame={frame}: \
                         4ch={} vs 1ch={expected} diff={diff}",
                        ch_data[frame]
                    );
                }
            }
        }
    }

    #[test_log::test(test_matrix(
        [
            SincInterpolationType::Cubic,
            SincInterpolationType::Quadratic,
            SincInterpolationType::Linear,
            SincInterpolationType::Nearest,
        ],
        [0.8f64, 1.2f64],
        [FixedAsync::Input, FixedAsync::Output]
    ))]
    fn sinc_4ch_matches_1ch(interp: SincInterpolationType, ratio: f64, fixed: FixedAsync) {
        compare_1ch_4ch_sinc_output(interp, ratio, fixed);
    }
}
