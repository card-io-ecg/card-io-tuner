use rustfft::num_traits::Float;

use crate::data::Cycle;

/// Pearson correlation coefficient
///
/// https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
pub fn corr_coeff(cycle: &[f32], avg: &[f32]) -> f32 {
    let mean_cycle = average(cycle.iter().map(|x| *x as f64)) as f32;
    let mean_avg = average(avg.iter().map(|x| *x as f64)) as f32;

    let mut covariance = 0.0;
    let mut sd_cycle = 0.0;
    let mut sd_avg = 0.0;

    for (sample, avg_sample) in cycle.iter().zip(avg.iter()) {
        let d_cycle = sample - mean_cycle;
        let d_avg = avg_sample - mean_avg;
        covariance += d_cycle * d_avg;
        sd_cycle += d_cycle * d_cycle;
        sd_avg += d_avg * d_avg;
    }

    covariance / (sd_cycle * sd_avg).sqrt()
}

pub fn average_cycle<'a>(mut cycles: impl Iterator<Item = &'a Cycle>) -> Option<Cycle> {
    let first = cycles.next()?;

    if first.as_slice().is_empty() {
        return None;
    }

    let mut count = 1;
    let mut average = first.as_slice().to_vec();

    for cycle in cycles {
        count += 1;
        for (avg, sample) in average.iter_mut().zip(cycle.as_slice().iter()) {
            *avg += *sample;
        }
    }

    let first_sample = average[0];
    for avg in average.iter_mut() {
        *avg -= first_sample;
        *avg /= count as f32;
    }

    let qrs = first.position - first.start;

    let search_width = first.as_slice().len() / 5;
    let search_start = qrs - search_width;

    let offset = max_pos(&average[search_start..][..2 * search_width]).unwrap_or(0);
    Some(Cycle::new_virtual(average, search_start + offset))
}

pub fn adjust_time(cycle: &[f32], average: &[f32]) -> isize {
    let mut offset = 0;
    let mut max_cc = f32::NEG_INFINITY;
    for (cycle_offset, signal) in cycle.windows(average.len()).enumerate() {
        let cc = corr_coeff(signal, average);
        if cc > max_cc {
            max_cc = cc;
            offset = cycle_offset as isize;
        }
    }

    let diff = cycle.len() as isize - average.len() as isize;
    offset - diff / 2
}

pub fn average<F: Float>(iter: impl Iterator<Item = F>) -> F {
    let (count, sum) = iter.fold((0, F::zero()), |(count, sum), y| (count + 1, sum + y));

    sum / F::from(count).unwrap()
}

pub fn max_pos(avg: &[f32]) -> Option<usize> {
    let mut max = f32::NEG_INFINITY;

    avg.iter()
        .enumerate()
        .filter_map(|(idx, y)| {
            max = max.max(*y);
            if *y == max {
                Some(idx)
            } else {
                None
            }
        })
        .last()
}
