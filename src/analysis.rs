use crate::data::Cycle;

pub fn corr_coeff(cycle: &[f32], avg: &[f32]) -> f32 {
    let mean_cycle = average(cycle.iter().map(|x| *x as f64)) as f32;
    let mean_avg = average(avg.iter().map(|x| *x as f64)) as f32;

    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;

    for (sample, avg_sample) in cycle.iter().zip(avg.iter()) {
        let diff1 = sample - mean_cycle;
        let diff2 = avg_sample - mean_avg;
        sum1 += diff1 * diff2;
        sum2 += diff1 * diff1;
        sum3 += diff2 * diff2;
    }

    sum1 / (sum2 * sum3).sqrt()
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

    for avg in average.iter_mut() {
        *avg /= count as f32;
    }

    Some(Cycle::new_virtual(average))
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

pub fn average(iter: impl Iterator<Item = f64>) -> f64 {
    let (count, sum) = iter.fold((0, 0.0), |(count, sum), y| (count + 1, sum + y));

    sum / count as f64
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
