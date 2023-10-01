pub fn cross_correlate(cycle: &[f32], average: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..cycle.len() {
        sum += cycle[i] * average[i];
    }
    sum
}

pub fn average_cycle<'a>(mut cycles: impl Iterator<Item = &'a [f32]>) -> Vec<f32> {
    let Some(first) = cycles.next() else {
        return vec![];
    };

    if first.is_empty() {
        return vec![];
    }

    let mut count = 1;
    let mut average = first.to_vec();

    for cycle in cycles {
        count += 1;
        for (avg, sample) in average.iter_mut().zip(cycle.iter()) {
            *avg += *sample;
        }
    }

    let first = average[0] / count as f32;
    for avg in average.iter_mut() {
        *avg /= count as f32;
        *avg -= first;
    }

    average
}

pub fn adjust_time(cycle: &[f32], average: &[f32]) -> isize {
    let mut offset = 0;
    let mut max_cc = f32::NEG_INFINITY;
    for (cycle_offset, signal) in cycle.windows(average.len()).enumerate() {
        let cc = cross_correlate(signal, average);
        if cc > max_cc {
            max_cc = cc;
            offset = cycle_offset as isize;
        }
    }

    let width = average.len() as isize;
    offset - width
}

pub fn average(iter: impl Iterator<Item = f64>) -> f64 {
    let (count, sum) = iter.fold((0, 0.0), |(count, sum), y| (count + 1, sum + y));

    sum / count as f64
}

pub fn similarity(corr: f32, max_corr: f32) -> f32 {
    1.0 - (1.0 - corr / max_corr).abs()
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
