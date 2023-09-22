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