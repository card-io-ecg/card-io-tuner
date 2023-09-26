use std::{cell::Ref, sync::Arc};

use rustfft::num_complex::{Complex, ComplexFloat};
use serde::{Deserialize, Serialize};
use signal_processing::{
    designfilt,
    filter::{
        iir::{HighPass, Iir, LowPass},
        pli::{adaptation_blocking::AdaptationBlocking, PowerLineFilter},
        Filter,
    },
    heart_rate::{HeartRateCalculator, SamplingFrequencyExt, Thresholds},
    moving::sum::Sum,
};

use crate::{
    analysis::{adjust_time, average, average_cycle, cross_correlate, similarity},
    data::{cell::DataCell, Cycle, Ekg},
};

pub struct HrData {
    pub detections: Vec<usize>,
    pub thresholds: Vec<Thresholds>,
    pub complex_lead: Vec<f32>,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Config {
    pub high_pass: bool,
    pub pli: bool,
    pub low_pass: bool,
    pub hr_debug: bool,
    pub ignored_start: usize,
    pub ignored_end: usize,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            high_pass: true,
            pli: true,
            low_pass: true,
            hr_debug: false,
            ignored_start: 0,
            ignored_end: 200,
        }
    }
}

pub struct Context {
    pub raw_ekg: Ekg,
    pub config: Config,
}

pub struct ProcessedSignal {
    raw_ekg: DataCell<Ekg>,
    filtered_ekg: DataCell<Ekg>,
    fft: DataCell<Vec<f32>>,
    hrs: DataCell<HrData>,
    cycles: DataCell<Vec<Cycle>>,
    adjusted_cycles: DataCell<Vec<Cycle>>,
    average_cycle: DataCell<Cycle>,
    majority_cycle: DataCell<Cycle>,
    rr_intervals: DataCell<Vec<f64>>,
    adjusted_rr_intervals: DataCell<Vec<f64>>,
}

impl ProcessedSignal {
    pub fn new() -> Self {
        Self {
            raw_ekg: DataCell::new("raw_ekg"),
            filtered_ekg: DataCell::new("filtered_ekg"),
            fft: DataCell::new("fft"),
            hrs: DataCell::new("hrs"),
            cycles: DataCell::new("cycles"),
            adjusted_cycles: DataCell::new("adjusted_cycles"),
            average_cycle: DataCell::new("average_cycle"),
            majority_cycle: DataCell::new("majority_cycle"),
            rr_intervals: DataCell::new("rr_intervals"),
            adjusted_rr_intervals: DataCell::new("adjusted_rr_intervals"),
        }
    }

    pub fn clear(&mut self) {
        self.raw_ekg.clear();
        self.filtered_ekg.clear();
        self.fft.clear();
        self.hrs.clear();
        self.cycles.clear();
        self.adjusted_cycles.clear();
        self.average_cycle.clear();
        self.majority_cycle.clear();
        self.rr_intervals.clear();
        self.adjusted_rr_intervals.clear();
    }

    pub fn raw_ekg(&self, context: &Context) -> Ref<'_, Ekg> {
        self.raw_ekg.get(|| {
            log::debug!("Data::raw_ekg");
            let ignore_start = context.config.ignored_start;
            let ignore_end = context.config.ignored_end;

            let sample_count = context.raw_ekg.samples.len();

            Ekg {
                samples: Arc::from(
                    context.raw_ekg.samples[ignore_start..sample_count - ignore_end].to_vec(),
                ),
                fs: context.raw_ekg.fs,
            }
        })
    }

    pub fn filtered_ekg(&self, context: &Context) -> Ref<'_, Ekg> {
        self.filtered_ekg.get(|| {
            log::debug!("Data::filtered_ekg");

            let mut samples = self.raw_ekg(context).samples.to_vec();

            if context.config.pli {
                apply_filter(
                    &mut samples,
                    PowerLineFilter::<AdaptationBlocking<Sum<1200>, 4, 19>, 1>::new(
                        context.raw_ekg.fs as f32,
                        [50.0],
                    ),
                );
            }

            if context.config.high_pass {
                #[rustfmt::skip]
                let high_pass = designfilt!(
                    "highpassiir",
                    "FilterOrder", 2,
                    "HalfPowerFrequency", 0.75,
                    "SampleRate", 1000
                );
                apply_zero_phase_filter(&mut samples, high_pass);
            }

            if context.config.low_pass {
                #[rustfmt::skip]
                let low_pass = designfilt!(
                    "lowpassiir",
                    "FilterOrder", 2,
                    "HalfPowerFrequency", 75,
                    "SampleRate", 1000
                );
                apply_zero_phase_filter(&mut samples, low_pass);
            }

            debias(&mut samples);

            Ekg {
                samples: Arc::from(samples),
                fs: context.raw_ekg.fs,
            }
        })
    }

    pub fn fft(&self, context: &Context) -> Ref<'_, Vec<f32>> {
        self.fft.get(|| {
            log::debug!("Data::fft");
            let mut samples = self
                .filtered_ekg(context)
                .samples
                .iter()
                .copied()
                .map(|y| Complex { re: y, im: 0.0 })
                .collect::<Vec<_>>();

            let mut planner = rustfft::FftPlanner::new();
            let fft = planner.plan_fft_forward(samples.len());

            fft.process(&mut samples);

            samples.iter().copied().map(|c| c.abs()).collect::<Vec<_>>()
        })
    }

    pub fn hrs(&self, context: &Context) -> Ref<'_, HrData> {
        self.hrs.get(|| {
            log::debug!("Data::hrs");
            let filtered = self.filtered_ekg(context);

            let ekg: &[f32] = &filtered.samples;
            let fs = filtered.fs as f32;
            let mut calculator = HeartRateCalculator::new(fs as f32);

            let mut qrs_idxs = Vec::new();
            let mut thresholds = Vec::new();
            let mut samples = Vec::new();

            for (idx, sample) in ekg.iter().enumerate() {
                if let Some(complex_lead) = calculator.update(*sample) {
                    thresholds.push(calculator.thresholds());
                    samples.push(complex_lead);

                    if calculator.is_beat() {
                        // We need to increase the index by the delay because the calculator isn't
                        // aware of the filtering on its input, which basically cuts of the first few
                        // samples.
                        qrs_idxs.push(idx);
                    }
                }
            }

            HrData {
                detections: qrs_idxs,
                thresholds,
                complex_lead: samples,
            }
        })
    }

    pub fn rr_intervals(&self, context: &Context) -> Ref<'_, Vec<f64>> {
        self.rr_intervals.get(|| {
            log::debug!("Data::rr_intervals");
            let fs = self.filtered_ekg(context).fs;

            self.hrs(context)
                .detections
                .iter()
                .map_windows(move |[a, b]| (*b - *a) as f64 / fs)
                .collect()
        })
    }

    pub fn adjusted_rr_intervals(&self, context: &Context) -> Ref<'_, Vec<f64>> {
        self.adjusted_rr_intervals.get(|| {
            log::debug!("Data::adjusted_rr_intervals");
            let fs = self.filtered_ekg(context).fs;

            self.adjusted_cycles(context)
                .iter()
                .map(|cycle| cycle.position)
                .map_windows(move |[a, b]| (*b - *a) as f64 / fs)
                .collect()
        })
    }

    pub fn cycles(&self, context: &Context) -> Ref<'_, Vec<Cycle>> {
        self.cycles.get(|| {
            log::debug!("Data::cycles");
            let filtered = self.filtered_ekg(context);
            let hrs = self.hrs(context);

            let fs = (filtered.fs as f32).sps();
            let avg_rr = self.avg_rr(context) as f32;

            let pre = fs.s_to_samples(avg_rr / 3.0);
            let post = fs.s_to_samples(avg_rr * 2.0 / 3.0);

            hrs.detections
                .iter()
                .copied()
                .filter_map(|idx| {
                    filtered.samples.get(idx - pre..idx + post).map(|_| Cycle {
                        samples: filtered.samples.clone(),
                        start: idx - pre,
                        position: idx,
                        end: idx + post,
                    })
                })
                .collect::<Vec<_>>()
        })
    }

    pub fn adjusted_cycles(&self, context: &Context) -> Ref<'_, Vec<Cycle>> {
        self.adjusted_cycles.get(|| {
            log::debug!("Data::adjusted_cycles");
            let filtered = self.filtered_ekg(context);

            let fs = (filtered.fs as f32).sps();
            let avg_rr = self.avg_rr(context) as f32;

            let pre = fs.s_to_samples(avg_rr / 3.0);
            let post = fs.s_to_samples(avg_rr * 2.0 / 3.0);

            let cycles = self.cycles(context);

            let cycle_idxs = cycles.iter().map(|cycle| cycle.position);
            let all_cycles = cycles.iter().map(|cycle| cycle.as_slice());

            let all_average = average_cycle(all_cycles.clone());

            let mut max = f32::NEG_INFINITY;
            let max_pos = all_average
                .iter()
                .enumerate()
                .filter_map(|(idx, y)| {
                    if *y > max {
                        max = *y;
                        Some(idx)
                    } else {
                        None
                    }
                })
                .last()
                .unwrap();

            // For QRS adjustment, we're using the 50-50 ms window around the peak of the QRS
            let avg_qrs_width = fs.ms_to_samples(25.0);
            let avg_qrs = &all_average[max_pos - avg_qrs_width..][..2 * avg_qrs_width];

            let avg_max_offset = max_pos as isize - pre as isize;

            let adjusted_idxs = cycle_idxs.map(|idx| {
                let idx = idx as isize;
                let offset_to_avg = adjust_time(
                    &filtered.samples[(idx + avg_max_offset) as usize - avg_qrs.len()..]
                        [..2 * avg_qrs.len()],
                    &avg_qrs,
                );

                (idx + avg_max_offset + offset_to_avg) as usize
            });

            adjusted_idxs
                .filter_map(|idx| {
                    filtered.samples.get(idx - pre..idx + post).map(|_| Cycle {
                        samples: filtered.samples.clone(),
                        start: idx - pre,
                        position: idx,
                        end: idx + post,
                    })
                })
                .collect::<Vec<_>>()
        })
    }

    pub fn average_cycle(&self, context: &Context) -> Ref<'_, Cycle> {
        self.average_cycle.get(|| {
            log::debug!("Data::average_cycle");
            let adjusted_cycles = self.adjusted_cycles(context);

            let avg = average_cycle(adjusted_cycles.iter().map(|cycle| cycle.as_slice()));

            let mut max = f32::NEG_INFINITY;
            let max_pos = avg
                .iter()
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
                .unwrap();

            Cycle {
                position: max_pos,
                start: 0,
                end: avg.len(),
                samples: Arc::from(avg),
            }
        })
    }

    pub fn majority_cycle(&self, context: &Context) -> Ref<'_, Cycle> {
        self.majority_cycle.get(|| {
            log::debug!("Data::majority_cycle");
            let adjusted_cycles = self.adjusted_cycles(context);

            let avg = self.average_cycle(context);

            let autocorr = cross_correlate(avg.as_slice(), avg.as_slice());

            let similarities = adjusted_cycles
                .iter()
                .map(|cycle| cross_correlate(cycle.as_slice(), avg.as_slice()))
                .map(|xcorr| similarity(xcorr, autocorr))
                .collect::<Vec<_>>();

            const SIMILARITY_THRESHOLD: f32 = 0.8;

            let similar_cycles = adjusted_cycles.iter().zip(similarities.iter()).filter_map(
                |(cycle, similarity)| {
                    (*similarity > SIMILARITY_THRESHOLD).then_some(cycle.as_slice())
                },
            );

            let majority_cycle = average_cycle(similar_cycles.clone());

            // TODO: this case can be avoided by using a better similarity metric based on clustering and artifact detection
            if majority_cycle.is_empty() {
                log::warn!("No similar cycles found");
                return avg.clone();
            }

            log::debug!(
                "Similarity with average: {}, based on {}/{} cycles",
                similarity(
                    cross_correlate(majority_cycle.as_slice(), avg.as_slice()),
                    autocorr
                ),
                similar_cycles.count(),
                adjusted_cycles.len(),
            );

            let mut max = f32::NEG_INFINITY;
            let max_pos = majority_cycle
                .iter()
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
                .unwrap();

            Cycle {
                position: max_pos,
                start: 0,
                end: majority_cycle.len(),
                samples: Arc::from(majority_cycle),
            }
        })
    }

    pub fn avg_rr(&self, context: &Context) -> f64 {
        average(self.rr_intervals(context).iter().copied())
    }

    pub fn adjusted_avg_rr(&self, context: &Context) -> f64 {
        average(self.adjusted_rr_intervals(context).iter().copied())
    }

    pub fn avg_hr(&self, context: &Context) -> f64 {
        60.0 / self.adjusted_avg_rr(context)
    }
}

fn debias(signal: &mut Vec<f32>) {
    let first = signal[0];
    signal.iter_mut().for_each(|x| *x = *x - first);
}

fn apply_filter<F: Filter>(signal: &mut Vec<f32>, mut filter: F) {
    debias(signal);
    *signal = signal
        .iter()
        .copied()
        .filter_map(|sample| filter.update(sample))
        .collect::<Vec<_>>();
}

fn apply_zero_phase_filter<F: Filter + Clone>(signal: &mut Vec<f32>, filter: F) {
    apply_filter(signal, filter.clone());
    signal.reverse();

    apply_filter(signal, filter);
    signal.reverse();
}
