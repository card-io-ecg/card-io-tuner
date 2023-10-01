use std::{cell::Ref, sync::Arc};

use rustfft::num_complex::{Complex, ComplexFloat};
use serde::{Deserialize, Serialize};
use signal_processing::{
    filter::{
        dyn_iir::DynIir,
        iir::{HighPass, LowPass},
        pli::{adaptation_blocking::AdaptationBlocking, PowerLineFilter},
        Filter,
    },
    heart_rate::{HeartRateCalculator, SamplingFrequencyExt, Thresholds},
    moving::sum::Sum,
};

use crate::{
    analysis::{adjust_time, average, average_cycle, cross_correlate, similarity},
    data::{cell::DataCell, Classification, Cycle, Ekg},
};

pub struct HrData {
    pub detections: Vec<usize>,
    pub thresholds: Vec<Thresholds>,
    pub complex_lead: Vec<f32>,
}

#[derive(Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct Config {
    pub high_pass: bool,
    pub pli: bool,
    pub low_pass: bool,
    pub hr_debug: bool,
    pub ignored_start: usize,
    pub ignored_end: usize,
    pub row_width: usize,
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
            row_width: 6000,
        }
    }
}

pub struct Context {
    pub(super) raw_ekg: Ekg,
    pub(super) config: Config,
}

pub struct ProcessedSignal {
    raw_ekg: DataCell<Ekg>,
    filtered_ekg: DataCell<Ekg>,
    fft: DataCell<Vec<f32>>,
    hrs: DataCell<HrData>,
    cycles: DataCell<Vec<Cycle>>,
    adjusted_cycles: DataCell<Vec<Cycle>>,
    classified_cycles: DataCell<Vec<(Cycle, Classification)>>,
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
            classified_cycles: DataCell::new("classified_cycles"),
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
        self.classified_cycles.clear();
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

            let fs = context.raw_ekg.fs as f32;
            let mut samples = self.raw_ekg(context).samples.to_vec();

            if context.config.pli {
                let pli = PowerLineFilter::<AdaptationBlocking<Sum<1200>, 4, 19>, _, 1>::design(
                    fs,
                    [50.0],
                );
                apply_filter(&mut samples, pli);
            }

            if context.config.high_pass {
                let high_pass = DynIir::<HighPass, 2>::design(fs, 0.75);
                apply_zero_phase_filter(&mut samples, high_pass);
            }

            if context.config.low_pass {
                let low_pass = DynIir::<LowPass, 2>::design(fs, 75.0);
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

            let mut ekg = filtered.samples.to_vec();

            let low_pass = DynIir::<LowPass, 2>::design(filtered.fs as f32, 20.0);
            apply_zero_phase_filter(&mut ekg, low_pass);

            let mut calculator = HeartRateCalculator::new_alloc(filtered.fs as f32);

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

            let avg_rr = self.avg_rr_samples(context);

            hrs.detections
                .iter()
                .copied()
                .filter_map(|idx| Cycle::at(&filtered.samples, idx, avg_rr))
                .collect::<Vec<_>>()
        })
    }

    pub fn adjusted_cycles(&self, context: &Context) -> Ref<'_, Vec<Cycle>> {
        self.adjusted_cycles.get(|| {
            log::debug!("Data::adjusted_cycles");
            let filtered = self.filtered_ekg(context);

            let fs = filtered.fs.sps();

            let avg_rr = self.avg_rr_samples(context);

            let cycles = self.cycles(context);

            let all_average = average_cycle(cycles.iter().map(|cycle| cycle.as_slice()));

            // For QRS adjustment, we're using the 50-50 ms window around the peak of the QRS
            let avg_qrs_width = fs.ms_to_samples(25.0);
            let mut max = f32::NEG_INFINITY;
            let max_pos = all_average
                .iter()
                .enumerate()
                .skip(avg_qrs_width)
                .take(all_average.len() - 2 * avg_qrs_width)
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

            let avg_qrs = &all_average[max_pos - avg_qrs_width..max_pos + avg_qrs_width];

            let pre = avg_rr / 3;
            let avg_max_offset = max_pos as isize - pre as isize;

            cycles
                .iter()
                .filter_map(|cycle| cycle.offset(avg_max_offset))
                .filter_map(|cycle| {
                    let offset_to_avg = adjust_time(cycle.middle(avg_qrs.len()), &avg_qrs);

                    cycle.offset(offset_to_avg)
                })
                .collect::<Vec<_>>()
        })
    }

    pub fn average_adjusted_cycle(&self, context: &Context) -> Ref<'_, Cycle> {
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

    pub fn classified_cycles(&self, context: &Context) -> Ref<'_, Vec<(Cycle, Classification)>> {
        self.classified_cycles.get(|| {
            log::debug!("Data::classified_cycles");

            log::debug!("Data::majority_cycle");
            let adjusted_cycles = self.adjusted_cycles(context);

            let avg = self.average_adjusted_cycle(context);

            let autocorr = cross_correlate(avg.as_slice(), avg.as_slice());

            const SIMILARITY_THRESHOLD: f32 = 0.8;

            adjusted_cycles
                .iter()
                .map(|cycle| {
                    (
                        cycle.clone(),
                        cross_correlate(cycle.as_slice(), avg.as_slice()),
                    )
                })
                .map(|(cycle, xcorr)| (cycle, similarity(xcorr, autocorr)))
                .map(|(cycle, similarity)| {
                    if similarity > SIMILARITY_THRESHOLD {
                        (cycle, Classification::Normal)
                    } else {
                        (cycle, Classification::Artifact)
                    }
                })
                .collect::<Vec<_>>()
        })
    }

    pub fn majority_cycle(&self, context: &Context) -> Ref<'_, Cycle> {
        self.majority_cycle.get(|| {
            log::debug!("Data::majority_cycle");
            let adjusted_cycles = self.adjusted_cycles(context);

            let avg = self.average_adjusted_cycle(context);

            let autocorr = cross_correlate(avg.as_slice(), avg.as_slice());
            let classified_cycles = self.classified_cycles(context);

            let similar_cycles = classified_cycles
                .iter()
                .filter_map(|(cycle, classification)| {
                    (*classification == Classification::Normal).then_some(cycle.as_slice())
                });

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

    pub fn avg_rr_samples(&self, context: &Context) -> usize {
        let filtered = self.filtered_ekg(context);

        let fs = (filtered.fs as f32).sps();
        fs.s_to_samples(self.avg_rr(context) as f32)
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
