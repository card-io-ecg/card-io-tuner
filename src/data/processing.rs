use std::cell::Ref;

use rustfft::num_complex::{Complex, ComplexFloat};
use serde::{Deserialize, Serialize};
use signal_processing::{
    filter::{
        dyn_iir::DynIir,
        iir::{HighPass, LowPass},
        pli::{adaptation_blocking::AdaptationBlocking, PowerLineFilter},
        Filter,
    },
    heart_rate::{HeartRateCalculator, SamplingFrequency, SamplingFrequencyExt, Thresholds},
    moving::sum::Sum,
};

use crate::{
    analysis::{adjust_time, average, average_cycle, corr_coeff, max_pos},
    data::{
        cell::DataCell,
        grouping::{group_cycles, GroupMap},
        matrix::Matrix,
        Classification, Cycle, Ekg,
    },
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
    pub high_pass_cutoff: f32,
    pub low_pass_cutoff: f32,
    pub similarity_threshold: f32,
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
            high_pass_cutoff: 0.75,
            low_pass_cutoff: 75.0,
            similarity_threshold: 0.85,
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
    classified_cycles: DataCell<Vec<Cycle>>,
    cycle_corr_coeffs: DataCell<Matrix<f32>>,
    cycle_groups: DataCell<GroupMap>,
    majority_cycle: DataCell<Cycle>,
    rr_intervals: DataCell<Vec<f32>>,
    adjusted_rr_intervals: DataCell<Vec<f32>>,
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
            cycle_corr_coeffs: DataCell::new("cycle_corr_coeffs"),
            cycle_groups: DataCell::new("cycle_groups"),
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
        self.cycle_corr_coeffs.clear();
        self.cycle_groups.clear();
        self.majority_cycle.clear();
        self.rr_intervals.clear();
        self.adjusted_rr_intervals.clear();
    }

    pub fn raw_ekg(&self, context: &Context) -> Ref<'_, Ekg> {
        self.raw_ekg.get(|| {
            log::debug!("raw_ekg");

            Ekg {
                samples: context.raw_ekg.samples.clone(),
                fs: context.raw_ekg.fs,
                ignore_start: context.config.ignored_start,
                ignore_end: context.config.ignored_end,
            }
        })
    }

    pub fn filtered_ekg(&self, context: &Context) -> Ref<'_, Ekg> {
        self.filtered_ekg.get(|| {
            log::debug!("filtered_ekg");

            let fs = self.fs(context).raw();
            let mut samples = self.raw_ekg(context).samples().to_vec();

            if context.config.high_pass {
                let high_pass = DynIir::<HighPass, 2>::design(fs, context.config.high_pass_cutoff);
                apply_zero_phase_filter(&mut samples, high_pass);
            }

            if context.config.low_pass {
                let low_pass = DynIir::<LowPass, 2>::design(fs, context.config.low_pass_cutoff);
                apply_zero_phase_filter(&mut samples, low_pass);
            }

            if context.config.pli {
                // TODO: adaptation blocking needs to be fs aware
                let pli = PowerLineFilter::<AdaptationBlocking<Sum<1200>, 4, 19>, _, 1>::design(
                    fs,
                    [50.0],
                );
                apply_filter(&mut samples, pli);
            }

            debias(&mut samples);

            Ekg::new(context.raw_ekg.fs, samples)
        })
    }

    pub fn fft(&self, context: &Context) -> Ref<'_, Vec<f32>> {
        self.fft.get(|| {
            log::debug!("fft");
            let mut samples = self
                .filtered_ekg(context)
                .samples()
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
            log::debug!("hrs");
            let filtered = self.filtered_ekg(context);
            let fs = self.fs(context).raw();

            let mut ekg = filtered.samples().to_vec();

            let low_pass = DynIir::<LowPass, 2>::design(fs, 20.0);
            apply_zero_phase_filter(&mut ekg, low_pass);

            let mut calculator = HeartRateCalculator::new_alloc(fs);

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

    pub fn rr_intervals(&self, context: &Context) -> Ref<'_, Vec<f32>> {
        self.rr_intervals.get(|| {
            log::debug!("rr_intervals");
            let fs = self.fs(context);

            self.hrs(context)
                .detections
                .iter()
                .map_windows(move |[a, b]| fs.samples_to_s(*b - *a))
                .collect()
        })
    }

    pub fn adjusted_rr_intervals(&self, context: &Context) -> Ref<'_, Vec<f32>> {
        self.adjusted_rr_intervals.get(|| {
            log::debug!("adjusted_rr_intervals");
            let fs = self.fs(context);

            self.adjusted_cycles(context)
                .iter()
                .map(|cycle| cycle.position)
                .map_windows(move |[a, b]| fs.samples_to_s(*b - *a))
                .collect()
        })
    }

    pub fn cycles(&self, context: &Context) -> Ref<'_, Vec<Cycle>> {
        self.cycles.get(|| {
            log::debug!("cycles");
            let filtered = self.filtered_ekg(context);
            let hrs = self.hrs(context);

            let avg_rr = self.avg_rr_samples(context);

            let result = hrs
                .detections
                .iter()
                .copied()
                .filter_map(|idx| Cycle::at(&filtered.samples, idx, avg_rr))
                .collect::<Vec<_>>();

            log::debug!("cycles: Processed {} cycles", result.len());

            result
        })
    }

    pub fn adjusted_cycles(&self, context: &Context) -> Ref<'_, Vec<Cycle>> {
        self.adjusted_cycles.get(|| {
            log::debug!("adjusted_cycles");

            let fs = self.fs(context);
            let cycles = self.classified_cycles(context);

            let result = if let Some(average) =
                average_cycle(cycles.iter().filter(|cycle| cycle.cycle_group().is_some()))
            {
                log::debug!("Average cycle: {:?}", average);

                // For QRS adjustment, we're using a smaller window around the peak of the QRS
                let avg_qrs_width = fs.ms_to_samples(40.0);

                cycles
                    .iter()
                    .filter_map(|cycle| {
                        if cycle.cycle_group().is_some() {
                            let offset_to_avg = adjust_time(
                                cycle.middle(2 * avg_qrs_width),
                                average.middle(avg_qrs_width),
                            );
                            cycle.offset(offset_to_avg)
                        } else {
                            Some(cycle.clone())
                        }
                    })
                    .collect::<Vec<_>>()
            } else {
                cycles.clone()
            };

            log::debug!("adjusted_cycles: Processed {} cycles", result.len());

            result
        })
    }

    pub fn cycle_corr_coeffs(&self, context: &Context) -> Ref<'_, Matrix<f32>> {
        self.cycle_corr_coeffs.get(|| {
            log::debug!("cycle_corr_coeffs");

            let cycles = self.cycles(context);

            let search_width = self.fs(context).ms_to_samples(40.0);

            let cycles = cycles.iter().map(|cycle| {
                let offset = max_pos(cycle.middle(search_width)).unwrap_or(0);
                cycle
                    .offset(offset as isize - search_width as isize)
                    .unwrap_or(cycle.clone())
            });

            let mut result = Matrix::<f32>::new(cycles.len(), cycles.len());

            let mut enumerated_cycles = cycles.enumerate();
            while let Some((x, cycle_a)) = enumerated_cycles.next() {
                result[(x, x)] = 1.0;

                let mut row = enumerated_cycles.clone();
                while let Some((y, cycle_b)) = row.next() {
                    let cc = corr_coeff(cycle_a.as_slice(), cycle_b.as_slice());
                    debug_assert!(!cc.is_nan());
                    result[(x, y)] = cc;
                    result[(y, x)] = cc;
                }
            }

            // pretty-print matrix
            // log::debug!("{:?}", result);

            result
        })
    }

    pub fn cycle_groups(&self, context: &Context) -> Ref<'_, GroupMap> {
        self.cycle_groups.get(|| {
            log::debug!("cycle_groups");

            let coeffs = self.cycle_corr_coeffs(context);

            group_cycles(&coeffs, context.config.similarity_threshold)
        })
    }

    pub fn classified_cycles(&self, context: &Context) -> Ref<'_, Vec<Cycle>> {
        self.classified_cycles.get(|| {
            log::debug!("classified_cycles");

            let cycles = self.cycles(context);
            let groups = self.cycle_groups(context);

            let result = cycles
                .iter()
                .enumerate()
                .map(|(cycle_idx, cycle)| {
                    cycle.classify(if groups.similar_count(cycle_idx) > 1 {
                        Classification::Normal(groups[cycle_idx])
                    } else {
                        Classification::Artifact
                    })
                })
                .collect::<Vec<_>>();

            log::debug!("classified_cycles: Processed {} cycles", result.len());

            result
        })
    }

    pub fn majority_cycle(&self, context: &Context) -> Ref<'_, Cycle> {
        self.majority_cycle.get(|| {
            log::debug!("majority_cycle");

            let cycles = self.adjusted_cycles(context);

            let majority_cycle =
                average_cycle(cycles.iter().filter(|cycle| cycle.cycle_group().is_some()))
                    .unwrap_or_else(|| {
                        log::warn!("Defaulting to the average cycle");
                        average_cycle(cycles.iter()).unwrap()
                    });

            // TODO: this case can be avoided by using an adaptive similarity threshold based on clustering and artifact detection
            if majority_cycle.as_slice().is_empty() {
                log::warn!("No similar cycles found");
                average_cycle(cycles.iter()).unwrap()
            } else {
                majority_cycle
            }
        })
    }

    pub fn avg_rr(&self, context: &Context) -> f64 {
        average(self.rr_intervals(context).iter().map(|rr| *rr as f64))
    }

    pub fn avg_rr_samples(&self, context: &Context) -> usize {
        let fs = self.fs(context);
        fs.s_to_samples(self.avg_rr(context) as f32)
    }

    pub fn adjusted_avg_rr(&self, context: &Context) -> f64 {
        average(
            self.adjusted_rr_intervals(context)
                .iter()
                .map(|rr| *rr as f64),
        )
    }

    pub fn avg_hr(&self, context: &Context) -> f64 {
        60.0 / self.adjusted_avg_rr(context)
    }

    pub fn fs(&self, context: &Context) -> SamplingFrequency {
        context.raw_ekg.fs.sps()
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
