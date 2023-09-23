#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![feature(iter_map_windows)]

use std::{cell::Ref, env, ops::Range, path::PathBuf, sync::Arc};

use eframe::{
    egui::{self, PointerButton, Ui},
    epaint::Color32,
};
use egui_plot::{AxisBools, GridInput, GridMark, Legend, Line, MarkerShape, PlotPoints, Points};
use rustfft::num_complex::{Complex, ComplexFloat};
use signal_processing::{
    compressing_buffer::EkgFormat,
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
    analysis::{adjust_time, average_cycle, cross_correlate},
    data_cell::DataCell,
};

mod analysis;
mod data_cell;

fn main() -> Result<(), eframe::Error> {
    env::set_var("RUST_LOG", "card_io_tuner=debug");
    env_logger::init();
    eframe::run_native(
        "EKG visualizer and filter tuner",
        eframe::NativeOptions {
            initial_window_size: Some(egui::vec2(640.0, 480.0)),
            ..Default::default()
        },
        Box::new(|_cc| Box::<EkgTuner>::default()),
    )
}

const EKG_COLOR: Color32 = Color32::from_rgb(100, 150, 250);

#[derive(Clone)]
struct Ekg {
    samples: Arc<[f32]>,
    fs: f64,
}

impl Ekg {
    fn load(bytes: Vec<u8>) -> Result<Self, ()> {
        let version: u32 = u32::from_le_bytes(bytes[0..4].try_into().map_err(|err| {
            log::warn!("Failed to read version: {}", err);
            ()
        })?);
        log::debug!("version: {}", version);

        match version {
            0 => Self::load_v0(&bytes[4..]),
            _ => {
                log::warn!("Unknown version: {}", version);
                Err(())
            }
        }
    }

    fn load_v0(mut bytes: &[u8]) -> Result<Self, ()> {
        pub const VOLTS_PER_LSB: f32 = -2.42 / (1 << 23) as f32; // ADS129x

        let mut reader = EkgFormat::new();
        let mut samples = Vec::new();
        while let Some(sample) = reader.read(&mut bytes).unwrap() {
            samples.push(sample as f32 * VOLTS_PER_LSB);
        }

        log::debug!("Loaded {} samples", samples.len());

        let ignore_start = 0;
        let ignore_end = 300;

        Ok(Self {
            fs: 1000.0,
            samples: Arc::from(
                samples
                    .get(ignore_start..samples.len() - ignore_end)
                    .ok_or(())?,
            ),
        })
    }
}

struct HrData {
    detections: Vec<usize>,
    thresholds: Vec<Thresholds>,
    complex_lead: Vec<f32>,
}

#[derive(Clone)]
pub struct Cycle {
    samples: Arc<[f32]>,
    start: usize,
    position: usize,
    end: usize,
}
impl Cycle {
    fn as_slice(&self) -> &[f32] {
        &self.samples[self.start..self.end]
    }
}

#[derive(Default)]
struct ProcessedSignal {
    filtered_ekg: DataCell<Ekg>,
    fft: DataCell<Vec<f32>>,
    hrs: DataCell<HrData>,
    cycles: DataCell<Vec<Cycle>>,
    adjusted_cycles: DataCell<Vec<Cycle>>,
    average_cycle: DataCell<Cycle>,
    majority_cycle: DataCell<Cycle>,
}

impl ProcessedSignal {
    fn new() -> Self {
        Self::default()
    }

    fn clear(&mut self) {
        self.filtered_ekg.clear();
        self.fft.clear();
        self.hrs.clear();
        self.cycles.clear();
        self.adjusted_cycles.clear();
        self.average_cycle.clear();
        self.majority_cycle.clear();
    }
}

struct Data {
    path: PathBuf,
    raw_ekg: Ekg,
    processed: ProcessedSignal,
    filter_config: FilterConfig,
    hr_debug: bool,
}

struct FilterConfig {
    high_pass: bool,
    pli: bool,
    low_pass: bool,
}

impl Data {
    fn load(path: PathBuf) -> Option<Self> {
        log::debug!("Loading {}", path.display());
        std::fs::read(&path).ok().and_then(|bytes| {
            let ekg = Ekg::load(bytes).ok()?;
            Some(Data {
                path,
                raw_ekg: ekg,
                processed: ProcessedSignal::new(),
                filter_config: FilterConfig {
                    high_pass: true,
                    pli: true,
                    low_pass: true,
                },
                hr_debug: false,
            })
        })
    }

    fn filtered_ekg(&self) -> Ref<'_, Ekg> {
        self.processed.filtered_ekg.get(|| {
            let mut samples = self.raw_ekg.samples.to_vec();

            if self.filter_config.pli {
                apply_filter(
                    &mut samples,
                    PowerLineFilter::<AdaptationBlocking<Sum<1200>, 4, 19>, 1>::new(
                        self.raw_ekg.fs as f32,
                        [50.0],
                    ),
                );
            }

            if self.filter_config.high_pass {
                #[rustfmt::skip]
                let high_pass = designfilt!(
                    "highpassiir",
                    "FilterOrder", 2,
                    "HalfPowerFrequency", 0.75,
                    "SampleRate", 1000
                );
                apply_zero_phase_filter(&mut samples, high_pass);
            }

            if self.filter_config.low_pass {
                #[rustfmt::skip]
                let low_pass = designfilt!(
                    "lowpassiir",
                    "FilterOrder", 2,
                    "HalfPowerFrequency", 75,
                    "SampleRate", 1000
                );
                apply_zero_phase_filter(&mut samples, low_pass);
            }

            Ekg {
                samples: Arc::from(samples),
                fs: self.raw_ekg.fs,
            }
        })
    }

    fn fft(&self) -> Ref<'_, Vec<f32>> {
        self.processed.fft.get(|| {
            let mut samples = self
                .filtered_ekg()
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

    fn hrs(&self) -> Ref<'_, HrData> {
        self.processed.hrs.get(|| {
            let filtered = self.filtered_ekg();
            let (qrs_idxs, thresholds, samples) =
                detect_beats(&filtered.samples, filtered.fs as f32);

            HrData {
                detections: qrs_idxs,
                thresholds,
                complex_lead: samples,
            }
        })
    }

    fn rr_intervals(&self) -> impl Iterator<Item = f64> {
        let fs = self.filtered_ekg().fs;

        self.hrs()
            .detections
            .clone()
            .into_iter()
            .map_windows(move |[a, b]| (*b - *a) as f64 / fs)
    }

    fn adjusted_rr_intervals(&self) -> impl Iterator<Item = f64> {
        let fs = self.filtered_ekg().fs;

        self.adjusted_cycles()
            .clone()
            .into_iter()
            .map(|cycle| cycle.position)
            .map_windows(move |[a, b]| (*b - *a) as f64 / fs)
    }

    fn avg_rr(&self) -> f32 {
        let (count, sum) = self
            .rr_intervals()
            .map(|rr| rr as f32)
            .fold((0, 0.0), |(count, sum), hr| (count + 1, sum + hr));

        sum / count as f32
    }

    fn adjusted_avg_rr(&self) -> f32 {
        let (count, sum) = self
            .adjusted_rr_intervals()
            .map(|rr| rr as f32)
            .fold((0, 0.0), |(count, sum), hr| (count + 1, sum + hr));

        sum / count as f32
    }

    fn avg_hr(&self) -> f32 {
        60.0 / self.adjusted_avg_rr()
    }

    fn cycles(&self) -> Ref<'_, Vec<Cycle>> {
        self.processed.cycles.get(|| {
            let filtered = self.filtered_ekg();
            let hrs = self.hrs();

            let fs = (filtered.fs as f32).sps();
            let avg_rr = self.avg_rr();

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

    fn adjusted_cycles(&self) -> Ref<'_, Vec<Cycle>> {
        self.processed.adjusted_cycles.get(|| {
            let filtered = self.filtered_ekg();

            let fs = (filtered.fs as f32).sps();
            let avg_rr = self.avg_rr();

            let pre = fs.s_to_samples(avg_rr / 3.0);
            let post = fs.s_to_samples(avg_rr * 2.0 / 3.0);

            let cycles = self.cycles();

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
                .map(|idx| Cycle {
                    samples: filtered.samples.clone(),
                    start: idx - pre,
                    position: idx,
                    end: idx + post,
                })
                .collect::<Vec<_>>()
        })
    }

    fn average_cycle(&self) -> Ref<'_, Cycle> {
        self.processed.average_cycle.get(|| {
            let adjusted_cycles = self.adjusted_cycles();

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

    fn majority_cycle(&self) -> Ref<'_, Cycle> {
        self.processed.majority_cycle.get(|| {
            let adjusted_cycles = self.adjusted_cycles();

            let avg = self.average_cycle();

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

    fn clear_processed(&mut self) {
        self.processed.clear();
    }
}

fn similarity(corr: f32, max_corr: f32) -> f32 {
    1.0 - (1.0 - corr / max_corr).abs()
}

fn detect_beats(ekg: &[f32], fs: f32) -> (Vec<usize>, Vec<Thresholds>, Vec<f32>) {
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

    (qrs_idxs, thresholds, samples)
}

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
enum Tabs {
    EKG,
    FFT,
    HRV,
    Cycle,
}

struct EkgTuner {
    data: Option<Data>,
    active_tab: Tabs,
}

impl Default for EkgTuner {
    fn default() -> Self {
        Self {
            data: None,
            active_tab: Tabs::EKG,
        }
    }
}

fn apply_filter<F: Filter>(signal: &mut Vec<f32>, mut filter: F) {
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

fn filter_menu(ui: &mut Ui, data: &mut Data) {
    egui::Grid::new("filter_opts").show(ui, |ui| {
        if ui
            .checkbox(&mut data.filter_config.high_pass, "High-pass filter")
            .changed()
        {
            data.clear_processed();
        }

        if ui
            .checkbox(&mut data.filter_config.pli, "PLI filter")
            .changed()
        {
            data.clear_processed();
        }

        if ui
            .checkbox(&mut data.filter_config.low_pass, "Low-pass filter")
            .changed()
        {
            data.clear_processed();
        }

        ui.checkbox(&mut data.hr_debug, "HR debug");
    });
}

fn generate_grid_marks(input: GridInput, scale: f64, steps: &[i32]) -> Vec<GridMark> {
    let mut marks = vec![];

    let (min, max) = input.bounds;
    let min = (min * scale).floor() as i32;
    let max = (max * scale).ceil() as i32;

    for i in min..=max {
        steps
            .iter()
            .copied()
            .find(|step| i % *step == 0)
            .map(|step| {
                marks.push(GridMark {
                    value: i as f64 / scale,
                    step_size: step as f64 / scale,
                })
            });
    }

    marks
}

struct SignalCharts<'a> {
    lines: &'a mut Vec<Line>,
    points: &'a mut Vec<Points>,
    offset: f64,
    fs: f64,
    samples: Range<usize>,
}
impl SignalCharts<'_> {
    fn to_point(&self, (x, y): (usize, f64)) -> [f64; 2] {
        [x as f64 / self.fs, y - self.offset]
    }

    fn push(
        &mut self,
        iter: impl Iterator<Item = f64>,
        color: impl Into<Color32>,
        name: impl ToString,
    ) {
        let line = iter
            .enumerate()
            .map(|p| self.to_point(p))
            .collect::<PlotPoints>();

        self.lines.push(Line::new(line).color(color).name(name));
    }

    fn push_points(
        &mut self,
        iter: impl Iterator<Item = (usize, f64)>,
        color: impl Into<Color32>,
        name: impl ToString,
    ) {
        let points = iter
            .filter(|(idx, _y)| self.samples.contains(idx))
            .map(|(idx, y)| self.to_point((idx - self.samples.start, y)))
            .collect::<PlotPoints>();

        self.points.push(
            Points::new(points)
                .color(color)
                .name(name)
                .shape(MarkerShape::Asterisk)
                .radius(4.0),
        );
    }
}

impl EkgTuner {
    fn ekg_tab(ui: &mut Ui, data: &mut Data) {
        Self::plot_signal(ui, data);
    }

    fn plot_signal(ui: &mut egui::Ui, data: &mut Data) {
        let mut lines = vec![];
        let mut points = vec![];

        let mut bottom = 0.0;
        let mut idx = 0;
        let mut marker_added = false;

        {
            let hr_data = data.hrs();
            let ekg_data = data.filtered_ekg();
            let adjusted_cycles = data.adjusted_cycles();

            let ekg = ekg_data.samples.chunks(6 * 1000);
            let threshold = hr_data.thresholds.chunks(6 * 1000);
            let complex_lead = hr_data.complex_lead.chunks(6 * 1000);

            for (ekg, (threshold, complex_lead)) in ekg.zip(threshold.zip(complex_lead)) {
                let (min, max) = ekg
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), y| {
                        (min.min(*y), max.max(*y))
                    });

                let height = max - min;
                let offset = (max - bottom) as f64;
                bottom -= height + 0.0005; // 500uV margin

                let mut signals = SignalCharts {
                    lines: &mut lines,
                    points: &mut points,
                    offset,
                    fs: ekg_data.fs,
                    samples: idx..idx + ekg.len(),
                };

                if !marker_added {
                    marker_added = true;
                    // to nearest 1mV
                    let marker_y =
                        ((max as f64 - offset) * ekg_data.fs).floor() / ekg_data.fs - 0.001;
                    let marker_x = -0.2;

                    signals.lines.push(
                        Line::new(
                            [
                                [marker_x + -0.04, marker_y + 0.0],
                                [marker_x + 0.0, marker_y + 0.0],
                                [marker_x + 0.0, marker_y + 0.001],
                                [marker_x + 0.16, marker_y + 0.001],
                                [marker_x + 0.16, marker_y + 0.0],
                                [marker_x + 0.2, marker_y + 0.0],
                            ]
                            .into_iter()
                            .collect::<PlotPoints>(),
                        )
                        .color(Color32::from_rgb(100, 200, 100)),
                    );
                }

                signals.push(ekg.iter().map(|y| *y as f64), EKG_COLOR, "EKG");

                if data.hr_debug {
                    signals.push(
                        threshold.iter().map(|y| y.total().unwrap_or(y.r) as f64),
                        Color32::YELLOW,
                        "Threshold",
                    );
                    signals.push(
                        threshold.iter().map(|y| y.m.unwrap_or(f32::NAN) as f64),
                        Color32::WHITE,
                        "M",
                    );
                    signals.push(
                        threshold.iter().map(|y| y.f.unwrap_or(f32::NAN) as f64),
                        Color32::GRAY,
                        "F",
                    );
                    signals.push(threshold.iter().map(|y| y.r as f64), Color32::GREEN, "R");
                    signals.push(
                        complex_lead.iter().map(|y| *y as f64),
                        Color32::LIGHT_RED,
                        "Complex lead",
                    );
                }

                // signals.push_points(
                //     hr_data
                //         .detections
                //         .iter()
                //         .map(|idx| (*idx, ekg_data.samples[*idx] as f64)),
                //     Color32::LIGHT_RED,
                //     "Raw HR",
                // );

                signals.push_points(
                    adjusted_cycles
                        .iter()
                        .map(|cycle| cycle.position)
                        .map(|idx| (idx, ekg_data.samples[idx] as f64)),
                    Color32::LIGHT_GREEN,
                    format!("HR: {}", data.avg_hr().round() as i32),
                );

                idx += ekg.len();
            }
        }

        egui_plot::Plot::new("ekg")
            .legend(Legend::default())
            .show_axes(false)
            .show_grid(true)
            .data_aspect(400.0) // 1 small square = 40ms = 0.1mV
            .allow_scroll(false)
            .boxed_zoom_pointer_button(PointerButton::Middle)
            .x_grid_spacer(|input| generate_grid_marks(input, 1.0 / 0.01, &[20, 4])) // 10ms resolution, 200ms, 40ms
            .y_grid_spacer(|input| generate_grid_marks(input, 1.0 / 0.000_1, &[5, 1])) // 100uV resolution, 500uV, 100uV
            .show(ui, |plot_ui| {
                for line in lines {
                    plot_ui.line(line);
                }
                for points in points {
                    plot_ui.points(points);
                }
            })
            .response
            .context_menu(|ui| filter_menu(ui, data));
    }

    fn fft_tab(ui: &mut Ui, data: &mut Data) {
        let fft = {
            let fft = data.fft();

            Line::new(
                fft.iter()
                    .skip(1 - data.filter_config.high_pass as usize) // skip DC if high-pass is off
                    .take(fft.len() / 2)
                    .enumerate()
                    .map(|(x, y)| [x as f64 * data.raw_ekg.fs / fft.len() as f64, *y as f64])
                    .collect::<PlotPoints>(),
            )
            .color(EKG_COLOR)
            .name("FFT")
        };

        egui_plot::Plot::new("fft")
            .legend(Legend::default())
            .show_axes(false)
            .show_grid(true)
            .auto_bounds_y()
            .allow_scroll(false)
            .allow_zoom(AxisBools { x: false, y: true })
            .show(ui, |plot_ui| {
                plot_ui.line(fft);
            })
            .response
            .context_menu(|ui| filter_menu(ui, data));
    }

    fn hrv_tab(ui: &mut Ui, data: &mut Data) {
        let fs = data.filtered_ekg().fs;
        let hr_data = data.adjusted_cycles();

        // Poincare plot to visualize heart-rate variability
        let rrs = hr_data
            .iter()
            .map(|cycle| cycle.position)
            .map_windows(|[x, y]| ((*y - *x) as f64 / fs) * 1000.0);

        let (min_rr, max_rr) = rrs
            .clone()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), y| {
                (min.min(y), max.max(y))
            });

        egui_plot::Plot::new("hrv")
            .legend(Legend::default())
            .show_axes(true)
            .show_grid(true)
            .include_x(min_rr - 100.0)
            .include_x(max_rr + 100.0)
            .include_y(min_rr - 100.0)
            .include_y(max_rr + 100.0)
            .data_aspect(1.0)
            .allow_scroll(false)
            .show(ui, |plot_ui| {
                plot_ui.points(
                    Points::new(rrs.map_windows(|[x, y]| [*x, *y]).collect::<PlotPoints>())
                        .color(EKG_COLOR),
                );
            });
    }

    fn cycle_tab(ui: &mut Ui, data: &mut Data) {
        let mut lines = vec![];

        let fs = data.filtered_ekg().fs;
        let mut add_cycle = |cycle: &Cycle, name: &str, color: Color32| {
            lines.push(
                Line::new(
                    cycle
                        .samples
                        .iter()
                        .enumerate()
                        .map(|(x, y)| [(x as f64 - cycle.position as f64) / fs, *y as f64])
                        .collect::<PlotPoints>(),
                )
                .color(color)
                .name(name),
            );
        };

        // add_cycle(&data.average_cycle(), "Average cycle", Color32::LIGHT_RED);
        add_cycle(&data.majority_cycle(), "Majority cycle", EKG_COLOR);

        egui_plot::Plot::new("cycle")
            .legend(Legend::default())
            .show_axes(false)
            .show_grid(true)
            .data_aspect(400.0) // 1 small square = 40ms = 0.1mV
            .allow_scroll(false)
            .boxed_zoom_pointer_button(PointerButton::Middle)
            .x_grid_spacer(|input| generate_grid_marks(input, 1.0 / 0.01, &[20, 4])) // 10ms resolution, 200ms, 40ms
            .y_grid_spacer(|input| generate_grid_marks(input, 1.0 / 0.000_1, &[5, 1])) // 100uV resolution, 500uV, 100uV
            .show(ui, |plot_ui| {
                for line in lines {
                    plot_ui.line(line);
                }
            });
    }
}

impl eframe::App for EkgTuner {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open file").clicked() {
                    if let Some(path) = rfd::FileDialog::new().pick_file() {
                        self.data = Data::load(path);
                    }
                }

                if let Some(data) = self.data.as_ref() {
                    ui.label(data.path.display().to_string());
                }
            });
            if let Some(data) = self.data.as_mut() {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.active_tab, Tabs::EKG, "EKG");
                    ui.selectable_value(&mut self.active_tab, Tabs::FFT, "FFT");
                    ui.selectable_value(&mut self.active_tab, Tabs::HRV, "HRV");
                    ui.selectable_value(&mut self.active_tab, Tabs::Cycle, "Cycle info");
                });

                match self.active_tab {
                    Tabs::EKG => Self::ekg_tab(ui, data),
                    Tabs::FFT => Self::fft_tab(ui, data),
                    Tabs::HRV => Self::hrv_tab(ui, data),
                    Tabs::Cycle => Self::cycle_tab(ui, data),
                }
            }
        });
    }
}
