#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![feature(iter_map_windows)]

use std::{cell::Ref, env, path::PathBuf};

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
    analysis::{adjust_time, average_cycle},
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

#[derive(Clone)]
struct Ekg {
    samples: Vec<f32>,
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
            samples: samples
                .get(ignore_start..samples.len() - ignore_end)
                .ok_or(())?
                .to_vec(),
        })
    }
}

struct HrData {
    detections: Vec<usize>,
    thresholds: Vec<Thresholds>,
    complex_lead: Vec<f32>,
    avg_hr: f32,
}

struct ProcessedSignal {
    filtered_ekg: DataCell<Ekg>,
    fft: DataCell<Vec<f32>>,
    hrs: DataCell<HrData>,
    cycles: DataCell<Vec<(usize, Vec<f32>)>>,
    adjusted_cycles: DataCell<Vec<(usize, Vec<f32>)>>,
    majority_cycle: DataCell<Ekg>,
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
                processed: ProcessedSignal {
                    filtered_ekg: DataCell::new(),
                    fft: DataCell::new(),
                    hrs: DataCell::new(),
                    cycles: DataCell::new(),
                    adjusted_cycles: DataCell::new(),
                    majority_cycle: DataCell::new(),
                },
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
            let mut filtered = self.raw_ekg.clone();

            if self.filter_config.pli {
                apply_filter(
                    &mut filtered,
                    &mut PowerLineFilter::<AdaptationBlocking<Sum<1200>, 4, 19>, 1>::new(
                        self.raw_ekg.fs as f32,
                        [50.0],
                    ),
                );
            }

            if self.filter_config.high_pass {
                #[rustfmt::skip]
                let mut high_pass = designfilt!(
                    "highpassiir",
                    "FilterOrder", 2,
                    "HalfPowerFrequency", 0.75,
                    "SampleRate", 1000
                );
                apply_zero_phase_filter(&mut filtered, &mut high_pass);
            }

            if self.filter_config.low_pass {
                #[rustfmt::skip]
                let mut low_pass = designfilt!(
                    "lowpassiir",
                    "FilterOrder", 2,
                    "HalfPowerFrequency", 75,
                    "SampleRate", 1000
                );
                apply_zero_phase_filter(&mut filtered, &mut low_pass);
            }

            filtered
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
            let (qrs_idxs, thresholds, samples, avg_hr) =
                detect_beats(&filtered.samples, filtered.fs as f32);

            HrData {
                detections: qrs_idxs,
                thresholds,
                complex_lead: samples,
                avg_hr,
            }
        })
    }

    fn avg_hr(&self) -> f32 {
        self.hrs().avg_hr
    }

    fn cycles(&self) -> Ref<'_, Vec<(usize, Vec<f32>)>> {
        self.processed.cycles.get(|| {
            let filtered = self.filtered_ekg();
            let hrs = self.hrs();

            let fs = (filtered.fs as f32).sps();
            let avg_rr = 60.0 / self.avg_hr();

            let pre = fs.s_to_samples(avg_rr / 3.0);
            let post = fs.s_to_samples(avg_rr * 2.0 / 3.0);

            hrs.detections
                .iter()
                .copied()
                .filter_map(|idx| {
                    filtered
                        .samples
                        .get(idx - pre..idx + post)
                        .map(|slice| (idx, slice.to_vec()))
                })
                .collect::<Vec<_>>()
        })
    }

    fn adjusted_cycles(&self) -> Ref<'_, Vec<(usize, Vec<f32>)>> {
        self.processed.adjusted_cycles.get(|| {
            let filtered = self.filtered_ekg();

            let fs = (filtered.fs as f32).sps();
            let avg_rr = 60.0 / self.avg_hr();

            let pre = fs.s_to_samples(avg_rr / 3.0);
            let post = fs.s_to_samples(avg_rr * 2.0 / 3.0);

            let cycles = self.cycles();

            let cycle_idxs = cycles.iter().map(|(idx, _)| *idx);
            let all_cycles = cycles.iter().map(|(_, cycle)| cycle.as_slice());

            let all_average = average_cycle(all_cycles.clone());

            // For QRS adjustment, we're using the 50-50 ms window around the peak of the QRS
            let avg_qrs_width = fs.ms_to_samples(25.0);
            let avg_qrs = &all_average[pre - avg_qrs_width..][..(2 * avg_qrs_width)];

            let adjusted_idxs = cycle_idxs.map(|idx| {
                (idx as isize
                    + adjust_time(
                        &filtered.samples[idx - (avg_qrs_width * 2)..idx + (avg_qrs_width * 2)],
                        &avg_qrs,
                    )) as usize
            });

            adjusted_idxs
                .map(|idx| (idx, filtered.samples[idx - pre..idx + post].to_vec()))
                .collect::<Vec<_>>()
        })
    }

    fn majority_cycle(&self) -> Ref<'_, Ekg> {
        self.processed.majority_cycle.get(|| {
            let filtered = self.filtered_ekg();
            let adjusted_cycles = self.adjusted_cycles();

            Ekg {
                samples: average_cycle(adjusted_cycles.iter().map(|(_, cycle)| cycle.as_slice())),
                fs: filtered.fs,
            }
        })
    }

    fn clear_processed(&mut self) {
        self.processed.filtered_ekg.clear();
        self.processed.fft.clear();
        self.processed.hrs.clear();
        self.processed.majority_cycle.clear();
    }
}

fn detect_beats(ekg: &[f32], fs: f32) -> (Vec<usize>, Vec<Thresholds>, Vec<f32>, f32) {
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

    let avg_hr = qrs_idxs
        .iter()
        .copied()
        .map_windows(|[a, b]| *b - *a)
        .map(|diff| 60.0 * fs as f32 / diff as f32)
        .sum::<f32>()
        / (qrs_idxs.len() as f32 - 1.0);

    (qrs_idxs, thresholds, samples, avg_hr)
}

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
enum Tabs {
    EKG,
    FFT,
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

fn apply_filter<F: Filter>(signal: &mut Ekg, filter: &mut F) {
    signal.samples = signal
        .samples
        .iter()
        .copied()
        .filter_map(|sample| filter.update(sample))
        .collect::<Vec<_>>();
}

fn apply_zero_phase_filter<F: Filter>(signal: &mut Ekg, filter: &mut F) {
    apply_filter(signal, filter);
    signal.samples.reverse();

    filter.clear();

    apply_filter(signal, filter);
    signal.samples.reverse();
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

impl EkgTuner {
    fn ekg_tab(ui: &mut Ui, data: &mut Data) {
        Self::plot_signal(ui, data);
    }

    fn plot_signal(ui: &mut egui::Ui, data: &mut Data) {
        let mut lines = vec![];
        let mut hrs = vec![];

        let mut bottom = 0.0;
        let mut idx = 0;
        let mut marker_added = false;

        {
            let hr_data = data.hrs();
            let ekg_data = data.filtered_ekg();

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
                let offset = max - bottom;
                bottom -= height + 0.0005; // 500uV margin

                if !marker_added {
                    marker_added = true;
                    // to nearest 1mV
                    let marker_y =
                        ((max - offset) as f64 * ekg_data.fs).floor() / ekg_data.fs - 0.001;
                    let marker_x = -0.2;

                    lines.push(
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

                lines.push(
                    Line::new(
                        ekg.iter()
                            .enumerate()
                            .map(|(x, y)| [x as f64 / ekg_data.fs, (*y - offset) as f64])
                            .collect::<PlotPoints>(),
                    )
                    .color(Color32::from_rgb(100, 150, 250))
                    .name("EKG"),
                );

                hrs.push(
                    Points::new(
                        hr_data
                            .detections
                            .iter()
                            .filter_map(|hr_idx| {
                                let hr_idx = *hr_idx as usize;
                                if (idx..idx + ekg.len()).contains(&hr_idx) {
                                    let x = hr_idx - idx;
                                    let y = ekg[x] as f64 - offset as f64;
                                    Some([x as f64 / ekg_data.fs, y])
                                } else {
                                    None
                                }
                            })
                            .collect::<PlotPoints>(),
                    )
                    .color(Color32::LIGHT_RED)
                    .shape(MarkerShape::Asterisk)
                    .radius(4.0)
                    .name(format!("HR: {}", hr_data.avg_hr.round() as i32)),
                );

                if data.hr_debug {
                    lines.push(
                        Line::new(
                            threshold
                                .iter()
                                .enumerate()
                                .map(|(x, y)| {
                                    [
                                        x as f64 / ekg_data.fs,
                                        (y.total().unwrap_or(y.r) - offset) as f64,
                                    ]
                                })
                                .collect::<PlotPoints>(),
                        )
                        .color(Color32::YELLOW)
                        .name("Threshold"),
                    );
                    lines.push(
                        Line::new(
                            threshold
                                .iter()
                                .enumerate()
                                .map(|(x, y)| {
                                    [
                                        x as f64 / ekg_data.fs,
                                        (y.m.unwrap_or(f32::NAN) - offset) as f64,
                                    ]
                                })
                                .collect::<PlotPoints>(),
                        )
                        .color(Color32::WHITE)
                        .name("M"),
                    );
                    lines.push(
                        Line::new(
                            threshold
                                .iter()
                                .enumerate()
                                .map(|(x, y)| {
                                    [
                                        x as f64 / ekg_data.fs,
                                        (y.f.unwrap_or(f32::NAN) - offset) as f64,
                                    ]
                                })
                                .collect::<PlotPoints>(),
                        )
                        .color(Color32::GRAY)
                        .name("F"),
                    );
                    lines.push(
                        Line::new(
                            threshold
                                .iter()
                                .enumerate()
                                .map(|(x, y)| [x as f64 / ekg_data.fs, (y.r - offset) as f64])
                                .collect::<PlotPoints>(),
                        )
                        .color(Color32::GREEN)
                        .name("R"),
                    );

                    lines.push(
                        Line::new(
                            complex_lead
                                .iter()
                                .enumerate()
                                .map(|(x, y)| [x as f64 / ekg_data.fs, (*y - offset) as f64])
                                .collect::<PlotPoints>(),
                        )
                        .color(Color32::LIGHT_RED)
                        .name("Complex lead"),
                    );
                }

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
                for points in hrs {
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
            .color(Color32::from_rgb(100, 150, 250))
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

    fn cycle_tab(ui: &mut Ui, data: &mut Data) {
        let mut lines = vec![];

        let cycle = data.majority_cycle();

        lines.push(
            Line::new(
                cycle
                    .samples
                    .iter()
                    .enumerate()
                    .map(|(x, y)| [x as f64 / cycle.fs, *y as f64])
                    .collect::<PlotPoints>(),
            )
            .color(Color32::from_rgb(100, 150, 250))
            .name("Majority cycle"),
        );

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
                    ui.selectable_value(&mut self.active_tab, Tabs::Cycle, "Cycle info");
                });

                match self.active_tab {
                    Tabs::EKG => Self::ekg_tab(ui, data),
                    Tabs::FFT => Self::fft_tab(ui, data),
                    Tabs::Cycle => Self::cycle_tab(ui, data),
                }
            }
        });
    }
}
