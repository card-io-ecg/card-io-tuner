#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![feature(iter_map_windows)]

use std::{env, path::PathBuf};

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
    heart_rate::{HeartRateCalculator, Thresholds},
    moving::sum::Sum,
};

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
            samples: samples
                .get(ignore_start..samples.len() - ignore_end)
                .ok_or(())?
                .to_vec(),
        })
    }
}

struct Data {
    path: PathBuf,
    raw_ekg: Ekg,
    fs: f64,
    filtered_ekg: Option<Ekg>,
    fft: Option<Vec<f32>>,
    hrs: Option<Vec<usize>>,
    hr_thresholds: Option<Vec<Thresholds>>,
    hr_complex_lead: Option<Vec<f32>>,
    avg_hr: f32,

    high_pass: bool,
    pli: bool,
    low_pass: bool,
    hr_debug: bool,
}

impl Data {
    fn load(path: PathBuf) -> Option<Self> {
        log::debug!("Loading {}", path.display());
        std::fs::read(&path).ok().and_then(|bytes| {
            let ekg = Ekg::load(bytes).ok()?;
            Some(Data {
                path,
                fs: 1000.0,
                raw_ekg: ekg,
                filtered_ekg: None,
                fft: None,
                hrs: None,
                hr_thresholds: None,
                hr_complex_lead: None,
                avg_hr: 0.0,
                high_pass: true,
                pli: true,
                low_pass: true,
                hr_debug: false,
            })
        })
    }

    fn update(&mut self) {
        if self.filtered_ekg.is_none() {
            let mut filtered = self.raw_ekg.clone();

            if self.pli {
                apply_filter(
                    &mut filtered,
                    &mut PowerLineFilter::<AdaptationBlocking<Sum<1200>, 4, 19>, 1>::new(
                        self.fs as f32,
                        [50.0],
                    ),
                );
            }

            if self.high_pass {
                #[rustfmt::skip]
                let mut high_pass = designfilt!(
                    "highpassiir",
                    "FilterOrder", 2,
                    "HalfPowerFrequency", 0.75,
                    "SampleRate", 1000
                );
                apply_zero_phase_filter(&mut filtered, &mut high_pass);
            }

            if self.low_pass {
                #[rustfmt::skip]
                let mut low_pass = designfilt!(
                    "lowpassiir",
                    "FilterOrder", 2,
                    "HalfPowerFrequency", 75,
                    "SampleRate", 1000
                );
                apply_zero_phase_filter(&mut filtered, &mut low_pass);
            }

            self.filtered_ekg = Some(filtered);
            self.fft = None;
            self.hrs = None;
        }

        if self.fft.is_none() {
            let mut samples = self
                .filtered_ekg
                .as_ref()
                .unwrap()
                .samples
                .iter()
                .copied()
                .map(|y| Complex { re: y, im: 0.0 })
                .collect::<Vec<_>>();

            let mut planner = rustfft::FftPlanner::new();
            let fft = planner.plan_fft_forward(samples.len());

            fft.process(&mut samples);

            let fft = samples.iter().copied().map(|c| c.abs()).collect::<Vec<_>>();
            self.fft = Some(fft);
        }

        if self.hrs.is_none() {
            const IGNORE_SAMPLES: usize = 500;
            let (qrs_idxs, mut thresholds, mut samples, avg_hr) = detect_beats(
                &self.filtered_ekg.as_ref().unwrap().samples[IGNORE_SAMPLES..],
                self.fs as f32,
            );

            for _ in 0..IGNORE_SAMPLES + 58 {
                thresholds.insert(
                    0,
                    Thresholds {
                        m: None,
                        f: None,
                        r: 0.0,
                    },
                );
                samples.insert(0, f32::NAN);
            }

            self.avg_hr = avg_hr;
            self.hrs = Some(qrs_idxs.into_iter().map(|hr| hr + IGNORE_SAMPLES).collect());
            self.hr_thresholds = Some(thresholds);
            self.hr_complex_lead = Some(samples);
        }
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
            .checkbox(&mut data.high_pass, "High-pass filter")
            .changed()
        {
            data.filtered_ekg = None;
        }

        if ui.checkbox(&mut data.pli, "PLI filter").changed() {
            data.filtered_ekg = None;
        }

        if ui.checkbox(&mut data.low_pass, "Low-pass filter").changed() {
            data.filtered_ekg = None;
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
        let mut marker = None;

        let mut lines = vec![];
        let mut hrs = vec![];

        let mut bottom = 0.0;
        let mut idx = 0;

        let ekg = data.filtered_ekg.as_ref().unwrap().samples.chunks(6 * 1000);
        let threshold = data.hr_thresholds.as_ref().unwrap().chunks(6 * 1000);
        let complex_lead = data.hr_complex_lead.as_ref().unwrap().chunks(6 * 1000);

        for (ekg, (threshold, complex_lead)) in ekg.zip(threshold.zip(complex_lead)) {
            let (min, max) = ekg
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), y| {
                    (min.min(*y), max.max(*y))
                });

            let height = max - min;
            let offset = max - bottom;
            bottom -= height + 0.0005; // 500uV margin

            if marker.is_none() {
                // to nearest 1mV
                let min = ((min - offset) as f64 * data.fs).ceil() / data.fs;

                marker = Some(
                    Line::new(
                        [
                            [-0.04, min + 0.0],
                            [0.0, min + 0.0],
                            [0.0, min + 0.001],
                            [0.16, min + 0.001],
                            [0.16, min + 0.0],
                            [0.2, min + 0.0],
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
                        .map(|(x, y)| [x as f64 / data.fs, (*y - offset) as f64])
                        .collect::<PlotPoints>(),
                )
                .color(Color32::from_rgb(100, 150, 250))
                .name("EKG"),
            );

            if data.hr_debug {
                lines.push(
                    Line::new(
                        threshold
                            .iter()
                            .enumerate()
                            .map(|(x, y)| {
                                [
                                    x as f64 / data.fs,
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
                        complex_lead
                            .iter()
                            .enumerate()
                            .map(|(x, y)| [x as f64 / data.fs, (*y - offset) as f64])
                            .collect::<PlotPoints>(),
                    )
                    .color(Color32::LIGHT_RED)
                    .name("Complex lead"),
                );
            }

            hrs.push(
                Points::new(
                    data.hrs
                        .as_ref()
                        .unwrap()
                        .iter()
                        .filter_map(|hr_idx| {
                            let hr_idx = *hr_idx as usize;
                            if (idx..idx + ekg.len()).contains(&hr_idx) {
                                let x = hr_idx - idx;
                                let y = ekg[x] as f64 - offset as f64;
                                Some([x as f64 / data.fs, y])
                            } else {
                                None
                            }
                        })
                        .collect::<PlotPoints>(),
                )
                .color(Color32::LIGHT_RED)
                .shape(MarkerShape::Asterisk)
                .radius(4.0)
                .name(format!("HR: {}", data.avg_hr.round() as i32)),
            );

            idx += ekg.len();
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
                if let Some(marker) = marker {
                    plot_ui.line(marker);
                }
            })
            .response
            .context_menu(|ui| filter_menu(ui, data));
    }

    fn fft_tab(ui: &mut Ui, data: &mut Data) {
        let fft = data.fft.as_ref().unwrap();

        let fft = Line::new(
            fft.iter()
                .skip(1 - data.high_pass as usize) // skip DC if high-pass is off
                .take(fft.len() / 2)
                .enumerate()
                .map(|(x, y)| [x as f64 * data.fs / fft.len() as f64, *y as f64])
                .collect::<PlotPoints>(),
        )
        .color(Color32::from_rgb(100, 150, 250))
        .name("FFT");

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
                });

                data.update();

                match self.active_tab {
                    Tabs::EKG => Self::ekg_tab(ui, data),
                    Tabs::FFT => Self::fft_tab(ui, data),
                }
            }
        });
    }
}
