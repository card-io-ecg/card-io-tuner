#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![feature(iter_map_windows)]

use std::{
    cell::{Ref, RefCell},
    env,
    ops::Range,
    path::PathBuf,
    sync::Arc,
};

use eframe::{
    egui::{self, PointerButton, Ui},
    epaint::Color32,
};
use egui_plot::{AxisBools, GridInput, GridMark, Legend, Line, MarkerShape, PlotPoints, Points};
use signal_processing::compressing_buffer::EkgFormat;

use crate::{
    app_config::AppConfig,
    processing::{Config, Context, Cycle, HrData, ProcessedSignal},
};

mod analysis;
mod app_config;
mod data_cell;
mod processing;

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
        let ignore_end = 200;

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

struct Data {
    path: PathBuf,
    processed: ProcessedSignal,
    context: Context,
}

macro_rules! query {
    ($name:ident: $ty:path) => {
        #[allow(dead_code)]
        fn $name(&self) -> Ref<'_, $ty> {
            self.processed.$name(&self.context)
        }
    };
}

impl Data {
    fn load(path: PathBuf) -> Option<Self> {
        log::debug!("Loading {}", path.display());
        std::fs::read(&path).ok().and_then(|bytes| {
            let ekg = Ekg::load(bytes).ok()?;
            Some(Self::new(path, ekg))
        })
    }

    fn new(path: PathBuf, ekg: Ekg) -> Self {
        Self {
            path,
            processed: ProcessedSignal::new(),
            context: Context {
                raw_ekg: ekg,
                config: Config {
                    high_pass: true,
                    pli: true,
                    low_pass: true,
                    hr_debug: false,
                },
            },
        }
    }

    query!(filtered_ekg: Ekg);
    query!(fft: Vec<f32>);
    query!(hrs: HrData);
    query!(rr_intervals: Vec<f64>);
    query!(adjusted_rr_intervals: Vec<f64>);
    query!(cycles: Vec<Cycle>);
    query!(adjusted_cycles: Vec<Cycle>);
    query!(average_cycle: Cycle);
    query!(majority_cycle: Cycle);

    fn avg_hr(&self) -> f64 {
        self.processed.avg_hr(&self.context)
    }

    fn clear_processed(&mut self) {
        self.processed.clear();
    }
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
    config: RefCell<AppConfig>,
}

impl Default for EkgTuner {
    fn default() -> Self {
        Self {
            data: None,
            active_tab: Tabs::EKG,
            config: RefCell::new(AppConfig::load()),
        }
    }
}

fn filter_menu(ui: &mut Ui, data: &mut Data) {
    egui::Grid::new("filter_opts").show(ui, |ui| {
        if ui
            .checkbox(&mut data.context.config.high_pass, "High-pass filter")
            .changed()
        {
            data.clear_processed();
        }

        if ui
            .checkbox(&mut data.context.config.pli, "PLI filter")
            .changed()
        {
            data.clear_processed();
        }

        if ui
            .checkbox(&mut data.context.config.low_pass, "Low-pass filter")
            .changed()
        {
            data.clear_processed();
        }

        ui.checkbox(&mut data.context.config.hr_debug, "HR debug");
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

                if data.context.config.hr_debug {
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

            let x_scale = data.context.raw_ekg.fs / fft.len() as f64;

            Line::new(
                fft.iter()
                    .skip(1 - data.context.config.high_pass as usize) // skip DC if high-pass is off
                    .take(fft.len() / 2)
                    .enumerate()
                    .map(|(x, y)| [x as f64 * x_scale, *y as f64])
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
                        .as_slice()
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
