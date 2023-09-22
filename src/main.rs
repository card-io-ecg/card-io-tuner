#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![feature(iter_map_windows)]

use std::{env, path::PathBuf};

use eframe::{
    egui::{self, PointerButton, Ui},
    epaint::Color32,
};
use egui_plot::{AxisBools, GridMark, Legend, Line, MarkerShape, PlotPoints, Points};
use rustfft::num_complex::{Complex, ComplexFloat};
use signal_processing::{
    compressing_buffer::EkgFormat,
    designfilt,
    filter::{
        iir::{HighPass, Iir, LowPass},
        pli::{adaptation_blocking::AdaptationBlocking, PowerLineFilter},
        Filter,
    },
    heart_rate::HeartRateCalculator,
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

        let ignore_start = 2000;
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
    avg_hr: f32,
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
                fs: 1000.0,
                raw_ekg: ekg,
                filtered_ekg: None,
                fft: None,
                hrs: None,
                avg_hr: 0.0,
                high_pass: true,
                pli: true,
                low_pass: true,
            })
        })
    }
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
        for section in data.filtered_ekg.as_ref().unwrap().samples.chunks(6 * 1000) {
            let (min, max) = section
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
                    section
                        .iter()
                        .enumerate()
                        .map(|(x, y)| [x as f64 / data.fs, (*y - offset) as f64])
                        .collect::<PlotPoints>(),
                )
                .color(Color32::from_rgb(100, 150, 250))
                .name("EKG"),
            );

            hrs.push(
                Points::new(
                    data.hrs
                        .as_ref()
                        .unwrap()
                        .iter()
                        .filter_map(|hr_idx| {
                            let hr_idx = *hr_idx as usize;
                            if (idx..idx + section.len()).contains(&hr_idx) {
                                let x = hr_idx - idx;
                                let y = section[x] as f64 - offset as f64;
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

            idx += section.len();
        }

        egui_plot::Plot::new("ekg")
            .legend(Legend::default())
            .show_axes(false)
            .show_grid(true)
            .data_aspect(400.0) // 1 small square = 40ms = 0.1mV
            .allow_scroll(false)
            .boxed_zoom_pointer_button(PointerButton::Middle)
            .x_grid_spacer(|input| {
                let mut marks = vec![];

                const SCALE: f64 = 100.0;

                let (min, max) = input.bounds;
                let min = (min * SCALE).floor() as i32;
                let max = (max * SCALE).ceil() as i32;

                for i in min..=max {
                    let step_size = if i % 20 == 0 {
                        0.200 // 200ms big square
                    } else if i % 4 == 0 {
                        0.04 // 40ms small square
                    } else {
                        continue;
                    };

                    marks.push(GridMark {
                        value: i as f64 / SCALE,
                        step_size,
                    });
                }

                marks
            })
            .y_grid_spacer(|input| {
                let mut marks = vec![];

                const SCALE: f64 = 1_0000.0;

                let (min, max) = input.bounds;
                let min = (min * SCALE).floor() as i32;
                let max = (max * SCALE).ceil() as i32;

                for i in min..=max {
                    let step_size = if i % 5 == 0 {
                        0.0005 // 500uV big square
                    } else if i % 1 == 0 {
                        0.0001 // 100uV small square
                    } else {
                        continue;
                    };

                    marks.push(GridMark {
                        value: i as f64 / SCALE,
                        step_size,
                    });
                }

                marks
            })
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
            .context_menu(|ui| {
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
                });
            });
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
            .context_menu(|ui| {
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
                });
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
                });

                if data.filtered_ekg.is_none() {
                    let mut filtered = data.raw_ekg.clone();

                    if data.pli {
                        apply_filter(
                            &mut filtered,
                            &mut PowerLineFilter::<AdaptationBlocking<Sum<1200>, 4, 19>, 1>::new(
                                data.fs as f32,
                                [50.0],
                            ),
                        );
                    }

                    if data.high_pass {
                        #[rustfmt::skip]
                        let mut high_pass = designfilt!(
                            "highpassiir",
                            "FilterOrder", 2,
                            "HalfPowerFrequency", 0.75,
                            "SampleRate", 1000
                        );
                        apply_zero_phase_filter(&mut filtered, &mut high_pass);
                    }

                    if data.low_pass {
                        #[rustfmt::skip]
                        let mut low_pass = designfilt!(
                            "lowpassiir",
                            "FilterOrder", 2,
                            "HalfPowerFrequency", 75,
                            "SampleRate", 1000
                        );
                        apply_zero_phase_filter(&mut filtered, &mut low_pass);
                    }

                    data.filtered_ekg = Some(filtered);
                    data.fft = None;
                    data.hrs = None;
                }

                if data.fft.is_none() {
                    let mut samples = data
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
                    data.fft = Some(fft);
                }

                if data.hrs.is_none() {
                    let mut calculator = HeartRateCalculator::new(data.fs as f32);

                    let mut hrs = Vec::new();
                    const DELAY: usize = 57; // The delay of the FIR filter in the calculator
                    for sample in data.filtered_ekg.as_ref().unwrap().samples.iter() {
                        if let Some(idx) = calculator.update(*sample) {
                            hrs.push(idx - DELAY);
                        }
                    }

                    let avg_hr = hrs
                        .iter()
                        .copied()
                        .map_windows(|[a, b]| *b - *a)
                        .map(|diff| 60.0 * data.fs as f32 / diff as f32)
                        .sum::<f32>()
                        / (hrs.len() as f32 - 1.0);

                    data.avg_hr = avg_hr;
                    data.hrs = Some(hrs);
                }

                match self.active_tab {
                    Tabs::EKG => Self::ekg_tab(ui, data),
                    Tabs::FFT => Self::fft_tab(ui, data),
                }
            }
        });
    }
}
