#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::{env, path::PathBuf};

use eframe::{
    egui::{self, Ui},
    epaint::Color32,
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

struct Ekg {
    samples: Vec<f64>,
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
        fn load_varint_diff(bytes: &mut &[u8], lsb: f64) -> Result<f64, ()> {
            const fn zigzag_decode(val: u32) -> i32 {
                (val >> 1) as i32 ^ -((val & 1) as i32)
            }

            fn pop(bytes: &mut &[u8]) -> Option<u8> {
                if bytes.is_empty() {
                    return None;
                }
                let byte = bytes[0];
                *bytes = &bytes[1..];
                Some(byte)
            }

            let mut diff = 0;
            let mut idx = 0;
            while let Some(byte) = pop(bytes) {
                diff |= ((byte & 0x7F) as u32) << (idx * 7);
                idx += 1;
                if byte & 0x80 == 0 {
                    break;
                }
            }
            let diff = zigzag_decode(diff);

            Ok(diff as f64 * lsb)
        }

        let mut samples = Vec::new();

        let mut last_sample = 0.0;
        pub const VOLTS_PER_LSB: f64 = -2.42 / (1 << 23) as f64; // ADS129x

        while !bytes.is_empty() {
            let diff = load_varint_diff(&mut bytes, VOLTS_PER_LSB)?;
            samples.push(last_sample + diff);
            last_sample += diff;
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
}
impl Data {
    fn load(path: PathBuf) -> Option<Self> {
        log::debug!("Loading {}", path.display());
        std::fs::read(&path).ok().and_then(|bytes| {
            Some(Data {
                path,
                raw_ekg: Ekg::load(bytes).ok()?,
            })
        })
    }
}

struct EkgTuner {
    data: Option<Data>,
    active_tab: Tabs,
}

impl Default for EkgTuner {
    fn default() -> Self {
        Self {
            data: None,
            active_tab: Tabs::First,
        }
    }
}

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
enum Tabs {
    First,
}

impl EkgTuner {
    fn first_tab(ui: &mut Ui, data: &mut Data) {
        ui.label(format!("Path: {}", data.path.display()));
        Self::plot_signal(ui, &data.raw_ekg.samples);
    }

    fn plot_signal(ui: &mut egui::Ui, ekg: &[f64]) -> egui::Response {
        use egui_plot::{AxisBools, GridMark, Line, PlotPoints};

        let mut marker = None;

        let mut lines = vec![];
        let mut bottom = 0.0;
        for section in ekg.chunks(10 * 1000) {
            let (min, max) = section
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), y| {
                    (min.min(*y), max.max(*y))
                });

            let height = max - min;
            let offset = max - bottom;
            bottom -= height + 0.001; // 1mV margin

            if marker.is_none() {
                // to nearest 1mV
                let min = ((min - offset) * 1000.0).ceil() / 1000.0;

                marker = Some(
                    Line::new(
                        [
                            [-0.04 * 1000.0, min + 0.0],
                            [0.0 * 1000.0, min + 0.0],
                            [0.0 * 1000.0, min + 0.001],
                            [0.16 * 1000.0, min + 0.001],
                            [0.16 * 1000.0, min + 0.0],
                            [0.2 * 1000.0, min + 0.0],
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
                        .map(|(x, y)| [x as f64, *y - offset])
                        .collect::<PlotPoints>(),
                )
                .color(Color32::from_rgb(100, 150, 250)),
            );
        }

        egui_plot::Plot::new("ekg")
            .show_axes([false, false])
            .show_grid(true)
            .auto_bounds_y()
            .allow_scroll(false)
            .allow_zoom(AxisBools { x: false, y: true })
            .x_grid_spacer(|input| {
                let mut marks = vec![];

                let (min, max) = input.bounds;
                let min = min.floor() as i32;
                let max = max.ceil() as i32;

                for i in min..=max {
                    let step_size = if i % 200 == 0 {
                        200.0 // 200ms big square
                    } else if i % 40 == 0 {
                        40.0 // 40ms small square
                    } else {
                        continue;
                    };

                    marks.push(GridMark {
                        value: i as f64,
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
                if let Some(marker) = marker {
                    plot_ui.line(marker);
                }
            })
            .response
    }
}

impl eframe::App for EkgTuner {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if ui.button("Open file…").clicked() {
                if let Some(path) = rfd::FileDialog::new().pick_file() {
                    self.data = Data::load(path);
                }
            }
            if let Some(data) = self.data.as_mut() {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.active_tab, Tabs::First, "First");
                });

                match self.active_tab {
                    Tabs::First => Self::first_tab(ui, data),
                }
            }
        });
    }
}
