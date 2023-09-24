#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![feature(iter_map_windows)]

use std::{
    cell::{Ref, RefCell},
    env,
    path::PathBuf,
    sync::Arc,
};

use eframe::egui;
use signal_processing::compressing_buffer::EkgFormat;

use crate::{
    app_config::AppConfig,
    processing::{Config, Context, Cycle, HrData, ProcessedSignal},
    signal_tab::SignalTab,
};

mod analysis;
mod app_config;
mod data_cell;
mod processing;
mod signal_tab;

fn main() -> Result<(), eframe::Error> {
    env::set_var("RUST_LOG", "card_io_tuner=debug");
    env_logger::init();

    eframe::run_native(
        "EKG visualizer and filter tuner",
        eframe::NativeOptions {
            drag_and_drop_support: true,
            initial_window_size: Some(egui::vec2(640.0, 480.0)),
            ..Default::default()
        },
        Box::new(|_cc| Box::<EkgTuner>::default()),
    )
}

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

struct EkgTuner {
    tabs: Vec<SignalTab>,
    selected_tab: usize,
    config: RefCell<AppConfig>,
}

impl Default for EkgTuner {
    fn default() -> Self {
        Self {
            tabs: Vec::new(),
            selected_tab: 0,
            config: RefCell::new(AppConfig::load()),
        }
    }
}

impl EkgTuner {
    fn load(&mut self, path: PathBuf) {
        if let Some(data) = Data::load(path) {
            self.tabs.push(SignalTab::new(data));
            self.selected_tab = self.tabs.len() - 1;
        }
    }
}

impl eframe::App for EkgTuner {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open file").clicked() {
                    if let Some(file) = rfd::FileDialog::new().pick_file() {
                        self.load(file);
                    }
                }

                for (i, _tab) in self.tabs.iter().enumerate() {
                    ui.selectable_value(&mut self.selected_tab, i, format!("{i}"));
                }
            });

            let close_current = if let Some(tab) = self.tabs.get_mut(self.selected_tab) {
                tab.display(ui)
            } else {
                false
            };

            if close_current {
                self.tabs.remove(self.selected_tab);
                self.selected_tab = self.selected_tab.min(self.tabs.len().saturating_sub(1));
            }
        });

        ctx.input(|i| {
            if !i.raw.dropped_files.is_empty() {
                if let Some(file) = i.raw.dropped_files[0].path.clone() {
                    self.load(file);
                }
            }
        });
    }
}
