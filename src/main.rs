#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![feature(iter_map_windows)]

use std::{cell::RefCell, env, path::PathBuf, rc::Rc};

use eframe::egui;

use crate::{
    app_config::AppConfig,
    data::Data,
    tabs::{remote::RemoteTab, signal_tab::SignalTab},
};

mod analysis;
mod app_config;
mod data;
mod tabs;

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

trait AppTab {
    fn label(&self) -> &str;
    fn display(&mut self, ui: &mut egui::Ui) -> bool;
}

struct EkgTuner {
    tabs: Vec<Box<dyn AppTab>>,
    selected_tab: usize,
    config: Rc<RefCell<AppConfig>>,
}

impl Default for EkgTuner {
    fn default() -> Self {
        let mut tabs = Vec::new();

        let config = Rc::new(RefCell::new(AppConfig::load()));
        tabs.push(RemoteTab::new_boxed(&config));

        Self {
            tabs,
            selected_tab: 0,
            config,
        }
    }
}

impl EkgTuner {
    fn load(&mut self, path: PathBuf) {
        if let Some(data) = Data::load(&path) {
            self.tabs.push(SignalTab::new_boxed(
                path.file_name().unwrap().to_string_lossy().to_string(),
                data,
            ));
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

                for (i, tab) in self.tabs.iter().enumerate() {
                    ui.selectable_value(&mut self.selected_tab, i, tab.label());
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
