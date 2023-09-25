#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![feature(iter_map_windows)]

use std::{env, path::PathBuf};

use eframe::egui;
use reqwest::{blocking::Client, redirect::Policy};

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
    fn display(&mut self, ui: &mut egui::Ui, context: &mut AppContext) -> bool;
}

struct AppContext {
    config: AppConfig,
    http_client: Client,
    messages: Vec<AppMessage>,
}

enum AppMessage {
    LoadFile(PathBuf),
}

struct EkgTuner {
    tabs: Vec<Box<dyn AppTab>>,
    selected_tab: usize,
    context: AppContext,
}

impl Default for EkgTuner {
    fn default() -> Self {
        let mut tabs = Vec::new();

        let context = AppContext {
            config: AppConfig::load(),
            http_client: Client::builder()
                .redirect(Policy::limited(3))
                .build()
                .unwrap(),
            messages: Vec::new(),
        };
        tabs.push(RemoteTab::new_boxed());

        Self {
            tabs,
            selected_tab: 0,
            context,
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
                        self.context.messages.push(AppMessage::LoadFile(file));
                    }
                }

                for (i, tab) in self.tabs.iter().enumerate() {
                    ui.selectable_value(&mut self.selected_tab, i, tab.label());
                }
            });

            let close_current = if let Some(tab) = self.tabs.get_mut(self.selected_tab) {
                tab.display(ui, &mut self.context)
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
                    self.context.messages.push(AppMessage::LoadFile(file));
                }
            }
        });

        for message in std::mem::take(&mut self.context.messages) {
            match message {
                AppMessage::LoadFile(file) => self.load(file),
            }
        }
    }
}
