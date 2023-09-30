#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![feature(iter_map_windows)]

use std::{env, path::PathBuf};

use eframe::egui::{self, CentralPanel, Id, TopBottomPanel};
use egui_dock::{DockArea, DockState, Style};

use crate::{
    app_config::AppConfig,
    app_context::AppContext,
    data::Data,
    tabs::{remote::RemoteTab, signal_tab::SignalTab},
};

mod analysis;
mod app_config;
mod app_context;
mod data;
mod tabs;
mod ui;

fn main() -> Result<(), eframe::Error> {
    env::set_var("RUST_LOG", "card_io_tuner=debug");
    env_logger::init();

    eframe::run_native(
        "EKG visualizer",
        eframe::NativeOptions {
            drag_and_drop_support: true,
            initial_window_size: Some(egui::vec2(640.0, 480.0)),
            ..Default::default()
        },
        Box::new(|_cc| Box::<EkgTuner>::default()),
    )
}

trait AppTab {
    fn id(&self) -> Id;
    fn label(&self) -> &str;
    fn display(&mut self, ui: &mut egui::Ui, context: &mut AppContext) -> bool;
}

impl PartialEq for Box<dyn AppTab> {
    fn eq(&self, other: &Self) -> bool {
        self.label() == other.label()
    }
}

enum AppMessage {
    LoadFile(PathBuf),
}

struct TabViewer<'a> {
    context: &'a mut AppContext,
}

impl egui_dock::TabViewer for TabViewer<'_> {
    type Tab = Box<dyn AppTab>;

    fn id(&mut self, tab: &mut Self::Tab) -> Id {
        tab.id()
    }

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        tab.label().into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        tab.display(ui, self.context);
    }
}

struct EkgTuner {
    tree: DockState<Box<dyn AppTab>>,
    context: AppContext,
}

impl Default for EkgTuner {
    fn default() -> Self {
        let tree = DockState::new(vec![]);

        let context = AppContext::new(AppConfig::load());

        Self { tree, context }
    }
}

impl EkgTuner {
    fn load(&mut self, path: PathBuf) {
        let title = path.file_name().unwrap().to_string_lossy().to_string();

        if let Some(data) = Data::load(&path) {
            self.tree
                .push_to_focused_leaf(SignalTab::new_boxed(title, data));
        }
    }
}

impl eframe::App for EkgTuner {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        TopBottomPanel::top("egui_dock::MenuBar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                if ui.button("Open file").clicked() {
                    if let Some(file) = rfd::FileDialog::new().pick_file() {
                        self.context.send_message(AppMessage::LoadFile(file));
                    }
                }
                if ui.button("Remote").clicked() {
                    let remote = RemoteTab::new_boxed();
                    if let Some(tab) = self.tree.find_tab(&remote) {
                        self.tree.set_active_tab(tab);
                    } else {
                        self.tree.push_to_focused_leaf(remote);
                    }
                }
            })
        });
        CentralPanel::default().show(ctx, |ui| {
            DockArea::new(&mut self.tree)
                .style(Style::from_egui(ui.ctx().style().as_ref()))
                .show(
                    ctx,
                    &mut TabViewer {
                        context: &mut self.context,
                    },
                );
        });

        ctx.input(|i| {
            if !i.raw.dropped_files.is_empty() {
                if let Some(file) = i.raw.dropped_files[0].path.clone() {
                    self.context.send_message(AppMessage::LoadFile(file));
                }
            }
        });

        for message in self.context.take_messages() {
            match message {
                AppMessage::LoadFile(file) => self.load(file),
            }
        }
    }
}
