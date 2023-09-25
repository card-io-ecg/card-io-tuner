use std::{cell::RefCell, io::Read, rc::Rc};

use eframe::{
    egui::{self, Layout, TextEdit, Ui},
    emath::Align,
};
use reqwest::{blocking::Client, redirect::Policy};
use serde_json::json;

use crate::{app_config::AppConfig, AppTab};

// Copied from egui examples
pub fn password_ui(ui: &mut egui::Ui, password: &mut String) -> egui::Response {
    // This widget has its own state — show or hide password characters (`show_plaintext`).
    // In this case we use a simple `bool`, but you can also declare your own type.
    // It must implement at least `Clone` and be `'static`.
    // If you use the `persistence` feature, it also must implement `serde::{Deserialize, Serialize}`.

    // Generate an id for the state
    let state_id = ui.id().with("show_plaintext");

    // Get state for this widget.
    // You should get state by value, not by reference to avoid borrowing of [`Memory`].
    let mut show_plaintext = ui.data_mut(|d| d.get_temp::<bool>(state_id).unwrap_or(false));

    // Process ui, change a local copy of the state
    // We want TextEdit to fill entire space, and have button after that, so in that case we can
    // change direction to right_to_left.
    let result = ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
        // Show the password field:
        ui.add(TextEdit::singleline(password).password(!show_plaintext));

        // Toggle the `show_plaintext` bool with a button:
        if ui
            .add(egui::SelectableLabel::new(show_plaintext, "👁"))
            .on_hover_text("Show/hide password")
            .clicked()
        {
            show_plaintext = !show_plaintext;
        }
    });

    // Store the (possibly changed) state:
    ui.data_mut(|d| d.insert_temp(state_id, show_plaintext));

    // All done! Return the interaction response so the user can check what happened
    // (hovered, clicked, …) and maybe show a tooltip:
    result.response
}

pub fn password(password: &mut String) -> impl egui::Widget + '_ {
    move |ui: &mut egui::Ui| password_ui(ui, password)
}

struct LoginData {
    username: String,
    password: String,
}

impl LoginData {
    fn new() -> Self {
        Self {
            username: String::new(),
            password: String::new(),
        }
    }

    fn display(&mut self, ui: &mut Ui, config: &RefCell<AppConfig>) {
        ui.with_layout(Layout::top_down(Align::Center), |ui| {
            ui.set_max_width(400.0);
            ui.group(|ui| {
                ui.heading("Log in to remote server");

                egui::Grid::new("login").num_columns(2).show(ui, |ui| {
                    ui.label("Name:");
                    ui.text_edit_singleline(&mut self.username);

                    ui.end_row();

                    ui.label("Password:");
                    ui.add(password(&mut self.password));
                });

                if ui.button("Sign in").clicked() {
                    let client_builder = Client::builder()
                        .redirect(Policy::limited(3))
                        .build()
                        .unwrap();

                    let mut jwt = String::new();
                    _ = client_builder
                        .post(config.borrow().backend_url("login"))
                        .json(&json!({
                            "username": &self.username,
                            "password": &self.password,
                        }))
                        .send()
                        .unwrap()
                        .read_to_string(&mut jwt);

                    let response = client_builder
                        .get(config.borrow().backend_url("validate"))
                        .header("Authorization", &format!("Bearer {jwt}"))
                        .send()
                        .unwrap();

                    if response.status().is_success() {
                        log::info!("Logged in. Token: {}", jwt);
                    } else {
                        log::error!(
                            "Failed to validate token: {:?}",
                            response.json::<Error>().unwrap().error
                        );
                    }
                }
            });
        });
    }
}

#[derive(serde::Deserialize)]
struct Error {
    pub error: String,
}

enum RemoteState {
    Login(LoginData),
}

impl RemoteState {
    fn display(&mut self, ui: &mut Ui, config: &RefCell<AppConfig>) {
        match self {
            Self::Login(data) => data.display(ui, config),
        }
    }
}

pub struct RemoteTab {
    state: RemoteState,
    config: Rc<RefCell<AppConfig>>,
}

impl RemoteTab {
    pub fn new_boxed(config: &Rc<RefCell<AppConfig>>) -> Box<dyn AppTab> {
        Box::new(Self {
            state: RemoteState::Login(LoginData::new()),
            config: config.clone(),
        })
    }
}

impl AppTab for RemoteTab {
    fn label(&self) -> &str {
        "Remote"
    }

    fn display(&mut self, ui: &mut Ui) -> bool {
        self.state.display(ui, &self.config);
        false
    }
}
