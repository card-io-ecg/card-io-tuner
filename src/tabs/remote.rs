use std::{io::Read, path::PathBuf};

use eframe::{
    egui::{self, Label, Layout, Sense, TextEdit, Ui},
    emath::Align,
};
use serde_json::json;

use crate::{AppContext, AppMessage, AppTab};

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

pub struct Token(String);
impl Token {
    fn new(jwt: &str) -> Self {
        Self(format!("Bearer {jwt}"))
    }

    fn header(&self) -> &str {
        &self.0
    }
}

impl LoginData {
    fn new() -> Self {
        Self {
            username: String::new(),
            password: String::new(),
        }
    }

    fn display(&mut self, ui: &mut Ui, context: &mut AppContext) {
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
                    let mut jwt = String::new();

                    _ = context
                        .http_client
                        .post(context.config.backend_url("login"))
                        .json(&json!({
                            "username": &self.username,
                            "password": &self.password,
                        }))
                        .send()
                        .unwrap()
                        .read_to_string(&mut jwt);

                    let token = Token::new(&jwt);

                    let response = context
                        .http_client
                        .get(context.config.backend_url("validate"))
                        .header("Authorization", token.header())
                        .send()
                        .unwrap();

                    if response.status().is_success() {
                        log::info!("Logged in. Token: {}", jwt);
                        context.auth_token = Some(token);
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

#[derive(serde::Deserialize)]
struct DeviceList {
    pub devices: Vec<String>,
}

#[derive(serde::Deserialize)]
struct RemoteMeasurementList {
    pub measurements: Vec<String>,
}

#[derive(serde::Deserialize)]
struct MeasurementList {
    pub measurements: Vec<(String, PathBuf)>,
}

enum RemotePage {
    Devices(DeviceList),
    Measurements(String, MeasurementList),
}
impl RemotePage {
    fn new(context: &AppContext) -> RemotePage {
        let response = context
            .http_client
            .get(context.config.backend_url("list_devices"))
            .header(
                "Authorization",
                context.auth_token.as_ref().unwrap().header(),
            )
            .send()
            .unwrap()
            .json::<DeviceList>()
            .unwrap();

        Self::Devices(response)
    }

    fn measurements(context: &AppContext, device: &str) -> RemotePage {
        log::info!("Getting measurements for {device}");

        let response = context
            .http_client
            .get(
                context
                    .config
                    .backend_url(format!("list_measurements/{device}")),
            )
            .header(
                "Authorization",
                context.auth_token.as_ref().unwrap().header(),
            )
            .send()
            .unwrap()
            .json::<RemoteMeasurementList>()
            .unwrap();

        Self::Measurements(
            device.to_string(),
            MeasurementList {
                measurements: response
                    .measurements
                    .into_iter()
                    .map(|m| (m.clone(), PathBuf::from(format!("data/{device}/{m}"))))
                    .collect(),
            },
        )
    }
}

enum RemoteState {
    Login(LoginData),
    Authenticated(RemotePage),
}

impl RemoteState {
    fn display(&mut self, ui: &mut Ui, context: &mut AppContext) {
        match self {
            Self::Login(data) => {
                data.display(ui, context);

                if context.auth_token.is_some() {
                    let page = RemotePage::new(context);
                    *self = Self::Authenticated(page);
                }
            }

            Self::Authenticated(page) => {
                ui.label("Logged in");
                if ui.button("Log out").clicked() {
                    context.auth_token = None;
                    *self = Self::Login(LoginData::new());
                    return;
                }

                let mut new_page = None;
                match page {
                    RemotePage::Devices(devices) => {
                        ui.vertical(|ui| {
                            for device in &devices.devices {
                                if ui.add(Label::new(device).sense(Sense::click())).clicked() {
                                    new_page = Some(RemotePage::measurements(context, &device));
                                    break;
                                }
                            }
                        });
                    }
                    RemotePage::Measurements(device, measurements) => {
                        ui.vertical(|ui| {
                            for (measurement, file) in &measurements.measurements {
                                if ui
                                    .add(Label::new(measurement).sense(Sense::click()))
                                    .clicked()
                                {
                                    let exists = std::path::Path::new(&file).exists();
                                    if !exists {
                                        log::info!("Downloading {device}/{measurement}");
                                        let ekg = context
                                            .http_client
                                            .get(context.config.backend_url(format!(
                                                "download_measurement/{device}/{measurement}"
                                            )))
                                            .header(
                                                "Authorization",
                                                context.auth_token.as_ref().unwrap().header(),
                                            )
                                            .send()
                                            .unwrap()
                                            .bytes()
                                            .unwrap();
                                        _ = std::fs::create_dir_all(file.parent().unwrap());
                                        std::fs::write(&file, ekg.as_ref()).unwrap();
                                    } else {
                                        log::info!("Already downloaded {device}/{measurement}");
                                    }

                                    context.messages.push(AppMessage::LoadFile(file.clone()));
                                }
                            }
                        });
                    }
                }

                if let Some(new_page) = new_page {
                    *page = new_page;
                }
            }
        }
    }
}

pub struct RemoteTab {
    state: RemoteState,
}

impl RemoteTab {
    pub fn new_boxed() -> Box<dyn AppTab> {
        Box::new(Self {
            state: RemoteState::Login(LoginData::new()),
        })
    }
}

impl AppTab for RemoteTab {
    fn label(&self) -> &str {
        "Remote"
    }

    fn display(&mut self, ui: &mut Ui, context: &mut AppContext) -> bool {
        self.state.display(ui, context);
        false
    }
}
