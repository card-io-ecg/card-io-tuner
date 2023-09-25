use std::{cell::RefCell, io::Read, path::PathBuf, rc::Rc};

use eframe::{
    egui::{self, Label, Layout, Sense, TextEdit, Ui},
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

struct Token(String);
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

    fn display(&mut self, ui: &mut Ui, config: &RefCell<AppConfig>) -> Option<Token> {
        let result = ui.with_layout(Layout::top_down(Align::Center), |ui| {
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

                    let token = Token::new(&jwt);

                    let response = client_builder
                        .get(config.borrow().backend_url("validate"))
                        .header("Authorization", token.header())
                        .send()
                        .unwrap();

                    if response.status().is_success() {
                        log::info!("Logged in. Token: {}", jwt);
                        Some(token)
                    } else {
                        log::error!(
                            "Failed to validate token: {:?}",
                            response.json::<Error>().unwrap().error
                        );
                        None
                    }
                } else {
                    None
                }
            })
            .inner
        });

        result.inner
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
struct MeasurementList {
    pub measurements: Vec<String>,
}

enum RemotePage {
    Devices(DeviceList),
    Measurements(String, MeasurementList),
}
impl RemotePage {
    fn new(config: &RefCell<AppConfig>, token: &Token) -> RemotePage {
        let client_builder = Client::builder()
            .redirect(Policy::limited(3))
            .build()
            .unwrap();

        let response = client_builder
            .get(config.borrow().backend_url("list_devices"))
            .header("Authorization", token.header())
            .send()
            .unwrap()
            .json::<DeviceList>()
            .unwrap();

        Self::Devices(response)
    }

    fn measurements(config: &RefCell<AppConfig>, token: &mut Token, device: String) -> RemotePage {
        log::info!("Getting measurements for {device}");
        let client_builder = Client::builder()
            .redirect(Policy::limited(3))
            .build()
            .unwrap();

        let response = client_builder
            .get(
                config
                    .borrow()
                    .backend_url(format!("list_measurements/{device}")),
            )
            .header("Authorization", token.header())
            .send()
            .unwrap()
            .json::<MeasurementList>()
            .unwrap();

        Self::Measurements(device, response)
    }
}

enum RemoteState {
    Login(LoginData),
    Authenticated(Token, RemotePage),
}

impl RemoteState {
    fn display(&mut self, ui: &mut Ui, config: &RefCell<AppConfig>) {
        match self {
            Self::Login(data) => {
                if let Some(token) = data.display(ui, config) {
                    let page = RemotePage::new(config, &token);
                    *self = Self::Authenticated(token, page);
                }
            }

            Self::Authenticated(token, page) => {
                ui.label("Logged in");
                if ui.button("Log out").clicked() {
                    *self = Self::Login(LoginData::new());
                } else {
                    let mut new_page = None;
                    match page {
                        RemotePage::Devices(devices) => {
                            ui.vertical(|ui| {
                                for device in &devices.devices {
                                    if ui.add(Label::new(device).sense(Sense::click())).clicked() {
                                        new_page = Some(RemotePage::measurements(
                                            config,
                                            token,
                                            device.clone(),
                                        ));
                                        break;
                                    }
                                }
                            });
                        }
                        RemotePage::Measurements(device, measurements) => {
                            ui.vertical(|ui| {
                                for measurement in &measurements.measurements {
                                    if ui
                                        .add(Label::new(measurement).sense(Sense::click()))
                                        .clicked()
                                    {
                                        log::info!("Downloading {device}/{measurement}");
                                        let client_builder = Client::builder()
                                            .redirect(Policy::limited(3))
                                            .build()
                                            .unwrap();

                                        let ekg = client_builder
                                            .get(config.borrow().backend_url(format!(
                                                "download_measurement/{device}/{measurement}"
                                            )))
                                            .header("Authorization", token.header())
                                            .send()
                                            .unwrap()
                                            .bytes()
                                            .unwrap();

                                        let file =
                                            PathBuf::from(format!("data/{device}/{measurement}"));
                                        if !std::path::Path::new(&file).exists() {
                                            _ = std::fs::create_dir_all(file.parent().unwrap());
                                            std::fs::write(file, ekg.as_ref()).unwrap();
                                        }

                                        break;
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
