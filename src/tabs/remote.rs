use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use eframe::egui::{self, Id, Ui};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    ui::{clickable_label, confirm, password, vertical_list},
    AppContext, AppMessage, AppTab,
};

struct LoginData {
    username: String,
    password: String,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Token(String);

impl Token {
    fn new(jwt: &str) -> Self {
        Self(format!("Bearer {jwt}"))
    }

    pub fn header(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
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
        ui.horizontal_centered(|ui| {
            ui.set_max_height(150.0);
            ui.vertical_centered(|ui| {
                ui.set_max_width(400.0);

                ui.group(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.heading("Log in to remote server");

                        let sent = egui::Grid::new("login")
                            .num_columns(2)
                            .show(ui, |ui| {
                                ui.label("Name:");
                                ui.text_edit_singleline(&mut self.username);

                                ui.end_row();

                                ui.label("Password:");
                                let re = ui.add(password(&mut self.password));

                                re.lost_focus() && re.ctx.input(|i| i.key_pressed(egui::Key::Enter))
                            })
                            .inner;

                        if ui.button("Sign in").clicked() || sent {
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

                            context.config.set_auth_token(Token::new(&jwt));
                            let Ok(response) = context.get_auth("validate") else {
                                log::error!("Failed to validate token");
                                return;
                            };

                            if response.status().is_success() {
                                log::info!("Logged in. Token: {}", jwt);
                                context.config.save();
                            } else {
                                log::error!(
                                    "Failed to validate token: {:?}",
                                    response.json::<Error>().unwrap().error
                                );
                            }
                        }
                    });
                });
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

impl DeviceList {
    fn ui(&self, ui: &mut Ui, context: &AppContext) -> Option<RemotePage> {
        if ui.button("Reload").clicked() {
            return Some(RemotePage::devices(context));
        }

        let mut new_page = None;
        ui.vertical(|ui| {
            ui.heading("Your devices");

            vertical_list("devices", ui, &self.devices, |ui, device| {
                ui.horizontal(|ui| {
                    ui.set_width(ui.available_width());
                    if ui.add(clickable_label(device)).clicked() {
                        new_page = Some(RemotePage::measurements(context, device));
                    }
                });
            });
        });

        new_page
    }
}

#[derive(serde::Deserialize)]
struct RemoteMeasurementList {
    pub measurements: Vec<String>,
}

#[derive(serde::Deserialize)]
struct MeasurementList {
    pub measurements: Vec<(String, PathBuf)>,
}
impl MeasurementList {
    fn ui(&self, ui: &mut Ui, context: &mut AppContext, device: &str) -> Option<RemotePage> {
        if let Some(page) = ui
            .horizontal(|ui| {
                if ui.button("Reload").clicked() {
                    return Some(RemotePage::measurements(context, device));
                }
                if ui.button("Back").clicked() {
                    return Some(RemotePage::devices(context));
                }

                None
            })
            .inner
        {
            return Some(page);
        }

        let mut new_page = None;
        ui.vertical(|ui| {
            ui.heading(format!("Measurements of {device}"));

            vertical_list(
                "measurements",
                ui,
                &self.measurements,
                |ui, (measurement, file)| {
                    ui.horizontal(|ui| {
                        ui.set_width(ui.available_width());

                        if ui.button("🗑").clicked() {
                            let message = format!(
                                r#"Are you sure you want to delete {measurement}?
This will remove the measurement from the cloud but it will not delete the local copy."#
                            );

                            if confirm("Delete measurement", message, "Delete", "Don't delete") {
                                delete_remote_file(device, measurement, context);

                                new_page = Some(RemotePage::measurements(context, device));
                            }
                        }

                        if ui.add(clickable_label(measurement)).clicked() {
                            open_remote_file(device, measurement, file, context);
                        }
                    });
                },
            );
        });

        new_page
    }
}

fn delete_remote_file(device: &str, measurement: &str, context: &AppContext) {
    let url = context
        .config
        .backend_url(format!("measurements/{device}/{measurement}"));

    context
        .http_client
        .delete(url)
        .header("Authorization", context.config.auth_token.header())
        .send()
        .unwrap();
}

fn open_remote_file(device: &str, measurement: &str, file: &Path, context: &mut AppContext) {
    if file.exists() {
        log::info!("Already downloaded {device}/{measurement}");
    } else {
        log::info!("Downloading {device}/{measurement}");

        let Ok(ekg) = context
            .get_auth(format!("measurements/{device}/{measurement}"))
            .and_then(|resp| resp.bytes().map_err(|_| ()))
        else {
            log::error!("Failed to download measurement");
            return;
        };

        _ = fs::create_dir_all(file.parent().unwrap());
        if fs::write(file, ekg.as_ref()).is_err() {
            log::error!("Failed to save measurement");
            return;
        }
    }

    context.send_message(AppMessage::LoadFile(file.to_owned()));
}

enum RemotePage {
    Login(LoginData),
    Devices(DeviceList),
    Measurements(String, MeasurementList),
}

impl Default for RemotePage {
    fn default() -> Self {
        Self::Login(LoginData::new())
    }
}

impl RemotePage {
    fn devices(context: &AppContext) -> RemotePage {
        match context
            .get_auth("list_devices")
            .and_then(|resp| resp.json::<DeviceList>().map_err(|_| ()))
        {
            Ok(devices) => Self::Devices(devices),
            Err(_) => Self::default(),
        }
    }

    fn measurements(context: &AppContext, device: &str) -> RemotePage {
        fn measurements_impl(context: &AppContext, device: &str) -> Result<MeasurementList, ()> {
            log::info!("Getting measurements for {device}");
            let mut response = context
                .get_auth(format!("measurements/{device}"))
                .and_then(|resp| resp.json::<RemoteMeasurementList>().map_err(|_| ()))?;

            response.measurements.sort_by(|a, b| b.cmp(a));

            Ok(MeasurementList {
                measurements: response
                    .measurements
                    .into_iter()
                    .map(|m| {
                        let path = PathBuf::from(format!("data/{device}/{m}"));
                        (m, path)
                    })
                    .collect(),
            })
        }

        match measurements_impl(context, device) {
            Ok(measurements) => Self::Measurements(device.to_string(), measurements),
            Err(_) => Self::default(),
        }
    }

    fn display(&mut self, ui: &mut Ui, context: &mut AppContext) {
        if self.is_logged_in() {
            let log_out_clicked = ui
                .horizontal(|ui| {
                    ui.label("Logged in");
                    ui.button("Log out").clicked()
                })
                .inner;

            if log_out_clicked {
                context.config.clear_auth_token();
                *self = Self::default();
                return;
            }
        }

        let new_page = match self {
            Self::Login(data) => {
                data.display(ui, context);

                if !context.config.auth_token.is_empty() {
                    Some(Self::devices(context))
                } else {
                    None
                }
            }

            Self::Devices(devices) => devices.ui(ui, context),
            Self::Measurements(device, measurements) => measurements.ui(ui, context, device),
        };

        if let Some(new_page) = new_page {
            *self = new_page;
        }
    }

    fn is_logged_in(&self) -> bool {
        !matches!(self, Self::Login(_))
    }
}

pub struct RemoteTab {
    id: Id,
    state: RemotePage,
}

impl RemoteTab {
    pub fn new_boxed() -> Box<dyn AppTab> {
        Box::new(Self {
            id: Id::new(rand::random::<u64>()),
            state: RemotePage::Login(LoginData::new()),
        })
    }
}

impl AppTab for RemoteTab {
    fn id(&self) -> Id {
        self.id
    }

    fn label(&self) -> &str {
        "Remote"
    }

    fn display(&mut self, ui: &mut Ui, context: &mut AppContext) {
        self.state.display(ui, context);
    }
}
