use eframe::egui::{self, Layout, TextEdit};

use crate::AppTab;

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

    fn display(&mut self, ui: &mut eframe::egui::Ui) {
        ui.with_layout(Layout::top_down(eframe::emath::Align::Center), |ui| {
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
                    log::debug!("Sign in with {}", self.password);
                }
            });
        });
    }
}

enum RemoteState {
    Login(LoginData),
}

impl RemoteState {
    fn display(&mut self, ui: &mut eframe::egui::Ui) {
        match self {
            Self::Login(data) => data.display(ui),
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

    fn display(&mut self, ui: &mut eframe::egui::Ui) -> bool {
        self.state.display(ui);
        false
    }
}
