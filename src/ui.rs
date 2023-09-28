use std::hash::Hash;

use eframe::{
    egui::{
        Grid, Label, Layout, Response, ScrollArea, SelectableLabel, Sense, TextEdit, Ui, Widget,
    },
    emath::Align,
};
use rfd::MessageDialogResult;

pub fn vertical_list<'a, E: 'a>(
    id: impl Hash,
    ui: &mut Ui,
    list: impl IntoIterator<Item = &'a E> + 'a,
    mut element: impl FnMut(&mut Ui, &'a E),
) {
    ScrollArea::vertical()
        .auto_shrink([false; 2])
        .show(ui, |ui| {
            Grid::new(id).num_columns(1).striped(true).show(ui, |ui| {
                for e in list.into_iter() {
                    element(ui, e);
                    ui.end_row();
                }
            });
        });
}

pub fn confirm(title: &str, message: String, yes: &str, no: &str) -> bool {
    let result = rfd::MessageDialog::new()
        .set_description(message)
        .set_buttons(rfd::MessageButtons::OkCancelCustom(
            yes.to_string(),
            no.to_string(),
        ))
        .set_level(rfd::MessageLevel::Info)
        .set_title(title)
        .show();

    if let MessageDialogResult::Custom(val) = result {
        if val == yes {
            return true;
        }
    }

    false
}

// Copied from egui examples
pub fn password_ui(ui: &mut Ui, password: &mut String) -> Response {
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
    let result = ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
        // Show the password field:
        ui.add(TextEdit::singleline(password).password(!show_plaintext));

        // Toggle the `show_plaintext` bool with a button:
        if ui
            .add(SelectableLabel::new(show_plaintext, "👁"))
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

pub fn password(password: &mut String) -> impl Widget + '_ {
    move |ui: &mut Ui| password_ui(ui, password)
}

pub fn clickable_label(measurement: &str) -> impl Widget + '_ {
    Label::new(measurement).sense(Sense::click())
}
