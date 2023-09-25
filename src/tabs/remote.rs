use crate::AppTab;

pub struct RemoteTab {}
impl RemoteTab {
    pub fn new_boxed() -> Box<dyn AppTab> {
        Box::new(Self {})
    }
}

impl AppTab for RemoteTab {
    fn label(&self) -> &str {
        "Remote"
    }

    fn display(&mut self, _ui: &mut eframe::egui::Ui) -> bool {
        false
    }
}
