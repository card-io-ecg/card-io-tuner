use reqwest::{blocking::Client, redirect::Policy};

use crate::{app_config::AppConfig, AppMessage};

pub struct AppContext {
    pub config: AppConfig,
    pub http_client: Client,
    messages: Vec<AppMessage>,
}

impl AppContext {
    pub fn new(config: AppConfig) -> Self {
        Self {
            config,
            http_client: Client::builder()
                .redirect(Policy::limited(3))
                .build()
                .unwrap(),
            messages: Vec::new(),
        }
    }

    pub fn send_message(&mut self, message: AppMessage) {
        self.messages.push(message);
    }

    pub fn take_messages(&mut self) -> impl Iterator<Item = AppMessage> {
        std::mem::take(&mut self.messages).into_iter()
    }
}
