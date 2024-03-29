use std::{fmt::Display, fs};

use log::warn;
use serde::{Deserialize, Serialize};

use crate::tabs::remote::Token;

fn default_backend_url() -> String {
    String::from("https://stingray-prime-monkey.ngrok-free.app")
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AppConfig {
    #[serde(default = "default_backend_url")]
    pub backend_url: String,

    #[serde(default)]
    pub auth_token: Token,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            backend_url: default_backend_url(),
            auth_token: Default::default(),
        }
    }
}

impl AppConfig {
    pub fn load() -> Self {
        let Ok(config_file) = fs::read_to_string("config.toml") else {
            warn!("Failed to read config.toml, using default config");
            return Self::default();
        };

        toml::from_str::<Self>(&config_file).unwrap_or_default()
    }

    pub fn save(&self) {
        let Ok(config_file) = toml::to_string_pretty(&self) else {
            warn!("Failed to serialize config, not saving");
            return;
        };

        if let Err(e) = fs::write("config.toml", config_file) {
            warn!("Failed to write config.toml: {}", e);
        }
    }

    pub fn backend_url(&self, path: impl Display) -> String {
        format!("{}/{}", self.backend_url, path)
    }

    pub fn set_auth_token(&mut self, token: Token) {
        self.auth_token = token;
    }

    pub fn clear_auth_token(&mut self) {
        self.set_auth_token(Token::default());
        self.save();
    }
}
