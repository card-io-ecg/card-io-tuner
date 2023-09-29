use std::{
    cell::Ref,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use signal_processing::compressing_buffer::EkgFormat;

use crate::data::processing::{Config, Context, HrData, ProcessedSignal};

pub mod processing;

mod cell;
mod standard;

#[derive(Clone)]
pub struct Ekg {
    pub samples: Arc<[f32]>,
    pub fs: f64,
}

impl Ekg {
    fn load(bytes: Vec<u8>) -> Result<Self, ()> {
        let version: u32 = u32::from_le_bytes(
            bytes[0..4]
                .try_into()
                .map_err(|err| log::warn!("Failed to read version: {}", err))?,
        );
        log::debug!("version: {}", version);

        match version {
            0 => Self::load_v0(&bytes[4..]),
            _ => {
                log::warn!("Unknown version: {}", version);
                Err(())
            }
        }
    }

    fn load_v0(mut bytes: &[u8]) -> Result<Self, ()> {
        pub const VOLTS_PER_LSB: f32 = -2.42 / (1 << 23) as f32; // ADS129x

        let mut reader = EkgFormat::new();
        let mut samples = Vec::new();
        while let Some(sample) = reader.read(&mut bytes).unwrap() {
            samples.push(sample as f32 * VOLTS_PER_LSB);
        }

        log::debug!("Loaded {} samples", samples.len());

        Ok(Self {
            fs: 1000.0,
            samples: Arc::from(samples),
        })
    }
}

#[derive(Clone)]
pub struct Cycle {
    samples: Arc<[f32]>,
    pub start: usize,
    pub position: usize,
    pub end: usize,
}

impl Cycle {
    pub fn as_slice(&self) -> &[f32] {
        &self.samples[self.start..self.end]
    }
}

pub struct Data {
    pub path: PathBuf,
    pub processed: ProcessedSignal,
    context: Context,
}

macro_rules! query {
    ($name:ident: $ty:path) => {
        #[allow(dead_code)]
        pub fn $name(&self) -> Ref<'_, $ty> {
            self.processed.$name(&self.context)
        }
    };
}

impl Data {
    pub fn load(path: &Path) -> Option<Self> {
        log::debug!("Loading {}", path.display());

        let config = Config::default();

        let ekg = match path.extension() {
            Some(ext) if ext == "ecg" => {
                fs::read(path).ok().and_then(|bytes| Ekg::load(bytes).ok())
            }
            Some(ext) if ext == "hea" && path.with_extension("dat").exists() => {
                standard::load(path)
            }
            Some(ext) if ext == "dat" && path.with_extension("hea").exists() => {
                standard::load(path)
            }
            _ => return None,
        };

        match ekg {
            Some(ekg) => {
                let mut this = Self::new(path.to_owned(), ekg, config);
                this.load_config();
                Some(this)
            }
            _ => None,
        }
    }

    fn new(path: PathBuf, raw_ekg: Ekg, config: Config) -> Self {
        Self {
            path,
            processed: ProcessedSignal::new(),
            context: Context { raw_ekg, config },
        }
    }

    query!(filtered_ekg: Ekg);
    query!(fft: Vec<f32>);
    query!(hrs: HrData);
    query!(rr_intervals: Vec<f64>);
    query!(adjusted_rr_intervals: Vec<f64>);
    query!(cycles: Vec<Cycle>);
    query!(adjusted_cycles: Vec<Cycle>);
    query!(average_cycle: Cycle);
    query!(majority_cycle: Cycle);

    pub fn avg_hr(&self) -> f64 {
        self.processed.avg_hr(&self.context)
    }

    pub fn clear_processed(&mut self) {
        self.processed.clear();
    }

    pub fn config(&self) -> &Config {
        &self.context.config
    }

    pub fn change_config(&mut self, f: impl FnOnce(&mut Config)) {
        let old = self.context.config;
        f(&mut self.context.config);

        let high_pass_changed = old.high_pass != self.context.config.high_pass;
        let pli_changed = old.pli != self.context.config.pli;
        let low_pass_changed = old.low_pass != self.context.config.low_pass;
        let hr_debug_changed = old.hr_debug != self.context.config.hr_debug;
        let ignored_start_changed = old.ignored_start != self.context.config.ignored_start;
        let ignored_end_changed = old.ignored_end != self.context.config.ignored_end;
        // let row_width_changed = old.row_width != self.context.config.row_width;

        let reprocess = high_pass_changed
            || pli_changed
            || low_pass_changed
            || hr_debug_changed
            || ignored_start_changed
            || ignored_end_changed;

        if reprocess {
            self.clear_processed();
        }
    }

    pub fn set_config(&mut self, config: Config) {
        self.change_config(|c| *c = config);
    }

    pub fn load_config(&mut self) {
        self.set_config(
            fs::read_to_string(self.path.with_extension("toml"))
                .ok()
                .map(|config| {
                    toml::from_str(&config).unwrap_or_else(|err| {
                        log::warn!("Failed to parse config: {}", err);
                        Config::default()
                    })
                })
                .unwrap_or_default(),
        );
    }

    pub fn save_config(&self) {
        let config = toml::to_string_pretty(&self.context.config).unwrap();
        _ = fs::write(self.path.with_extension("toml"), config);
    }
}
