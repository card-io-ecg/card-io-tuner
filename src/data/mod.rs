use std::{
    cell::Ref,
    path::{Path, PathBuf},
    sync::Arc,
};

use signal_processing::compressing_buffer::EkgFormat;

use crate::data::processing::{Config, Context, HrData, ProcessedSignal};

mod cell;
pub mod processing;

#[derive(Clone)]
pub struct Ekg {
    pub samples: Arc<[f32]>,
    pub fs: f64,
}

impl Ekg {
    fn load(bytes: Vec<u8>, config: &Config) -> Result<Self, ()> {
        let version: u32 = u32::from_le_bytes(
            bytes[0..4]
                .try_into()
                .map_err(|err| log::warn!("Failed to read version: {}", err))?,
        );
        log::debug!("version: {}", version);

        match version {
            0 => Self::load_v0(&bytes[4..], config),
            _ => {
                log::warn!("Unknown version: {}", version);
                Err(())
            }
        }
    }

    fn load_v0(mut bytes: &[u8], config: &Config) -> Result<Self, ()> {
        pub const VOLTS_PER_LSB: f32 = -2.42 / (1 << 23) as f32; // ADS129x

        let mut reader = EkgFormat::new();
        let mut samples = Vec::new();
        while let Some(sample) = reader.read(&mut bytes).unwrap() {
            samples.push(sample as f32 * VOLTS_PER_LSB);
        }

        log::debug!("Loaded {} samples", samples.len());

        let ignore_start = config.ignored_start;
        let ignore_end = config.ignored_end;

        Ok(Self {
            fs: 1000.0,
            samples: Arc::from(
                samples
                    .get(ignore_start..samples.len() - ignore_end)
                    .ok_or(())?,
            ),
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
    pub context: Context,
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

        let config = std::fs::read_to_string(path.with_extension("toml"))
            .ok()
            .map(|config| {
                toml::from_str(&config).unwrap_or_else(|err| {
                    log::warn!("Failed to parse config: {}", err);
                    Config::default()
                })
            })
            .unwrap_or_default();

        std::fs::read(path).ok().and_then(|bytes| {
            let ekg = Ekg::load(bytes, &config).ok()?;
            Some(Self::new(path.to_owned(), ekg, config))
        })
    }

    fn new(path: PathBuf, ekg: Ekg, config: Config) -> Self {
        Self {
            path,
            processed: ProcessedSignal::new(),
            context: Context {
                raw_ekg: ekg,
                config,
            },
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
}
