use std::{fs, io::Read, path::Path, sync::Arc};

use crate::data::Ekg;

pub fn load(file: &Path) -> Option<Ekg> {
    let hea = file.with_extension("hea");
    let dat = file.with_extension("dat");

    let header = load_hea(&hea)?;

    if header.channels.len() != 1 {
        return None;
    }

    let data = fs::read(dat).ok()?;

    let format = header.channels[0].format()?;

    let mut data = data.as_slice();
    let mut samples = vec![];
    while !data.is_empty() {
        let sample = format.read(&mut data)?;
        samples.push(sample);
    }

    if samples.len() != header.samples {
        return None;
    }

    Some(Ekg {
        fs: header.fs as f64,
        samples: Arc::from(samples),
    })
}

#[derive(Clone, Debug)]
struct Hea {
    _signal_name: String,
    fs: usize,
    samples: usize,
    channels: Vec<Channel>,
}

#[derive(Clone, Debug)]
struct Channel {
    _file_name: String,
    resolution: usize,
}

impl Channel {
    fn format(&self) -> Option<Format> {
        match self.resolution {
            16 => Some(Format::Int16LE),
            _ => None,
        }
    }
}

fn load_hea(hea: &Path) -> Option<Hea> {
    let header = fs::read_to_string(hea).ok()?;
    let mut lines = header.lines().skip_while(|l| l.starts_with('#'));

    let header_line = lines.next()?;
    let mut parts = header_line.split_whitespace();

    let _signal_name = parts.next()?.to_string();
    let channel_cnt = parts.next()?.parse().ok()?;
    let fs = parts.next()?.parse().ok()?;
    let samples = parts.next()?.parse().ok()?;

    let mut channels = vec![];
    for _ in 0..channel_cnt {
        let channel_line = lines.next()?;
        let mut parts = channel_line.split_whitespace();

        let _file_name = parts.next()?.to_string();
        let resolution = parts.next()?.parse().ok()?;

        channels.push(Channel {
            _file_name,
            resolution,
        });
    }

    Some(Hea {
        _signal_name,
        channels,
        fs,
        samples,
    })
}

#[derive(Clone, Copy)]
enum Format {
    Int16LE,
}

impl Format {
    pub fn read(self, data: impl Read) -> Option<f32> {
        match self {
            Format::Int16LE => {
                let mut data = data.bytes();
                let low = data.next()?;
                let high = data.next()?;
                let low = low.ok()? as i16;
                let high = high.ok()? as i16;
                let sample = (high << 8) | low;
                Some(sample as f32)
            }
        }
    }
}
