use std::{fs, io::Read, path::Path};

use crate::data::Ekg;

pub fn load(file: &Path) -> Option<Ekg> {
    let hea = file.with_extension("hea");
    let dat = file.with_extension("dat");

    let header = load_hea(&hea)?;

    if header.channels.len() != 1 {
        return None;
    }

    let data = fs::read(dat).ok()?;

    let channel = &header.channels[0];

    let mut data = data.as_slice();
    let mut samples = vec![];
    while !data.is_empty() {
        let sample = channel.read(&mut data)? * 0.001;
        samples.push(sample);
    }

    if samples.len() != header.samples {
        return None;
    }

    Some(Ekg::new(header.fs as f64, samples))
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
    lsb_per_mv: f32,
    zero: isize,
}

impl Channel {
    fn format(&self) -> Option<Format> {
        match self.resolution {
            16 => Some(Format::Int16LE),
            _ => None,
        }
    }

    fn read(&self, data: impl Read) -> Option<f32> {
        let format = self.format()?;
        Some((format.read(data)? - self.zero) as f32 / self.lsb_per_mv)
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
        let lsb_per_mv = parts.next()?.parse::<usize>().ok()? as f32;
        let _bits = parts.next()?.parse::<usize>().ok()?;
        let baseline = parts.next()?.parse::<isize>().ok()?;

        channels.push(Channel {
            _file_name,
            resolution,
            lsb_per_mv,
            zero: baseline,
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
    pub fn read(self, data: impl Read) -> Option<isize> {
        match self {
            Format::Int16LE => {
                let mut data = data.bytes();
                let low = data.next()?.ok()? as i16;
                let high = data.next()?.ok()? as i16;
                let sample = (high << 8) | low;
                Some(sample as isize)
            }
        }
    }
}
