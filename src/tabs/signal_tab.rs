use std::ops::{Not, Range};

use eframe::{
    egui::{DragValue, Grid, Id, PointerButton, Ui, WidgetText},
    epaint::Color32,
};
use egui_dock::{DockArea, DockState, NodeIndex, Style};
use egui_plot::{AxisBools, GridInput, GridMark, Legend, Line, MarkerShape, PlotPoints, Points};
use signal_processing::heart_rate::SamplingFrequency;

use crate::{
    data::{Cycle, Data},
    AppContext, AppTab,
};

const EKG_COLOR: Color32 = Color32::from_rgb(100, 150, 250);
const CYCLE_GROUP_COLORS: [Color32; 3] = [
    Color32::LIGHT_GREEN,
    Color32::LIGHT_YELLOW,
    Color32::LIGHT_GRAY,
];

struct SignalCharts<'a> {
    lines: &'a mut Vec<Line>,
    points: &'a mut Vec<Points>,
    offset: f64,
    fs: SamplingFrequency,
    samples: Range<usize>,
}

impl SignalCharts<'_> {
    fn to_point(&self, (x, y): (usize, f64)) -> [f64; 2] {
        [self.fs.samples_to_s(x) as f64, y - self.offset]
    }

    fn push(
        &mut self,
        iter: impl Iterator<Item = f64>,
        color: impl Into<Color32>,
        name: impl ToString,
    ) {
        let line = iter
            .enumerate()
            .map(|p| self.to_point(p))
            .collect::<PlotPoints>();

        self.lines.push(Line::new(line).color(color).name(name));
    }

    fn push_points(
        &mut self,
        iter: impl Iterator<Item = (usize, f64)>,
        color: impl Into<Color32>,
        name: impl ToString,
    ) {
        let points = iter
            .filter(|(idx, _y)| self.samples.contains(idx))
            .map(|(idx, y)| self.to_point((idx - self.samples.start, y)))
            .collect::<PlotPoints>();

        self.points.push(
            Points::new(points)
                .color(color)
                .name(name)
                .shape(MarkerShape::Asterisk)
                .radius(4.0),
        );
    }
}

fn generate_grid_marks(input: GridInput, scale: f64, steps: &[i32]) -> Vec<GridMark> {
    let mut marks = vec![];

    let (min, max) = input.bounds;
    let min = (min * scale).floor() as i32;
    let max = (max * scale).ceil() as i32;

    for i in min..=max {
        if let Some(step) = steps.iter().copied().find(|step| i % *step == 0) {
            marks.push(GridMark {
                value: i as f64 / scale,
                step_size: step as f64 / scale,
            });
        }
    }

    marks
}

pub struct SignalTab {
    id: Id,
    label: String,
    data: Data,
    tree: DockState<Box<dyn SignalSubTab>>,
}

impl SignalTab {
    pub fn new_boxed(label: String, data: Data) -> Box<Self> {
        let id = Id::new(rand::random::<u64>());

        let mut tree = DockState::<Box<dyn SignalSubTab>>::new(vec![Box::new(EkgTab::new(id))]);

        let [_, right] = tree.main_surface_mut().split_right(
            NodeIndex::root(),
            2.0 / 3.0,
            vec![Box::new(CycleTab::new(id))],
        );
        let [_, right] =
            tree.main_surface_mut()
                .split_below(right, 1.0 / 3.0, vec![Box::new(HrvTab::new(id))]);
        let [_, _] =
            tree.main_surface_mut()
                .split_below(right, 0.5, vec![Box::new(FftTab::new(id))]);

        Box::new(Self {
            id,
            label,
            data,
            tree,
        })
    }

    fn signal_editor(&mut self, ui: &mut Ui) {
        ui.collapsing("Edit signal", |ui| {
            ui.horizontal(|ui| {
                self.data.change_config(|config| {
                    ui.vertical(|ui| {
                        ui.heading("Trim");

                        Grid::new((&self.label, "trim"))
                            .num_columns(2)
                            .show(ui, |ui| {
                                ui.label("Start");
                                ui.add(DragValue::new(&mut config.ignored_start).speed(1));

                                ui.end_row();

                                ui.label("End");
                                ui.add(DragValue::new(&mut config.ignored_end).speed(1));

                                ui.end_row();

                                ui.label("Row");
                                ui.add(DragValue::new(&mut config.row_width).speed(1));
                            });
                    });

                    ui.vertical(|ui| {
                        ui.heading("Filter");

                        Grid::new("filters").num_columns(2).show(ui, |ui| {
                            ui.checkbox(&mut config.high_pass, "High-pass filter");
                            ui.add(
                                DragValue::new(&mut config.high_pass_cutoff)
                                    .speed(0.01)
                                    .clamp_range(0.01..=f32::INFINITY),
                            );

                            ui.end_row();

                            ui.checkbox(&mut config.low_pass, "Low-pass filter");
                            ui.add(
                                DragValue::new(&mut config.low_pass_cutoff)
                                    .speed(0.1)
                                    .clamp_range(0.1..=f32::INFINITY),
                            );
                            ui.end_row();

                            ui.checkbox(&mut config.pli, "PLI filter");
                            ui.end_row();
                        });
                    });

                    #[cfg(feature = "debug")]
                    ui.vertical(|ui| {
                        ui.heading("Debug");
                        Grid::new((&self.label, "debug_opts"))
                            .num_columns(2)
                            .show(ui, |ui| {
                                ui.checkbox(&mut config.hr_debug, "HR debug");
                                ui.end_row();

                                ui.label("Similarity threshold");
                                ui.add(
                                    DragValue::new(&mut config.similarity_threshold)
                                        .speed(0.001)
                                        .clamp_range(0.0..=1.0),
                                );
                                ui.end_row();
                            });
                    });
                });
            });

            ui.horizontal(|ui| {
                if ui.button("Default").clicked() {
                    self.data.set_config(Default::default());
                }

                if ui.button("Reload").clicked() {
                    self.data.load_config();
                }

                if ui.button("Save").clicked() {
                    self.data.save_config();
                }
            });
        });
    }
}

impl AppTab for SignalTab {
    fn id(&self) -> Id {
        self.id
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn display(&mut self, ui: &mut Ui, _: &mut AppContext) {
        ui.vertical(|ui| {
            self.signal_editor(ui);
            let id = self.id().with("dock");
            DockArea::new(&mut self.tree)
                .id(id)
                .style(Style::from_egui(ui.ctx().style().as_ref()))
                .show_inside(
                    ui,
                    &mut SignalTabViewer {
                        data: &mut self.data,
                    },
                );
        });
    }
}

trait SignalSubTab {
    fn id(&self) -> Id;
    fn label(&self) -> &str;
    fn display(&mut self, ui: &mut Ui, data: &mut Data);
}

struct SignalTabViewer<'a> {
    data: &'a mut Data,
}

impl egui_dock::TabViewer for SignalTabViewer<'_> {
    type Tab = Box<dyn SignalSubTab>;

    fn id(&mut self, tab: &mut Self::Tab) -> Id {
        tab.id()
    }

    fn title(&mut self, tab: &mut Self::Tab) -> WidgetText {
        tab.label().into()
    }

    fn ui(&mut self, ui: &mut Ui, tab: &mut Self::Tab) {
        tab.display(ui, self.data);
    }

    fn allowed_in_windows(&self, _tab: &mut Self::Tab) -> bool {
        false
    }

    fn closeable(&mut self, _tab: &mut Self::Tab) -> bool {
        false
    }
}

struct EkgTab {
    id: Id,
}

impl EkgTab {
    fn new(id: Id) -> Self {
        Self {
            id: id.with("ekg_tab"),
        }
    }
}

impl SignalSubTab for EkgTab {
    fn id(&self) -> Id {
        self.id
    }

    fn label(&self) -> &str {
        "EKG"
    }

    fn display(&mut self, ui: &mut Ui, data: &mut Data) {
        let mut lines = vec![];
        let mut points = vec![];

        let mut bottom = 0.0;
        let mut idx = 0;
        let mut marker_added = false;

        {
            let hr_data = data.hrs();
            let ekg_data = data.filtered_ekg();
            let classified_cycles = data.adjusted_cycles();
            let groups = data.cycle_groups();

            let chunk = data.config().row_width.max(1);

            let ekg = ekg_data.samples.chunks(chunk);
            let threshold = hr_data.thresholds.chunks(chunk);
            let complex_lead = hr_data.complex_lead.chunks(chunk);

            let fs = data.fs();

            for (ekg, (threshold, complex_lead)) in ekg.zip(threshold.zip(complex_lead)) {
                let (min, max) = ekg
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), y| {
                        (min.min(*y), max.max(*y))
                    });

                let height = max - min;
                let offset = (max - bottom) as f64;
                bottom -= height + 0.0005; // 500uV margin

                let mut signals = SignalCharts {
                    lines: &mut lines,
                    points: &mut points,
                    offset,
                    fs,
                    samples: idx..idx + ekg.len(),
                };

                if !marker_added {
                    marker_added = true;
                    // to nearest 1mV
                    let marker_y =
                        fs.samples_to_s(fs.s_to_samples(max - offset as f32)) as f64 - 0.001;
                    let marker_x = -0.2;

                    signals.lines.push(
                        Line::new(
                            [
                                [marker_x + -0.04, marker_y + 0.0],
                                [marker_x + 0.0, marker_y + 0.0],
                                [marker_x + 0.0, marker_y + 0.001],
                                [marker_x + 0.16, marker_y + 0.001],
                                [marker_x + 0.16, marker_y + 0.0],
                                [marker_x + 0.2, marker_y + 0.0],
                            ]
                            .into_iter()
                            .collect::<PlotPoints>(),
                        )
                        .color(Color32::from_rgb(100, 200, 100)),
                    );
                }

                signals.push(
                    ekg.iter().map(|y| *y as f64),
                    EKG_COLOR,
                    format!("EKG (HR: {} bpm)", data.avg_hr().round() as i32,),
                );

                if data.config().hr_debug {
                    signals.push(
                        threshold.iter().map(|y| y.total().unwrap_or(y.r) as f64),
                        Color32::YELLOW,
                        "Threshold",
                    );
                    signals.push(
                        threshold.iter().map(|y| y.m.unwrap_or(f32::NAN) as f64),
                        Color32::WHITE,
                        "M",
                    );
                    signals.push(
                        threshold.iter().map(|y| y.f.unwrap_or(f32::NAN) as f64),
                        Color32::GRAY,
                        "F",
                    );
                    signals.push(threshold.iter().map(|y| y.r as f64), Color32::GREEN, "R");
                    signals.push(
                        complex_lead.iter().map(|y| *y as f64),
                        Color32::LIGHT_RED,
                        "Complex lead",
                    );
                }

                // signals.push_points(
                //     hr_data
                //         .detections
                //         .iter()
                //         .map(|idx| (*idx, ekg_data.samples[*idx] as f64)),
                //     Color32::LIGHT_RED,
                //     "Raw HR",
                // );

                for (group_idx, group) in groups.iter().filter(|g| g.len() > 1).enumerate() {
                    signals.push_points(
                        group
                            .cycles()
                            .map(|idx| classified_cycles[idx].position)
                            .map(|idx| (idx, ekg_data.samples[idx] as f64)),
                        CYCLE_GROUP_COLORS[group_idx % CYCLE_GROUP_COLORS.len()],
                        format!("Group {} (cycles: {})", group_idx, group.len()),
                    );
                }

                signals.push_points(
                    classified_cycles
                        .iter()
                        .filter_map(|cycle| cycle.cycle_group().is_none().then_some(cycle.position))
                        .map(|idx| (idx, ekg_data.samples[idx] as f64)),
                    Color32::LIGHT_RED,
                    "Artifacts",
                );

                idx += ekg.len();
            }
        }

        egui_plot::Plot::new(ui.id().with("ekg"))
            .legend(Legend::default())
            .show_axes(false)
            .show_grid(true)
            .data_aspect(400.0) // 1 small square = 40ms = 0.1mV
            .allow_scroll(false)
            .boxed_zoom_pointer_button(PointerButton::Middle)
            .x_grid_spacer(|input| generate_grid_marks(input, 1.0 / 0.01, &[20, 4])) // 10ms resolution, 200ms, 40ms
            .y_grid_spacer(|input| generate_grid_marks(input, 1.0 / 0.000_1, &[5, 1])) // 100uV resolution, 500uV, 100uV
            .show(ui, |plot_ui| {
                for line in lines {
                    plot_ui.line(line);
                }
                for points in points {
                    plot_ui.points(points);
                }
            });
    }
}

struct FftTab {
    id: Id,
}

impl FftTab {
    fn new(id: Id) -> Self {
        Self {
            id: id.with("fft_tab"),
        }
    }
}

impl SignalSubTab for FftTab {
    fn id(&self) -> Id {
        self.id
    }

    fn label(&self) -> &str {
        "FFT"
    }

    fn display(&mut self, ui: &mut Ui, data: &mut Data) {
        let fft = {
            let fft = data.fft();

            let x_scale = data.fs().raw() as f64 / fft.len() as f64;

            Line::new(
                fft.iter()
                    .skip(1 - data.config().high_pass as usize) // skip DC if high-pass is off
                    .take(fft.len() / 2)
                    .enumerate()
                    .map(|(x, y)| [x as f64 * x_scale, *y as f64])
                    .collect::<PlotPoints>(),
            )
            .color(EKG_COLOR)
            .name("FFT")
        };

        egui_plot::Plot::new(ui.id().with("fft"))
            .legend(Legend::default())
            .show_axes(false)
            .show_grid(true)
            .auto_bounds_y()
            .allow_scroll(false)
            .boxed_zoom_pointer_button(PointerButton::Middle)
            .allow_zoom(AxisBools { x: false, y: true })
            .show(ui, |plot_ui| {
                plot_ui.line(fft);
            });
    }
}

struct HrvTab {
    id: Id,
}

impl HrvTab {
    fn new(id: Id) -> Self {
        Self {
            id: id.with("hrv_tab"),
        }
    }
}

impl SignalSubTab for HrvTab {
    fn id(&self) -> Id {
        self.id
    }

    fn label(&self) -> &str {
        "HRV"
    }

    fn display(&mut self, ui: &mut Ui, data: &mut Data) {
        let fs = data.fs();
        let cycles = data.adjusted_cycles();

        // Poincare plot to visualize heart-rate variability
        let rrs = cycles
            .iter()
            .map(|cycle| (cycle.position, cycle.cycle_group()))
            .map_windows(|[(xp, x_normal), (yp, y_normal)]| {
                let is_nn = x_normal.is_some() && y_normal.is_some();
                (is_nn, fs.samples_to_s(*yp - *xp) as f64 * 1000.0)
            });

        let nns = rrs.clone().filter_map(|(is_nn, rr)| is_nn.then_some(rr));
        let extras = rrs
            .clone()
            .filter_map(|(is_nn, rr)| is_nn.not().then_some(rr));

        let (min_rr, max_rr) = nns
            .clone()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), y| {
                (min.min(y), max.max(y))
            });

        egui_plot::Plot::new(ui.id().with("hrv"))
            .legend(Legend::default())
            .show_axes(true)
            .show_grid(true)
            .include_x(min_rr - 100.0)
            .include_x(max_rr + 100.0)
            .include_y(min_rr - 100.0)
            .include_y(max_rr + 100.0)
            .data_aspect(1.0)
            .allow_scroll(false)
            .boxed_zoom_pointer_button(PointerButton::Middle)
            .show(ui, |plot_ui| {
                plot_ui.points(
                    Points::new(nns.map_windows(|[x, y]| [*x, *y]).collect::<PlotPoints>())
                        .name("NN intervals")
                        .color(EKG_COLOR),
                );
                plot_ui.points(
                    Points::new(
                        extras
                            .map_windows(|[x, y]| [*x, *y])
                            .collect::<PlotPoints>(),
                    )
                    .name("Extras")
                    .color(Color32::LIGHT_RED),
                );
            });
    }
}

struct CycleTab {
    id: Id,
}

impl CycleTab {
    fn new(id: Id) -> Self {
        Self {
            id: id.with("cycle_tab"),
        }
    }
}

impl SignalSubTab for CycleTab {
    fn id(&self) -> Id {
        self.id
    }

    fn label(&self) -> &str {
        "Cycle"
    }

    fn display(&mut self, ui: &mut Ui, data: &mut Data) {
        let mut lines = vec![];

        let fs = data.fs();
        let groups = data.cycle_groups();
        let mut add_cycle = |cycle: &Cycle, name: String, color: Color32| {
            let offset = fs.samples_to_s(cycle.position) as f64;
            lines.push(
                Line::new(
                    cycle
                        .as_slice()
                        .iter()
                        .enumerate()
                        .map(|(x, y)| (fs.samples_to_s(x) as f64, *y as f64))
                        .map(|(x, y)| [x - offset, y])
                        .collect::<PlotPoints>(),
                )
                .color(color)
                .name(name),
            );
        };

        // add_cycle(&data.average_cycle(), "Average cycle", Color32::LIGHT_RED);
        for (displayed_group, (group_idx, average)) in data.average_cycles().iter().enumerate() {
            add_cycle(
                average,
                format!(
                    "Group {} average (cycles: {})",
                    displayed_group,
                    groups.group(*group_idx).len()
                ),
                CYCLE_GROUP_COLORS[displayed_group % CYCLE_GROUP_COLORS.len()],
            );
        }

        egui_plot::Plot::new(ui.id().with("cycle"))
            .legend(Legend::default())
            .show_axes(false)
            .show_grid(true)
            .data_aspect(400.0) // 1 small square = 40ms = 0.1mV
            .allow_scroll(false)
            .boxed_zoom_pointer_button(PointerButton::Middle)
            .x_grid_spacer(|input| generate_grid_marks(input, 1.0 / 0.01, &[20, 4])) // 10ms resolution, 200ms, 40ms
            .y_grid_spacer(|input| generate_grid_marks(input, 1.0 / 0.000_1, &[5, 1])) // 100uV resolution, 500uV, 100uV
            .show(ui, |plot_ui| {
                for line in lines {
                    plot_ui.line(line);
                }
            });
    }
}
