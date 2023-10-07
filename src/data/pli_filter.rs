use signal_processing::{
    filter::{pli::adaptation_blocking::AdaptationBlockingTrait, Filter},
    moving::{
        sum::DynSum,
        variance::{MovingVariance, MovingVarianceOfErgodic},
    },
    sliding::AllocSlidingWindow,
};

#[derive(Clone)]
pub struct DynCombFilter {
    window: AllocSlidingWindow,
}

impl DynCombFilter {
    pub fn new(window_size: usize) -> Self {
        Self {
            window: AllocSlidingWindow::new(window_size),
        }
    }
}

impl Filter for DynCombFilter {
    fn update(&mut self, sample: f32) -> Option<f32> {
        self.window.push(sample).map(|old| sample - old)
    }

    fn clear(&mut self) {
        self.window.clear();
    }
}

#[derive(Clone)]
pub struct DynAdaptationBlocking {
    delay: AllocSlidingWindow,
    comb_filter: DynCombFilter,
    variance: MovingVarianceOfErgodic<DynSum>,
    delay_cnt: usize,
}

impl AdaptationBlockingTrait for DynAdaptationBlocking {
    fn new(fs: f32) -> Self {
        Self {
            delay: AllocSlidingWindow::new((fs * 0.005).ceil() as usize - 1),
            comb_filter: DynCombFilter::new((fs * 0.002).ceil() as usize - 1),
            variance: MovingVarianceOfErgodic::new(DynSum::new((fs * 1.2).ceil() as usize)),
            delay_cnt: 0,
        }
    }

    fn update(&mut self, sample: f32) -> Option<(f32, bool)> {
        let delayed_sample = self.delay.push(sample);
        let comb_filtered = self.comb_filter.update(sample)?;
        let variance = self.variance.update(comb_filtered)?;

        self.delay_cnt = if comb_filtered.abs() > (2.0 * variance).sqrt() {
            2 * self.delay.capacity()
        } else {
            self.delay_cnt.saturating_sub(1)
        };

        delayed_sample.map(|delayed_sample| (delayed_sample, self.delay_cnt > 0))
    }

    fn clear(&mut self) {
        self.comb_filter.clear();
        self.delay.clear();
        self.delay_cnt = 0;
        self.variance.clear();
    }
}
