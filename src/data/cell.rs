use std::{
    cell::{Ref, RefCell},
    process::abort,
    time::Instant,
};

pub struct DataCell<T> {
    name: &'static str,
    data: RefCell<Option<T>>,
}

impl<T> DataCell<T> {
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            data: RefCell::new(None),
        }
    }

    pub fn get(&self, initializer: impl FnOnce() -> T) -> Ref<'_, T> {
        let is_initialized = {
            let Ok(value) = self.data.try_borrow() else {
                log::error!("Query cycle error detected in {}", self.name);
                abort();
            };
            value.is_some()
        };

        if !is_initialized {
            let start = Instant::now();
            let mut value = self.data.borrow_mut();
            *value = Some(initializer());
            log::debug!(
                "{}: {:.3}ms",
                self.name,
                start.elapsed().as_micros() as f32 / 1000.0
            );
        }

        Ref::map(self.data.borrow(), |data| {
            data.as_ref().expect("data should be Some")
        })
    }

    pub fn clear(&mut self) {
        self.data.borrow_mut().take();
    }
}
