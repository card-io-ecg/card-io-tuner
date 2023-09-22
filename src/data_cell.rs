use std::cell::{Ref, RefCell};

pub struct DataCell<T> {
    data: RefCell<Option<T>>,
}

impl<T> Default for DataCell<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> DataCell<T> {
    pub const fn new() -> Self {
        Self {
            data: RefCell::new(None),
        }
    }

    pub fn get(&self, initializer: impl FnOnce() -> T) -> Ref<'_, T> {
        if self.data.borrow().is_none() {
            *self.data.borrow_mut() = Some(initializer());
        }

        Ref::map(self.data.borrow(), |data| {
            data.as_ref().expect("data should be Some")
        })
    }

    pub fn clear(&mut self) {
        self.data.borrow_mut().take();
    }
}
