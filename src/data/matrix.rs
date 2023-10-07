use std::{
    fmt::{Debug, Write},
    ops::{Index, IndexMut},
};

pub struct Matrix<T> {
    rows: usize,
    columns: usize,
    data: Vec<T>,
}

impl<T, const C: usize, const R: usize> From<[[T; C]; R]> for Matrix<T> {
    fn from(value: [[T; C]; R]) -> Self {
        Self {
            rows: R,
            columns: C,
            data: value.into_iter().flatten().collect(),
        }
    }
}

impl<T: Default + Copy> Matrix<T> {
    pub fn new(rows: usize, columns: usize) -> Self {
        Self {
            rows,
            columns,
            data: vec![T::default(); rows * columns],
        }
    }
}

impl<T: Copy> Matrix<T> {
    #[allow(dead_code)]
    pub fn columns(&self) -> usize {
        self.columns
    }

    #[allow(dead_code)]
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn get(&self, index: (usize, usize)) -> Option<&T> {
        let coords = self.coords_to_index(index);
        self.data.get(coords)
    }

    pub fn get_mut(&mut self, index: (usize, usize)) -> Option<&mut T> {
        let coords = self.coords_to_index(index);
        self.data.get_mut(coords)
    }

    pub fn coords_to_index(&self, (row, col): (usize, usize)) -> usize {
        row * self.columns + col
    }

    pub fn row(&self, row: usize) -> Row<'_, T> {
        Row::new(self, row)
    }

    pub fn column(&self, col: usize) -> Column<'_, T> {
        Column::new(self, col)
    }

    #[allow(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item = Row<'_, T>> {
        (0..self.rows).map(|row| self.row(row))
    }

    #[allow(dead_code)]
    pub fn iter_columns(&self) -> impl Iterator<Item = Column<'_, T>> {
        (0..self.columns).map(|col| self.column(col))
    }
}

impl<T: Copy> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<T: Copy> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

impl<T: Debug + Copy> Debug for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.columns {
                f.write_fmt(format_args!("{:?} ", self[(i, j)]))?;
            }
            f.write_char('\n')?;
        }
        Ok(())
    }
}

pub struct Row<'a, T> {
    matrix: &'a Matrix<T>,
    row: usize,
    col: usize,
}

impl<'a, T: Copy> Row<'a, T> {
    fn new(matrix: &'a Matrix<T>, row: usize) -> Self {
        Self {
            matrix,
            row,
            col: 0,
        }
    }
}

impl<'a, T: Copy> Iterator for Row<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.matrix.get((self.row, self.col)).copied().map(|val| {
            self.col += 1;
            val
        })
    }
}

pub struct Column<'a, T> {
    matrix: &'a Matrix<T>,
    row: usize,
    col: usize,
}

impl<'a, T: Copy> Column<'a, T> {
    fn new(matrix: &'a Matrix<T>, col: usize) -> Self {
        Self {
            matrix,
            row: 0,
            col,
        }
    }
}

impl<'a, T: Copy> Iterator for Column<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.matrix.get((self.row, self.col)).copied().map(|val| {
            self.row += 1;
            val
        })
    }
}
