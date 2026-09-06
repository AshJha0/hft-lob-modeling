//! Minimal row-major dense matrix used by the feature pipeline and the
//! deterministic neural network.  Not a general linear-algebra library — just
//! the storage plus the accessors the pinned algorithms need, with all inner
//! loops written out explicitly so the floating-point evaluation order is
//! under our control.  Public constructors validate their arguments and
//! return [`Result`] — this module never panics on caller input.

use crate::error::{LobError, Result};

/// Row-major `rows x cols` matrix of `f64`.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// All-zeros matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Build from a flat row-major buffer.
    ///
    /// # Errors
    /// [`LobError::InvalidInput`] if `data.len() != rows * cols`.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(LobError::InvalidInput(format!(
                "matrix buffer length {} does not match {rows} x {cols}",
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Element at `(r, c)`.
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /// Set element at `(r, c)`.
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    /// Row `r` as a slice.
    #[inline]
    pub fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Row `r` as a mutable slice.
    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [f64] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// The flat row-major buffer.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Mutable flat row-major buffer.
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Copy of the row range `[start, end)` as a new matrix.
    ///
    /// # Errors
    /// [`LobError::InvalidInput`] unless `start <= end <= rows`.
    pub fn slice_rows(&self, start: usize, end: usize) -> Result<Matrix> {
        if !(start <= end && end <= self.rows) {
            return Err(LobError::InvalidInput(format!(
                "row range [{start}, {end}) out of bounds for {} rows",
                self.rows
            )));
        }
        Ok(Matrix {
            rows: end - start,
            cols: self.cols,
            data: self.data[start * self.cols..end * self.cols].to_vec(),
        })
    }

    /// True if every element is finite.
    pub fn all_finite(&self) -> bool {
        self.data.iter().all(|v| v.is_finite())
    }
}
