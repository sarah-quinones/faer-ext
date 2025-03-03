#![cfg_attr(docsrs, feature(doc_cfg))]

/// Conversions from external library matrix views into `faer` types.
pub trait IntoFaer {
    type Faer;
    fn into_faer(self) -> Self::Faer;
}

#[cfg(feature = "nalgebra")]
#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
/// Conversions from external library matrix views into `nalgebra` types.
pub trait IntoNalgebra {
    type Nalgebra;
    fn into_nalgebra(self) -> Self::Nalgebra;
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
/// Conversions from external library matrix views into `ndarray` types.
pub trait IntoNdarray {
    type Ndarray;
    fn into_ndarray(self) -> Self::Ndarray;
}

#[cfg(feature = "nalgebra")]
#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
const _: () = {
    use faer::prelude::*;
    use nalgebra::{Dim, Dyn, MatrixView, MatrixViewMut, ViewStorage, ViewStorageMut};

    impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoFaer
        for MatrixView<'a, T, R, C, RStride, CStride>
    {
        type Faer = MatRef<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides = self.strides();
            let ptr = self.as_ptr();
            unsafe {
                faer::MatRef::from_raw_parts(
                    ptr,
                    nrows,
                    ncols,
                    strides.0.try_into().unwrap(),
                    strides.1.try_into().unwrap(),
                )
            }
        }
    }

    impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoFaer
        for MatrixViewMut<'a, T, R, C, RStride, CStride>
    {
        type Faer = MatMut<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides = self.strides();
            let ptr = { self }.as_mut_ptr();
            unsafe {
                faer::MatMut::from_raw_parts_mut(
                    ptr,
                    nrows,
                    ncols,
                    strides.0.try_into().unwrap(),
                    strides.1.try_into().unwrap(),
                )
            }
        }
    }

    impl<'a, T> IntoNalgebra for MatRef<'a, T> {
        type Nalgebra = MatrixView<'a, T, Dyn, Dyn, Dyn, Dyn>;

        #[track_caller]
        fn into_nalgebra(self) -> Self::Nalgebra {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride = self.row_stride();
            let col_stride = self.col_stride();
            let ptr = self.as_ptr();
            unsafe {
                MatrixView::<'_, T, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorage::<
                    '_,
                    T,
                    Dyn,
                    Dyn,
                    Dyn,
                    Dyn,
                >::from_raw_parts(
                    ptr,
                    (Dyn(nrows), Dyn(ncols)),
                    (
                        Dyn(row_stride.try_into().unwrap()),
                        Dyn(col_stride.try_into().unwrap()),
                    ),
                ))
            }
        }
    }

    impl<'a, T> IntoNalgebra for MatMut<'a, T> {
        type Nalgebra = MatrixViewMut<'a, T, Dyn, Dyn, Dyn, Dyn>;

        #[track_caller]
        fn into_nalgebra(self) -> Self::Nalgebra {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride = self.row_stride();
            let col_stride = self.col_stride();
            let ptr = self.as_ptr_mut();
            unsafe {
                MatrixViewMut::<'_, T, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorageMut::<
                    '_,
                    T,
                    Dyn,
                    Dyn,
                    Dyn,
                    Dyn,
                >::from_raw_parts(
                    ptr,
                    (Dyn(nrows), Dyn(ncols)),
                    (
                        Dyn(row_stride.try_into().unwrap()),
                        Dyn(col_stride.try_into().unwrap()),
                    ),
                ))
            }
        }
    }
};

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
const _: () = {
    use faer::prelude::*;
    use ndarray::{ArrayView, ArrayViewMut, Ix2, ShapeBuilder};

    impl<'a, T> IntoFaer for ArrayView<'a, T, Ix2> {
        type Faer = MatRef<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = self.as_ptr();
            unsafe { faer::MatRef::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
        }
    }

    impl<'a, T> IntoFaer for ArrayViewMut<'a, T, Ix2> {
        type Faer = MatMut<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = { self }.as_mut_ptr();
            unsafe { faer::MatMut::from_raw_parts_mut(ptr, nrows, ncols, strides[0], strides[1]) }
        }
    }

    impl<'a, T> IntoNdarray for MatRef<'a, T> {
        type Ndarray = ArrayView<'a, T, Ix2>;

        #[track_caller]
        fn into_ndarray(self) -> Self::Ndarray {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride: usize = self.row_stride().try_into().unwrap();
            let col_stride: usize = self.col_stride().try_into().unwrap();
            let ptr = self.as_ptr();
            unsafe {
                ArrayView::<'_, T, Ix2>::from_shape_ptr(
                    (nrows, ncols).strides((row_stride, col_stride)),
                    ptr,
                )
            }
        }
    }

    impl<'a, T> IntoNdarray for MatMut<'a, T> {
        type Ndarray = ArrayViewMut<'a, T, Ix2>;

        #[track_caller]
        fn into_ndarray(self) -> Self::Ndarray {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride: usize = self.row_stride().try_into().unwrap();
            let col_stride: usize = self.col_stride().try_into().unwrap();
            let ptr = self.as_ptr_mut();
            unsafe {
                ArrayViewMut::<'_, T, Ix2>::from_shape_ptr(
                    (nrows, ncols).strides((row_stride, col_stride)),
                    ptr,
                )
            }
        }
    }
};

#[cfg(all(feature = "nalgebra", feature = "ndarray"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "nalgebra", feature = "ndarray"))))]
const _: () = {
    use nalgebra::{Dim, Dyn, MatrixView, MatrixViewMut, ViewStorage, ViewStorageMut};
    use ndarray::{ArrayView, ArrayViewMut, Ix2, ShapeBuilder};

    impl<'a, T> IntoNalgebra for ArrayView<'a, T, Ix2> {
        type Nalgebra = MatrixView<'a, T, Dyn, Dyn, Dyn, Dyn>;

        #[track_caller]
        fn into_nalgebra(self) -> Self::Nalgebra {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let [row_stride, col_stride]: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = self.as_ptr();

            unsafe {
                MatrixView::<'_, T, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorage::<
                    '_,
                    T,
                    Dyn,
                    Dyn,
                    Dyn,
                    Dyn,
                >::from_raw_parts(
                    ptr,
                    (Dyn(nrows), Dyn(ncols)),
                    (
                        Dyn(row_stride.try_into().unwrap()),
                        Dyn(col_stride.try_into().unwrap()),
                    ),
                ))
            }
        }
    }
    impl<'a, T> IntoNalgebra for ArrayViewMut<'a, T, Ix2> {
        type Nalgebra = MatrixViewMut<'a, T, Dyn, Dyn, Dyn, Dyn>;

        #[track_caller]
        fn into_nalgebra(self) -> Self::Nalgebra {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let [row_stride, col_stride]: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = { self }.as_mut_ptr();

            unsafe {
                MatrixViewMut::<'_, T, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorageMut::<
                    '_,
                    T,
                    Dyn,
                    Dyn,
                    Dyn,
                    Dyn,
                >::from_raw_parts(
                    ptr,
                    (Dyn(nrows), Dyn(ncols)),
                    (
                        Dyn(row_stride.try_into().unwrap()),
                        Dyn(col_stride.try_into().unwrap()),
                    ),
                ))
            }
        }
    }

    impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoNdarray
        for MatrixView<'a, T, R, C, RStride, CStride>
    {
        type Ndarray = ArrayView<'a, T, Ix2>;

        #[track_caller]
        fn into_ndarray(self) -> Self::Ndarray {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let (row_stride, col_stride) = self.strides();
            let ptr = self.as_ptr();

            unsafe {
                ArrayView::<'_, T, Ix2>::from_shape_ptr(
                    (nrows, ncols).strides((row_stride, col_stride)),
                    ptr,
                )
            }
        }
    }
    impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoNdarray
        for MatrixViewMut<'a, T, R, C, RStride, CStride>
    {
        type Ndarray = ArrayViewMut<'a, T, Ix2>;

        #[track_caller]
        fn into_ndarray(self) -> Self::Ndarray {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let (row_stride, col_stride) = self.strides();
            let ptr = { self }.as_mut_ptr();

            unsafe {
                ArrayViewMut::<'_, T, Ix2>::from_shape_ptr(
                    (nrows, ncols).strides((row_stride, col_stride)),
                    ptr,
                )
            }
        }
    }
};

#[cfg(feature = "polars")]
#[cfg_attr(docsrs, doc(cfg(feature = "polars")))]
pub mod polars {
    use faer::Mat;
    use polars::prelude::*;

    pub trait Frame {
        fn is_valid(self) -> PolarsResult<LazyFrame>;
    }

    impl Frame for LazyFrame {
        fn is_valid(self) -> PolarsResult<LazyFrame> {
            let test_dtypes: bool = self
                .clone()
                .limit(0)
                .collect()?
                .dtypes()
                .into_iter()
                .map(|e| {
                    matches!(
                        e,
                        DataType::UInt8
                            | DataType::UInt16
                            | DataType::UInt32
                            | DataType::UInt64
                            | DataType::Int8
                            | DataType::Int16
                            | DataType::Int32
                            | DataType::Int64
                            | DataType::Float32
                            | DataType::Float64
                    )
                })
                .all(|e| e);
            let test_no_nulls: bool = self
                .clone()
                .null_count()
                .cast_all(DataType::UInt64, true)
                .with_column(
                    fold_exprs(
                        lit(0).cast(DataType::UInt64),
                        |acc, x| Ok(Some(acc + x)),
                        [col("*")],
                    )
                    .alias("sum"),
                )
                .select(&[col("sum")])
                .collect()?
                .column("sum")?
                .u64()?
                .into_iter()
                .map(|e| e.eq(&Some(0u64)))
                .collect::<Vec<_>>()[0];
            match (test_dtypes, test_no_nulls) {
                (true, true) => Ok(self),
                (false, true) => Err(PolarsError::InvalidOperation(
                    "frame contains non-numerical data".into(),
                )),
                (true, false) => Err(PolarsError::InvalidOperation(
                    "frame contains null entries".into(),
                )),
                (false, false) => Err(PolarsError::InvalidOperation(
                    "frame contains non-numerical data and null entries".into(),
                )),
            }
        }
    }

    macro_rules! polars_impl {
        ($ty: ident, $dtype: ident, $fn_name: ident) => {
            /// Converts a `polars` lazyframe into a [`Mat`].
            ///
            /// Note that this function expects that the frame passed "looks like"
            /// a numerical array and all values will be cast to either f32 or f64
            /// prior to building [`Mat`].
            ///
            /// Passing a frame with either non-numerical column data or null
            /// entries will result in a error. Users are expected to reolve
            /// these issues in `polars` prior calling this function.
            #[cfg(feature = "polars")]
            #[cfg_attr(docsrs, doc(cfg(feature = "polars")))]
            pub fn $fn_name(
                frame: impl Frame,
            ) -> PolarsResult<Mat<$ty>> {
                use core::{iter::zip, mem::MaybeUninit};

                fn implementation(
                    lf: LazyFrame,
                ) -> PolarsResult<Mat<$ty>> {
                    let df = lf
                        .select(&[col("*").cast(DataType::$dtype)])
                        .collect()?;

                    let nrows = df.height();
                    let ncols = df.get_column_names().len();

                    let mut out = Mat::<$ty>::with_capacity(df.height(), df.get_column_names().len());
                    let stride = out.col_stride() as usize;
                    let out_ptr = out.as_ptr_mut();

                    df.get_column_names().iter()
                        .enumerate()
                        .try_for_each(|(j, col)| -> PolarsResult<()> {
                            let mut row_start = 0usize;

                            // SAFETY: this is safe since we allocated enough space for `ncols` columns and
                            // `nrows` rows
                            let out_col = unsafe {
                                core::slice::from_raw_parts_mut(
                                    out_ptr.wrapping_add(stride * j) as *mut MaybeUninit<$ty>,
                                    nrows,
                                )
                            };

                            df.column(col)?.$ty()?.downcast_iter().try_for_each(
                                |chunk| -> PolarsResult<()> {
                                    let len = chunk.len();
                                    if len == 0 {
                                        return Ok(());
                                    }

                                    match row_start.checked_add(len) {
                                        Some(next_row_start) => {
                                            if next_row_start <= nrows {
                                                let mut out_slice = &mut out_col[row_start..next_row_start];
                                                let mut values = chunk.values_iter().as_slice();
                                                let validity = chunk.validity();

                                                assert_eq!(values.len(), len);

                                                match validity {
                                                    Some(bitmap) => {
                                                        let (mut bytes, offset, bitmap_len) = bitmap.as_slice();
                                                        assert_eq!(bitmap_len, len);
                                                        const BITS_PER_BYTE: usize = 8;

                                                        if offset > 0 {
                                                            let first_byte_len = Ord::min(len, 8 - offset);

                                                            let (out_prefix, out_suffix) = out_slice.split_at_mut(first_byte_len);
                                                            let (values_prefix, values_suffix) = values.split_at(first_byte_len);

                                                            for (out_elem, value_elem) in zip(
                                                                out_prefix,
                                                                values_prefix,
                                                            ) {
                                                                *out_elem = MaybeUninit::new(*value_elem)
                                                            }

                                                            bytes = &bytes[1..];
                                                            values = values_suffix;
                                                            out_slice = out_suffix;
                                                        }

                                                        if bytes.len() > 0 {
                                                            for (out_slice8, values8) in zip(
                                                                out_slice.chunks_exact_mut(BITS_PER_BYTE),
                                                                values.chunks_exact(BITS_PER_BYTE),
                                                            ) {
                                                                for (out_elem, value_elem) in zip(out_slice8, values8) {
                                                                    *out_elem = MaybeUninit::new(*value_elem);
                                                                }
                                                            }

                                                            for (out_elem, value_elem) in zip(
                                                                out_slice.chunks_exact_mut(BITS_PER_BYTE).into_remainder(),
                                                                values.chunks_exact(BITS_PER_BYTE).remainder(),
                                                            ) {
                                                                *out_elem = MaybeUninit::new(*value_elem);
                                                            }
                                                        }
                                                    }
                                                    None => {
                                                        // SAFETY: T and MaybeUninit<T> have the same layout
                                                        // NOTE: This state should not be reachable
                                                        let values = unsafe {
                                                            core::slice::from_raw_parts(
                                                                values.as_ptr() as *const MaybeUninit<$ty>,
                                                                values.len(),
                                                            )
                                                        };
                                                        out_slice.copy_from_slice(values);
                                                    }
                                                }

                                                row_start = next_row_start;
                                                Ok(())
                                            } else {
                                                Err(PolarsError::ShapeMismatch(
                                                    format!("too many values in column {col}").into(),
                                                ))
                                            }
                                        }
                                        None => Err(PolarsError::ShapeMismatch(
                                            format!("too many values in column {col}").into(),
                                        )),
                                    }
                                },
                            )?;

                            if row_start < nrows {
                                Err(PolarsError::ShapeMismatch(
                                    format!("not enough values in column {col} (column has {row_start} values, while dataframe has {nrows} rows)").into(),
                                ))
                            } else {
                                Ok(())
                            }
                        })?;

                    // SAFETY: we initialized every `ncols` columns, and each one was initialized with `nrows`
                    // elements
                    unsafe { out.set_dims(nrows, ncols) };

                    Ok(out)
                }

                implementation(frame.is_valid()?)
            }
        };
    }

    polars_impl!(f32, Float32, polars_to_faer_f32);
    polars_impl!(f64, Float64, polars_to_faer_f64);
}

#[cfg(feature = "numpy")]
#[cfg_attr(docsrs, doc(cfg(feature = "numpy")))]
const _: () = {
    use faer::prelude::*;
    use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods};
    use numpy::Element;

    impl<'a, T: Element + 'a> IntoFaer for PyReadonlyArray2<'a, T> {
        type Faer = MatRef<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let raw_arr = self.as_raw_array();
            let nrows = raw_arr.nrows();
            let ncols = raw_arr.ncols();
            let strides: [isize; 2] = raw_arr.strides().try_into().unwrap();
            let ptr = raw_arr.as_ptr();
            unsafe { MatRef::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
        }
    }

    impl<'a, T: Element + 'a> IntoFaer for PyReadonlyArray1<'a, T> {
        type Faer = ColRef<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let raw_arr = self.as_raw_array();
            let nrows = raw_arr.len();
            let strides: [isize; 1] = raw_arr.strides().try_into().unwrap();
            let ptr = raw_arr.as_ptr();
            unsafe { ColRef::from_raw_parts(ptr, nrows, strides[0]) }
        }
    }

};

#[cfg(test)]
mod tests {
    #![allow(unused_imports)]
    #![allow(non_snake_case)]

    use super::*;
    use faer::mat;
    use faer::prelude::*;

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_ext_ndarray() {
        let mut I_faer = Mat::<f32>::identity(8, 7);
        let mut I_ndarray = ndarray::Array2::<f32>::zeros([8, 7]);
        I_ndarray.diag_mut().fill(1.0);

        assert_eq!(I_ndarray.view().into_faer(), I_faer);
        assert_eq!(I_faer.as_ref().into_ndarray(), I_ndarray);

        assert_eq!(I_ndarray.view_mut().into_faer(), I_faer);
        assert_eq!(I_faer.as_mut().into_ndarray(), I_ndarray);
    }

    #[cfg(feature = "nalgebra")]
    #[test]
    fn test_ext_nalgebra() {
        let mut I_faer = Mat::<f32>::identity(8, 7);
        let mut I_nalgebra = nalgebra::DMatrix::<f32>::identity(8, 7);

        assert_eq!(I_nalgebra.view_range(.., ..).into_faer(), I_faer);
        assert_eq!(I_faer.as_ref().into_nalgebra(), I_nalgebra);

        assert_eq!(I_nalgebra.view_range_mut(.., ..).into_faer(), I_faer);
        assert_eq!(I_faer.as_mut().into_nalgebra(), I_nalgebra);
    }

    #[cfg(feature = "polars")]
    #[test]
    fn test_polars_pos() {
        use crate::polars::{polars_to_faer_f32, polars_to_faer_f64};
        #[rustfmt::skip]
        use ::polars::prelude::*;

        let s0: Series = Series::new("a", [1, 2, 3]);
        let s1: Series = Series::new("b", [10, 11, 12]);

        let lf = DataFrame::new(vec![s0, s1]).unwrap().lazy();

        let arr_32 = polars_to_faer_f32(lf.clone()).unwrap();
        let arr_64 = polars_to_faer_f64(lf).unwrap();

        let expected_32 = mat![[1f32, 10f32], [2f32, 11f32], [3f32, 12f32]];
        let expected_64 = mat![[1f64, 10f64], [2f64, 11f64], [3f64, 12f64]];

        assert_eq!(arr_32, expected_32);
        assert_eq!(arr_64, expected_64);
    }

    #[cfg(feature = "polars")]
    #[test]
    #[should_panic(expected = "frame contains null entries")]
    fn test_polars_neg_32_null() {
        use crate::polars::polars_to_faer_f32;
        #[rustfmt::skip]
        use ::polars::prelude::*;

        let s0: Series = Series::new("a", [1, 2, 3]);
        let s1: Series = Series::new("b", [Some(10), Some(11), None]);

        let lf = DataFrame::new(vec![s0, s1]).unwrap().lazy();

        polars_to_faer_f32(lf).unwrap();
    }

    #[cfg(feature = "polars")]
    #[test]
    #[should_panic(expected = "frame contains non-numerical data")]
    fn test_polars_neg_32_strl() {
        use crate::polars::polars_to_faer_f32;
        #[rustfmt::skip]
        use ::polars::prelude::*;

        let s0: Series = Series::new("a", [1, 2, 3]);
        let s1: Series = Series::new("b", ["fish", "dog", "crocodile"]);

        let lf = DataFrame::new(vec![s0, s1]).unwrap().lazy();

        polars_to_faer_f32(lf).unwrap();
    }

    #[cfg(feature = "polars")]
    #[test]
    #[should_panic(expected = "frame contains non-numerical data and null entries")]
    fn test_polars_neg_32_combo() {
        use crate::polars::polars_to_faer_f32;
        #[rustfmt::skip]
        use ::polars::prelude::*;

        let s0: Series = Series::new("a", [1, 2, 3]);
        let s1: Series = Series::new("b", [Some(10), Some(11), None]);
        let s2: Series = Series::new("c", [Some("fish"), Some("dog"), None]);

        let lf = DataFrame::new(vec![s0, s1, s2]).unwrap().lazy();

        polars_to_faer_f32(lf).unwrap();
    }

    #[cfg(feature = "polars")]
    #[test]
    #[should_panic(expected = "frame contains null entries")]
    fn test_polars_neg_64_null() {
        use crate::polars::polars_to_faer_f64;
        #[rustfmt::skip]
        use ::polars::prelude::*;

        let s0: Series = Series::new("a", [1, 2, 3]);
        let s1: Series = Series::new("b", [Some(10), Some(11), None]);

        let lf = DataFrame::new(vec![s0, s1]).unwrap().lazy();

        polars_to_faer_f64(lf).unwrap();
    }

    #[cfg(feature = "polars")]
    #[test]
    #[should_panic(expected = "frame contains non-numerical data")]
    fn test_polars_neg_64_strl() {
        use crate::polars::polars_to_faer_f64;
        #[rustfmt::skip]
        use ::polars::prelude::*;

        let s0: Series = Series::new("a", [1, 2, 3]);
        let s1: Series = Series::new("b", ["fish", "dog", "crocodile"]);

        let lf = DataFrame::new(vec![s0, s1]).unwrap().lazy();

        polars_to_faer_f64(lf).unwrap();
    }

    #[cfg(feature = "polars")]
    #[test]
    #[should_panic(expected = "frame contains non-numerical data and null entries")]
    fn test_polars_neg_64_combo() {
        use crate::polars::polars_to_faer_f64;
        #[rustfmt::skip]
        use ::polars::prelude::*;

        let s0: Series = Series::new("a", [1, 2, 3]);
        let s1: Series = Series::new("b", [Some(10), Some(11), None]);
        let s2: Series = Series::new("c", [Some("fish"), Some("dog"), None]);

        let lf = DataFrame::new(vec![s0, s1, s2]).unwrap().lazy();

        polars_to_faer_f64(lf).unwrap();
    }

    #[cfg(feature = "numpy")]
    #[test]
    fn test_ext_numpy() {
        use pyo3::{Bound, Python};
        use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods};
        use numpy::ndarray::array;
        use numpy::pyarray;
        use pyo3::prelude::*;
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let arr_1 = PyArray1::from_vec(py, vec![1., 0., 1., 0.]);
            let py_array1: PyReadonlyArray1<f64> = arr_1.readonly();
            let expected_f64: Mat<f64> = mat![[1., 0., 1., 0.]];
            assert_eq!(py_array1.into_faer(), expected_f64.transpose().col(0));

            let arr_2 = pyarray![py, [1., 0.], [0., 1.]];
            let py_array2: PyReadonlyArray2<f64> = arr_2.readonly();
            let expected_f64 = Mat::<f64>::identity(2, 2);
            assert_eq!(py_array2.into_faer(), expected_f64.as_ref());

            let arr_1 = PyArray1::from_vec(py, vec![1., 0., 1., 0.]);
            let py_array1: PyReadonlyArray1<f32> = arr_1.readonly();
            let expected_f32: Mat<f32> = mat![[1., 0., 1., 0.]];
            assert_eq!(py_array1.into_faer(), expected_f32.transpose().col(0));

            let arr_2 = pyarray![py, [1., 0.], [0., 1.]];
            let py_array2: PyReadonlyArray2<f32> = arr_2.readonly();
            let expected_f32 = Mat::<f32>::identity(2, 2);
            assert_eq!(py_array2.into_faer(), expected_f32.as_ref());
        });
    }
}
