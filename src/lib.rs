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

/// Conversions from external library matrix views into complex `faer` types.
pub trait IntoFaerComplex {
    type Faer;
    fn into_faer_complex(self) -> Self::Faer;
}

#[cfg(feature = "nalgebra")]
#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
/// Conversions from external library matrix views into complex `nalgebra` types.
pub trait IntoNalgebraComplex {
    type Nalgebra;
    fn into_nalgebra_complex(self) -> Self::Nalgebra;
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
/// Conversions from external library matrix views into complex `ndarray` types.
pub trait IntoNdarrayComplex {
    type Ndarray;
    fn into_ndarray_complex(self) -> Self::Ndarray;
}

#[cfg(feature = "nalgebra")]
#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
const _: () = {
    use faer::complex_native::*;
    use faer::prelude::*;
    use faer::SimpleEntity;
    use nalgebra::{Dim, Dyn, MatrixView, MatrixViewMut, ViewStorage, ViewStorageMut};
    use num_complex::{Complex32, Complex64};

    impl<'a, T: SimpleEntity, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoFaer
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
                faer::mat::from_raw_parts(
                    ptr,
                    nrows,
                    ncols,
                    strides.0.try_into().unwrap(),
                    strides.1.try_into().unwrap(),
                )
            }
        }
    }

    impl<'a, T: SimpleEntity, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoFaer
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
                faer::mat::from_raw_parts_mut::<'_, T>(
                    ptr,
                    nrows,
                    ncols,
                    strides.0.try_into().unwrap(),
                    strides.1.try_into().unwrap(),
                )
            }
        }
    }

    impl<'a, T: SimpleEntity> IntoNalgebra for MatRef<'a, T> {
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

    impl<'a, T: SimpleEntity> IntoNalgebra for MatMut<'a, T> {
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

    impl<'a, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoFaerComplex
        for MatrixView<'a, Complex32, R, C, RStride, CStride>
    {
        type Faer = MatRef<'a, c32>;

        #[track_caller]
        fn into_faer_complex(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides = self.strides();
            let ptr = self.as_ptr() as *const c32;
            unsafe {
                faer::mat::from_raw_parts(
                    ptr,
                    nrows,
                    ncols,
                    strides.0.try_into().unwrap(),
                    strides.1.try_into().unwrap(),
                )
            }
        }
    }

    impl<'a, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoFaerComplex
        for MatrixViewMut<'a, Complex32, R, C, RStride, CStride>
    {
        type Faer = MatMut<'a, c32>;

        #[track_caller]
        fn into_faer_complex(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides = self.strides();
            let ptr = { self }.as_mut_ptr() as *mut c32;
            unsafe {
                faer::mat::from_raw_parts_mut(
                    ptr,
                    nrows,
                    ncols,
                    strides.0.try_into().unwrap(),
                    strides.1.try_into().unwrap(),
                )
            }
        }
    }

    impl<'a> IntoNalgebraComplex for MatRef<'a, c32> {
        type Nalgebra = MatrixView<'a, Complex32, Dyn, Dyn, Dyn, Dyn>;

        #[track_caller]
        fn into_nalgebra_complex(self) -> Self::Nalgebra {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride = self.row_stride();
            let col_stride = self.col_stride();
            let ptr = self.as_ptr() as *const Complex32;
            unsafe {
                MatrixView::<'_, Complex32, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorage::<
                    '_,
                    Complex32,
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

    impl<'a> IntoNalgebraComplex for MatMut<'a, c32> {
        type Nalgebra = MatrixViewMut<'a, Complex32, Dyn, Dyn, Dyn, Dyn>;

        #[track_caller]
        fn into_nalgebra_complex(self) -> Self::Nalgebra {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride = self.row_stride();
            let col_stride = self.col_stride();
            let ptr = self.as_ptr_mut() as *mut Complex32;
            unsafe {
                MatrixViewMut::<'_, Complex32, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorageMut::<
                    '_,
                    Complex32,
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

    impl<'a, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoFaerComplex
        for MatrixView<'a, Complex64, R, C, RStride, CStride>
    {
        type Faer = MatRef<'a, c64>;

        #[track_caller]
        fn into_faer_complex(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides = self.strides();
            let ptr = self.as_ptr() as *const c64;
            unsafe {
                faer::mat::from_raw_parts(
                    ptr,
                    nrows,
                    ncols,
                    strides.0.try_into().unwrap(),
                    strides.1.try_into().unwrap(),
                )
            }
        }
    }

    impl<'a, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoFaerComplex
        for MatrixViewMut<'a, Complex64, R, C, RStride, CStride>
    {
        type Faer = MatMut<'a, c64>;

        #[track_caller]
        fn into_faer_complex(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides = self.strides();
            let ptr = { self }.as_mut_ptr() as *mut c64;
            unsafe {
                faer::mat::from_raw_parts_mut(
                    ptr,
                    nrows,
                    ncols,
                    strides.0.try_into().unwrap(),
                    strides.1.try_into().unwrap(),
                )
            }
        }
    }

    impl<'a> IntoNalgebraComplex for MatRef<'a, c64> {
        type Nalgebra = MatrixView<'a, Complex64, Dyn, Dyn, Dyn, Dyn>;

        #[track_caller]
        fn into_nalgebra_complex(self) -> Self::Nalgebra {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride = self.row_stride();
            let col_stride = self.col_stride();
            let ptr = self.as_ptr() as *const Complex64;
            unsafe {
                MatrixView::<'_, Complex64, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorage::<
                    '_,
                    Complex64,
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

    impl<'a> IntoNalgebraComplex for MatMut<'a, c64> {
        type Nalgebra = MatrixViewMut<'a, Complex64, Dyn, Dyn, Dyn, Dyn>;

        #[track_caller]
        fn into_nalgebra_complex(self) -> Self::Nalgebra {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride = self.row_stride();
            let col_stride = self.col_stride();
            let ptr = self.as_ptr_mut() as *mut Complex64;
            unsafe {
                MatrixViewMut::<'_, Complex64, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorageMut::<
                    '_,
                    Complex64,
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
    use faer::complex_native::*;
    use faer::prelude::*;
    use faer::SimpleEntity;
    use ndarray::{ArrayView, ArrayViewMut, IntoDimension, Ix2, ShapeBuilder};
    use num_complex::{Complex32, Complex64};

    impl<'a, T: SimpleEntity> IntoFaer for ArrayView<'a, T, Ix2> {
        type Faer = MatRef<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = self.as_ptr();
            unsafe { faer::mat::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
        }
    }

    impl<'a, T: SimpleEntity> IntoFaer for ArrayViewMut<'a, T, Ix2> {
        type Faer = MatMut<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = { self }.as_mut_ptr();
            unsafe {
                faer::mat::from_raw_parts_mut::<'_, T>(ptr, nrows, ncols, strides[0], strides[1])
            }
        }
    }

    impl<'a, T: SimpleEntity> IntoNdarray for MatRef<'a, T> {
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
                    (nrows, ncols)
                        .into_shape()
                        .strides((row_stride, col_stride).into_dimension()),
                    ptr,
                )
            }
        }
    }

    impl<'a, T: SimpleEntity> IntoNdarray for MatMut<'a, T> {
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
                    (nrows, ncols)
                        .into_shape()
                        .strides((row_stride, col_stride).into_dimension()),
                    ptr,
                )
            }
        }
    }

    impl<'a> IntoFaerComplex for ArrayView<'a, Complex32, Ix2> {
        type Faer = MatRef<'a, c32>;

        #[track_caller]
        fn into_faer_complex(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = self.as_ptr() as *const c32;
            unsafe { faer::mat::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
        }
    }

    impl<'a> IntoFaerComplex for ArrayViewMut<'a, Complex32, Ix2> {
        type Faer = MatMut<'a, c32>;

        #[track_caller]
        fn into_faer_complex(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = { self }.as_mut_ptr() as *mut c32;
            unsafe { faer::mat::from_raw_parts_mut(ptr, nrows, ncols, strides[0], strides[1]) }
        }
    }

    impl<'a> IntoNdarrayComplex for MatRef<'a, c32> {
        type Ndarray = ArrayView<'a, Complex32, Ix2>;

        #[track_caller]
        fn into_ndarray_complex(self) -> Self::Ndarray {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride: usize = self.row_stride().try_into().unwrap();
            let col_stride: usize = self.col_stride().try_into().unwrap();
            let ptr = self.as_ptr() as *const Complex32;
            unsafe {
                ArrayView::<'_, Complex32, Ix2>::from_shape_ptr(
                    (nrows, ncols)
                        .into_shape()
                        .strides((row_stride, col_stride).into_dimension()),
                    ptr,
                )
            }
        }
    }

    impl<'a> IntoNdarrayComplex for MatMut<'a, c32> {
        type Ndarray = ArrayViewMut<'a, Complex32, Ix2>;

        #[track_caller]
        fn into_ndarray_complex(self) -> Self::Ndarray {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride: usize = self.row_stride().try_into().unwrap();
            let col_stride: usize = self.col_stride().try_into().unwrap();
            let ptr = self.as_ptr_mut() as *mut Complex32;
            unsafe {
                ArrayViewMut::<'_, Complex32, Ix2>::from_shape_ptr(
                    (nrows, ncols)
                        .into_shape()
                        .strides((row_stride, col_stride).into_dimension()),
                    ptr,
                )
            }
        }
    }

    impl<'a> IntoFaerComplex for ArrayView<'a, Complex64, Ix2> {
        type Faer = MatRef<'a, c64>;

        #[track_caller]
        fn into_faer_complex(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = self.as_ptr() as *const c64;
            unsafe { faer::mat::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
        }
    }

    impl<'a> IntoFaerComplex for ArrayViewMut<'a, Complex64, Ix2> {
        type Faer = MatMut<'a, c64>;

        #[track_caller]
        fn into_faer_complex(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = { self }.as_mut_ptr() as *mut c64;
            unsafe { faer::mat::from_raw_parts_mut(ptr, nrows, ncols, strides[0], strides[1]) }
        }
    }

    impl<'a> IntoNdarrayComplex for MatRef<'a, c64> {
        type Ndarray = ArrayView<'a, Complex64, Ix2>;

        #[track_caller]
        fn into_ndarray_complex(self) -> Self::Ndarray {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride: usize = self.row_stride().try_into().unwrap();
            let col_stride: usize = self.col_stride().try_into().unwrap();
            let ptr = self.as_ptr() as *const Complex64;
            unsafe {
                ArrayView::<'_, Complex64, Ix2>::from_shape_ptr(
                    (nrows, ncols)
                        .into_shape()
                        .strides((row_stride, col_stride).into_dimension()),
                    ptr,
                )
            }
        }
    }

    impl<'a> IntoNdarrayComplex for MatMut<'a, c64> {
        type Ndarray = ArrayViewMut<'a, Complex64, Ix2>;

        #[track_caller]
        fn into_ndarray_complex(self) -> Self::Ndarray {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride: usize = self.row_stride().try_into().unwrap();
            let col_stride: usize = self.col_stride().try_into().unwrap();
            let ptr = self.as_ptr_mut() as *mut Complex64;
            unsafe {
                ArrayViewMut::<'_, Complex64, Ix2>::from_shape_ptr(
                    (nrows, ncols)
                        .into_shape()
                        .strides((row_stride, col_stride).into_dimension()),
                    ptr,
                )
            }
        }
    }
};

#[cfg(all(feature = "nalgebra", feature = "ndarray"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "nalgebra", feature = "ndarray"))))]
const _: () =
    {
        use nalgebra::{Dim, Dyn, MatrixView, MatrixViewMut, ViewStorage, ViewStorageMut};
        use ndarray::{ArrayView, ArrayViewMut, IntoDimension, Ix2, ShapeBuilder};
        use num_complex::Complex;

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
                        (nrows, ncols)
                            .into_shape()
                            .strides((row_stride, col_stride).into_dimension()),
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
                        (nrows, ncols)
                            .into_shape()
                            .strides((row_stride, col_stride).into_dimension()),
                        ptr,
                    )
                }
            }
        }

        impl<'a, T> IntoNalgebraComplex for ArrayView<'a, Complex<T>, Ix2> {
            type Nalgebra = MatrixView<'a, Complex<T>, Dyn, Dyn, Dyn, Dyn>;

            #[track_caller]
            fn into_nalgebra_complex(self) -> Self::Nalgebra {
                let nrows = self.nrows();
                let ncols = self.ncols();
                let [row_stride, col_stride]: [isize; 2] = self.strides().try_into().unwrap();
                let ptr = self.as_ptr();

                unsafe {
                    MatrixView::<'_, Complex<T>, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorage::<
                        '_,
                        Complex<T>,
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
        impl<'a, T> IntoNalgebraComplex for ArrayViewMut<'a, Complex<T>, Ix2> {
            type Nalgebra = MatrixViewMut<'a, Complex<T>, Dyn, Dyn, Dyn, Dyn>;

            #[track_caller]
            fn into_nalgebra_complex(self) -> Self::Nalgebra {
                let nrows = self.nrows();
                let ncols = self.ncols();
                let [row_stride, col_stride]: [isize; 2] = self.strides().try_into().unwrap();
                let ptr = { self }.as_mut_ptr();

                unsafe {
                    MatrixViewMut::<'_, Complex<T>, Dyn, Dyn, Dyn, Dyn>::from_data(
                        ViewStorageMut::<'_, Complex<T>, Dyn, Dyn, Dyn, Dyn>::from_raw_parts(
                            ptr,
                            (Dyn(nrows), Dyn(ncols)),
                            (
                                Dyn(row_stride.try_into().unwrap()),
                                Dyn(col_stride.try_into().unwrap()),
                            ),
                        ),
                    )
                }
            }
        }

        impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoNdarrayComplex
            for MatrixView<'a, Complex<T>, R, C, RStride, CStride>
        {
            type Ndarray = ArrayView<'a, Complex<T>, Ix2>;

            #[track_caller]
            fn into_ndarray_complex(self) -> Self::Ndarray {
                let nrows = self.nrows();
                let ncols = self.ncols();
                let (row_stride, col_stride) = self.strides();
                let ptr = self.as_ptr();

                unsafe {
                    ArrayView::<'_, Complex<T>, Ix2>::from_shape_ptr(
                        (nrows, ncols)
                            .into_shape()
                            .strides((row_stride, col_stride).into_dimension()),
                        ptr,
                    )
                }
            }
        }
        impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoNdarrayComplex
            for MatrixViewMut<'a, Complex<T>, R, C, RStride, CStride>
        {
            type Ndarray = ArrayViewMut<'a, Complex<T>, Ix2>;

            #[track_caller]
            fn into_ndarray_complex(self) -> Self::Ndarray {
                let nrows = self.nrows();
                let ncols = self.ncols();
                let (row_stride, col_stride) = self.strides();
                let ptr = { self }.as_mut_ptr();

                unsafe {
                    ArrayViewMut::<'_, Complex<T>, Ix2>::from_shape_ptr(
                        (nrows, ncols)
                            .into_shape()
                            .strides((row_stride, col_stride).into_dimension()),
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

                    df.get_column_names().iter()
                        .enumerate()
                        .try_for_each(|(j, col)| -> PolarsResult<()> {
                            let mut row_start = 0usize;

                            // SAFETY: this is safe since we allocated enough space for `ncols` columns and
                            // `nrows` rows
                            let out_col = unsafe {
                                core::slice::from_raw_parts_mut(
                                    out.as_mut().ptr_at_mut(0, j) as *mut MaybeUninit<$ty>,
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
}
