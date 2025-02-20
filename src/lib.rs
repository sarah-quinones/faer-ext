#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "ndarray")]
use ndarray::{IntoDimension, ShapeArg};
use pyo3::{Bound, Python};

/// Conversions from external library matrix views into `faer` types.
pub trait IntoFaer {
    type Faer;
    fn into_faer(self) -> Self::Faer;
}

/// Conversions from external library matrix views into complex `faer` types.
pub trait IntoFaerComplex {
    type Faer;
    fn into_faer_complex(self) -> Self::Faer;
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
/// Conversions from external library matrix views into `ndarray` types.
pub trait IntoNdarray {
    type Ndarray;
    fn into_ndarray(self) -> Self::Ndarray;
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
/// Conversions from external library matrix views into complex `ndarray` types.
pub trait IntoNdarrayComplex {
    type Ndarray;
    fn into_ndarray_complex(self) -> Self::Ndarray;
}

// pub trait IntoPyArray2<'py> {
//     type NpArray;
//     fn into_pyarray(self, py: Python<'py>) -> Self::NpArray;
// }


const _: () = {
    use faer::prelude::*;
    use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods};
    use numpy::Element;

    //impl<'a> IntoFaer for PyReadonlyArray2<'a, f64> {
    //    type Faer = MatRef<'a, f64>;
    impl<'a, T: Element + 'a> IntoFaer for PyReadonlyArray2<'a, T> {
        type Faer = MatRef<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let raw_arr = self.as_raw_array();
            let nrows = raw_arr.nrows();
            let ncols = raw_arr.ncols();
            let strides = raw_arr.strides();
            let ptr = raw_arr.as_ptr();
            println!("{:?}", strides);
            unsafe { MatRef::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
        }
    }

    impl<'a, T: Element + 'a> IntoFaer for PyReadonlyArray1<'a, T> {
        type Faer = ColRef<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let raw_arr = self.as_raw_array();
            let nrows = raw_arr.len();
            let strides = raw_arr.strides();
            let ptr = raw_arr.as_ptr();
            unsafe { ColRef::from_raw_parts(ptr, nrows, strides[0]) }
        }
    }

};

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
const _: () = {
    // use faer::complex_native::*;
    use faer::prelude::*;
    use faer_traits::RealField;
    // use faer::SimpleEntity;
    use ndarray::{ArrayView, ArrayViewMut, Ix2, ShapeBuilder};
    use num_complex::{Complex32, Complex64};

    impl<'a, T: RealField> IntoFaer for ArrayView<'a, T, Ix2> {
        type Faer = MatRef<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = self.as_ptr();
            unsafe { MatRef::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
        }
    }

    impl<'a, T: RealField> IntoFaer for ArrayViewMut<'a, T, Ix2> {
        type Faer = MatMut<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = { self }.as_mut_ptr();
            unsafe {
                MatMut::from_raw_parts_mut(
                    ptr, nrows, ncols, strides[0], strides[1],
                )
            }
        }
    }

    impl<'a, T: RealField> IntoNdarray for MatRef<'a, T> {
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

    impl<'a, T: RealField> IntoNdarray for MatMut<'a, T> {
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

    impl<'a> IntoFaerComplex for ArrayView<'a, Complex32, Ix2> {
        type Faer = MatRef<'a, c32>;

        #[track_caller]
        fn into_faer_complex(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = self.as_ptr() as *const c32;
            unsafe { MatRef::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
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
            unsafe { MatMut::from_raw_parts_mut(ptr, nrows, ncols, strides[0], strides[1]) }
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
                    (nrows, ncols).strides((row_stride, col_stride)),
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
                    (nrows, ncols).strides((row_stride, col_stride)),
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
            unsafe { MatRef::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
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
            unsafe { MatMut::from_raw_parts_mut(ptr, nrows, ncols, strides[0], strides[1]) }
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
                    (nrows, ncols).strides((row_stride, col_stride)),
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
                    (nrows, ncols).strides((row_stride, col_stride)),
                    ptr,
                )
            }
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

    #[test]
    fn test_ext_pyarray() {
        use numpy::{IntoPyArray, PyArray2, PyArray, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods};
        use numpy::ndarray::array;
        use numpy::pyarray;
        use pyo3::prelude::*;
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let py_array: PyReadonlyArray2<f64> = pyarray![py, [1.0, 3.], [3., 2.]].extract().unwrap();
            // let py_array: PyReadonlyArray2<f64> = array![[1.0f64, 0.], [0., 1.]].into_pyarray(py).extract().unwrap();
            let I_faer = Mat::<f64>::identity(2, 2);
            // println!("{:?}", py_array.into_faer());
            assert_eq!(py_array.into_faer(), I_faer.as_ref());

        });
    }

}
