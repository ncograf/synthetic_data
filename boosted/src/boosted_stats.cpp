#include <Eigen/Dense>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <chrono>
#include <iostream>

namespace p = boost::python;
namespace np = boost::python::numpy;

namespace boosted_stats {

template <typename Scalar>
np::ndarray lag_prod_mean(np::ndarray arr, int max_lag) {

  int n_col = arr.shape(1);
  int n_row = arr.shape(0);
  int scalar_size = sizeof(Scalar);
  int col_stride = arr.strides(1) / scalar_size;
  int row_stride = arr.strides(0) / scalar_size;
  Scalar *data = (Scalar *)arr.get_data();

  // RowMajor Layout because it is given in numpy and we use if for faster
  // computation
  auto t1 = std::chrono::high_resolution_clock::now();
  Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> mat =
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::ColMajor>,
                 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(
          data, n_row, n_col,
          Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(col_stride,
                                                        row_stride));
  auto t2 = std::chrono::high_resolution_clock::now();
  auto d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  std::cout << "Map Time :" << d1.count() << "ms\n";

  // Count non nan entries
  t1 = std::chrono::high_resolution_clock::now();
  Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic> zero_mat =
      Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(n_row, n_col);

  Eigen::Array<Scalar, 1, Eigen::Dynamic> count =
      (mat.array() == mat.array()).select(1, zero_mat).colwise().sum();
  Eigen::Array<Scalar, 1, Eigen::Dynamic> ones =
      Eigen::Array<Scalar, 1, Eigen::Dynamic>::Ones(n_col);

  // std::cout << "Matrix \n" << mat << std::endl;
  mat = (mat.isNaN()).select(0, mat);
  data = mat.data();

  Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out(
      max_lag, n_col);
  out.setZero();

  t2 = std::chrono::high_resolution_clock::now();
  d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  std::cout << "Prep Time :" << d1.count() << "ms\n";

  t1 = std::chrono::high_resolution_clock::now();
  for (int lag = 1; lag <= max_lag; lag++) {

    // r_t * r_{t-lag}
    out.row(lag - 1) = (mat.bottomRows(n_row - lag) * mat.topRows(n_row - lag))
                           .colwise()
                           .sum() /
                       (count - ones * lag);
  }
  t2 = std::chrono::high_resolution_clock::now();
  d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  std::cout << "Loop Time :" << d1.count() << "ms\n";

  np::dtype dtype = arr.get_dtype();
  p::tuple shape = p::make_tuple(out.rows(), out.cols());
  p::tuple stride = p::make_tuple(1, 1);
  np::ndarray out_arr = np::zeros(shape, dtype);

  t1 = std::chrono::high_resolution_clock::now();
  // copy data to the newly generated numpy array
  std::copy(out.data(), out.data() + max_lag * n_col,
            reinterpret_cast<Scalar *>(out_arr.get_data()));
  t2 = std::chrono::high_resolution_clock::now();
  d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  std::cout << "Post Time :" << d1.count() << "ms\n";

  return out_arr;
}

BOOST_PYTHON_MODULE(boosted_stats) {
  Py_Initialize();
  np::initialize();
  p::def("lag_prod_mean_double", lag_prod_mean<double>);
  p::def("lag_prod_mean_float", lag_prod_mean<float>);
}

} // namespace boosted_stats
