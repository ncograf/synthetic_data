#include <Eigen/Dense>
#include <boost/multi_array.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace p = boost::python;
namespace np = boost::python::numpy;

namespace boosted_stats {

/** Comptue Leverage effect
 *
 *  ```
 *  $L(k) = \frac{\langle r_t r_{t+k}² \rangle}{\langle r_{t+k}^2 \rangle^2}$
 *  ```
 *
 *  This formula is obtained from DOI:10.1016/j.physa.2016.12.021, note that
 *  it deviates from other papers.
 *
 */
template <typename Scalar>
np::ndarray leverage_effect(np::ndarray arr, int max_lag, bool verbose) {

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
  if (verbose)
    std::cout << "Map Time :" << d1.count() << "ms\n";

  // Count non nan entries
  t1 = std::chrono::high_resolution_clock::now();
  Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic> zero_mat =
      Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(n_row, n_col);

  Eigen::Array<Scalar, 1, Eigen::Dynamic> count =
      (mat.isNaN()).select(zero_mat, 1).colwise().sum();
  Eigen::Array<Scalar, 1, Eigen::Dynamic> ones =
      Eigen::Array<Scalar, 1, Eigen::Dynamic>::Ones(n_col);

  mat = (mat.isNaN()).select(0, mat);
  data = mat.data();

  // squared array
  Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
      squared = mat * mat;

  // out array
  Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out(
      max_lag, n_col);
  out.setZero();

  t2 = std::chrono::high_resolution_clock::now();
  d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  if (verbose)
    std::cout << "Prep Time :" << d1.count() << "ms\n";

  // std::cout << "Matrix \n" << mat << std::endl;

  t1 = std::chrono::high_resolution_clock::now();
  for (int lag = 1; lag <= max_lag; lag++) {

    // r_t * r_{t-lag}
    Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic> enumerator =
        (mat.topRows(n_row - lag) * squared.bottomRows(n_row - lag))
            .colwise()
            .sum() /
        (count - ones * lag);
    Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic> denominator =
        (squared.bottomRows(n_row - lag).colwise().sum() /
         (count - ones * lag));
    out.row(lag - 1) = enumerator / (denominator * denominator);
    // std::cout << "Enum \n" << enumerator << std::endl;
    // std::cout << "Top Squared \n" << squared.topRows(n_row - lag) <<
    // std::endl;
  }

  t2 = std::chrono::high_resolution_clock::now();
  d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  if (verbose)
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

template <typename Scalar>
p::tuple gain_loss_asym(np::ndarray arr, int max_lag, Scalar theta,
                        bool verbose) {

  auto t1 = std::chrono::high_resolution_clock::now();

  int n_col = arr.shape(1);
  int n_row = arr.shape(0);
  int scalar_size = sizeof(Scalar);
  int col_stride = arr.strides(1) / scalar_size;
  int row_stride = arr.strides(0) / scalar_size;
  Scalar *in_data = (Scalar *)arr.get_data();

  np::dtype dtype = arr.get_dtype();
  p::tuple shape = p::make_tuple(max_lag + 1, n_col);
  np::ndarray gain = np::zeros(shape, dtype);
  np::ndarray loss = np::zeros(shape, dtype);
  Scalar *gain_p = (Scalar *)gain.get_data();
  Scalar *loss_p = (Scalar *)loss.get_data();
  int col_stride_gain = gain.strides(1) / scalar_size;
  int row_stride_gain = gain.strides(0) / scalar_size;
  int col_stride_loss = loss.strides(1) / scalar_size;
  int row_stride_loss = loss.strides(0) / scalar_size;

  for (int c = 0; c < n_col; c++) {
    for (int t = 0; t < n_row; t++) {
      Scalar p_t = *(in_data + c * col_stride + row_stride * t);
      for (int t_ = t + 1; (t_ <= max_lag + t) && (t_ < n_row); t_++) {
        Scalar p_t_ = *(in_data + c * col_stride + row_stride * t_);
        if (p_t_ - p_t >= theta) {
          Scalar *data_loc =
              gain_p + c * col_stride_gain + row_stride_gain * (t_ - t);
          *data_loc += 1;
          break; // go to the next t
        }
      }
    }
  }

  for (int c = 0; c < n_col; c++) {
    for (int t = 0; t < n_row; t++) {
      Scalar p_t = *(in_data + c * col_stride + row_stride * t);
      for (int t_ = t + 1; (t_ <= max_lag + t) && (t_ < n_row); t_++) {
        Scalar p_t_ = *(in_data + c * col_stride + row_stride * t_);
        if (p_t_ - p_t <= -theta) {
          Scalar *data_loc =
              loss_p + c * col_stride_loss + row_stride_loss * (t_ - t);
          *data_loc += 1;
          break; // go to the next t
        }
      }
    }
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  auto d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  if (verbose)
    std::cout << "Loops Time :" << d1.count() << "ms\n";

  return p::make_tuple(gain, loss);
}

/**
 * Computes the laged product mean over the colums for all lags in 1 to
 * max_lag The lagged product mean is defined as
 *
 * ```
 *  1/n \sum_{t=1}^{n} arr_pos_lag[t + lag] * arr[t]
 * ```
 * Note that the second argument will be treated as lagging behind
 *
 */
template <typename Scalar>
np::ndarray lag_prod_mean_two(np::ndarray arr_pos_lag, np::ndarray arr,
                              int max_lag) {

  int scalar_size = sizeof(Scalar);

  int n_col = arr.shape(1);
  int n_row = arr.shape(0);

  if (arr_pos_lag.shape(1) != n_col || arr_pos_lag.shape(0) != n_row) {
    throw std::invalid_argument("The two input matrices have different sizes");
  }

  int col_stride = arr.strides(1) / scalar_size;
  int row_stride = arr.strides(0) / scalar_size;
  Scalar *data_arr = (Scalar *)arr.get_data();

  // RowMajor Layout because it is given in numpy and we use if for faster
  // computation
  auto t1 = std::chrono::high_resolution_clock::now();
  Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
      mat_arr = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::ColMajor>,
                           0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(
          data_arr, n_row, n_col,
          Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(col_stride,
                                                        row_stride));

  int col_stride_pos_lag = arr_pos_lag.strides(1) / scalar_size;
  int row_stride_pos_lag = arr_pos_lag.strides(0) / scalar_size;
  Scalar *data_arr_pos_lag = (Scalar *)arr_pos_lag.get_data();
  Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
      mat_pos_lag =
          Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::ColMajor>,
                     0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(
              data_arr_pos_lag, n_row, n_col,
              Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                  col_stride_pos_lag, row_stride_pos_lag));

  auto t2 = std::chrono::high_resolution_clock::now();
  auto d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  std::cout << "Map Time :" << d1.count() << "ms\n";

  // Count non nan entries
  t1 = std::chrono::high_resolution_clock::now();

  Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic> zero_mat =
      Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(n_row, n_col);

  Eigen::Array<Scalar, 1, Eigen::Dynamic> count =
      (mat_arr.isNaN()).select(zero_mat, 1).colwise().sum();
  Eigen::Array<Scalar, 1, Eigen::Dynamic> count_pos_lag =
      (mat_pos_lag.isNaN()).select(zero_mat, 1).colwise().sum();

  Eigen::Array<Scalar, 1, Eigen::Dynamic> ones =
      Eigen::Array<Scalar, 1, Eigen::Dynamic>::Ones(n_col);

  // std::cout << "Matrix \n" << mat_arr << std::endl;
  // std::cout << "Count \n" << count << std::endl;
  // std::cout << "Count_pos \n" << count_pos_lag << std::endl;
  mat_arr = (mat_arr.isNaN()).select(0, mat_arr);
  mat_pos_lag = (mat_pos_lag.isNaN()).select(0, mat_pos_lag);

  Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out(
      max_lag, n_col);
  out.setZero();

  t2 = std::chrono::high_resolution_clock::now();
  d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  std::cout << "Prep Time :" << d1.count() << "ms\n";

  Eigen::Array<Scalar, 1, Eigen::Dynamic> curr_count = count;

  t1 = std::chrono::high_resolution_clock::now();
  for (int lag = 1; lag <= max_lag; lag++) {

    // lag of the positive count nans are ignored
    curr_count = count.min(count_pos_lag + lag);

    // r_t * r_{t-lag}
    out.row(lag - 1) =
        (mat_pos_lag.bottomRows(n_row - lag) * mat_arr.topRows(n_row - lag))
            .colwise()
            .sum() /
        (curr_count - ones * lag);
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
      (mat.isNaN()).select(zero_mat, 1).colwise().sum();
  Eigen::Array<Scalar, 1, Eigen::Dynamic> ones =
      Eigen::Array<Scalar, 1, Eigen::Dynamic>::Ones(n_col);

  // std::cout << "Matrix \n" << mat << std::endl;
  // std::cout << "Count \n" << count << std::endl;
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
  p::def("lag_prod_mean_double", lag_prod_mean<double>,
         "Compute 1/n \\sum data[t-lag] data[t]");
  p::def("lag_prod_mean_float", lag_prod_mean<float>);
  p::def("lag_prod_mean_double", lag_prod_mean_two<double>);
  p::def("lag_prod_mean_float", lag_prod_mean_two<float>);
  p::def("leverage_effect_double", leverage_effect<double>);
  p::def("leverage_effect_float", leverage_effect<float>);
  p::def("gain_loss_asym_double", gain_loss_asym<double>);
  p::def("gain_loss_asym_float", gain_loss_asym<float>);
}

} // namespace boosted_stats
