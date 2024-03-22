#include "lag_prod_mean.h"

namespace boosted_stats {

template <typename Scalar>
np::ndarray lag_prod_mean(np::ndarray arr, int max_lag) {

  int n_col = arr.shape(1);
  int n_row = arr.shape(0);
  Scalar *data = (Scalar *)arr.get_data();

  // RowMajor Layout because it is given in numpy and we use if for faster
  // computation
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat =
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>(data, n_row, n_col);
  // clean data

  // Count non nan entries
  Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      zero_mat = Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>::Zero(n_row, n_col);

  Eigen::Array<Scalar, 1, Eigen::Dynamic, Eigen::RowMajor> count =
      (mat.array() == mat.array()).select(1, zero_mat).colwise().sum();
  Eigen::Array<Scalar, 1, Eigen::Dynamic, Eigen::RowMajor> ones =
      Eigen::Array<Scalar, 1, Eigen::Dynamic, Eigen::RowMajor>::Ones(n_col);

  mat = (mat.array() == mat.array()).select(mat, 0).matrix();
  data = mat.data();
  std::cout << "mat\n" << mat << std::endl << data << std::endl << std::endl;

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out(
      max_lag, n_col);
  out.setZero();

  for (int lag = 1; lag <= max_lag; lag++) {

    std::cout << "Lag" << lag << std::endl;
    std::cout << "Row" << n_row << std::endl;

    // r_t * r_{t-lag}
    out.row(max_lag - lag) =
        (mat.bottomRows(n_row - lag).array() * mat.topRows(n_row - lag).array())
            .colwise()
            .sum() /
        (count - ones * lag);

    std::cout << "r_t_1\n"
              << mat.topRows(n_row - lag) << std::endl
              << mat.topRows(n_row - lag).data() << std::endl
              << std::endl;

    std::cout << "r_t\n"
              << mat.bottomRows(n_row - lag) << std::endl
              << mat.bottomRows(n_row - lag).data() << std::endl
              << std::endl;
    std::cout << out << std::endl << std::endl;
  }

  np::dtype dtype = arr.get_dtype();
  std::cout << p::extract<char const *>(p::str(dtype)) << std::endl;
  std::cout << p::extract<char const *>(
                   p::str(np::dtype::get_builtin<Scalar>()))
            << std::endl;
  p::tuple shape = p::make_tuple(out.rows(), out.cols());
  std::cout << p::extract<char const *>(p::str(shape)) << std::endl;
  p::tuple stride = p::make_tuple(1, 1);
  std::cout << p::extract<char const *>(p::str(stride)) << std::endl;
  np::ndarray out_arr = np::zeros(shape, dtype);

  // copy data to the newly generated numpy array
  std::copy(out.data(), out.data() + max_lag * n_col,
            reinterpret_cast<Scalar *>(out_arr.get_data()));

  return out_arr;
}

} // namespace boosted_stats
