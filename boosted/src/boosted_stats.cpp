#include <Eigen/Dense>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <lag_prod_mean.h>

namespace p = boost::python;
namespace np = boost::python::numpy;

namespace boosted_stats {

template <typename Scalar>
np::ndarray lag_prod_mean_wrap(np::ndarray arr, int max_lag) {
  return lag_prod_mean<Scalar>(arr, max_lag);
}

BOOST_PYTHON_MODULE(boosted_stats) {
  Py_Initialize();
  np::initialize();
  p::def("lag_prod_mean", lag_prod_mean_wrap<double>);
  p::def("lag_prod_mean", lag_prod_mean_wrap<float>);
}

} // namespace boosted_stats
