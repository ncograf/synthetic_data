#ifndef LAG_PROD_MEAN_H
#define LAG_PROD_MEAN_H

#include <Eigen/Dense>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>

namespace p = boost::python;
namespace np = boost::python::numpy;

namespace boosted_stats {

/**
 * Helper function to compute \E_{t}[ r_t * r_{t-k} ] for all 1 <= k <=
 * `max_lag` on the sample data `arr`
 *
 * @param arr numpy array containing the time data in the columns
 * @param max_lag number of k to be computed
 */
template <typename Scalar>
np::ndarray lag_prod_mean(np::ndarray arr, int max_lag);

} // namespace boosted_stats

#endif // LAG_PROD_MEAN_H
