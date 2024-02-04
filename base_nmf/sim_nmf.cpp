#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>
#include <random>

Eigen::MatrixXd simulateNMF(int nrow, int ncol, int k, double noise = 0.5,
                            double dropout = 0.5, int seed = NULL) {
  #include <cmath>

  if (seed != NULL) {
    std::srand(seed);
  }

  // Number of non-zeros
  Eigen::VectorXd num_nzh = (Eigen::VectorXd::Random(k).array() * ncol / 4 + ncol / 2).abs().round();
  Eigen::VectorXd num_nzw = (Eigen::VectorXd::Random(k).array() * nrow / 4 + nrow / 2).abs().round();
  num_nzw = num_nzw.cwiseMax(2).cwiseMin(nrow);
  num_nzh = num_nzh.cwiseMax(2).cwiseMin(ncol);

  // Initialize factors
  Eigen::MatrixXd h = Eigen::MatrixXd::Zero(k, ncol);
  Eigen::MatrixXd w = Eigen::MatrixXd::Zero(nrow, k);

  // Assign non-zero elements
  for (int i = 0; i < k; ++i) {
    Eigen::VectorXi non_zero_indices = Eigen::VectorXi::LinSpaced(num_nzh(i), 0, ncol - 1).unaryExpr([num_nzh, ncol](int x) { return std::rand() % ncol; });
    h.row(i).segment(non_zero_indices(0), num_nzh(i)) = Eigen::VectorXd::Random(num_nzh(i)).array().abs();

    non_zero_indices = Eigen::VectorXi::LinSpaced(num_nzw(i), 0, nrow - 1).unaryExpr([num_nzw, nrow](int x) { return std::rand() % nrow; });
    w.col(i).segment(non_zero_indices(0), num_nzw(i)) = Eigen::VectorXd::Random(num_nzw(i)).array().abs();
  }

  // Normalize factors
  w = w * w.colwise().sum().inverse().asDiagonal();
  h = h.rowwise().sum().inverse().asDiagonal() * h;

  // Build input matrix
  Eigen::MatrixXd res = w * h;

  // Add noise
  if (noise > 0) {
    Eigen::MatrixXd noise_matrix = Eigen::MatrixXd::Random(nrow, ncol) * noise;
    res = res + noise_matrix;
    res = (res.array() < 0).select(0, res);
  }

  // Introduce dropout
  if (dropout > 0) {
    Eigen::SparseMatrix<double> d(nrow, ncol);
    d.reserve(nrow * ncol * (1 - dropout));
    for (int i = 0; i < nrow; ++i) {
      for (int j = 0; j < ncol; ++j) {
        if (std::rand() % 100 > dropout * 100) {
          d.insert(i, j) = 1;
        }
      }
    }
    res = res.cwiseProduct(d);
  }

  return res;
}
