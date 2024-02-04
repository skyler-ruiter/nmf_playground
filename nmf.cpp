#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>
// #include "nmf.hpp"

#define TINY_NUM 1e-15

struct NMFResult {
  Eigen::MatrixXd w;
  Eigen::VectorXd d;
  Eigen::MatrixXd h;
};

template <typename T>
double mean(const T& mat) {
  return mat.sum() / mat.size();
}

// scale rows in w (or h) to sum to 1 and put previous rowsums in d
void scale(Eigen::MatrixXd& w, Eigen::VectorXd& d) {
  d = w.rowwise().sum();
  d.array() += 1e-15;
  for (long int i = 0; i < w.rows(); ++i)
    for (long int j = 0; j < w.cols(); ++j) w(i, j) /= d(i);
};


// sequential coordinate descent for NMF with KL-divergence (need to add regularization)
void scd_kl_update(Eigen::VectorXd& hj, const Eigen::MatrixXd& Wt, const Eigen::VectorXd& aj, const Eigen::VectorXd& sumW, const unsigned int max_iter, const double rel_tol) {

  double sumHj = hj.sum();
  Eigen::VectorXd Ajt = Wt.transpose() * hj;
  Eigen::VectorXd mu;
  double a; // second derivative
  double b; // first derivative
  double tmp, etmp;
  double rel_err = 1 + rel_tol;

  unsigned int t = 0;

  for (; t < max_iter && rel_err > rel_tol; t++)
  {
    rel_err = 0;
    for (long int k = 0; k < Wt.rows(); k++) {
      mu = Wt.row(k).transpose().array() / (Ajt.array() + TINY_NUM);
      // update a to the dot product of aj and mu squared
      //square mu
      Eigen::VectorXd mu_squared = mu.array() * mu.array();
      a = aj.dot(mu_squared);
      b = aj.dot(mu) - sumW(k);
      // a += beta(0);
      // b += a * Hj(k) - beta(2) - beta(1) * (sumHj - Hj(k));
      tmp = b/(a + TINY_NUM);
      if (tmp < 0) tmp = 0;
      if (tmp != hj(k)) {
        Ajt += (tmp - hj(k)) * Wt.row(k).transpose();
        etmp = 2 * std::abs(tmp - hj(k)) / (tmp + hj(k) + TINY_NUM);
        if (etmp > rel_err) rel_err = etmp;
        sumHj += tmp - hj(k);
        hj(k) = tmp;
      }
    }
  }
  // return t;
}

// update h given A and w
inline void update(Eigen::MatrixXd A, const Eigen::MatrixXd& h, Eigen::MatrixXd& wt, const double L1, const double L2, const int threads) {
  
  unsigned int max_iter = 100;
  double rel_tol = 1e-4;

  #ifdef _OPENMP
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  #endif
  for (long int i = 0; i < h.cols(); ++i) {
    // get sum of W
    Eigen::VectorXd sumW = wt.rowwise().sum();
    // get the ith column of A
    Eigen::VectorXd aj = A.col(i);
    // get the ith column of H
    Eigen::VectorXd hj = h.col(i);
    // update based on kl-divergence
    scd_kl_update(hj, wt, aj, sumW, max_iter, rel_tol);
  }
}

NMFResult c_nmf(const Eigen::MatrixXd& A, const Eigen::MatrixXd& At, const unsigned int k, const unsigned int maxit, const double tol, const double L1_h, const double L2_h, const double L1_w, const double L2_w, const unsigned int threads, const unsigned int seed) {
  
  
  // initialize random W and H
  Eigen::MatrixXd w = Eigen::MatrixXd::Random(k, A.rows());
  Eigen::MatrixXd h = Eigen::MatrixXd::Zero(w.rows(), A.cols());
  Eigen::VectorXd d = Eigen::VectorXd::Ones(w.rows());

  double rel_err = tol + 1;
  double klerror_last = 1e99;
  std::vector<double> kl_error(maxit);
  // initialize kl_error to zero
  for (size_t i = 0; i < maxit; ++i) kl_error[i] = 0;

  for (uint16_t iter_ = 0; iter_ < maxit && rel_err > tol; ++iter_) {
    // update h
    update(A, w, h, L1_h, L2_h, threads);
    scale(h, d);

    // update w
    update(At, h, w, L1_w, L2_w, threads);
    scale(w, d);

    // calculate fit
    Eigen::MatrixXd A_hat = w.transpose() * h;
    kl_error[iter_] += ((-(A.array() + TINY_NUM).log() * (A_hat.array() + TINY_NUM) + A_hat.array()).mean());

    // calculate relative error
    rel_err = std::abs((klerror_last - kl_error[iter_]) / (klerror_last + kl_error[iter_] + TINY_NUM));
    klerror_last = kl_error[iter_];
  }

  return {w, d, h};
}


int main()
{
    std::cout << "Hello, world!" << std::endl;
    
    
    // run a simple test
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(100, 100);
    A = A.cwiseAbs();

    Eigen::MatrixXd At = A.transpose();

    NMFResult result = c_nmf(A, At, 10, 100, 1e-4, 0, 0, 0, 0, 0, 1);
    
    
    return 0;
}