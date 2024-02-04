#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>
#include "helpers.cpp"
#include "sim_nmf.cpp"

#define TINY_NUM 1e-15

struct NMFResult {
  Eigen::MatrixXd w;
  Eigen::VectorXd d;
  Eigen::MatrixXd h;
};

NMFResult c_nmf(const Eigen::MatrixXd& A, const Eigen::MatrixXd& At, const unsigned int k, const unsigned int maxit, const double tol, const double L1_h, const double L2_h, const double L1_w, const double L2_w, const unsigned int threads, const unsigned int seed) {
  
  // initialize random W and H
  Eigen::MatrixXd w = Eigen::MatrixXd::Random(k, A.rows());
  Eigen::MatrixXd h = Eigen::MatrixXd::Zero(w.rows(), A.cols());
  Eigen::VectorXd d = Eigen::VectorXd::Ones(w.rows());
  double tol_ = 1;

  // alternating least squares update loop
  for (uint16_t iter_ = 0; iter_ < maxit && tol_ > tol; ++iter_) {
    Eigen::MatrixXd w_it = w;
    // update h
    predict(A, w, h, L1_h, L2_h, threads);
    scale(h, d);
    // update w
    predict(At, h, w, L1_w, L2_w, threads);
    scale(w, d);

    // calculate tolerance of the model fit to detect convergence
    // correlation between "w" across consecutive iterations
    tol_ = cor(w, w_it);
  }

  return {w, d, h};
}

int main()
{
    std::cout << "Hello, world!" << std::endl;
    
    // run a simple test
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(10, 10);
    // impose non-negativity
    A = A.cwiseAbs();
  
    
    Eigen::MatrixXd At = A.transpose();

    NMFResult result = c_nmf(A, At, 10, 100, 1e-4, 0, 0, 0, 0, 0, 1);

    // // print a
    // std::cout << "A = " << std::endl;
    // std::cout << A << std::endl;

    // print w
    // std::cout << "w = " << std::endl;
    // std::cout << result.w << std::endl;


    // print A - w * h
    // Eigen::MatrixXd A_wh = A - result.w.transpose() * result.h;
    // std::cout << "A - w * h = " << std::endl;
    // std::cout << A_wh << std::endl;

    // print the dimensions of w and h and a
    std::cout << "w dimensions = " << result.w.rows() << " x " << result.w.cols() << std::endl;
    std::cout << "h dimensions = " << result.h.rows() << " x " << result.h.cols() << std::endl;
    std::cout << "A dimensions = " << A.rows() << " x " << A.cols() << std::endl;


    // try and get a simulated NMF matrix
    Eigen::MatrixXd sim = simulateNMF(10, 10, 10, 0.5, 0.5, 1);
    
    return 0;
}