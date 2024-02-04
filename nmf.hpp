#include <Eigen/Dense>
#include <iostream>
#include <vector>

#define TINY_NUM 1e-15

template <class T>
class nmf {

  private:
    // Original matrix (and transpose)
    T& A;
    T t_A;

    // Factorization matrices
    Eigen::MatrixXd W;
    Eigen::VectorXd D;
    Eigen::MatrixXd H;

    // Matrix properties
    bool symmetric = false;
    bool transposed = false;

    double tol_ = -1;
    unsigned int iter_ = 0, best_model_ = 0;

   public:
    unsigned int maxit = 100;
    unsigned int threads = 0;
    std::vector<double> L1 = std::vector<double>(2);
    std::vector<double> L2 = std::vector<double>(2);

    double tol = 1e-4;

    // Constructors
    nmf(T& A, const unsigned int k, const unsigned int seed = 0) : A(A) {
      W = randomMatrix(k, A.rows(), seed);
      H = Eigen::MatrixXd(k, A.cols());
      D = Eigen::VectorXd::Ones(k);
    }


    // Getters
    Eigen::MatrixXd matrixW() const { return W; }
    Eigen::MatrixXd matrixH() const { return H; }
    Eigen::VectorXd vectorD() const { return D; }
    double fit_tol() { return tol_; }
    unsigned int fit_iter() { return iter_; }
    unsigned int best_model() { return best_model_; }


    // Functions

    // scale rows in "w" to sum to 1, where "d" is rowsums of "w"
    void scaleW() {
      D = W.rowwise().sum();
      D.array() += TINY_NUM;
      for (unsigned int i = 0; i < W.rows(); ++i)
        for (unsigned int j = 0; j < W.cols(); ++j) W(i, j) /= D(i);
    };

    // scale rows in "h" to sum to 1, where "d" is rowsums of "h"
    void scaleH() {
      D = H.rowwise().sum();
      D.array() += TINY_NUM;
      for (unsigned int i = 0; i < H.rows(); ++i)
        for (unsigned int j = 0; j < H.cols(); ++j) H(i, j) /= D(i);
    };

    // project "w" onto "A" to solve for "h" in the equation "A = wh"
    void predictH() {
      predict(A, W, H, L1[1], L2[1], threads);
    }

    // project "h" onto "t(A)" to solve for "w"
    void predictW() {
      if (symmetric)
        predict(A, H, W, L1[0], L2[0], threads);
      else {
        if (!transposed) {
          t_A = A.transpose();
          transposed = true;
        }
        predict(t_A, H, W, L1[0], L2[0], threads);
      }
    };

    void fit() {
      // alternating least squares updates
      for (; iter_ < maxit; ++iter_) {
        Eigen::MatrixXd W_it = W;
        predictH();  // update "h"
        scaleH();
        predictW();  // update "w"
        scaleW();
        tol_ = cor(W, W_it);  // correlation between "w" across consecutive iterations
        if (tol_ < tol) break;
      }
    }

    // Helper functions

    Eigen::MatrixXd randomMatrix(const unsigned int nrow,
                                 const unsigned int ncol,
                                 const unsigned int seed) {
      std::vector<double> random_values = getRandomValues(nrow * ncol, seed);
      Eigen::MatrixXd x(nrow, ncol);
      unsigned int indx = 0;
      for (unsigned int r = 0; r < nrow; ++r)
        for (unsigned int c = 0; c < ncol; ++c, ++indx)
          x(r, c) = random_values[indx];
      return x;
    }

};