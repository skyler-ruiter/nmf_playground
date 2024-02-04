#include <Eigen/Dense>

// Pearson correlation between two matrices
inline double cor(Eigen::MatrixXd& x, Eigen::MatrixXd& y) {
  double x_i, y_i, sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
  const size_t n = x.size();
  for (size_t i = 0; i < n; ++i) {
    x_i = (*(x.data() + i));
    y_i = (*(y.data() + i));
    sum_x += x_i;
    sum_y += y_i;
    sum_xy += x_i * y_i;
    sum_x2 += x_i * x_i;
    sum_y2 += y_i * y_i;
  }
  return 1 - (n * sum_xy - sum_x * sum_y) / std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
}

// fast symmetric matrix multiplication, A * A.transpose()
inline Eigen::MatrixXd AAt(const Eigen::MatrixXd& A) {
  Eigen::MatrixXd AAt = Eigen::MatrixXd::Zero(A.rows(), A.rows());
  AAt.selfadjointView<Eigen::Lower>().rankUpdate(A);
  AAt.triangularView<Eigen::Upper>() = AAt.transpose();
  AAt.diagonal().array() += 1e-15;
  return AAt;
}

// scale rows in w (or h) to sum to 1 and put previous rowsums in d
void scale(Eigen::MatrixXd& w, Eigen::VectorXd& d) {
  d = w.rowwise().sum();
  d.array() += 1e-15;
  for (long int i = 0; i < w.rows(); ++i)
    for (long int j = 0; j < w.cols(); ++j) w(i, j) /= d(i);
};

// optimized and modified from github.com/linxihui/NNLM "c_nnls" function
inline void nnls(Eigen::MatrixXd& a, Eigen::VectorXd& b, Eigen::MatrixXd& x, const size_t col, const double L1 = 0, const double L2 = 0) {
  double tol = 1;
  for (uint8_t it = 0; it < 100 && (tol / b.size()) > 1e-8; ++it) {
    tol = 0;
    for (long int i = 0; i < x.rows(); ++i) {
      double diff = b(i) / a(i, i);
      if (L1 != 0) diff -= L1;
      if (L2 != 0) diff += L2 * x(i, col);
      if (-diff > x(i, col)) {
        if (x(i, col) != 0) {
          b -= a.col(i) * -x(i, col);
          tol = 1;
          x(i, col) = 0;
        }
      } else if (diff != 0) {
        x(i, col) += diff;
        b -= a.col(i) * diff;
        tol += std::abs(diff / (x(i, col) + 1e-15));
      }
    }
  }
}

// update h given A and w
inline void predict(Eigen::MatrixXd A, const Eigen::MatrixXd& w, Eigen::MatrixXd& h, const double L1, const double L2, const int threads) {
    Eigen::MatrixXd a = AAt(w);
    // if (L2 != 0) a.diagonal().array() *= (1 - L2);
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(threads)
    #endif
    for (long int i = 0; i < h.cols(); ++i) {
        Eigen::VectorXd b = w * A.col(i);
        // b.array() -= L1;
        nnls(a, b, h, i, L1, L2);
    }
}