#include <Eigen/Dense>
#include <iostream>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif 
#include <cmath>
#include <fstream>
// #include "nmf.hpp"
#include <chrono>

#define TINY_NUM 1e-15
#define BLUE "\u001b[34m"
#define RESET "\u001b[0m"
#define GREEN "\u001b[32m"
#define CYAN "\u001b[36m"
#define MAGENTA "\u001b[35m"
#define YELLOW "\u001b[33m"

struct NMFResult {
    Eigen::MatrixXd w;
    Eigen::VectorXd d;
    Eigen::MatrixXd h;
};


// Pearson correlation between two matrices
inline double cor(const Eigen::MatrixXd& x, Eigen::MatrixXd& y) {
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

// scale rows in w (or h) to sum to 1 and put previous rowsums in d
void scale(Eigen::MatrixXd& w, Eigen::VectorXd& d) {
    d = w.rowwise().sum();
    d.array() += 1e-15;
    for (long int i = 0; i < w.rows(); ++i)
        for (long int j = 0; j < w.cols(); ++j) w(i, j) /= d(i);
};

int scd_kl_update(Eigen::Ref<Eigen::VectorXd> Hj, const Eigen::MatrixXd& Wt, const Eigen::VectorXd& Aj, const Eigen::VectorXd& sumW, const unsigned int& max_iter, const double& rel_tol) {
    
    // Problem:  Aj = W * Hj
    // Method: Sequentially minimize KL distance using quadratic approximation
    // Wt = W^T
    // sumW = column sum of W
    // mask: skip updating
    // beta: a vector of 3, for L2, angle, L1 regularization

    double sumHj = Hj.sum();
    Eigen::VectorXd Ajt = Wt.transpose() * Hj;
    Eigen::VectorXd mu;
    double a; // 2nd-order-derivative
    double b; // 1st-order-derivative
    double tmp, etmp;
    double rel_err = 1 + rel_tol;

    unsigned int t = 0;
    for (; t < max_iter && rel_err > rel_tol; t++) {
        rel_err = 0;
        for (unsigned int k = 0; k < Wt.rows(); k++) {

            mu = Wt.row(k).transpose().cwiseQuotient((Ajt.array() + TINY_NUM).matrix());

            a = Aj.dot(mu.cwiseProduct(mu));
            b = Aj.dot(mu) - sumW(k); // 0.5*ax^2 - bx + beta(0)

            // a += beta(0);
            // b += a * eigenHj(k) - beta(2) - beta(1) * (sumHj - Hj(k));
            b += a * Hj(k);
            
            tmp = b / (a + TINY_NUM);

            if (tmp < 0) tmp = 0;
            if (tmp != Hj(k)) {
                Ajt += (tmp - Hj(k)) * Wt.row(k).transpose();
                etmp = 2 * std::abs(Hj(k) - tmp) / (tmp + Hj(k) + TINY_NUM);
                if (etmp > rel_err)
                    rel_err = etmp;
                sumHj += tmp - Hj(k);
                Hj(k) = tmp;
            }
        }
    }
    return int(t);
}

// update h given A and w
inline void update(Eigen::MatrixXd& H, const Eigen::MatrixXd& Wt, const Eigen::MatrixXd A, const double L1, const double L2, const int threads) {
    unsigned int max_iter = 100;
    double rel_tol = 1e-4;
    Eigen::VectorXd sumW = Wt.rowwise().sum();

    #pragma omp parallel for
    for (long int j = 0; j < A.cols(); ++j) {
        
        scd_kl_update(H.col(j), Wt, A.col(j), sumW, max_iter, rel_tol);
    
    }
}

NMFResult c_nmf(const Eigen::MatrixXd& A, const unsigned int k, const unsigned int maxit, const double rel_tol, const double L1_h, const double L2_h, const double L1_w, const double L2_w, const unsigned int threads, const unsigned int seed) {

    printf("%sIter %s    KL Err   / Relative Change %s\n", BLUE, GREEN, RESET);

    // initialize random W and H
    Eigen::MatrixXd W = Eigen::MatrixXd::Random(k, A.rows()); // These **both** should be initialized 
    Eigen::MatrixXd H = Eigen::MatrixXd::Random(k, A.cols()); // with random values. See paper under Eq. 6
    Eigen::VectorXd d = Eigen::VectorXd::Ones(W.rows());

    double rel_err = rel_tol + 1; // relative error 
    double klerror_last = 1e99; // last kl error
    std::vector<double> kl_error(maxit);
    
    //* alternating least squares update loop
    for (uint16_t iter_ = 0; iter_ < maxit && rel_err > rel_tol; ++iter_) {
        
        // update w
        update(W, H, A.transpose(), L1_w, L2_w, threads);
        // scale(w, d);

        // update h
        update(H, W, A, L1_h, L2_h, threads);
        // scale(h, d);

        // calculate fit
        Eigen::MatrixXd A_hat = W.transpose() * H;
        kl_error[iter_] = ((-(A.array() + TINY_NUM) * (A_hat.array() + TINY_NUM).log()).array() + A_hat.array()).mean();
        
        //print kl error
        rel_err = 2 * ((klerror_last - kl_error[iter_]) / (klerror_last + kl_error[iter_] + TINY_NUM));
        klerror_last = kl_error[iter_];

        printf("%s%4d |%s %10lf / %10lf%% %s\n", BLUE, iter_, GREEN, kl_error[iter_], rel_err * 100, RESET);
    }

    return { W, d, H };
}


void readCSV(std::string filename, Eigen::Matrix<double, -1, -1>& A) {

    std::vector<std::vector<double>> data;
    data.reserve(1183);
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        row.reserve(184);
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(atof(cell.c_str()));
        }
        data.push_back(row);
    }
    std::cout << "Read " << data.size() << " rows and " << data[0].size() << " columns" << std::endl;

    A = Eigen::Matrix<double, -1, -1>(185, 1183);
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[i].size(); j++) {
            A(i, j) = data[i][j];
        }

    }

}


int main() {
    // std::cout << "Hello, world!" << std::endl;


    // run a simple test
    // Eigen::MatrixXd A = Eigen::MatrixXd::Random(50, 100);
    Eigen::Matrix<double, -1, -1> A;
    readCSV("birds.csv", A);

    A = A.cwiseAbs();
    NMFResult result = c_nmf(A, 10, 100, 1e-4, 0, 0, 0, 0, 0, 1); // run nmf

    // std::cout << "W: " << result.w << std::endl;
    // std::cout << "D: " << result.d << std::endl;
    // std::cout << "H: " << result.h << std::endl;
    Eigen::MatrixXd A_hat = result.w.transpose() * result.h;
    
    std::cout << "Absolute Residual: " << (A - A_hat).norm() << std::endl;
    std::cout << "Relative Residual: " << (A - A_hat).norm() / A.norm() << std::endl;
    std::cout << "Pearson Correlation: " << cor(A, A_hat) << std::endl;
    // std::cout << A - A_hat << std::endl;
    // Eigen::MatrixXd diff = A - result.w.transpose() * result.h;
    // diff = Eigen::square(diff.array());
    // std::cout << "Res: " << diff.mean() << std::endl;


    return 0;
}
