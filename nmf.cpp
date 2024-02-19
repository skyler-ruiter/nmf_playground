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

// sequential coordinate descent for NMF with KL-divergence (need to add regularization)
void scd_kl_update(Eigen::VectorXd& Hj, const Eigen::MatrixXd& Wt, const Eigen::VectorXd& Aj, const Eigen::VectorXd& sumW, const unsigned int max_iter, const double rel_tol) {

    double sumHj = Hj.sum();
    Eigen::VectorXd Ajt = Wt.transpose() * Hj;
    Eigen::VectorXd mu;
    double a; // second derivative
    double b; // first derivative
    double tmp, etmp;
    double rel_err = 1 + rel_tol;
    std::vector<double> beta; // l2 regularization, angle, l1 regularization

    // initialize beta with defaults
    beta.push_back(0);
    beta.push_back(0);
    beta.push_back(0);

    unsigned int t = 0;
    for (; t < max_iter && rel_err > rel_tol; t++) {
        rel_err = 0;
        for (long int k = 0; k < Wt.rows(); k++) {
            mu = Wt.row(k).transpose().array() / (Ajt.array() + TINY_NUM);

            Eigen::VectorXd mu_squared = mu.array() * mu.array();

            a = Aj.dot(mu_squared);
            // b = sumW(k) - Aj.dot(mu); //! new
            b = Aj.dot(mu) - sumW(k); //! old

            // double numerator = a*Hj(k) - b - beta[2] - beta[1]*(sumHj - Hj(k)); //! new
            a += beta[0];
            b += a*Hj(k) - beta[2] - beta[1] * (sumHj - Hj(k)); //! old

            tmp = b / (a + TINY_NUM);
            // std::cout << "tmp: " << tmp << std::endl;

            if (tmp < 0) tmp = 0;

            if (tmp != Hj(k)) {
                Ajt += (tmp - Hj(k)) * Wt.row(k).transpose();
                etmp = 2 * std::abs((Hj(k) - tmp) / (tmp + Hj(k) + TINY_NUM));
                if (etmp > rel_err) rel_err = etmp;
                sumHj += (tmp - Hj(k));
                Hj(k) = tmp;
            }
            // print rel_err
        }
        // std::cout << "Rel Err: " << rel_err << std::endl;

    }
    // return int(t);
}

// update h given A and w
inline void update(Eigen::MatrixXd& h, const Eigen::MatrixXd& wt, const Eigen::MatrixXd A, const double L1, const double L2, const int threads) {
    unsigned int max_iter = 100;
    double rel_tol = 1e-4;
    Eigen::VectorXd mu;
    #pragma omp parallel for num_threads(threads) schedule(dynamic) private(mu)
    for (long int j = 0; j < h.cols(); ++j) {
        Eigen::VectorXd Hj = h.col(j);
        scd_kl_update(Hj, wt, A.col(j), wt.colwise().sum(), max_iter, rel_tol);
    }
}

NMFResult c_nmf(const Eigen::MatrixXd& A, const Eigen::MatrixXd& At, const unsigned int k, const unsigned int maxit, const double rel_tol, const double L1_h, const double L2_h, const double L1_w, const double L2_w, const unsigned int threads, const unsigned int seed) {
    
    printf("%sIter %s    KL Err   / Rel %s            P Corr   / Rel     %s        MSE      / Rel     %s\n", BLUE, GREEN, CYAN, MAGENTA, RESET);
    //  0 |   0.593055 / 200.000000% |   0.962781 / 200.000000% |   0.017833 / 200.000000% |
    
    // initialize random W and H
    Eigen::MatrixXd w = Eigen::MatrixXd::Random(k, A.rows());
    Eigen::MatrixXd h = Eigen::MatrixXd::Random(k, A.cols());
    Eigen::VectorXd d = Eigen::VectorXd::Ones(w.rows());

    // ensure non-negative
    w = w.cwiseAbs();
    h = h.cwiseAbs();

    int iterations = 0; // number of iterations
    double rel_err = rel_tol + 1; // relative error 
    double rel_MSE = rel_tol + 1; // relative error for mean squared error
    double rel_Corr = rel_tol + 1; // relative error for pearson correlation

    double MSE_last = 1e99; // last mean squared error
    double corr_last = 1e99; // last pearson correlation
    double klerror_last = 1e99; // last kl error
    std::vector<double> kl_error(maxit);
    for (size_t i = 0; i < maxit; ++i) {
        // initialize kl error to mean of A componentwise multiplied by log of A componentwise subtracted by A
        kl_error[i] = ((A.array() + TINY_NUM) * (A.array() + TINY_NUM).log() - A.array()).mean();
        // kl_error[i] = 0;
    }

    // print kl error 0
    std::cout << "Iter: 0 KL Error: " << kl_error[0] << " relative error: " << rel_err << std::endl;


    //* alternating least squares update loop
    for (uint16_t iter_ = 0; iter_ < maxit && rel_err > rel_tol; ++iter_) {
        // update w
        update(w, h, At, L1_w, L2_w, threads);
        scale(w, d);

        // update h
        update(h, w, A, L1_h, L2_h, threads);
        scale(h, d);

        // calculate fit
        Eigen::MatrixXd A_hat = w.transpose() * h;
        // kl_error = -log(A) * A_hat + Ahat
        // kl_error[iter_] += ((-(A.array() + TINY_NUM).log() * (A_hat.array() + TINY_NUM) + A_hat.array()).mean()); // nnlms kl error formula

        // check if any negative values are in a_hat
        if (A_hat.minCoeff() < 0) {
            std::cout << "Negative values in A_hat" << std::endl;
        }

        // regular mean kl error = A * log(A / A_hat) - A + A_hat
        kl_error[iter_] += ((-(A.array() + TINY_NUM) * (A_hat.array() + TINY_NUM).log()).array() + A_hat.array()).mean();
        // calculate relative error
        //print kl error
        rel_err = 2 * ((klerror_last - kl_error[iter_]) / (klerror_last + kl_error[iter_] + TINY_NUM));
        klerror_last = kl_error[iter_];

        // print kl error 
        std::cout << "Iter: " << iter_ << " KL Error: " << kl_error[iter_] << " relative error: " << rel_err << " Corr: " << cor(A, A_hat) << std::endl;
        
        
        /*
        Black: 30
        Red: 31
        Green: 32
        Yellow: 33
        Blue: 34
        Magenta: 35
        Cyan: 36
        White: 37
        */

        double MSE = Eigen::square((A - w.transpose() * h).array()).mean();
        rel_MSE = 2 * ((MSE_last - MSE) / (MSE_last + MSE + TINY_NUM));
        MSE_last = MSE;

        double corr = cor(A, A_hat);
        rel_Corr = 2 * ((corr_last - corr) / (corr_last + corr + TINY_NUM));
        corr_last = corr;

        // std::cout << "Iter: \u001b[34m" << iter_ << "\u001b[0m KL Error: " << kl_error[iter_] << " relative error: " << rel_err << " Corr: " << cor(A, A_hat) << std::endl;
        printf("%s%4d |%s %10lf / %10lf\% |%s %10lf / %10lf\% |%s %10lf / %10lf\% |%s\n", BLUE, iter_, GREEN, kl_error[iter_], rel_err * 100, CYAN, corr, rel_Corr * 100, MAGENTA,  MSE, rel_MSE * 100, RESET);
        // std::cout << "KL Error: " << klerror_last << std::endl;
        // std::cout << "relative error: " << rel_err << std::endl;
        // std::cout << "Corr: " << cor(A, A_hat) << std::endl;
    }

    return { w, d, h };
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

    Eigen::MatrixXd At = A.transpose();

    NMFResult result = c_nmf(A, At, 10, 100, 1e-4, 0, 0, 0, 0, 0, 1); // run nmf

    // std::cout << "W: " << result.w << std::endl;
    // std::cout << "D: " << result.d << std::endl;
    // std::cout << "H: " << result.h << std::endl;

    // std::cout << "Res: " << (A - (result.w.transpose() * result.h)).norm() << std::endl;
    // std::cout << A - result.w.transpose() * result.h << std::endl;
    Eigen::MatrixXd diff = A - result.w.transpose() * result.h;
    diff = Eigen::square(diff.array());
    std::cout << "Res: " << diff.mean() << std::endl;


    return 0;
}
