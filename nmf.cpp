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


// add_penalty to the target error 'terr'
void add_penalty(const unsigned int& i_e, std::vector<double> terr, const Eigen::MatrixXd W, const Eigen::MatrixXd H, Eigen::VectorXd& alpha, Eigen::VectorXd& beta) {
    uint N_non_missing = W.rows() * W.cols();

    // add penalty term back to the loss function (terr)
    if (alpha(0) != alpha(1)) {
        terr[i_e] += 0.5 * (alpha(0) - alpha(1)) * (W.cwiseProduct(W)).sum() / N_non_missing;
    }
    if (beta(0) != beta(1)) {
        terr[i_e] += 0.5 * (beta(0) - beta(1)) * (H.cwiseProduct(H)).sum() / N_non_missing;
    }
    if (alpha(1) != 0) {
        terr[i_e] += 0.5 * alpha(1) * (W * W.transpose()).sum() / N_non_missing;
    }
    if (beta(1) != 0) {
        terr[i_e] += 0.5 * beta(1) * (H * H.transpose()).sum() / N_non_missing;
    }
    if (alpha(2) != 0) {
        terr[i_e] += alpha(2) * W.sum() / N_non_missing;
    }
    if (beta(2) != 0) {
        terr[i_e] += beta(2) * H.sum() / N_non_missing;
    }
}



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

int scd_kl_update(Eigen::Ref<Eigen::VectorXd> Hj, const Eigen::MatrixXd& Wt, const Eigen::VectorXd& Aj, const Eigen::VectorXd& sumW, Eigen::VectorXd& beta, const double& rel_tol) {

    // Problem:  Aj = W * Hj
    // Method: Sequentially minimize KL distance using quadratic approximation
    // Wt = W^T
    // sumW = column sum of W
    // mask: skip updating
    // beta: a vector of 3, for L2, angle, L1 regularization

    double sumHj = Hj.sum();
    Eigen::VectorXd Ajt = Wt * Hj;
    Eigen::VectorXd mu;
    double tmp;
    double rel_err = 1 + rel_tol;

    unsigned int t = 0;
    for (; t < 100 && rel_err / Hj.rows() > rel_tol; t++) {
        rel_err = 0;
        for (uint k = 0; k < Wt.cols(); k++) {

            // mu = Wt.row(k).transpose().cwiseQuotient((Ajt.array() + TINY_NUM).matrix());
            mu = Wt.col(k).array() / (Ajt.array() + TINY_NUM);

            double b = Aj.dot(mu) - sumW(k);
            mu.array().square();

            double a = Aj.dot(mu) + beta(0);

            b += a * Hj(k) - beta(2) - beta(1) * (sumHj - Hj(k));
            b /= (a + TINY_NUM);

            if (b < 0) {
                rel_err = 1;

                Ajt -= Hj(k) * Wt.col(k);
                sumHj -= Hj(k);
                Hj(k) = 0;
            }
            else if (b != Hj(k)) {
                double diff = b - Hj(k);
                Ajt += diff * Wt.col(k);

                rel_err += std::abs(diff) / (Hj(k) + TINY_NUM);

                sumHj += diff;
                Hj(k) = b;
            }
        }
    }
    return int(t);
}

// update h given A and w
inline void update(Eigen::MatrixXd& H, const Eigen::MatrixXd& Wt, const Eigen::MatrixXd A, Eigen::VectorXd& beta, const int threads) {
    unsigned int max_iter = 100;
    double rel_tol = 1e-4;
    Eigen::VectorXd sumW = Wt.rowwise().sum();
    Eigen::MatrixXd W = Wt.transpose();

    #pragma omp parallel for
    for (long int j = 0; j < A.cols(); ++j) {

        scd_kl_update(H.col(j), W, A.col(j), sumW, beta, rel_tol);
    }
}

NMFResult c_nmf(const Eigen::MatrixXd& A, const unsigned int k, const double L1_h, const double L2_h, const double L1_w, const double L2_w, const unsigned int threads, const unsigned int seed) {

    printf("%sIter %s    KL Err   / Relative Change %s\n", BLUE, GREEN, RESET);

    // initialize random W and H
    Eigen::MatrixXd W = Eigen::MatrixXd::Random(k, A.rows()); // These **both** should be initialized 
    Eigen::MatrixXd H = Eigen::MatrixXd::Random(k, A.cols()); // with random values. See paper under Eq. 6
    Eigen::VectorXd d = Eigen::VectorXd::Ones(W.rows());

    Eigen::VectorXd alpha = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(3);
    Eigen::MatrixXd W_last = H;

    const uint maxit = 1000;

    const double rel_tol = 1e-4;
    double rel_err = rel_tol + 1;
    double klerror_last = 1e99; // last kl error
    std::vector<double> kl_error(maxit, ((A.array() + TINY_NUM) * (A.array() + TINY_NUM).log() - A.array()).mean());
    double cor_last = cor(W_last, H);

    for (uint16_t iter_ = 0; iter_ < maxit && rel_err > rel_tol; ++iter_) {
        W_last = H;

        // update h
        update(H, W, A, beta, threads);
        // scale(h, d);

        // update w
        update(W, H, A.transpose(), alpha, threads);
        // scale(w, d);

        // calculate fit
        Eigen::MatrixXd A_hat = W.transpose() * H;
        double corr = cor(W_last, W);
        std::cout << "Pearson Correlation: " << corr << " Relative Change: " << (cor_last - corr) / (cor_last + corr + TINY_NUM) << std::endl;
        cor_last = corr;
        // NNLM Armadillo version
        // mkl_err(i_e) += mean(mean(-(A + TINY_NUM) % log(Ahat + TINY_NUM) + Ahat));
        // Eigen Conversion
        kl_error[iter_] += ((-(A.array() + TINY_NUM) * (A_hat.array() + TINY_NUM).log()).array() + A_hat.array()).mean();
        add_penalty(iter_, kl_error, W, H, alpha, beta);

        //print kl error
        rel_err = (klerror_last - kl_error[iter_]) / (klerror_last + kl_error[iter_] + TINY_NUM);
        klerror_last = kl_error[iter_];

        printf("%s%4d |%s %10lf / %10lf%% %s\n", BLUE, iter_, GREEN, kl_error[iter_], rel_err * 100, RESET);
    }

    return { W, d, H };
}


void readCSV(std::string filename, Eigen::Matrix<double, -1, -1>& A) {

    std::vector<std::vector<double>> data;
    // data.reserve(1183);
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        // row.reserve(184);
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(atof(cell.c_str()));
        }
        data.push_back(row);
    }
    std::cout << "Read " << data.size() << " rows and " << data[0].size() << " columns" << std::endl;

    A = Eigen::Matrix<double, -1, -1>(data.size(), data[0].size());
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[0].size(); j++) {
            A(i, j) = data[i][j];
        }
    }
}


int main() {
    // std::cout << "Hello, world!" << std::endl;
    srand(time(NULL));

    // run a simple test
    // Eigen::MatrixXd A = Eigen::MatrixXd::Random(50, 100);
    Eigen::Matrix<double, -1, -1> A;
    // readCSV("birds.csv", A);
    readCSV("/home/sethwolfgang/Downloads/NNLM-master/nsclc.csv", A);
    A = A.cwiseAbs();
    // NMFResult result = c_nmf(A, 10, 0, 0, 0, 0, 0, 1); // run nmf
    NMFResult result = c_nmf(A, 15, 0, 0, 0, 0, 0, 1); // run nmf

    // std::cout << "W: " << result.w << std::endl;
    // std::cout << "D: " << result.d << std::endl;
    // std::cout << "H: " << result.h << std::endl;
    Eigen::MatrixXd A_hat = result.w.transpose() * result.h;

    std::cout << "Absolute Residual: " << (A - A_hat).norm() << std::endl;
    std::cout << "Relative Residual: " << (A - A_hat).norm() / A.norm() << std::endl;
    std::cout << "MSE: " << (A - A_hat).squaredNorm() / A.size() << std::endl;
    std::cout << "Pearson Correlation: " << cor(A, A_hat) << std::endl;
    // std::cout << A - A_hat << std::endl;
    // Eigen::MatrixXd diff = A - result.w.transpose() * result.h;
    // diff = Eigen::square(diff.array());
    // std::cout << "Res: " << diff.mean() << std::endl;


    return 0;
}
