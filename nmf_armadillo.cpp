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
#include "armadillo-12.8.1/include/armadillo"

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

arma::mat toArma(Eigen::MatrixXd eigen) {
    arma::mat arma(eigen.rows(), eigen.cols());

    #pragma omp parallel for
    for (int i = 0; i < eigen.rows(); i++) {
        for (int j = 0; j < eigen.cols(); j++) {
            arma(i, j) = eigen(i, j);
        }
    }

    return arma;
}

arma::vec toArma(Eigen::VectorXd eigen) {
    arma::vec arma(eigen.data(), eigen.size());
    return arma;
}

Eigen::MatrixXd toEigen(arma::mat arma) {
    Eigen::MatrixXd eigen(arma.n_rows, arma.n_cols);

    #pragma omp parallel for
    for (int i = 0; i < arma.n_rows; i++) {
        for (int j = 0; j < arma.n_cols; j++) {
            eigen(i, j) = arma(i, j);
        }
    }

    return eigen;
}

Eigen::VectorXd toEigen(arma::vec arma) {
    Eigen::VectorXd eigen = Eigen::Map<Eigen::VectorXd>(arma.memptr(), arma.n_elem);
    return eigen;
}

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


void scale(arma::mat& w, arma::vec& d) {
    d = arma::sum(w, 1);

    #pragma omp parallel for
    for (int i = 0; i < w.n_rows; i++) {
        for (int j = 0; j < w.n_cols; j++) {
            w(i, j) /= d(i);
        }
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

int scd_kl_update(arma::subview_col<double> Hj, const arma::mat& Wt, const arma::vec& Aj, const arma::vec& sumW, const unsigned int& max_iter, const double& rel_tol) {
    // Problem:  Aj = W * Hj
    // Method: Sequentially minimize KL distance using quadratic approximation
    // Wt = W^T
    // sumW = column sum of W
    // mask: skip updating
    // beta: a vector of 3, for L2, angle, L1 regularization

    double sumHj = sum(Hj);
    arma::vec Ajt = Wt.t() * Hj;
    arma::vec mu;
    double a; // 2nd-order-derivative
    double b; // 1st-order-derivative
    double tmp, etmp;
    double rel_err = 1 + rel_tol;
    // bool is_masked = mask.n_elem > 0;

    unsigned int t = 0;
    for (; t < max_iter && rel_err > rel_tol; t++) {
        rel_err = 0;
        for (unsigned int k = 0; k < Wt.n_rows; k++) {
            // if (is_masked && mask(k) > 0) continue;
            mu = Wt.row(k).t() / (Ajt + TINY_NUM);
            a = dot(Aj, square(mu));
            b = dot(Aj, mu) - sumW(k); // 0.5*ax^2 - bx
            // a += -0.1;
            // b += a * Hj(k) - 0 - 0.1 * (sumHj - Hj(k));
            b += a * Hj(k) * (sumHj - Hj(k));

            tmp = b / (a + TINY_NUM);
            // std::cout << "a: " << a << " b: " << b << " tmp: " << tmp << std::endl;
            // std::cout << "tmp: " << tmp << std::endl;

            if (tmp < 0) tmp = 0;
            if (tmp != Hj(k)) {
                Ajt += (tmp - Hj(k)) * Wt.row(k).t();
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
int update(arma::mat& H, const arma::mat& Wt, const arma::mat& A, unsigned int max_iter, double rel_tol) {

    // A = W H, solve H
    // No missing in A, Wt = W^T
    // method: 1 = scd, 2 = lee_ls, 3 = scd_kl, 4 = lee_kl

    unsigned int m = A.n_cols;
    int total_raw_iter = 0;

    arma::mat WtW;
    arma::vec mu, sumW;
    sumW = sum(Wt, 1);

    #pragma omp parallel for private(mu)
    for (unsigned int j = 0; j < m; j++) // by columns of H
    {
        int iter = 0;
        iter = scd_kl_update(H.col(j), Wt, A.col(j), sumW, max_iter, rel_tol);

        #pragma omp critical
        total_raw_iter += iter;
    }

    return total_raw_iter;
}

// for birds.csv k = 10, tol = 1e-4, maxit = 100
NMFResult c_nmf(const arma::mat& A, const unsigned int k = 10, const unsigned int max_iter = 100, const double rel_tol = 1e-4) {

    unsigned int n = A.n_rows;
    unsigned int m = A.n_cols;
    arma::mat W = arma::randu(n, k);
    arma::mat H = arma::zeros(k, m);
    const arma::vec beta = { 0, 0, 0 };
    arma::vec d = arma::ones(k);
    //int k = H.n_rows; // decomposition rank k
    int err_len = max_iter;

    arma::vec mse_err(err_len), mkl_err(err_len), terr(err_len), ave_epoch(err_len);

    // check progression
    double rel_err = rel_tol + 1;
    double terr_last = 1e99;
    mkl_err.fill(arma::mean(arma::mean((A + TINY_NUM) % log(A + TINY_NUM) - A))); // fixed part in KL-dist, mean(A log(A) - A)

    if (W.empty()) {
        W.randu(k, n);
        W *= 0.01;
    }
    else
        arma::inplace_trans(W);

    if (H.empty()) {
        H.randu(k, m);
        H *= 0.01;
    }

    printf("\n%10s | %10s | %10s | %10s | %10s\n", "Iteration", "MSE", "MKL", "Target", "Rel. Err.");
    printf("--------------------------------------------------------------\n");


    int total_raw_iter = 0;
    unsigned int i = 0;
    unsigned int i_e = 0; // index for error checking
    for (; i < max_iter && std::abs(rel_err) > rel_tol; i++) {

        // update W
        total_raw_iter += update(W, H, A.t(), max_iter, rel_tol);
        // scale(W, d);

        // update H
        total_raw_iter += update(H, W, A, max_iter, rel_tol);
        // scale(H, d);



        const arma::mat& Ahat = W.t() * H;
        mse_err(i_e) = arma::mean(arma::mean(arma::square((A - Ahat))));
        mkl_err(i_e) += arma::mean(arma::mean(-(A + TINY_NUM) % arma::log(Ahat + TINY_NUM) + Ahat));


        ave_epoch(i_e) = double(total_raw_iter) / (n + m);

        terr(i_e) = mkl_err(i_e);


        rel_err = 2 * (terr_last - terr(i_e)) / (terr_last + terr(i_e) + TINY_NUM);
        terr_last = terr(i_e);

        printf("%10d | %10.4f | %10.4f | %10.4f | %10.g\n", i + 1, mse_err(i_e), mkl_err(i_e), terr(i_e), rel_err);

        ++i_e;

        printf("--------------------------------------------------------------\n");
        printf("%10s | %10s | %10s | %10s | %10s\n\n", "Iteration", "MSE", "MKL", "Target", "Rel. Err.");
    }

    if (i_e < err_len) {
        mse_err.resize(i_e);
        mkl_err.resize(i_e);
        terr.resize(i_e);
        ave_epoch.resize(i_e);
    }

    arma::mat Wt = W.t();
    Eigen::MatrixXd w = toEigen(Wt);
    Eigen::MatrixXd h = toEigen(H);
    Eigen::VectorXd eigen_d = toEigen(d);

    return { w, eigen_d, h };
}



void readCSV(std::string filename, Eigen::Matrix<double, -1, -1>& A) {

    std::vector<std::vector<double>> data;
    data.reserve(1183);
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        row.reserve(183);
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(atof(cell.c_str()));
        }
        data.push_back(row);
    }
    std::cout << "Read " << data.size() << " rows and " << data[0].size() << " columns" << std::endl;

    A = Eigen::Matrix<double, -1, -1>(183, 1183);
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
    arma::mat armaA = toArma(A);

    NMFResult result = c_nmf(armaA); // run nmf

    // std::cout << "W: " << result.w << std::endl;
    // std::cout << "D: " << result.d << std::endl;
    // std::cout << "H: " << result.h << std::endl;
    for(int i = 0; i < result.h.rows(); i++) {
        result.h.row(i) *= result.d(i);
    }

    std::cout << "Rel Res: " << (A - (result.w * result.h)).norm() / A.norm() << std::endl;
    // std::cout << A - result.w * result.h << std::endl;
    // Eigen::MatrixXd diff = A - result.w.transpose() * result.h;
    // diff = Eigen::square(diff.array());
    // std::cout << "Res: " << diff.mean() << std::endl;


    return 0;
}
