
// add_penalty to the target error 'terr'
void add_penalty(const unsigned int& i_e, std::vector<double> terr, const Eigen::MatrixXd W, const Eigen::MatrixXd H,  const double L1_w, const double L1_h, const double L2_w, const double L2_h) {
    uint N_non_missing = W.rows() * W.cols();

    // add penalty term back to the loss function (terr)
    // if (L2_w != alpha(1)) {
        terr[i_e] += 0.5 * (L2_w /*- alpha(1)*/) * (W.cwiseProduct(W)).sum() / N_non_missing;
    // }
    // if (L2_h != beta(1)) {
        terr[i_e] += 0.5 * (L2_h /*- beta(1)*/) * (H.cwiseProduct(H)).sum() / N_non_missing;
    // }
    // if (alpha(1) != 0) {
    //     terr[i_e] += 0.5 * alpha(1) * (W * W.transpose()).sum() / N_non_missing;
    // }
    // if (beta(1) != 0) {
    //     terr[i_e] += 0.5 * beta(1) * (H * H.transpose()).sum() / N_non_missing;
    // }
    if (L1_w != 0) {
        terr[i_e] += L1_w * W.sum() / N_non_missing;
    }
    if (L1_h != 0) {
        terr[i_e] += L1_h * H.sum() / N_non_missing;
    }
}

int scd_kl_update(Eigen::Ref<Eigen::VectorXd> Hj, const Eigen::MatrixXd& Wt, const Eigen::VectorXd& Aj, const Eigen::VectorXd& sumW, const double L1_w, const double L2_w, const double& rel_tol) {

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

            // mu = Wt.row(k).transpose().cwiseQuotient((Ajt.array() + 1e-15).matrix());
            mu = Wt.col(k).array() / (Ajt.array() + 1e-15);

            double b = Aj.dot(mu) - sumW(k);
            mu.array().square();

            double a = Aj.dot(mu) + L2_w;

            b += a * Hj(k) - L1_w; //- beta(1) * (sumHj - Hj(k));
            b /= (a + 1e-15);

            if (b < 0) {
                rel_err = 1;

                Ajt -= Hj(k) * Wt.col(k);
                sumHj -= Hj(k);
                Hj(k) = 0;
            }
            else if (b != Hj(k)) {
                double diff = b - Hj(k);
                Ajt += diff * Wt.col(k);

                rel_err += std::abs(diff) / (Hj(k) + 1e-15);

                sumHj += diff;
                Hj(k) = b;
            }
        }
    }
    return int(t);
}

// update h given A and w
inline void kl_update(Eigen::MatrixXd& H, const Eigen::MatrixXd& Wt, const Eigen::MatrixXd A, const double L1_w, const double L2_w, const int threads) {
    unsigned int max_iter = 100;
    double rel_tol = 1e-4;
    Eigen::VectorXd sumW = Wt.rowwise().sum();
    Eigen::MatrixXd W = Wt.transpose();

    #pragma omp parallel for
    for (long int j = 0; j < A.cols(); ++j) {

        scd_kl_update(H.col(j), W, A.col(j), sumW, L1_w, L2_w, rel_tol);
    }
}

Rcpp::List c_kl_nmf_base(Eigen::MatrixXd& A, const double tol, const uint16_t maxit, const bool verbose, const double L1_w, const double L1_h, const double L2_w, const double L2_h, const uint16_t threads, Eigen::MatrixXd W) {
    // initialize random W and H
    Eigen::MatrixXd H = Eigen::MatrixXd::Random(W.rows(), A.cols()); // with random values. See paper under Eq. 6
    Eigen::VectorXd d = Eigen::VectorXd::Ones(W.rows());

    double tol_ = 1;
    if (verbose)
        printf("%sIter %s    KL Err   / Relative Change %s\n", "\u001b[34m", "\u001b[32m", "\u001b[0m");

    const double rel_tol = 1e-4;
    double rel_err = rel_tol + 1;
    double klerror_last = 1e99; // last kl error
    std::vector<double> kl_error(maxit, ((A.array() + 1e-15) * (A.array() + 1e-15).log() - A.array()).mean());

    // alternating least squares update loop
    for (uint16_t iter_ = 0; iter_ < maxit && tol_ > tol; ++iter_) {
        // Eigen::MatrixXd w_it = W;
        // update W
        kl_update(W, H, A.transpose(), L1_w, L2_w, threads);
        // scale(W, d);
        Rcpp::checkUserInterrupt();
        // update H
        kl_update(H, W, A, L1_h, L2_h, threads);
        // scale(H, d);

        Eigen::MatrixXd A_hat = W.transpose() * H; // (H * d.asDiagonal());
        kl_error[iter_] += ((-(A.array() + 1e-15) * (A_hat.array() + 1e-15).log()).array() + A_hat.array()).mean();
        add_penalty(iter_, kl_error, W, H, L1_w, L2_w, L1_h, L2_h);

        rel_err = (klerror_last - kl_error[iter_]) / (klerror_last + kl_error[iter_] + 1e-15);
        if (verbose)
            printf("%s%4d |%s %10lf / %10lf%% %s\n", "\u001b[34m", iter_, "\u001b[32m", kl_error[iter_], rel_err * 100, "\u001b[0m");
        Rcpp::checkUserInterrupt();
    }
    return Rcpp::List::create(Rcpp::Named("w") = W, Rcpp::Named("d") = d, Rcpp::Named("h") = H);
}
//[[Rcpp::export]]
Rcpp::List c_kl_nmf(Eigen::MatrixXd& A, const double tol, const uint16_t maxit, const bool verbose, const double L1_w, const double L1_h, const double L2_w, const double L2_h, 
                    const uint16_t threads, Eigen::MatrixXd W) {
    return c_kl_nmf_base(A, tol, maxit, verbose, L1_w, L1_h, L2_w, L2_h, threads, W);
}

