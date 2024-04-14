// add_penalty to the target error 'terr'
void add_penalty(const unsigned int& i_e, std::vector<double> terr, const Eigen::MatrixXd W, const Eigen::MatrixXd H, const double L1_w, const double L1_h, const double L2_w, const double L2_h) {
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

            double a = Aj.dot(mu); //+ L2_w;

            b += a * Hj(k); //- L1_w; //- beta(1) * (sumHj - Hj(k));
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
inline void kl_update(Eigen::MatrixXd& H, const Eigen::MatrixXd& Wt, const Eigen::MatrixXd A, const double L1, const double L2, const int threads) {
    unsigned int max_iter = 100;
    double rel_tol = 1e-4;
    Eigen::VectorXd sumW = Wt.rowwise().sum();
    Eigen::MatrixXd W = Wt.transpose();

    #pragma omp parallel for
    for (long int j = 0; j < A.cols(); ++j) {

        scd_kl_update(H.col(j), W, A.col(j), sumW, L1, L2, rel_tol);

    }
}

Rcpp::List c_kl_nmf_base(const Eigen::MatrixXd& A, const unsigned int k, int maxiter, int verbose, const double L1_h, const double L2_h, const double L1_w, const double L2_w, const unsigned int threads) {

    Rprintf("%sIter %s    KL Err   / Relative Change %s\n", "\u001b[34m", "\u001b[32m", "\u001b[0m");
    // initialize random W and H
    Eigen::MatrixXd W = Eigen::MatrixXd::Random(k, A.rows()); // These **both** should be initialized 
    Eigen::MatrixXd H = Eigen::MatrixXd::Random(k, A.cols()); // with random values. See paper under Eq. 6
    Eigen::VectorXd d = Eigen::VectorXd::Ones(W.rows());

    // Eigen::MatrixXd W_last = H;

    const uint maxit = 1000;

    const double rel_tol = 1e-4;
    double rel_err = rel_tol + 1;
    double klerror_last = 1e99; // last kl error
    // double cor_last = cor(W_last, W);
    std::vector<double> kl_error(maxit, ((A.array() + 1e-15) * (A.array() + 1e-15).log() - A.array()).mean());
    std::cout << "KL Error: " << kl_error[0] << std::endl;

    for (uint16_t iter_ = 0; iter_ < maxit && rel_err > rel_tol; ++iter_) {
        // W_last = W;

        // update h
        kl_update(H, W, A, L1_h, L2_h, threads);
        scale(H, d);
        Rcpp::checkUserInterrupt();

        // update w
        kl_update(W, H, A.transpose(), L1_w, L2_w, threads);

        scale(W, d);
        Eigen::MatrixXd A_hat = W.transpose() * d.asDiagonal() * H;

        kl_error[iter_] += ((-(A.array() + 1e-15) * (A_hat.array() + 1e-15).log()).array() + A_hat.array()).mean();
        // add_penalty(iter_, kl_error, W, H, L1_w, L2_w, L1_h, L2_h);

        //print kl error
        rel_err = std::fabs((klerror_last - kl_error[iter_]) / (klerror_last + kl_error[iter_] + 1e-15));
        klerror_last = kl_error[iter_];

        printf("%4ld | %10f / %10f\n", iter_, kl_error[iter_], rel_err);

        Rcpp::checkUserInterrupt();

    }

    return Rcpp::List::create(Rcpp::Named("w") = W, Rcpp::Named("d") = d, Rcpp::Named("h") = H);
}

//[[Rcpp::export]]
Rcpp::List c_kl_nmf(Eigen::MatrixXd& A, int k, const double tol, const uint16_t maxit, const bool verbose, const double L1_w, const double L1_h, const double L2_w, const double L2_h,
                    const uint16_t threads) {
    return c_kl_nmf_base(A, k, maxit, verbose, L1_h, L2_h, L1_w, L2_w, threads);
                         // A, k, tol, maxit, verbose, L1_h, L2_h, L1_w, L2_w, threads
        
}
