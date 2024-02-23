#ifdef _OPENMP
#include <omp.h>
#endif

// include armadillo
#include <armadillo>

#define TINY_NUM 1e-15

using namespace arma;


int scd_kl_update(subview_col<double> Hj, const mat & Wt, const vec & Aj, const vec & sumW, const subview_col<uword> mask, const vec & beta, const unsigned int & max_iter, const double & rel_tol) {
	// Problem:  Aj = W * Hj
	// Method: Sequentially minimize KL distance using quadratic approximation
	// Wt = W^T
	// sumW = column sum of W
	// mask: skip updating
	// beta: a vector of 3, for L2, angle, L1 regularization

	double sumHj = sum(Hj);
	vec Ajt = Wt.t()*Hj; // column of A computed again from W and a column of H
	vec mu; // col of W / iterative column of A
	double a; // 2nd-order-derivative
	double b; // 1st-order-derivative
	double tmp, etmp; // tmp for updating, etmp for error checking
	double rel_err = 1 + rel_tol;
	bool is_masked = mask.n_elem > 0;

	unsigned int t = 0; 
	for (; t < max_iter && rel_err > rel_tol; t++)
	{
		rel_err = 0;
		for (unsigned int k = 0; k < Wt.n_rows; k++)
		{
			if (is_masked && mask(k) > 0) continue;
			
			
			mu = Wt.row(k).t()/(Ajt + TINY_NUM);
			a = dot(Aj, square(mu)); 
			b = dot(Aj, mu) - sumW(k); // 0.5*ax^2 - bx
			//? should go after update to be (otherwise l2 is added to b in the next line) 
			a += beta(0); // L2 regularization 
			//? should be -= instead of +=
			b += a*Hj(k) - beta(2) - beta(1)*(sumHj - Hj(k)); // L1 regularization?
			tmp = b/(a+TINY_NUM);  

			if (tmp < 0) tmp = 0;
			
			if (tmp != Hj(k))
			{
				Ajt += (tmp - Hj(k)) * Wt.row(k).t();
				etmp = 2*std::abs(Hj(k)-tmp) / (tmp+Hj(k) + TINY_NUM);
				if (etmp > rel_err)
					rel_err = etmp;
				sumHj += tmp - Hj(k);
				Hj(k) = tmp;
			}
		}
	}
	return int(t);
}


int update(mat & H, const mat & Wt, const mat & A, const umat & mask, const vec & beta, unsigned int max_iter, double rel_tol, int n_threads, int method)
{
	// A = W H, solve H
	// No missing in A, Wt = W^T
	// method: 1 = scd, 2 = lee_ls, 3 = scd_kl, 4 = lee_kl

	unsigned int m = A.n_cols;
	int total_raw_iter = 0;

	if (n_threads < 0) n_threads = 0;
	bool is_masked = !mask.empty();
	mat WtW;
	vec mu, sumW;
	if (method == 1 || method == 2)
	{
		WtW = Wt*Wt.t();
		if (beta(0) != beta(1))
			WtW.diag() += beta(0) - beta(1);
		if (beta(1) != 0)
			WtW += beta(1);
		WtW.diag() += TINY_NUM; // for stability: avoid divided by 0 in scd_ls, scd_kl
	}
	else
		sumW = sum(Wt, 1);

	#pragma omp parallel for num_threads(n_threads) schedule(dynamic) private(mu)
	for (unsigned int j = 0; j < m; j++) // by columns of H
	{
		// break if all entries of col_j are masked
		if (is_masked && arma::all(mask.col(j)))
			continue;

		int iter = 0;

		if (method == 3) {
			iter = scd_kl_update(H.col(j), Wt, A.col(j), sumW, mask.col(j), beta, max_iter, rel_tol);
    }

		#pragma omp critical
		total_raw_iter += iter;
	}
	return total_raw_iter;
}

void add_penalty(const unsigned int & i_e, vec & terr, const mat & W, const mat & H,
	const unsigned int & N_non_missing, const vec & alpha, const vec & beta)
{
	// add penalty term back to the loss function (terr)
	if (alpha(0) != alpha(1))
		terr(i_e) += 0.5*(alpha(0)-alpha(1))*accu(square(W))/N_non_missing;
	if (beta(0) != beta(1))
		terr(i_e) += 0.5*(beta(0)-beta(1))*accu(square(H))/N_non_missing;
	if (alpha(1) != 0)
		terr(i_e) += 0.5*alpha(1)*accu(W*W.t())/N_non_missing;
	if (beta(1) != 0)
		terr(i_e) += 0.5*beta(1)*accu(H*H.t())/N_non_missing;
	if (alpha(2) != 0)
		terr(i_e) += alpha(2)*accu(W)/N_non_missing;
	if (beta(2) != 0)
		terr(i_e) += beta(2)*accu(H)/N_non_missing;
}


int c_nnmf(const arma::mat & A, const unsigned int k, arma::mat W, arma::mat H, arma::umat Wm, arma::umat Hm, const arma::vec & alpha, const arma::vec & beta, const unsigned int max_iter, const double rel_tol, const int n_threads, const int verbose, const bool show_warning, const unsigned int inner_max_iter,	const double inner_rel_tol, const int method, unsigned int trace)
{
	/******************************************************************************************************
	 *              Non-negative Matrix Factorization(NNMF) using alternating scheme
	 *              ----------------------------------------------------------------
	 * Description:
	 * 	Decompose matrix A such that
	 * 		A = W H
	 * Arguments:
	 * 	A              : Matrix to be decomposed
	 * 	W, H           : Initial matrices of W and H, where ncol(W) = nrow(H) = k. # of rows/columns of W/H could be 0
	 * 	Wm, Hm         : Masks of W and H, s.t. masked entries are no-updated and fixed to initial values
	 * 	alpha          : [L2, angle, L1] regularization on W (non-masked entries)
	 * 	beta           : [L2, angle, L1] regularization on H (non-masked entries)
	 * 	max_iter       : Maximum number of iteration
	 * 	rel_tol        : Relative tolerance between two successive iterations, = |e2-e1|/avg(e1, e2)
	 * 	n_threads      : Number of threads (openMP)
	 * 	verbose        : Either 0 = no any tracking, 1 == progression bar, 2 == print iteration info
	 * 	show_warning   : If to show warning if targeted `tol` is not reached
	 * 	inner_max_iter : Maximum number of iterations passed to each inner W or H matrix updating loop
	 * 	inner_rel_tol  : Relative tolerance passed to inner W or H matrix updating loop, = |e2-e1|/avg(e1, e2)
	 * 	method         : Integer of 1, 2, 3 or 4, which encodes methods
	 * 	               : 1 = sequential coordinate-wise minimization using square loss
	 * 	               : 2 = Lee's multiplicative update with square loss, which is re-scaled gradient descent
	 * 	               : 3 = sequentially quadratic approximated minimization with KL-divergence
	 * 	               : 4 = Lee's multiplicative update with KL-divergence, which is re-scaled gradient descent
	 * 	trace          : A positive integer, error will be checked very 'trace' iterations. Computing WH can be very expansive,
	 * 	               : so one may not want to check error A-WH every single iteration
	 * Return:
	 * 	A list (Rcpp::List) of
	 * 		W, H          : resulting W and H matrices
	 * 		mse_error     : a vector of mean square error (divided by number of non-missings)
	 * 		mkl_error     : a vector (length = number of iterations) of mean KL-distance
	 * 		target_error  : a vector of loss (0.5*mse or mkl), plus constraints
	 * 		average_epoch : a vector of average epochs (one complete swap over W and H)
	 * Author:
	 * 	Eric Xihui Lin <xihuil.silence@gmail.com>
	 * Version:
	 * 	2015-12-11
	 ******************************************************************************************************/

	unsigned int n = A.n_rows; // n = # of rows
	unsigned int m = A.n_cols; // m = # of columns
	//int k = H.n_rows; // decomposition rank k
	unsigned int N_non_missing = n*m; // total number of non-missing entries

	if (trace < 1) trace = 1; // check once per iteration at least

	// low long of a vector to store error
	unsigned int err_len = (unsigned int)std::ceil(double(max_iter)/double(trace)) + 1;
	
	// vectors to store errors (mse, mkl, target, average_epoch)
	vec mse_err(err_len), mkl_err(err_len), terr(err_len), ave_epoch(err_len);

	// initialize errors
	double rel_err = rel_tol + 1; // relative error
	double terr_last = 1e99; // last target error
	uvec non_missing; // indices of non-missing entries
	bool any_missing = !A.is_finite(); // if any missing entries
	if (any_missing)
	{
		non_missing = find_finite(A);
		N_non_missing = non_missing.n_elem;
		mkl_err.fill(mean((A.elem(non_missing)+TINY_NUM) % log(A.elem(non_missing)+TINY_NUM) - A.elem(non_missing)));
	}
	else
		mkl_err.fill(mean(mean((A+TINY_NUM) % log(A+TINY_NUM) - A))); // fixed part in KL-dist, mean(A log(A) - A)

	if (Wm.empty())
		Wm.resize(0, n);
	else
		inplace_trans(Wm);
	if (Hm.empty())
		Hm.resize(0, m);

	if (W.empty())
	{
		W.randu(k, n); // random initialization
		W *= 0.01; // scale down
		if (!Wm.empty())
			W.elem(find(Wm > 0)).fill(0.0); 
	}
	else
		inplace_trans(W); // if W is given, then transpose it

	if (H.empty())
	{
		H.randu(k, m);
		H *= 0.01;
		if (!Hm.empty())
			H.elem(find(Hm > 0)).fill(0.0);
	}


	// main loop -----------------------------------------------------

	int total_raw_iter = 0;
	unsigned int i = 0;
	unsigned int i_e = 0; // index for error checking
	for(; i < max_iter && std::abs(rel_err) > rel_tol; i++)
	{
    // update W
    total_raw_iter += update(W, H, A.t(), Wm, alpha, inner_max_iter, inner_rel_tol, n_threads, method);
    // update H
    total_raw_iter += update(H, W, A, Hm, beta, inner_max_iter, inner_rel_tol, n_threads, method);

    if (i % trace == 0)
    {
      const mat & Ahat = W.t()*H;
      mse_err(i_e) = mean(mean(square((A - Ahat))));
      mkl_err(i_e) += mean(   mean(   -(A+TINY_NUM) % log(Ahat+TINY_NUM) + Ahat   )   );
    }

		if (i % trace == 0)
		{
			ave_epoch(i_e) = double(total_raw_iter)/(n+m);
			terr(i_e) = mkl_err(i_e);
			add_penalty(i_e, terr, W, H, N_non_missing, alpha, beta);

			rel_err = 2*(terr_last - terr(i_e)) / (terr_last + terr(i_e) + TINY_NUM );
			terr_last = terr(i_e);

			total_raw_iter = 0; // reset to 0
			++i_e;
		}
	}

	// --------------------------------------------------------------


	// compute error of the last iteration
	if ((i-1) % trace != 0)
	{
    const mat & Ahat = W.t()*H;
    mse_err(i_e) = mean(mean(square((A - Ahat))));
    mkl_err(i_e) += mean(mean(-(A+TINY_NUM) % log(Ahat+TINY_NUM) + Ahat));

		ave_epoch(i_e) = double(total_raw_iter)/(n+m);
		terr(i_e) = mkl_err(i_e);
		add_penalty(i_e, terr, W, H, N_non_missing, alpha, beta);

		rel_err = 2*(terr_last - terr(i_e)) / (terr_last + terr(i_e) + TINY_NUM );
		terr_last = terr(i_e);

		++i_e;
	}

	if (i_e < err_len)
	{
		mse_err.resize(i_e);
		mkl_err.resize(i_e);
		terr.resize(i_e);
		ave_epoch.resize(i_e);
	}

	if (show_warning && rel_err > rel_tol)
		printf("Target tolerance not reached. Try a larger max.iter.");

	return 0;
}



// main
int main() {
    // set the number of threads
    #ifdef _OPENMP
    omp_set_num_threads(4);
    #endif

    // create a random matrix
    arma::mat A = arma::randu(4,5);
    A.print("A:");

    // return 0
    return 0;
}
