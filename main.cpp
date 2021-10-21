#include <iostream>
#include <thread>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <mutex>
#include <fstream>
#include <iomanip>
#include <array>
#include <chrono>
#include <atomic>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using Eigen::internal::BandMatrix;
using Eigen::ComplexEigenSolver;
using Eigen::MatrixXcf;
using Eigen::MatrixXcd;
using Eigen::ArrayXcf;
using Eigen::ArrayXcd;


/**
 * 
 * Summary:
 *  Program to compute eigenvalue density of the Bohemian family of
 *   skew-symmetric tridiagonal matrices via its trivial parallelization
 * 
 *  Uses population P = [1, i]
 * 
 * Options:
 *      "N" : dimension of matrices
 * 
 * 
 * Output:
 *  int[NROWS][NCOLS] containing the counts of eigenvalues within each grid
 *   of a NROWSxNCOLS mesh from (MINREAL, MINIMAG) to (MAXREAL, MAXIMAG)
 *   with uniform step sizes, written to OUTFILE
 *  
 * 
 *
 *
 *  
**/


/* macros */

#define N 31

#define T std::pow(2, std::floor(std::log2(std::thread::hardware_concurrency())))

#define NROWS 4096
#define NCOLS 4096
#define MINREAL -2.0
#define MAXREAL 2.0
#define MINIMAG -2.0
#define MAXIMAG 2.0

#define OUTFILE std::to_string(N) + std::string("x") + std::to_string(N) + std::string(".dat")


// returns kth bit of t
inline uint64_t kth_bit(uint64_t t, uint64_t k) { return (t & (1 << k)) >> k; }


// prototypes
void solve(int tid);
void update(double x, double y);
void print_tof(const std::string& fname);


// image
int img[NROWS][NCOLS] = {0};

// lock for each row of image
std::mutex locks[NROWS];

// grid spacing
double real_step = (MAXREAL - MINREAL) / (NCOLS - 1);
double im_step = (MAXIMAG - MINIMAG) / (NROWS - 1);

// count of eigenvalues computed
std::atomic<uint64_t> toteigs{0};

int main(int argc, const char* argv[]) {

    // time process
    auto start_t = std::chrono::high_resolution_clock::now();

    // vector of threads
    std::vector<std::thread> threads;

    // solve function computes eigenvalues for matrices with the first
    // log_2(T) entries encoded by tid
    for (int tid = 0; tid < T; ++tid) {
        threads.push_back(std::thread(solve, tid));
    }
    
    // sync threads
    for (auto& th : threads) { th.join(); }

    // write img to file
    print_tof(OUTFILE);

    // output some stats
    auto stop_t = std::chrono::high_resolution_clock::now();
    auto t_hrs = std::chrono::duration_cast<std::chrono::hours>(stop_t - start_t);
    auto t_mins = std::chrono::duration_cast<std::chrono::minutes>(stop_t - start_t);
    std::cout << toteigs << " eigenvalues computed from all " << (uint64_t) std::pow(2, N - 1);
    std::cout << " skew-symmetric tridiagonal matrices with population P = [1, i]";
    std::cout << " of size " << N << "x" << N << std::endl;
    std::cout << "Total execution time: " << t_hrs.count() << " hours (";
    std::cout << t_mins.count() << " mins)" << std::endl;
    std::cout << "Total threads: " << T << std::endl;
    std::cout << "Matrix written to: " << OUTFILE << std::endl;

    
    return 0;
}


void solve(int tid) {

    // tid encodes first log_2(T) entries stored in nums
    std::vector<std::complex<double>> nums;

    // first log_2(T) entries
    int m = std::floor(std::log2(T));
    for (int i = 0; i < m; ++i) {
        if (kth_bit(tid, i + 1) == 0) {
            nums.push_back(std::complex<double>(0.0, 1.0));
        }
        else {
            nums.push_back(std::complex<double>(1.0, 0.0));
        }
    }

    
    // solve eigenvalues of matrices encoded by N-1 binary vector
    // starting with nums of length log_2(T)
    std::vector<std::complex<double>> b;

    // 2^(N-1-log_2(T)) encodings of matrices for this tid to solve
    uint64_t mm = std::pow(2L, N - m - 1);
    for (uint64_t i = 0; i < mm; ++i) {
        b = nums;
        for (uint64_t j = 0; j < N - m - 1; ++j) {
            if (kth_bit(i, j + 1) == 0) {
                b.push_back(std::complex<double>(0.0, 1.0));
            }
            else {
                b.push_back(std::complex<double>(1.0, 0.0));
            }
        }

        // map off-diagonal vector
        Eigen::Map<Eigen::Matrix<std::complex<double>, 1, N - 1>> a(b.data());
        
        // make skew-symetric tridiagonal mat by setting main-diagonal to 0
        // and using off-diagonal vector mapping a for sub and super diagonals

        //  -------------------------------------------------
        // |   0   -a(1)     0      0      0      0      0   |
        // | a(1)     0   -a(2)     0      0      0      0   |
        // |   0    a(2)     0   -a(3)     0      0      0   |
        // |   0      0      .      0      .      0      0   |
        // |   0      0      0      .      0      .      0   |
        // |   0      0      0      0  a(n-2)     0 -a(n-1)  |
        // |   0      0      0      0      0  a(n-1)     0   |
        //  -------------------------------------------------

        BandMatrix<std::complex<double>> mat(N, N, 1, 1);
        mat.diagonal(0).setConstant(std::complex<double>(0.0, 0.0));
        mat.diagonal(-1) = a;
        mat.diagonal(1) = -a;


        // create eigensolver ces that computes eigenvalues of mat
        ComplexEigenSolver<MatrixXcd> ces(mat.toDenseMatrix(), /* computeEigenvectors = */ false);

        // get eigenvalues of mat from ces
        MatrixXcd eigs = ces.eigenvalues();

        // update img with eigs
        for (int j = 0; j < eigs.size(); ++j) {
            double x = eigs(j).real();
            double y = eigs(j).imag();
            update(x, y);
        }

    }
    
}


void update(double x, double y) {
    
    int xsteps = 0;
    int ysteps = 0;

    while (x > MINREAL + real_step * xsteps && xsteps < NCOLS - 2) { xsteps++; }
    while (y > MINIMAG + im_step * ysteps && ysteps < NROWS - 2) { ysteps++; }

    locks[ysteps].lock();
    img[ysteps][xsteps]++;
    locks[ysteps].unlock();

    toteigs++;

}


void print_tof(const std::string& fname) {

    std::ofstream out_file(fname);

	if (!out_file.is_open()) {
		std::cout << "Unable to open/create file for writing!" << std::endl;
		return;
	}


	for (int i = 0; i < NROWS; i++) {
		for (int j = 0; j < NCOLS; j++) {
			out_file << img[i][j];
			if (j != NCOLS - 1) { out_file << " "; }
		}
		out_file << std::endl;
	}
	out_file.close();

}
