// propagate.h
#pragma once

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <complex>
#include <vector>

#include "grid.h"

namespace splitop {

using cd     = std::complex<double>;
using SpMat  = Eigen::SparseMatrix<cd>;
using Triplet= Eigen::Triplet<cd>;

// Build tridiagonal sparse matrix from diagonals (sub, main, super)
static inline SpMat tridiag(const std::vector<cd>& a,
                            const std::vector<cd>& b,
                            const std::vector<cd>& c,
                            int n) {
    std::vector<Triplet> trips;
    trips.reserve((n - 1) + n + (n - 1));

    for (int i = 0; i < n - 1; ++i) trips.emplace_back(i + 1, i, a[i]);     // sub
    for (int i = 0; i < n;     ++i) trips.emplace_back(i,     i, b[i]);     // main
    for (int i = 0; i < n - 1; ++i) trips.emplace_back(i, i + 1, c[i]);     // super

    SpMat M(n, n);
    M.setFromTriplets(trips.begin(), trips.end());
    return M;
}

// X-part matrices
static inline std::pair<SpMat, SpMat> sparse_matrices_x(
    int Nx, double dt, double dx, double hbar,
    const std::vector<double>& mux
) {
    std::vector<cd> MLdiag1(Nx - 1, cd(0, 0));
    std::vector<cd> MLdiag2(Nx,     cd(0, 0));
    std::vector<cd> MLdiag3(Nx - 1, cd(0, 0));
    std::vector<cd> MRdiag1(Nx - 1, cd(0, 0));
    std::vector<cd> MRdiag2(Nx,     cd(0, 0));
    std::vector<cd> MRdiag3(Nx - 1, cd(0, 0));

    const cd I(0.0, 1.0);

    for (int i = 0; i < Nx - 1; ++i) {
        cd D1 = 1.0 + I * (hbar * dt) / (2.0 * mux[i] * (dx * dx));
        cd D2 = -I * (hbar * dt) / (4.0 * mux[i] * (dx * dx));
        MLdiag1[i] = D2;
        MLdiag2[i] = D1;
        MLdiag3[i] = D2;

        cd D3 = 1.0 - I * (hbar * dt) / (2.0 * mux[i] * (dx * dx));
        cd D4 =  I * (hbar * dt) / (4.0 * mux[i] * (dx * dx));
        MRdiag1[i] = D4;
        MRdiag2[i] = D3;
        MRdiag3[i] = D4;
    }

    MLdiag2[Nx - 1] = 1.0 + I * (hbar * dt) / (2.0 * mux.back() * (dx * dx));
    MRdiag2[Nx - 1] = 1.0 - I * (hbar * dt) / (2.0 * mux.back() * (dx * dx));

    SpMat ML = tridiag(MLdiag1, MLdiag2, MLdiag3, Nx);
    SpMat MR = tridiag(MRdiag1, MRdiag2, MRdiag3, Nx);
    return {ML, MR};
}

// Y-part matrices (single muy value per region)
static inline std::pair<SpMat, SpMat> sparse_matrices_y(
    int Ny, double dt, double dy, double hbar,
    double muy_i
) {
    const cd I(0.0, 1.0);

    cd D1 = 1.0 + I * (hbar * dt) / (2.0 * muy_i * (dy * dy));
    cd D2 = -I * (hbar * dt) / (4.0 * muy_i * (dy * dy));

    std::vector<cd> MLdiag1(Ny - 1, D2);
    std::vector<cd> MLdiag2(Ny,     D1);
    std::vector<cd> MLdiag3(Ny - 1, D2);

    cd D3 = 1.0 - I * (hbar * dt) / (2.0 * muy_i * (dy * dy));
    cd D4 =  I * (hbar * dt) / (4.0 * muy_i * (dy * dy));

    std::vector<cd> MRdiag1(Ny - 1, D4);
    std::vector<cd> MRdiag2(Ny,     D3);
    std::vector<cd> MRdiag3(Ny - 1, D4);

    SpMat ML = tridiag(MLdiag1, MLdiag2, MLdiag3, Ny);
    SpMat MR = tridiag(MRdiag1, MRdiag2, MRdiag3, Ny);
    return {ML, MR};
}

// XY-part matrices
static inline std::pair<SpMat, SpMat> sparse_matrices_xy(
    int Nx, int Ny, double dt, double dx, double dy, double hbar,
    const std::vector<double>& gxy
) {
    const cd I(0.0, 1.0);

    std::vector<cd> SLdiag1(Nx - 1, cd(0, 0));
    std::vector<cd> SLdiag3(Nx - 1, cd(0, 0));
    std::vector<cd> SRdiag1(Nx - 1, cd(0, 0));
    std::vector<cd> SRdiag3(Nx - 1, cd(0, 0));

    for (int i = 0; i < Nx - 1; ++i) {
        cd D = -I * gxy[i] * (hbar * dt) / (8.0 * dx * dy);
        SLdiag1[i] = -D;
        SLdiag3[i] =  D;
        SRdiag1[i] =  D;
        SRdiag3[i] = -D;
    }

    std::vector<cd> zerosNx(Nx, cd(0, 0));
    SpMat SL = tridiag(SLdiag1, zerosNx, SLdiag3, Nx);
    SpMat SR = tridiag(SRdiag1, zerosNx, SRdiag3, Nx);

    const int N = Nx * Ny;
    std::vector<Triplet> tripsML;
    std::vector<Triplet> tripsMR;

    tripsML.reserve(static_cast<size_t>(Ny) * Nx
                    + static_cast<size_t>(2 * (Ny - 1)) * 2 * (Nx - 1));
    tripsMR.reserve(static_cast<size_t>(Ny) * Nx
                    + static_cast<size_t>(2 * (Ny - 1)) * 2 * (Nx - 1));

    auto addBlock = [&](std::vector<Triplet>& trips, int blockRow, int blockCol,
                        const SpMat& B, cd scale) {
        const int r0 = blockRow * Nx;
        const int c0 = blockCol * Nx;
        for (int k = 0; k < B.outerSize(); ++k) {
            for (SpMat::InnerIterator it(B, k); it; ++it) {
                trips.emplace_back(r0 + it.row(), c0 + it.col(), scale * it.value());
            }
        }
    };

    // diagonal identity blocks
    for (int by = 0; by < Ny; ++by) {
        const int r0 = by * Nx;
        const int c0 = by * Nx;
        for (int i = 0; i < Nx; ++i) {
            tripsML.emplace_back(r0 + i, c0 + i, cd(1.0, 0.0));
            tripsMR.emplace_back(r0 + i, c0 + i, cd(1.0, 0.0));
        }
    }

    // off-diagonal blocks
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Ny; ++i) {
            if (i == j) continue;
            if ((i - j) == 1) {
                addBlock(tripsML, j, i, SL, cd(1.0, 0.0));
                addBlock(tripsMR, j, i, SR, cd(1.0, 0.0));
            } else if ((i - j) == -1) {
                addBlock(tripsML, j, i, SL, cd(-1.0, 0.0));
                addBlock(tripsMR, j, i, SR, cd(-1.0, 0.0));
            }
        }
    }

    SpMat ML(N, N), MR(N, N);
    ML.setFromTriplets(tripsML.begin(), tripsML.end());
    MR.setFromTriplets(tripsMR.begin(), tripsMR.end());
    return {ML, MR};
}

// ============================================================================
// SplitOperatorPropagator (non-copyable/non-movable; build once, reuse)
// ============================================================================
struct SplitOperatorPropagator {
    SplitOperatorPropagator(const Params& P, const Grid& G)
        : Nx(G.Nx), Ny(G.Ny), dt(P.dt), dx(P.dx), dy(P.dy), hbar(P.hbar),
          muyL(G.uy1), muyR(G.uy2)
    {
        // Build & factorize MLxy
        {
            auto mats = sparse_matrices_xy(Nx, Ny, dt, dx, dy, hbar, G.gxy);
            MLxy = std::move(mats.first);
            MRxy = std::move(mats.second);
            solverMLxy.analyzePattern(MLxy);
            solverMLxy.factorize(MLxy);
        }

        // Build & factorize MLx
        {
            auto mats = sparse_matrices_x(Nx, dt, dx, hbar, G.mux);
            MLx = std::move(mats.first);
            MRx = std::move(mats.second);
            solverMLx.analyzePattern(MLx);
            solverMLx.factorize(MLx);
        }

        // Build & factorize MLy for left/right
        {
            auto matsL = sparse_matrices_y(Ny, dt, dy, hbar, muyL);
            MLyL = std::move(matsL.first);
            MRyL = std::move(matsL.second);
            solverMLyL.analyzePattern(MLyL);
            solverMLyL.factorize(MLyL);

            auto matsR = sparse_matrices_y(Ny, dt, dy, hbar, muyR);
            MLyR = std::move(matsR.first);
            MRyR = std::move(matsR.second);
            solverMLyR.analyzePattern(MLyR);
            solverMLyR.factorize(MLyR);
        }
    }

    SplitOperatorPropagator(const SplitOperatorPropagator&) = delete;
    SplitOperatorPropagator& operator=(const SplitOperatorPropagator&) = delete;
    SplitOperatorPropagator(SplitOperatorPropagator&&) = delete;
    SplitOperatorPropagator& operator=(SplitOperatorPropagator&&) = delete;

    // Apply one split-operator kinetic step:
    //   input  : xi  (Nx x Ny)
    //   outputs: etax (Nx x Ny), plus intermediate etaxy, etay
    //
    // Temporaries are provided by caller to avoid reallocation each step.
    void apply(const Grid& G,
               const Eigen::MatrixXcd& xi,
               Eigen::MatrixXcd& etaxy,
               Eigen::MatrixXcd& etay,
               Eigen::MatrixXcd& etax,
               Eigen::VectorXcd& axy,
               Eigen::VectorXcd& bxy,
               Eigen::VectorXcd& cxy,
               Eigen::VectorXcd& rhsY,
               Eigen::VectorXcd& solY,
               Eigen::MatrixXcd& bx) const
    {
        (void)G; // not used currently (kept for future potentials etc.)

        // ---- Txy ----
        {
            int idx = 0;
            for (int j = 0; j < Ny; ++j)
                for (int i = 0; i < Nx; ++i)
                    axy[idx++] = xi(i, j);
        }

        bxy = MRxy * axy;
        cxy = solverMLxy.solve(bxy);

        // Keep same indexing logic as your original code: skips last element in each block
        for (int j = 0; j < Ny; ++j) {
            for (int i = j * Nx; i < (j + 1) * Nx - 1; ++i) {
                etaxy(i - j * Nx, j) = cxy[i];
            }
        }

        // ---- Ty ----
        for (int i = 0; i < Nx; ++i) {
            const bool isLeft = (i < Nx / 2);

            rhsY = (isLeft ? (MRyL * etaxy.row(i).transpose())
                           : (MRyR * etaxy.row(i).transpose()));

            solY = (isLeft ? solverMLyL.solve(rhsY)
                           : solverMLyR.solve(rhsY));

            etay.row(i) = solY.transpose();
        }

        // ---- Tx ----
        for (int j = 0; j < Ny; ++j)
            bx.col(j) = MRx * etay.col(j);

        for (int j = 0; j < Ny; ++j)
            etax.col(j) = solverMLx.solve(bx.col(j));
    }

private:
    int Nx = 0, Ny = 0;
    double dt = 0.0, dx = 0.0, dy = 0.0, hbar = 0.0;
    double muyL = 0.0, muyR = 0.0;

    SpMat MLxy, MRxy;
    SpMat MLx,  MRx;
    SpMat MLyL, MRyL;
    SpMat MLyR, MRyR;

    Eigen::SparseLU<SpMat> solverMLxy;
    Eigen::SparseLU<SpMat> solverMLx;
    Eigen::SparseLU<SpMat> solverMLyL;
    Eigen::SparseLU<SpMat> solverMLyR;
};

} // namespace splitop
