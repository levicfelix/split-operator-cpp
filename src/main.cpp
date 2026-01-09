// main.cpp
#include <Eigen/Dense>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "grid.h"
#include "propagate.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n  " << argv[0] << " filename.inp\n";
        return 1;
    }

    const fs::path inp_path = argv[1];

    // 1) Read params (defaults + overrides)
    Params P;
    if (!read_input(inp_path, P)) {
        return 1;
    }

    // 2) Snapshots directory: filename_snapshots/
    const fs::path snapdir = snapshots_dir_from_inp(inp_path);
    fs::create_directories(snapdir);

    // 3) Build grid + write wave_header.bin (done inside build_grid)
    Grid G = build_grid(P, snapdir);

    // Optional, but useful for debugging/human-readable info:
    write_input_dat("input.dat", P);

    // 4) Initialize wavepacket
    const int Nx = G.Nx;
    const int Ny = G.Ny;

    Eigen::MatrixXcd psi0(Nx, Ny), psi(Nx, Ny);
    Eigen::MatrixXcd xi(Nx, Ny);
    initialize_wavepacket(P, G, psi0, xi);

    // 5) Build propagator (factorizes matrices once)
    splitop::SplitOperatorPropagator S(P, G);

    // 6) Allocate intermediates / temporaries
    Eigen::MatrixXcd etaxy = Eigen::MatrixXcd::Zero(Nx, Ny);
    Eigen::MatrixXcd etay  = Eigen::MatrixXcd::Zero(Nx, Ny);
    Eigen::MatrixXcd etax  = Eigen::MatrixXcd::Zero(Nx, Ny);

    Eigen::VectorXcd axy(Nx * Ny), bxy(Nx * Ny), cxy(Nx * Ny);
    Eigen::VectorXcd rhsY(Ny), solY(Ny);
    Eigen::MatrixXcd bx(Nx, Ny);

    // 7) average.dat (same style as your old code)
    std::ofstream favg("average.dat");
    favg << "# t  xCM  yCM  xCMl  yCMl  xCMr  yCMr  R  T\n";

    const int init_time = P.init_time;
    const int Nsteps    = P.Nsteps;

    std::cout << "Snapshots will be written to: " << snapdir << "\n";
    std::cout << "Starting wave packet propagation...\n";

    for (int k = init_time; k < Nsteps; ++k) {
        std::cout << k << "/" << (Nsteps - 1) << std::endl;

        // Apply kinetic split operator (potential is currently off, same as before)
        S.apply(G, xi, etaxy, etay, etax, axy, bxy, cxy, rhsY, solY, bx);

        // Expectation values + optional snapshot
        double xCM=0.0, yCM=0.0, xCMl=0.0, yCMl=0.0, xCMr=0.0, yCMr=0.0;
        double norm=0.0, R=0.0, T=0.0;

        std::ofstream fout;
        const bool do_snapshot = (k % P.snapshots_step == 0);
        if (do_snapshot) {
            fout.open(snapdir / ("wave_" + std::to_string(k) + ".bin"), std::ios::binary);
            if (!fout) {
                std::cerr << "Error: could not open snapshot for writing.\n";
                return 1;
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                // No potential: psi = etax, xi = psi (same logic as your original)
                psi(i,j) = etax(i,j);
                xi(i,j)  = psi(i,j);

                const double psi2 = std::real(psi(i,j) * std::conj(psi(i,j)));

                if (do_snapshot) {
                    float p = static_cast<float>(psi2);
                    fout.write(reinterpret_cast<const char*>(&p), sizeof(float));
                }

                xCM  += G.x[i] * psi2;
                yCM  += G.y[j] * psi2;
                norm += psi2;

                if (i < Nx/2) {
                    xCMl += G.x[i] * psi2;
                    yCMl += G.y[j] * psi2;
                    R    += psi2;
                } else {
                    xCMr += G.x[i] * psi2;
                    yCMr += G.y[j] * psi2;
                    T    += psi2;
                }
            }
        }

        if (do_snapshot) fout.close();

        xCM  /= norm; yCM  /= norm;
        xCMl /= norm; yCMl /= norm;
        xCMr /= norm; yCMr /= norm;
        R    /= norm; T    /= norm;

        const double t = k * P.dt;
        favg.setf(std::ios::fixed);
        favg.precision(6);
        favg << t << "  "
             << xCM << "  " << yCM << "  "
             << xCMl << "  " << yCMl << "  "
             << xCMr << "  " << yCMr << "  "
             << R << "  " << T << "\n";
    }

    favg.close();
    std::cout << "End of the calculations!\n";
    return 0;
}
