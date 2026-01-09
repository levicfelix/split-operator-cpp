// grid.h
#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;
using cd = std::complex<double>;

// -------------------- constants (same as your original code) --------------------
static constexpr double hbar_SI_eVs = 6.582119569e-16;        // [eV*s]
static constexpr double e_charge    = 1.602176634e-19;        // [C]
static constexpr double m_e_kg      = 9.1093837015e-31;       // [kg]
static constexpr double pi_val      = 3.14159265358979323846; // pi

// Convert to your units:
static constexpr double hbar_meV_fs     = hbar_SI_eVs * 1e18;          // [meV*fs]
static constexpr double m0_meV_fs2_A2   = m_e_kg * 1e13 / e_charge;    // [meV*fs^2/Ang^2]

// -------------------- helpers --------------------
static inline std::string trim(std::string s) {
    auto not_space = [](unsigned char c){ return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

static inline std::string strip_comment(std::string s) {
    auto pos = s.find('#');
    if (pos != std::string::npos) s = s.substr(0, pos);
    return s;
}

static inline bool parse_key_value(const std::string& line, std::string& key, std::string& val) {
    std::string s = trim(strip_comment(line));
    if (s.empty()) return false;

    size_t p = s.find('=');
    if (p == std::string::npos) p = s.find(':');
    if (p == std::string::npos) return false;

    key = trim(s.substr(0, p));
    val = trim(s.substr(p + 1));
    return (!key.empty() && !val.empty());
}

static inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

static inline fs::path snapshots_dir_from_inp(const fs::path& inp_path) {
    // filename.inp -> filename_snapshots
    fs::path stem = inp_path.stem();              // "filename"
    return stem.string() + "_snapshots";          // relative path in cwd
}

// -------------------- Params --------------------
struct Params {
    // core numerics
    double dt = 0.7;
    int    Nsteps = 400;

    double dx = 7.0;
    double dy = 7.0;
    double Lx = 3000.0;
    double Ly = 3000.0;

    int snapshots_step = 1;
    int init_time = 0;

    // grain orientations (degrees)
    double alpha1_deg = 90.0;
    double alpha2_deg = 0.0;

    // effective masses (in units of m0)
    double mx_over_m0 = 0.846;
    double my_over_m0 = 0.166;

    std::string material = "Monolayer Phosphorene";

    // wavepacket
    double d = 100.0;
    double phi1_deg = 30.0;
    double E = 300.0;
    double V0 = 0.0;
    double x0 = -400.0;
    double y0 = 200.0;

    // constants exposed (some code may reference these)
    double hbar = hbar_meV_fs;
    double m0   = m0_meV_fs2_A2;
    double pi   = pi_val;
};

// -------------------- Grid --------------------
struct Grid {
    int Nx = 0;
    int Ny = 0;
    double dx = 0.0;
    double dy = 0.0;

    std::vector<double> x;     // Nx
    std::vector<double> y;     // Ny

    std::vector<double> mux;   // Nx
    std::vector<double> muy;   // Nx
    std::vector<double> gxy;   // Nx

    // region constants
    double ux1 = 0.0, uy1 = 0.0, gxy1 = 0.0;
    double ux2 = 0.0, uy2 = 0.0, gxy2 = 0.0;

    Eigen::MatrixXcd xi0;      // Nx x Ny
};

// -------------------- read .inp (only overrides keys present) --------------------
static inline bool read_input(const fs::path& inp, Params& P) {
    std::ifstream fin(inp);
    if (!fin) {
        std::cerr << "Error: could not open input file: " << inp << "\n";
        return false;
    }

    std::string line;
    int lineno = 0;

    try {
        while (std::getline(fin, line)) {
            ++lineno;
            std::string key, val;
            if (!parse_key_value(line, key, val)) continue;
            key = lower(key);

            auto to_double = [&](const std::string& s)->double { return std::stod(s); };
            auto to_int    = [&](const std::string& s)->int    { return std::stoi(s); };

            // numerics
            if      (key == "dt")             P.dt = to_double(val);
            else if (key == "nsteps")         P.Nsteps = to_int(val);
            else if (key == "dx")             P.dx = to_double(val);
            else if (key == "dy")             P.dy = to_double(val);
            else if (key == "lx")             P.Lx = to_double(val);
            else if (key == "ly")             P.Ly = to_double(val);
            else if (key == "snapshots_step") P.snapshots_step = to_int(val);
            else if (key == "init_time")      P.init_time = to_int(val);

            // geometry/material
            else if (key == "alpha1")         P.alpha1_deg = to_double(val);
            else if (key == "alpha2")         P.alpha2_deg = to_double(val);
            else if (key == "mx")             P.mx_over_m0 = to_double(val);
            else if (key == "my")             P.my_over_m0 = to_double(val);
            else if (key == "material")       P.material = val;

            // wavepacket
            else if (key == "d")              P.d = to_double(val);
            else if (key == "phi1")           P.phi1_deg = to_double(val);
            else if (key == "e")              P.E = to_double(val);
            else if (key == "v0")             P.V0 = to_double(val);
            else if (key == "x0")             P.x0 = to_double(val);
            else if (key == "y0")             P.y0 = to_double(val);

            // optional constant overrides
            else if (key == "hbar")           P.hbar = to_double(val);
            else if (key == "m0")             P.m0 = to_double(val);

            else {
                // unknown keys: ignore, but you can warn if you want
                // std::cerr << "Warning: unknown key '" << key << "' ignored\n";
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing " << inp << " at/near line " << lineno << ": " << e.what() << "\n";
        return false;
    }

    // minimal sanity checks (not "must provide keys", just "must be valid values")
    if (P.dt <= 0)             { std::cerr << "Error: dt must be > 0\n"; return false; }
    if (P.Nsteps <= 0)         { std::cerr << "Error: Nsteps must be > 0\n"; return false; }
    if (P.dx <= 0 || P.dy <= 0){ std::cerr << "Error: dx,dy must be > 0\n"; return false; }
    if (P.Lx <= 0 || P.Ly <= 0){ std::cerr << "Error: Lx,Ly must be > 0\n"; return false; }
    if (P.snapshots_step <= 0) { std::cerr << "Error: snapshots_step must be > 0\n"; return false; }
    if (P.d <= 0)              { std::cerr << "Error: d must be > 0\n"; return false; }

    return true;
}

// -------------------- wave_header.bin (needed by python) --------------------
static inline void write_wave_header_bin(const fs::path& snapdir, int Nx, int Ny, double dx, double dy) {
    fs::create_directories(snapdir);
    std::ofstream hb(snapdir / "wave_header.bin", std::ios::binary);
    int32_t nx = static_cast<int32_t>(Nx);
    int32_t ny = static_cast<int32_t>(Ny);
    float fdx  = static_cast<float>(dx);
    float fdy  = static_cast<float>(dy);
    hb.write(reinterpret_cast<const char*>(&nx), sizeof(int32_t));
    hb.write(reinterpret_cast<const char*>(&ny), sizeof(int32_t));
    hb.write(reinterpret_cast<const char*>(&fdx), sizeof(float));
    hb.write(reinterpret_cast<const char*>(&fdy), sizeof(float));
}

// -------------------- optional input.dat (human readable) --------------------
static inline void write_input_dat(const fs::path& outpath, const Params& P) {
    std::ofstream out(outpath);
    out << "# Plot metadata (generated from .inp)\n";
    out << "# Units: meV, fs, Angstrom, degrees\n";
    out << "# Material: " << P.material << "\n";
    out.setf(std::ios::fixed);
    out.precision(6);

    out << P.dt << " # dt\n";
    out << P.Nsteps << " # Nsteps\n";
    out << P.dx << " # dx\n";
    out << P.dy << " # dy\n";
    out << P.Lx << " # Lx\n";
    out << P.Ly << " # Ly\n";
    out << P.alpha1_deg << " # alpha1 (deg)\n";
    out << P.alpha2_deg << " # alpha2 (deg)\n";
    out << P.mx_over_m0 << " # mx/m0\n";
    out << P.my_over_m0 << " # my/m0\n";
    out << P.d << " # d\n";
    out << P.phi1_deg << " # phi1 (deg)\n";
    out << P.E << " # E\n";
    out << P.V0 << " # V0\n";
    out << P.x0 << " # x0\n";
    out << P.y0 << " # y0\n";
    out << P.snapshots_step << " # snapshots_step\n";
    out << P.init_time << " # init_time\n";
}

// -------------------- build grid --------------------
static inline Grid build_grid(const Params& P, const fs::path& snapdir) {
    Grid G;
    G.dx = P.dx;
    G.dy = P.dy;

    G.Nx = static_cast<int>(P.Lx / P.dx);
    G.Ny = static_cast<int>(P.Ly / P.dy);

    const int Nx = G.Nx;
    const int Ny = G.Ny;

    // centered at 0
    G.x.resize(Nx);
    G.y.resize(Ny);
    for (int i = 0; i < Nx; ++i) G.x[i] = (i - Nx/2) * P.dx;
    for (int j = 0; j < Ny; ++j) G.y[j] = (j - Ny/2) * P.dy;

    const double alpha1 = P.alpha1_deg * P.pi / 180.0;
    const double alpha2 = P.alpha2_deg * P.pi / 180.0;

    const double mx = P.mx_over_m0 * P.m0;
    const double my = P.my_over_m0 * P.m0;

    // region 1
    G.ux1  = 1.0 / ( (std::cos(alpha1)*std::cos(alpha1))/mx + (std::sin(alpha1)*std::sin(alpha1))/my );
    G.uy1  = 1.0 / ( (std::sin(alpha1)*std::sin(alpha1))/mx + (std::cos(alpha1)*std::cos(alpha1))/my );
    G.gxy1 = (1.0/mx - 1.0/my) * std::cos(alpha1) * std::sin(alpha1);

    // region 2
    G.ux2  = 1.0 / ( (std::cos(alpha2)*std::cos(alpha2))/mx + (std::sin(alpha2)*std::sin(alpha2))/my );
    G.uy2  = 1.0 / ( (std::sin(alpha2)*std::sin(alpha2))/mx + (std::cos(alpha2)*std::cos(alpha2))/my );
    G.gxy2 = (1.0/mx - 1.0/my) * std::cos(alpha2) * std::sin(alpha2);

    // x-dependent arrays
    G.mux.assign(Nx, 0.0);
    G.muy.assign(Nx, 0.0);
    G.gxy.assign(Nx, 0.0);
    for (int i = 0; i < Nx; ++i) {
        if (i < Nx/2) { G.mux[i]=G.ux1; G.muy[i]=G.uy1; G.gxy[i]=G.gxy1; }
        else         { G.mux[i]=G.ux2; G.muy[i]=G.uy2; G.gxy[i]=G.gxy2; }
    }

    // header for python
    write_wave_header_bin(snapdir, Nx, Ny, P.dx, P.dy);

    G.xi0 = Eigen::MatrixXcd::Zero(Nx, Ny);
    return G;
}

// -------------------- initial wavepacket --------------------
static inline void initialize_wavepacket(const Params& P, Grid& G,
                                         Eigen::MatrixXcd& psi0,
                                         Eigen::MatrixXcd& xi) {
    const int Nx = G.Nx;
    const int Ny = G.Ny;

    psi0 = Eigen::MatrixXcd::Zero(Nx, Ny);
    xi   = Eigen::MatrixXcd::Zero(Nx, Ny);

    const double phi1 = P.phi1_deg * P.pi / 180.0;

    const double ux1 = G.ux1;
    const double uy1 = G.uy1;
    const double gxy1 = G.gxy1;

    const double theta1 = std::atan((gxy1 - std::tan(phi1)/ux1) /
                                    (std::tan(phi1)*gxy1 - 1.0/uy1));

    const double kk0 = ( (std::cos(theta1)*std::cos(theta1))/ux1
                       + (std::sin(theta1)*std::sin(theta1))/uy1
                       + 2.0*std::cos(theta1)*std::sin(theta1)*gxy1 );

    const double k0  = std::sqrt(2.0 * P.E / kk0) / P.hbar;
    const double k0x = k0 * std::cos(theta1);
    const double k0y = k0 * std::sin(theta1);

    const double d  = P.d;
    const double x0 = P.x0;
    const double y0 = P.y0;

    for (int i = 0; i < Nx; ++i) {
        const double x = G.x[i];
        for (int j = 0; j < Ny; ++j) {
            const double y = G.y[j];
            const double k0dotr = k0x*x + k0y*y;

            const cd gaussexp = std::exp(cd(
                -0.5*(x - x0)*(x - x0)/(d*d) - 0.5*(y - y0)*(y - y0)/(d*d),
                k0dotr
            ));

            psi0(i,j) = gaussexp / (d * std::sqrt(2.0 * P.pi));
            xi(i,j)   = psi0(i,j);
        }
    }

    G.xi0 = xi;
}
