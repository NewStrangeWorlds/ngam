#include "species_definition.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <iomanip>


namespace ngam{


bool GasCO2CIA::calcContinuumAbsorption(
  const double temperature, 
  const std::vector<double>& number_densities, 
  std::vector<double>& absorption_coeff)
{
  const size_t nb_wavelengths = spectral_grid->nbSpectralPoints();
  
  const double number_density = number_densities[_CO2];
  if (number_density == 0) return false;

  //partial pressure of CO2 in bar
  const double partial_pressure = 
    number_densities[_CO2] * constants::boltzmann_k * temperature * 1e-6; 
  
  // Limit temperature to the valid range of the CIA data (100-800 K).
  double temp_calc = temperature;

  if (temp_calc < 100) temp_calc = 100;
  if (temp_calc > 800) temp_calc = 800;

  #pragma omp parallel for
  for (size_t i=0; i<nb_wavelengths; ++i)
  {
    //volume absorption coefficient in cm-1; the CO2 density is already
    //included via the partial pressure (k * amagat^2)
    absorption_coeff[i] = compute(
      temp_calc,
      spectral_grid->wavenumber_list[i],
      partial_pressure);
  }

  return true;
}


void GasCO2CIA::loadDimerData(const std::string& filepath) 
{
  std::ifstream file(filepath, std::ios::binary);
  if (!file) {
    throw std::runtime_error(
        "CO2Continuum::loadDimerData: could not open file: " + filepath);
  }

  // FORTRAN unformatted files wrap each record with 4-byte length markers.
  int32_t rec_len;

  // Record 1: wn_arr  (nS_ doubles)
  file.read(reinterpret_cast<char*>(&rec_len), sizeof(rec_len));
  file.read(reinterpret_cast<char*>(wn_arr_.data()), nS_ * sizeof(double));
  file.read(reinterpret_cast<char*>(&rec_len), sizeof(rec_len));

  // Record 2: temp_arr  (nT_ doubles)
  file.read(reinterpret_cast<char*>(&rec_len), sizeof(rec_len));
  file.read(reinterpret_cast<char*>(temp_arr_.data()), nT_ * sizeof(double));
  file.read(reinterpret_cast<char*>(&rec_len), sizeof(rec_len));

  // Record 3: dim_arr  (nS_ * nT_ doubles, column-major)
  file.read(reinterpret_cast<char*>(&rec_len), sizeof(rec_len));
  file.read(reinterpret_cast<char*>(dim_arr_.data()),
            nS_ * nT_ * sizeof(double));
  file.read(reinterpret_cast<char*>(&rec_len), sizeof(rec_len));

  if (!file) {
    throw std::runtime_error(
        "CO2Continuum::loadDimerData: error reading file: " + filepath);
  }

  data_loaded_ = true;
}


double GasCO2CIA::compute(double T, double nu, double Ps) const 
{
  double k1 = 0.0;
  double k2 = 0.0;

  // Convert partial pressure from bar to atm and compute amagats of CO2.
  double amg = 273.15 / T * Ps * 0.986923;

  if (nu >= 0.0 && nu <= 500.0) {
    k1 = getspc(T, nu);  // cm^{-1} amagat^{-2}
  }

  if (nu >= 1000.0 && nu <= 2000.0) {
    k2 = baranov(T, nu);  // cm^{-1} amagat^{-2}
  }

  double k = k1 + k2;  // cm^{-1} amagat^{-2}
  return k * amg * amg; // cm^{-1}
}


// ---------------------------------------------------------------------------
// Analytical roto-translational CIA spectrum (Gruszka & Borysow 1997/1998).
// Valid for 0-500 cm^{-1}, temperatures 200-800 K.
// ---------------------------------------------------------------------------
double GasCO2CIA::getspc(double temp, double wn) {
  // Fitting coefficients (A, B, C parameters from the paper).
  // tau^{L}_{1}:
  static constexpr double ah[3] = {0.1586914382e+13, -0.9344296879e+01,
                                    0.6943966881e+00};
  // tau^{L}_{2}:
  static constexpr double bh[3] = {0.1285676961e-12,  0.9420973263e+01,
                                   -0.7855988401e+00};
  // tau^{H}_{1}:
  static constexpr double at[3] = {0.3312598766e-09,  0.7285659464e+01,
                                   -0.6732642658e+00};
  // tau^{H}_{2}:
  static constexpr double bt[3] = {0.1960966173e+09, -0.6834613750e+01,
                                    0.5516825232e+00};
  // gamma_{1}:
  static constexpr double gam[3] = {0.1059151675e+17, -0.1048630307e+02,
                                     0.7321430968e+00};

  // S-shape function transition parameters (cm^{-1}).
  constexpr double w1 = 50.0;
  constexpr double w2 = 100.0;

  double incrmt = 250.0;
  int p1 = static_cast<int>(std::lround(w1 / incrmt));
  int p2 = static_cast<int>(std::lround(w2 / incrmt));

  double lntemp = std::log(temp);

  double a1 = 1.0 / (ah[0] * std::exp(ah[1] * lntemp + ah[2] * lntemp * lntemp));
  double b1 = bh[0] * std::exp(bh[1] * lntemp + bh[2] * lntemp * lntemp);
  double a2 = 1.0 / (at[0] * std::exp(at[1] * lntemp + at[2] * lntemp * lntemp));
  double b2 = bt[0] * std::exp(bt[1] * lntemp + bt[2] * lntemp * lntemp);
  double gamma = gam[0] * std::exp(gam[1] * lntemp + gam[2] * lntemp * lntemp);

  constexpr double icm = 0.1885;
  double frq = wn;

  double x1 = b1 * std::sqrt(a1 * a1 + icm * icm * frq * frq);
  double bc1 = std::exp(a1 * b1) * a1 * xk1(x1) /
               (a1 * a1 + icm * icm * frq * frq);

  double x2 = b2 * std::sqrt(a2 * a2 + icm * icm * frq * frq);
  double bc2 = std::exp(a2 * b2) * a2 * xk1(x2) /
               (a2 * a2 + icm * icm * frq * frq);

  // S-shape blending between low-frequency (bc1) and high-frequency (bc2).
  double bcbc;
  if (p1 >= 1) {
    bcbc = bc1;
  } else if (p2 < 1) {
    bcbc = bc2;
  } else {
    double ratio =
        static_cast<double>(1 - p1) / static_cast<double>(p2 - p1);
    bcbc = std::exp((1.0 - ratio) * std::log(bc1) + ratio * std::log(bc2));
  }

  constexpr double spunit = 1.296917e55;
  constexpr double gm0con = 1.259009e-6;
  double scon = spunit / temp;
  double mtot = gm0con * temp * gamma * 1.0e-56;

  return scon * mtot * bcbc * wn * wn;
}


// ---------------------------------------------------------------------------
// Modified Bessel function K1(x) * x.
// Precision better than 2.2e-7 everywhere.
// Reference: Abramowitz & Stegun, p.379; tables p.417.
// ---------------------------------------------------------------------------
double GasCO2CIA::xk1(double x) {
  if (x <= 2.0) {
    double t = (x / 3.75) * (x / 3.75);
    double fi1 =
        x * ((((((0.00032411 * t + 0.00301532) * t + 0.02658733) * t +
                 0.15084934) *
                    t +
                0.51498869) *
                   t +
               0.87890594) *
                  t +
              0.5);
    t = (x / 2.0) * (x / 2.0);
    double p = ((((((-0.00004686 * t - 0.00110404) * t - 0.01919402) * t -
                    0.18156897) *
                       t -
                   0.67278579) *
                      t +
                  0.15443144) *
                     t +
                 1.0);
    return x * std::log(x / 2.0) * fi1 + p;
  } else {
    double t = 2.0 / x;
    double p = (((((-0.00068245 * t + 0.00325614) * t - 0.00780353) * t +
                   0.01504268) *
                      t -
                  0.03655620) *
                     t +
                 0.23498619) *
                    t +
                1.25331414;
    double x_clamped = std::min(x, 330.0);
    return std::sqrt(x_clamped) * std::exp(-x_clamped) * p;
  }
}


// ---------------------------------------------------------------------------
// CIA dimer spectrum from Baranov (2004) data (added by R.D. Wordsworth 2009).
// Valid for 1000-2000 cm^{-1}.
// ---------------------------------------------------------------------------
double GasCO2CIA::baranov(double temp, double wn) const {
  if (!data_loaded_) {
    throw std::runtime_error(
        "CO2Continuum::baranov: dimer data not loaded. "
        "Call loadDimerData() first.");
  }

  return bilinear(wn_arr_.data(), temp_arr_.data(), nS_, nT_, dim_arr_.data(),
                  wn, temp);
}


// ---------------------------------------------------------------------------
// 2D bilinear interpolation for an irregular grid.
// f2d_arr is stored column-major with dimensions (nX, nY).
// ---------------------------------------------------------------------------
double GasCO2CIA::bilinear(const double* x_arr, const double* y_arr,
                              int nX, int nY, const double* f2d_arr, double x,
                              double y) {
  // Check wavenumber range (margin of 2 points from each boundary,
  // matching the original FORTRAN bounds x_arr(2) .. x_arr(nX-2)).
  if (x < x_arr[1] || x > x_arr[nX - 3]) {
    return 0.0;
  }

  // Find x bracket: x_arr[a] <= x < x_arr[a+1]  (0-based indices).
  int a = 0;
  for (int i = 1; i < nX; ++i) {
    if (x_arr[i] > x) {
      a = i - 1;
      break;
    }
  }

  // Clamp temperature to the valid range.
  if (y < y_arr[0]) y = y_arr[0] + 1.0;
  if (y > y_arr[nY - 1]) y = y_arr[nY - 1] - 1.0;

  // Find y bracket: y_arr[b] <= y < y_arr[b+1]  (0-based indices).
  int b = 0;
  for (int j = 1; j < nY; ++j) {
    if (y_arr[j] > y) {
      b = j - 1;
      break;
    }
  }

  double x1 = x_arr[a];
  double x2 = x_arr[a + 1];
  double y1 = y_arr[b];
  double y2 = y_arr[b + 1];

  // Column-major access: FORTRAN f(i,j) -> C++ f2d_arr[j * nX + i].
  double f11 = f2d_arr[b * nX + a];
  double f21 = f2d_arr[b * nX + (a + 1)];
  double f12 = f2d_arr[(b + 1) * nX + a];
  double f22 = f2d_arr[(b + 1) * nX + (a + 1)];

  // Interpolate in x-direction first.
  double fA = f11 * (x2 - x) / (x2 - x1) + f21 * (x - x1) / (x2 - x1);
  double fB = f12 * (x2 - x) / (x2 - x1) + f22 * (x - x1) / (x2 - x1);

  // Then in y-direction.
  return fA * (y2 - y) / (y2 - y1) + fB * (y - y1) / (y2 - y1);
}


}