#ifndef STELLAR_SPECTRUM_H
#define STELLAR_SPECTRUM_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../additional/aux_functions.h"
#include "../additional/physical_const.h"
#include "../additional/quadrature.h"


namespace ngam {


class StellarSpectrum {
  public:
    virtual ~StellarSpectrum() {}
    virtual std::vector<double> calcFlux(
      const std::vector<double>& wavenumber_list) const = 0;
};


class BlackbodyStar : public StellarSpectrum {
  public:
    BlackbodyStar(double stellar_temperature, double instellation_flux)
      : stellar_temperature(stellar_temperature),
        instellation_flux(instellation_flux)
    {}

    // Returns spectral flux [erg/cm^2/s/cm^-1] at each wavenumber.
    // The spectrum is a Planck function scaled so that its integral
    // over wavenumber equals instellation_flux.
    std::vector<double> calcFlux(
      const std::vector<double>& wavenumber_list) const override
    {
      const size_t n = wavenumber_list.size();
      std::vector<double> flux(n);

      for (size_t i = 0; i < n; ++i)
        flux[i] = constants::pi
                * aux::planckFunctionWavenumber(stellar_temperature, wavenumber_list[i]);

      // integrate to get the bolometric flux of the raw Planck spectrum
      double bolometric = aux::quadratureTrapezoidal(wavenumber_list, flux);

      if (bolometric > 0)
      {
        double scale = instellation_flux / bolometric;
        for (size_t i = 0; i < n; ++i)
          flux[i] *= scale;
      }

      return flux;
    }

  private:
    double stellar_temperature;
    double instellation_flux;
};


class TabulatedStar : public StellarSpectrum {
  public:
    TabulatedStar(const std::string& file_path, double instellation_flux)
      : instellation_flux(instellation_flux)
    {
      std::ifstream file(file_path);

      if (!file)
        throw std::runtime_error(
          "TabulatedStar: could not open file: " + file_path);

      std::cout << "- Stellar spectrum from file: " << file_path << "\n";

      std::vector<double> wavelength_data;
      std::vector<double> flux_data;

      std::string line;

      while (std::getline(file, line))
      {
        if (line.empty() || line[0] == '#')
          continue;

        std::istringstream iss(line);
        double wl, fl;

        if (!(iss >> wl >> fl))
          continue;

        wavelength_data.push_back(wl);
        flux_data.push_back(fl);
      }

      if (wavelength_data.size() < 2)
        throw std::runtime_error(
          "TabulatedStar: need at least 2 data points in " + file_path);

      // Convert from wavelength (mu) + F_lambda to wavenumber (cm^-1) + F_nu
      // nu = 1e4 / lambda
      // F_nu = F_lambda * lambda^2 / 1e4  (from F_lambda dlambda = F_nu dnu)
      file_wavenumber.resize(wavelength_data.size());
      file_flux.resize(wavelength_data.size());

      for (size_t i = 0; i < wavelength_data.size(); ++i)
      {
        file_wavenumber[i] = 1e4 / wavelength_data[i];
        file_flux[i] = flux_data[i]
          * wavelength_data[i] * wavelength_data[i] / 1e4;
      }

      // Ensure wavenumber is in ascending order
      if (file_wavenumber.front() > file_wavenumber.back())
      {
        std::reverse(file_wavenumber.begin(), file_wavenumber.end());
        std::reverse(file_flux.begin(), file_flux.end());
      }

      std::cout << "  - Wavelength range: "
                << wavelength_data.front() << " - " << wavelength_data.back()
                << " mu (" << wavelength_data.size() << " points)\n";
    }

    // Returns spectral flux [erg/cm^2/s/cm^-1] at each wavenumber.
    // The tabulated spectrum is interpolated and scaled so that its
    // integral over wavenumber equals instellation_flux.
    std::vector<double> calcFlux(
      const std::vector<double>& wavenumber_list) const override
    {
      const size_t n = wavenumber_list.size();
      std::vector<double> flux(n, 0.0);

      // Interpolate file data onto the model wavenumber grid
      // wavenumber_list is ascending; file_wavenumber is ascending
      size_t j = 0;

      for (size_t i = 0; i < n; ++i)
      {
        const double nu = wavenumber_list[i];

        // Outside file range: flux stays 0
        if (nu < file_wavenumber.front() || nu > file_wavenumber.back())
          continue;

        // Advance to the bracketing interval
        while (j < file_wavenumber.size() - 1 && file_wavenumber[j + 1] < nu)
          ++j;

        if (j >= file_wavenumber.size() - 1)
        {
          flux[i] = file_flux.back();
          continue;
        }

        const double t = (nu - file_wavenumber[j])
          / (file_wavenumber[j + 1] - file_wavenumber[j]);

        flux[i] = file_flux[j] + t * (file_flux[j + 1] - file_flux[j]);
      }

      // Scale so the integral over wavenumber equals instellation_flux
      double bolometric = aux::quadratureTrapezoidal(wavenumber_list, flux);

      if (bolometric > 0)
      {
        double scale = instellation_flux / bolometric;
        for (size_t i = 0; i < n; ++i)
          flux[i] *= scale;
      }

      return flux;
    }

  private:
    double instellation_flux;
    std::vector<double> file_wavenumber;
    std::vector<double> file_flux;
};


} // namespace ngam

#endif // STELLAR_SPECTRUM_H
