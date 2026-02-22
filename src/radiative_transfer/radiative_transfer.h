/*
* This file is part of the BeAR code (https://github.com/newstrangeworlds/BeAR).
* Copyright (C) 2024 Daniel Kitzmann
*
* BeAR is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* BeAR is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You find a copy of the GNU General Public License in the main
* BeAR directory under <LICENSE>. If not, see
* <http://www.gnu.org/licenses/>.
*/


#ifndef _radiative_transfer_h
#define _radiative_transfer_h

#include <vector>

#include "../atmosphere/atmosphere.h"
#include "../transport_coeff/opacity_calc.h"
#include "../spectral_grid/spectral_grid.h"
#include "../additional/quadrature.h"


namespace ngam {


struct RadiativeTransferOutput {
  RadiativeTransferOutput(SpectralGrid* spectral_grid_, size_t nb_grid_points)
    : spectral_grid(spectral_grid_)
  {
    const size_t nb_spectral_points = spectral_grid->nbSpectralPoints();

    spectrum.assign(nb_spectral_points, 0.0);

    flux_total.assign(nb_grid_points, 0.0);
    flux_up_total.assign(nb_grid_points, 0.0);
    flux_down_total.assign(nb_grid_points, 0.0);
    mean_intensity_total.assign(nb_grid_points, 0.0);
    flux_divergence.assign(nb_grid_points, 0.0);

    flux.assign(nb_grid_points, std::vector<double>(nb_spectral_points, 0.0));
    flux_up.assign(nb_grid_points, std::vector<double>(nb_spectral_points, 0.0));
    flux_down.assign(nb_grid_points, std::vector<double>(nb_spectral_points, 0.0));
    mean_intensity.assign(nb_grid_points, std::vector<double>(nb_spectral_points, 0.0));
  }

  void integrateQuantities()
  {
    const size_t n = flux_total.size();

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i)
    {
      flux_up_total[i] = aux::quadratureTrapezoidal(
        spectral_grid->wavenumber_list, flux_up[i]);

      flux_down_total[i] = aux::quadratureTrapezoidal(
        spectral_grid->wavenumber_list, flux_down[i]);

      flux_total[i] = flux_up_total[i] - flux_down_total[i];

      mean_intensity_total[i] = aux::quadratureTrapezoidal(
        spectral_grid->wavenumber_list, mean_intensity[i]);
    }
  }

  void calcFluxDivergence(const std::vector<double>& pressure)
  {
    const size_t n = flux_total.size();

    if (n < 2) return;

    // forward difference at the bottom boundary
    flux_divergence[0] = (flux_total[1] - flux_total[0])
                        / (pressure[1] - pressure[0]);

    // centered differences for interior points
    for (size_t i = 1; i < n - 1; ++i)
    {
      flux_divergence[i] = (flux_total[i+1] - flux_total[i-1])
                          / (pressure[i+1] - pressure[i-1]);
    }

    // backward difference at the top boundary
    flux_divergence[n-1] = (flux_total[n-1] - flux_total[n-2])
                          / (pressure[n-1] - pressure[n-2]);
  }

  SpectralGrid* spectral_grid = nullptr;

  std::vector<double> spectrum;

  std::vector<double> flux_total;
  std::vector<double> flux_up_total;
  std::vector<double> flux_down_total;
  std::vector<double> mean_intensity_total;
  std::vector<double> flux_divergence;

  std::vector< std::vector<double> > flux;
  std::vector< std::vector<double> > flux_up;
  std::vector< std::vector<double> > flux_down;
  std::vector< std::vector<double> > mean_intensity;
};



class RadiativeTransfer{
  public:
    RadiativeTransfer(SpectralGrid* spectral_grid_) {
      spectral_grid = spectral_grid_;};
    virtual ~RadiativeTransfer() {}
    virtual void calculate(
      const Atmosphere& atmosphere,
      const OpacityCalculation& opacity,
      RadiativeTransferOutput& output) = 0;
    
    //change units of high-res spectrum from cm to micron^-1
    void changeSpectrumUnits(std::vector<double>& spectrum) {
      for (size_t i=0; i<spectrum.size(); ++i)
        spectrum[i] = spectrum[i]/spectral_grid->wavelength_list[i]/spectral_grid->wavelength_list[i]*10000.0;};
  
  protected:
    SpectralGrid* spectral_grid = nullptr;
};


}
#endif

