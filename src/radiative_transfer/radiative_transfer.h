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

    flux_net_thermal_total.assign(nb_grid_points, 0.0);
    flux_net_thermal.assign(nb_grid_points, std::vector<double>(nb_spectral_points, 0.0));

    net_heating.assign(nb_grid_points, 0.0);
    net_heating_jacobian.assign(nb_grid_points, std::vector<double>(nb_grid_points, 0.0));
    net_flux_jacobian.assign(nb_grid_points, std::vector<double>(nb_grid_points, 0.0));
    meanint_kappa_jacobian.assign(nb_grid_points, std::vector<double>(nb_grid_points, 0.0));
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

      flux_net_thermal_total[i] = aux::quadratureTrapezoidal(
        spectral_grid->wavenumber_list, flux_net_thermal[i]);
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

  // When true, the radiative-transfer backend also fills the frequency-integrated
  // temperature Jacobians below (only the DISORT backend supports this). Set by the
  // linearised temperature correction before calling RadiativeTransfer::calculate.
  bool compute_jacobian = false;

  // Frequency-integrated net radiative heating and temperature Jacobians, ngam grid
  // indexing ([i] / [i][j], 0 = bottom). DISORT's own flux divergence is used (rather than
  // a kappa*(J-B) reconstruction) so that net_heating = 0 is exactly equivalent to a
  // constant DISORT net flux -- avoiding the J/flux stencil mismatch.
  //   net_heating[i]            = sum_nu w_nu * dF_net/dtau(nu,i)        (heating rate, cgs)
  //   net_heating_jacobian[i][j]= sum_nu w_nu * d(dF_net/dtau)(nu,i)/dT_j
  //   net_flux_jacobian[i][j]   = sum_nu w_nu * d(F_up - F_down)(nu,i)/dT_j  (flux anchor)
  // The derivative w.r.t. the deep/surface boundary temperature is folded into j = 0
  // (the bottom grid point), since temperature_bottom tracks atmosphere.temperature[0].
  //
  // kappa-weighted mean-intensity Jacobian for the ratio-form local-RE residual
  // g_i = (sum_nu w_nu kappa_nu,i J_nu,i)/(sum_nu w_nu kappa_nu,i B_nu,i) - 1. Its
  // numerator derivative (opacity frozen) is exactly this contraction:
  //   meanint_kappa_jacobian[i][j] = sum_nu w_nu * kappa_nu,i * d(mean_intensity)(nu,i)/dT_j
  // (cgs). The kappa weighting matches the residual's numerator/denominator so that
  // kappa cancels in the diagonal (the property that keeps the ratio Newton conditioned
  // as kappa->0). The 1/den_i row scaling is applied by the corrector, not here.
  std::vector<double> net_heating;
  std::vector< std::vector<double> > net_heating_jacobian;
  std::vector< std::vector<double> > net_flux_jacobian;
  std::vector< std::vector<double> > meanint_kappa_jacobian;

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

  // Thermal (longwave/IR, Planck-source-driven) NET upward flux per level. The adding-doubling backend
  // splits the net flux into thermal + stellar parts (net_flux_thermal + net_flux_stellar = flux_total);
  // the THERMAL part is the diffusion flux F = -(4/3 kappa) d(sigma T^4)/dz that carries the greenhouse /
  // internal flux through the optically-thick deep, and is the F_star the deep gradient integration needs
  // (the total net flux -> 0 for a terrestrial planet, so it cannot be used). Filled only by the adding-
  // doubling backend (empty / zero otherwise).
  std::vector<double> flux_net_thermal_total;
  std::vector< std::vector<double> > flux_net_thermal;
};



struct RadiativeBoundaryConditions {
  std::vector<double> incident_flux;   // per wavenumber [erg/cm^2/s/cm^-1], empty = none
  double zenith_angle = 0.5;           // cos(theta) of incident beam
  std::vector<double> surface_albedo;  // per wavenumber, empty = zero everywhere
  double surface_temperature = 0.0;    // K (0 = use atmosphere.temperature[0])
  bool has_surface = false;            // whether to include surface emission in the radiative transfer calculation
};


class RadiativeTransfer{
  public:
    RadiativeTransfer(SpectralGrid* spectral_grid_) {
      spectral_grid = spectral_grid_;};
    virtual ~RadiativeTransfer() {}
    virtual void calculate(
      const Atmosphere& atmosphere,
      const OpacityCalculation& opacity,
      RadiativeTransferOutput& output,
      const RadiativeBoundaryConditions& bc = RadiativeBoundaryConditions{}) = 0;
    
    //change units of high-res spectrum from cm to micron^-1
    void changeSpectrumUnits(std::vector<double>& spectrum) {
      for (size_t i=0; i<spectrum.size(); ++i)
        spectrum[i] = spectrum[i]/spectral_grid->wavelength_list[i]/spectral_grid->wavelength_list[i]*10000.0;};
  
  protected:
    SpectralGrid* spectral_grid = nullptr;
};


}
#endif

