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


#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

#include "adding_doubling.h"

#include "../atmosphere/atmosphere.h"
#include "../additional/aux_functions.h"
#include "../additional/physical_const.h"
#include "../additional/quadrature.h"
#include "../spectral_grid/spectral_grid.h"
#include "../additional/exceptions.h"

namespace ngam{


AddingDoubling::AddingDoubling(
  SpectralGrid* spectral_grid_ptr,
  const size_t num_quadrature,
  const size_t nb_grid_points)
   : RadiativeTransfer(spectral_grid_ptr),
      num_quadrature(num_quadrature),
      nb_grid_points(nb_grid_points),
      nb_layers(nb_grid_points - 1)
{
  if (num_quadrature < 2)
    throw InvalidInput(
      "AddingDoubling",
      "Number of quadrature points must be >= 2, got "
      + std::to_string(num_quadrature) + ".");

  nb_threads = omp_get_max_threads();

  initSolver();
}



void AddingDoubling::calculate(
  const Atmosphere& atmosphere,
  const OpacityCalculation& opacity,
  RadiativeTransferOutput& output,
  const RadiativeBoundaryConditions& bc)
{
  surface_temperature = (bc.surface_temperature > 0)
    ? bc.surface_temperature : atmosphere.temperature[0];

  // The caller passes bc.surface_temperature < 0 (folded sentinel) for a BOA-tied surface: it then has
  // NO independent DOF and emits at the bottom-level temperature, folding into the bottom-level Jacobian
  // column. A positive value is a genuinely independent surface temperature (its own DOF). This is a
  // FIXED decision (sign of bc), so it cannot flip under a temperature perturbation in a finite diff.
  surface_temperature_config = (bc.surface_temperature < 0.0) ? -1.0 : surface_temperature;

  setTemperatureStructure(atmosphere.temperature);

  const double max_temperature = std::max(
    surface_temperature,
    *std::max_element(atmosphere.temperature.begin(), atmosphere.temperature.end()));

  const bool has_irradiation = !bc.incident_flux.empty();

  // CGS [erg/cm^2/s/cm^-1] -> SI [W/m^2/cm^-1]: divide by 1e3
  constexpr double cgs_to_si = 1e-3;

  // The per-layer net heating VALUE is always accumulated (cheap, and needed by collocated
  // residuals like the local-RE corrector on EVERY RT solve, not only when a Jacobian is asked
  // for). The expensive n*n heating/flux Jacobians are accumulated only when requested.
  quad_weights = aux::trapezoidalWeights(spectral_grid->wavenumber_list);
  heating_value_thread.assign(
    nb_threads, std::vector<double>(nb_grid_points, 0.0));
  if (output.compute_jacobian)
  {
    jac_net_heating_thread.assign(
      nb_threads,
      std::vector<std::vector<double>>(nb_grid_points, std::vector<double>(nb_grid_points, 0.0)));
    jac_net_flux_thread.assign(
      nb_threads,
      std::vector<std::vector<double>>(nb_grid_points, std::vector<double>(nb_grid_points, 0.0)));
    jac_meanint_kappa_thread.assign(
      nb_threads,
      std::vector<std::vector<double>>(nb_grid_points, std::vector<double>(nb_grid_points, 0.0)));
  }

  #pragma omp parallel for
  for (size_t i=0; i<output.spectrum.size(); ++i)
  {
    const double incident = has_irradiation
      ? bc.incident_flux[i] * cgs_to_si : 0.0;

    const double albedo_i = bc.surface_albedo.empty() ? 0.0 : bc.surface_albedo[i];

    calculate(
      opacity,
      atmosphere.altitude,
      albedo_i,
      bc.has_surface,
      incident,
      bc.zenith_angle,
      i,
      max_temperature,
      output);
  }

  output.integrateQuantities();

  // deterministic reduction of the per-thread Jacobian accumulators (summed in
  // ascending thread order for run-to-run reproducibility)
  // always reduce the net-heating VALUE
  for (size_t i=0; i<nb_grid_points; ++i) output.net_heating[i] = 0.0;
  for (int t=0; t<nb_threads; ++t)
    for (size_t i=0; i<nb_grid_points; ++i)
      output.net_heating[i] += heating_value_thread[t][i];

  if (output.compute_jacobian)
  {
    for (size_t i=0; i<nb_grid_points; ++i)
      for (size_t j=0; j<nb_grid_points; ++j)
      {
        output.net_heating_jacobian[i][j] = 0.0;
        output.net_flux_jacobian[i][j] = 0.0;
        output.meanint_kappa_jacobian[i][j] = 0.0;
      }

    for (int t=0; t<nb_threads; ++t)
      for (size_t i=0; i<nb_grid_points; ++i)
        for (size_t j=0; j<nb_grid_points; ++j)
        {
          output.net_heating_jacobian[i][j]  += jac_net_heating_thread[t][i][j];
          output.net_flux_jacobian[i][j]     += jac_net_flux_thread[t][i][j];
          output.meanint_kappa_jacobian[i][j] += jac_meanint_kappa_thread[t][i][j];
        }
  }
}



void AddingDoubling::calcTotalTransportCoeff(
  const OpacityCalculation& opacity,
  const std::vector<double>& altitude,
  const size_t nu_index,
  std::vector<double>& optical_depth,
  std::vector<double>& single_scattering_albedo,
  std::vector<double>& asymmetry_parameter)
{

  for (size_t j=0; j<nb_layers; ++j)
  {
    const double gas_abs_coeff_1 = opacity.absorption_coeff[nu_index][j];
    const double gas_abs_coeff_2 = opacity.absorption_coeff[nu_index][j+1];

    const double gas_scat_coeff_1 = opacity.scattering_coeff[nu_index][j];
    const double gas_scat_coeff_2 = opacity.scattering_coeff[nu_index][j+1];

    const double gas_tau_abs = (altitude[j+1] - altitude[j])
      * (gas_abs_coeff_2 + gas_abs_coeff_1)/2.;
    const double gas_tau_scat = (altitude[j+1] - altitude[j])
      * (gas_scat_coeff_2 + gas_scat_coeff_1)/2.;

    const double cloud_tau_abs = opacity.cloud_optical_depths[nu_index][j]
      * (1.0 - opacity.cloud_single_scattering[nu_index][j]);

    const double cloud_tau_scat = opacity.cloud_optical_depths[nu_index][j]
      * opacity.cloud_single_scattering[nu_index][j];

    const double total_tau_abs = gas_tau_abs + cloud_tau_abs;

    const double total_tau_scat = gas_tau_scat + cloud_tau_scat;
    const double total_tau = total_tau_abs + total_tau_scat;

    optical_depth[j] = total_tau;

    if (total_tau > 0)
      single_scattering_albedo[j] = total_tau_scat / total_tau;
    else
      single_scattering_albedo[j] = 0.0;

    if (single_scattering_albedo[j] > 0.999999)
      single_scattering_albedo[j] = 0.999999;

    if (gas_tau_scat + cloud_tau_scat > 0)
      asymmetry_parameter[j] = cloud_tau_scat / (gas_tau_scat + cloud_tau_scat)
        * opacity.cloud_asym_param[nu_index][j];
    else
      asymmetry_parameter[j] = 0.0;
  }
}



void AddingDoubling::calculate(
  const OpacityCalculation& opacity,
  const std::vector<double>& vertical_grid,
  const double surface_albedo,
  const bool has_surface,
  const double incident_radiation,
  const double zenith_angle,
  const size_t nu_index,
  const double max_temperature,
  RadiativeTransferOutput& output)
{
  std::vector<double> optical_depth(nb_layers, 0.0);
  std::vector<double> single_scattering_albedo(nb_layers, 0.0);
  std::vector<double> asymmetry_parameter(nb_layers, 0.0);

  calcTotalTransportCoeff(
    opacity,
    vertical_grid,
    nu_index,
    optical_depth,
    single_scattering_albedo,
    asymmetry_parameter);

  const double wavenumber = spectral_grid->wavenumber_list[nu_index];
  const int thread_id = omp_get_thread_num();

  setSolverParam(
    thread_id,
    wavenumber,
    optical_depth,
    single_scattering_albedo,
    asymmetry_parameter,
    incident_radiation,
    zenith_angle,
    surface_albedo,
    has_surface);

  const bool thermal_on =
    aux::planckFunctionWavenumber(max_temperature, wavenumber) > 1.e-35;

  configs[thread_id].use_thermal_emission = thermal_on;

  // a point with no thermal emission anywhere in the column has zero incoming flux at a
  // diffusion (self-luminous) lower boundary as well -- disable the BC instead of tripping
  // the solver's consistency check (which, inside the OpenMP region, is a process abort)
  if (!thermal_on)
    configs[thread_id].use_diffusion_lower_bc = false;

  // an explicit (Lambertian) surface emits at surface_temperature; otherwise the
  // bottom level temperature is used (surface_temperature < 0). When thermal
  // emission is off the surface temperature must stay < 0 (ADConfig::validate).
  configs[thread_id].surface_temperature =
    (has_surface && thermal_on) ? surface_temperature_config : -1.0;

  // no downwelling thermal radiation at the top boundary (atmosphere open to cold
  // space), matching DISORT: top_temperature = 0 -> B(0) = 0. Must stay < 0 when
  // thermal emission is off (ADConfig::validate).
  configs[thread_id].top_temperature = thermal_on ? 0.0 : -1.0;

  configs[thread_id].compute_temperature_jacobian = output.compute_jacobian;
  configs[thread_id].compute_flux_components = true;   // split the net flux into thermal + stellar parts

  auto results = adrt::solve(configs[thread_id], workspaces[thread_id]);

  // adrt returns fluxes in W/m²; convert to CGS (erg/cm²/s): 1 W/m² = 1e3 erg/cm²/s
  constexpr double si_to_cgs = 1e3;

  for (size_t j=0; j<nb_grid_points; ++j)
  {
    const size_t j_rev = nb_grid_points - j - 1;

    output.flux_up[j][nu_index]   = results.flux_up[j_rev] * si_to_cgs;
    output.flux_down[j][nu_index] = (results.flux_down[j_rev]
      + results.flux_direct[j_rev]) * si_to_cgs;

    output.flux[j][nu_index] = output.flux_up[j][nu_index] - output.flux_down[j][nu_index];
    output.mean_intensity[j][nu_index] = results.mean_intensity[j_rev] * si_to_cgs;

    if (!results.net_flux_thermal.empty())                            // thermal (IR) net flux per level
      output.flux_net_thermal[j][nu_index] = results.net_flux_thermal[j_rev] * si_to_cgs;
  }

  // accumulate the frequency-integrated net heating and its temperature Jacobian for
  // this wavenumber (only when thermal emission is active, so the arrays are filled).
  if (thermal_on)
  {
    const double w_nu = quad_weights[nu_index];

    // net-heating VALUE: always (a collocated residual needs it on every solve)
    auto& hv = heating_value_thread[thread_id];
    for (size_t i=0; i<nb_grid_points; ++i)
      hv[i] += w_nu * results.flux_divergence[nb_grid_points - i - 1] * si_to_cgs;

    // heating/flux JACOBIANS: only when requested (the expensive n*n part)
    if (output.compute_jacobian && !results.flux_divergence_temperature_jac.empty())
    {
      // adrt temperature DOF columns: 0..nb_layers are levels (top->bottom),
      // column nb_layers+1 (= nb_grid_points) is the surface temperature, which
      // equals atmosphere.temperature[0], so it folds onto ngam j = 0.
      const size_t nb_dof = nb_grid_points + 1;
      auto& nh = jac_net_heating_thread[thread_id];
      auto& nf = jac_net_flux_thread[thread_id];
      auto& mk = jac_meanint_kappa_thread[thread_id];

      for (size_t i=0; i<nb_grid_points; ++i)
      {
        const size_t l_d = nb_grid_points - i - 1;
        const auto& fdiv_row = results.flux_divergence_temperature_jac[l_d];
        const auto& fu_row   = results.flux_up_temperature_jac[l_d];
        const auto& fd_row   = results.flux_down_temperature_jac[l_d];
        const auto& mij_row  = results.mean_intensity_temperature_jac[l_d];
        // kappa weight at ngam grid point i (same absorption_coeff the corrector uses
        // to build num/den), so kappa cancels in the ratio-residual diagonal.
        const double kappa_i = opacity.absorption_coeff[nu_index][i];

        for (size_t c=0; c<nb_dof; ++c)
        {
          const size_t j = (c < nb_grid_points) ? (nb_grid_points - c - 1) : 0;
          nh[i][j] += w_nu * fdiv_row[c] * si_to_cgs;
          nf[i][j] += w_nu * (fu_row[c] - fd_row[c]) * si_to_cgs;
          mk[i][j] += w_nu * kappa_i * mij_row[c] * si_to_cgs;
        }
      }
    }
  }

  //convert from W m-2 cm to W m-2 micron-1
  output.spectrum[nu_index] = output.flux_up.back()[nu_index]
    /spectral_grid->wavelength_list[nu_index]/spectral_grid->wavelength_list[nu_index]*10000.0;
}



void AddingDoubling::setTemperatureStructure(
  const std::vector<double>& temperature_structure)
{
  std::vector<double> temp_structure_reversed(
    temperature_structure.rbegin(),
    temperature_structure.rend());

  for (size_t j=0; j<configs.size(); ++j)
    for (size_t i=0; i<nb_grid_points; i++)
      configs[j].temperature[i] = temp_structure_reversed[i];
}



void AddingDoubling::setSolverParam(
  const int thread_id,
  const double wavenumber_input,
  const std::vector<double>& optical_depth,
  const std::vector<double>& single_scattering_albedo,
  const std::vector<double>& asymmetry_parameter,
  const double incident_radiation,
  const double zenith_angle,
  const double surface_albedo,
  const bool has_surface)
{
  configs[thread_id].surface_albedo = surface_albedo;
  configs[thread_id].solar_flux = incident_radiation;
  configs[thread_id].solar_mu = zenith_angle;

  configs[thread_id].wavenumber_low = wavenumber_input;
  configs[thread_id].wavenumber_high = wavenumber_input;

  if (has_surface)
    configs[thread_id].use_diffusion_lower_bc = false;
  else
    configs[thread_id].use_diffusion_lower_bc = true;

  for (size_t lc = 0; lc < nb_layers; lc++)
  {
    configs[thread_id].delta_tau[lc] = optical_depth[nb_layers - lc - 1];
    configs[thread_id].single_scat_albedo[lc] = single_scattering_albedo[nb_layers - lc - 1];
  }

  for (size_t lc = 0; lc < nb_layers; lc++)
  {
    double gg = asymmetry_parameter[nb_layers - lc - 1];

    if (gg > 0)
      configs[thread_id].setHenyeyGreenstein(gg, lc);
    else
      configs[thread_id].setRayleigh(lc);
  }
}



void AddingDoubling::initSolver()
{
  adrt::ADConfig config(nb_grid_points-1, num_quadrature);

  config.use_thermal_emission = true;
  config.allocate();

  configs.assign(nb_threads, config);
  workspaces.resize(nb_threads);
}


}
