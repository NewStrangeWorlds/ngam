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
#include <memory>
#include <omp.h>

#include "discrete_ordinate.h"

#include "../atmosphere/atmosphere.h"
#include "../additional/aux_functions.h"
#include "../additional/physical_const.h"
#include "../additional/quadrature.h"
#include "../spectral_grid/spectral_grid.h"
#include "../additional/exceptions.h"
#include "../additional/quadrature.h"

#include "../../_deps/disortpp-src/src/DisortFluxConfig.hpp"
#include "../../_deps/disortpp-src/src/FluxResult.hpp"
#include "../../_deps/disortpp-src/src/FluxSolver.hpp"

namespace ngam{


DiscreteOrdinates::DiscreteOrdinates(
  SpectralGrid* spectral_grid_ptr,
  const size_t nb_streams,
  const size_t nb_grid_points)
   : RadiativeTransfer(spectral_grid_ptr),
      nb_streams(nb_streams),
      nb_grid_points(nb_grid_points),
      nb_layers(nb_grid_points - 1)
{ 
  nb_threads = omp_get_max_threads();

  initDISORT();
}



void DiscreteOrdinates::calculate(
  const Atmosphere& atmosphere,
  const OpacityCalculation& opacity,
  RadiativeTransferOutput& output)
{ 
  for (size_t i=0; i<configs.size(); ++i)
  {
    setTemperatureStructure(atmosphere.temperature, atmosphere.temperature[0]);
  }

  const double max_temperature = *std::max_element(
    atmosphere.temperature.begin(), 
    atmosphere.temperature.end());


  double surface_albedo = 0;
  double incident_radiation = 0;
  double zenith_angle = 0.5;
  
  #pragma omp parallel for
  for (size_t i=0; i<output.spectrum.size(); ++i)
    calculate(
      opacity, 
      atmosphere.altitude, 
      surface_albedo,
      incident_radiation,
      zenith_angle,
      i,
      max_temperature,
      output);

  output.integrateQuantities();
}



void DiscreteOrdinates::calcTotalTransportCoeff(
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



void DiscreteOrdinates::calculate(
  const OpacityCalculation& opacity,
  const std::vector<double>& vertical_grid,
  const double surface_albedo,
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
  
  setDISORTParam(
    thread_id,
    wavenumber, 
    optical_depth, 
    single_scattering_albedo, 
    asymmetry_parameter, 
    incident_radiation,
    zenith_angle,
    surface_albedo);
  if (aux::planckFunctionWavenumber(max_temperature, wavenumber) > 1.e-35)
    configs[thread_id].use_thermal_emission = true;
  else
    configs[thread_id].use_thermal_emission = false;
  
  auto results = solvers[thread_id](configs[thread_id]);

  // DisORT returns fluxes in W/m²; convert to CGS (erg/cm²/s): 1 W/m² = 1e3 erg/cm²/s
  constexpr double si_to_cgs = 1e3;

  for (size_t j=0; j<nb_grid_points; ++j)
  {
    const size_t j_rev = nb_grid_points - j - 1;

    output.flux_up[j][nu_index]   = results.flux_up[j_rev] * si_to_cgs;
    output.flux_down[j][nu_index] = (results.flux_down[j_rev]
      + results.flux_direct_beam[j_rev]) * si_to_cgs;

    output.flux[j][nu_index] = output.flux_up[j][nu_index] - output.flux_down[j][nu_index];
    output.mean_intensity[j][nu_index] = results.mean_intensity[j_rev] * si_to_cgs;
  }

  output.spectrum[nu_index] = output.flux_up.back()[nu_index];
}



void DiscreteOrdinates::setTemperatureStructure(
  const std::vector<double>& temperature_structure,
  const double& surface_temperature)
{
  std::vector<double> temp_structure_reversed(
    temperature_structure.rbegin(), 
    temperature_structure.rend());
  
  for (size_t j=0; j<configs.size(); ++j)
  {
    configs[j].temperature_bottom = surface_temperature;
  
    for (size_t i=0; i<nb_grid_points; i++)
      configs[j].temperature[i] = temp_structure_reversed[i];
  }

}



void DiscreteOrdinates::setDISORTParam(
  const int thread_id,
  const double wavenumber_input,
  const std::vector<double>& optical_depth,
  const std::vector<double>& single_scattering_albedo,
  const std::vector<double>& asymmetry_parameter,
  const double incident_radiation,
  const double zenith_angle,
  const double surface_albedo)
{ 
  configs[thread_id].surface_albedo  = surface_albedo;
  configs[thread_id].direct_beam_flux = incident_radiation;
  configs[thread_id].direct_beam_mu = zenith_angle;

  configs[thread_id].wavenumber_low = wavenumber_input;
  configs[thread_id].wavenumber_high = wavenumber_input;

  for (int lc = 0; lc < nb_layers; lc++)
  {
    configs[thread_id].delta_tau[lc] = optical_depth[nb_layers - lc - 1];
    configs[thread_id].single_scat_albedo[lc] = single_scattering_albedo[nb_layers - lc - 1];
  }

  for (int lc = 0; lc < nb_layers; lc++)
  {
    double gg = asymmetry_parameter[nb_layers - lc - 1];

    if (gg > 0)
      configs[thread_id].setHenyeyGreenstein(gg, lc);
    else
      configs[thread_id].setRayleigh(lc);
  }

}


template<int NStr>
static void assignFluxSolvers(
  std::vector<std::function<disortpp::FluxResult(const disortpp::DisortFluxConfig&)>>& solvers,
  int nb_threads)
{
  solvers.resize(nb_threads);

  for (auto& s : solvers)
  {
    auto solver = std::make_shared<disortpp::DisortFluxSolver<NStr>>();
    s = [solver](const disortpp::DisortFluxConfig& config) {
      return solver->solve(config);
    };
  }
}


void DiscreteOrdinates::initDISORT()
{
  disortpp::DisortFluxConfig config(nb_grid_points-1, nb_streams, nb_streams);

  config.use_thermal_emission = true;
  config.use_diffusion_lower_bc = true;

  configs.assign(nb_threads, config);

  for (auto & c : configs)
  {
    c.allocate();
  }

  switch(nb_streams) {
    case 4:  assignFluxSolvers<4>(solvers, nb_threads);  break;
    case 6:  assignFluxSolvers<6>(solvers, nb_threads);  break;
    case 8:  assignFluxSolvers<8>(solvers, nb_threads);  break;
    case 10: assignFluxSolvers<10>(solvers, nb_threads); break;
    case 12: assignFluxSolvers<12>(solvers, nb_threads); break;
    case 14: assignFluxSolvers<14>(solvers, nb_threads); break;
    case 16: assignFluxSolvers<16>(solvers, nb_threads); break;
    case 32: assignFluxSolvers<32>(solvers, nb_threads); break;
    case 64: assignFluxSolvers<64>(solvers, nb_threads); break;
    default:
      throw InvalidInput(
        "DiscreteOrdinates",
        "Unsupported number of streams: " + std::to_string(nb_streams) +
        ". DisortFluxSolver supports: 4, 6, 8, 10, 12, 14, 16, 32, 64.");
  }
}


}
