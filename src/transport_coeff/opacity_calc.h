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


#ifndef OPACITY_CALC_H
#define OPACITY_CALC_H


#include <vector>

#include "transport_coeff.h"
#include "../config/global_config.h"
#include "../spectral_grid/spectral_grid.h"
#include "../atmosphere/atmosphere.h"



namespace ngam{


class OpacityCalculation {
  public:
    OpacityCalculation(
      GlobalConfig* config_, 
      SpectralGrid* spectral_grid_,
      Atmosphere* atmosphere_,
      const std::vector<std::string> opacity_species_symbol,
      const std::vector<std::string> opacity_species_folder,
      const bool use_cloud_)
      : spectral_grid(spectral_grid_)
      , atmosphere(atmosphere_)
      , transport_coeff(
          config_, 
          spectral_grid_, 
          opacity_species_symbol, 
          opacity_species_folder)
      , use_cloud(use_cloud_)
      {}
    ~OpacityCalculation();

    void calculate();
      //std::vector<CloudModel*>& cloud_models,
      //const std::vector<double>& cloud_parameter);

    std::vector< std::vector<double> > absorption_coeff;
    std::vector< std::vector<double> > scattering_coeff;

    std::vector< std::vector<double> > cloud_optical_depths;
    std::vector< std::vector<double> > cloud_single_scattering;
    std::vector< std::vector<double> > cloud_asym_param;

  private:
    SpectralGrid* spectral_grid;
    Atmosphere* atmosphere;
    TransportCoefficients transport_coeff;

    bool use_cloud = false;
};



inline void OpacityCalculation::calculate()
  //std::vector<CloudModel*>& cloud_models,
  //const std::vector<double>& cloud_parameter)
{ 
  const size_t nb_grid_points = atmosphere->nb_grid_points;
  const size_t nb_spectral_points = spectral_grid->nbSpectralPoints();

  //calculate gas transport coefficients
  absorption_coeff.assign(nb_spectral_points, std::vector<double>(nb_grid_points, 0.0));
  scattering_coeff.assign(nb_spectral_points, std::vector<double>(nb_grid_points, 0.0));

  for (size_t i=0; i<nb_grid_points; ++i)
  {
    std::vector<double> absorption_coeff_level(nb_spectral_points, 0.0);
    std::vector<double> scattering_coeff_level(nb_spectral_points, 0.0);

    transport_coeff.calculate(
      atmosphere->temperature[i],
      atmosphere->pressure[i], 
      atmosphere->number_densities[i], 
      absorption_coeff_level,
      scattering_coeff_level);

    for (size_t j=0; j<nb_spectral_points; ++j)
    {
      absorption_coeff[j][i] = absorption_coeff_level[j];
      scattering_coeff[j][i] = scattering_coeff_level[j];
    }
  }

  cloud_optical_depths.assign(nb_spectral_points, std::vector<double>(nb_grid_points-1, 0.0));
  cloud_single_scattering.assign(nb_spectral_points, std::vector<double>(nb_grid_points-1, 0.0));
  cloud_asym_param.assign(nb_spectral_points, std::vector<double>(nb_grid_points-1, 0.0));


  // if (use_cloud)
  // { 
  //   size_t nb_param = 0;

  //   for (auto & i : cloud_models)
  //   {
  //     std::vector<double> parameter(
  //       cloud_parameter.begin() + nb_param, 
  //       cloud_parameter.begin() + nb_param + i->nbParameters());

  //     nb_param += i->nbParameters();

  //     std::vector<std::vector<double>> optical_depths;
  //     std::vector<std::vector<double>> single_scattering;
  //     std::vector<std::vector<double>> asym_param;

  //     i->opticalProperties(
  //       parameter, 
  //       *atmosphere, 
  //       spectral_grid, 
  //       optical_depths, 
  //       single_scattering, 
  //       asym_param);

  //     for (size_t j=0; j<nb_spectral_points; ++j)
  //       for (size_t k=0; k<nb_grid_points-1; ++k)
  //       {
  //         const double tau_scattering1 = cloud_optical_depths[j][k] * cloud_single_scattering[j][k];
  //         const double tau_scattering2 = optical_depths[j][k] * cloud_single_scattering[j][k];
  //         const double tau_scattering_mix = tau_scattering1 + tau_scattering2;
          
  //         const double optical_depth_mix = cloud_optical_depths[j][k] + optical_depths[j][k];
          
  //         double single_scattering_mix = 0;
          
  //         if (optical_depth_mix > 0)
  //           single_scattering_mix = tau_scattering_mix / optical_depth_mix;

  //         const double mix_asym_param = tau_scattering1/tau_scattering_mix * cloud_asym_param[j][k]
  //                                     + tau_scattering2/tau_scattering_mix * asym_param[j][k];

  //         cloud_optical_depths[j][k] = optical_depth_mix;
  //         cloud_single_scattering[j][k] = single_scattering_mix;
  //         cloud_asym_param[j][k] = mix_asym_param;
  //       }
  //   }
  //}


}



inline OpacityCalculation::~OpacityCalculation()
{
 
}


}

#endif
