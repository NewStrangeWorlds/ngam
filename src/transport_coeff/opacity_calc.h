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
#include <string>
#include <cstdlib>

#include "transport_coeff.h"
#include "../spectral_grid/spectral_grid.h"
#include "../atmosphere/atmosphere.h"
#include "../additional/quadrature.h"



namespace ngam{


class OpacityCalculation {
  public:
    OpacityCalculation(
      const std::string& cross_section_file_path,
      SpectralGrid* spectral_grid_,
      Atmosphere* atmosphere_,
      const std::vector<std::string>& opacity_species_symbol,
      const std::vector<std::string>& opacity_species_folder,
      const bool use_cloud_)
      : spectral_grid(spectral_grid_)
      , atmosphere(atmosphere_)
      , transport_coeff(
          cross_section_file_path,
          spectral_grid_,
          opacity_species_symbol,
          opacity_species_folder)
      , use_cloud(use_cloud_)
      {}
    void calculate();

    std::vector< std::vector<double> > absorption_coeff;
    std::vector< std::vector<double> > scattering_coeff;

    //band-closure correction for the local energy balance (clima_rce_correction):
    //band_correction[b][level] = sum over line species of n_s*(int_band sigma dnu - sampled
    //representation), i.e. the absorption the sampled grid misses (can be negative where the
    //sampling overcounts). Empty when disabled (CLIMA_BANDCORR=0). Units: cm^-1 * cm^-1(nu).
    std::vector< std::vector<double> > band_correction;
    //peak absorption coefficient per band per level [cm^-1] (max over line species): the core
    //optical depth to space built from this decides the escape-probability weight of the closure
    std::vector< std::vector<double> > band_peak_coeff;
    std::vector<double> band_wavenumber;   //band centres [cm^-1]

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

  //band-closure correction setup (DEFAULT ON since the v3 escape-probability weighting;
  //kill switch CLIMA_BANDCORR=0): trapezoid weights of the sampled list and the correction
  //band of each point, built once per grid
  static const bool bandcorr_enabled = [] {
    const char* e = std::getenv("CLIMA_BANDCORR");
    return !(e != nullptr && std::string(e) == "0"); }();

  const size_t nb_bands = bandcorr_enabled ? spectral_grid->nbCorrectionBands() : 0;
  BandCorrectionSpec band_spec;
  std::vector<double> point_weights;
  std::vector<int> band_of_point;

  if (nb_bands > 0)
  {
    point_weights = aux::trapezoidalWeights(spectral_grid->wavenumber_list);
    band_of_point.assign(nb_spectral_points, -1);
    for (size_t k=0; k<nb_spectral_points; ++k)
      band_of_point[k] = static_cast<int>(
        spectral_grid->wavenumber_list[k]/SpectralGrid::correction_band_width);

    band_spec.point_weights = &point_weights;
    band_spec.band_of_point = &band_of_point;
    band_spec.nb_bands = nb_bands;

    band_correction.assign(nb_bands, std::vector<double>(nb_grid_points, 0.0));
    band_peak_coeff.assign(nb_bands, std::vector<double>(nb_grid_points, 0.0));
    band_wavenumber.assign(nb_bands, 0.0);
    for (size_t b=0; b<nb_bands; ++b)
      band_wavenumber[b] = (b + 0.5)*SpectralGrid::correction_band_width;
  }
  else
  {
    band_correction.clear();
    band_peak_coeff.clear();
    band_wavenumber.clear();
  }

  //calculate gas transport coefficients
  absorption_coeff.assign(nb_spectral_points, std::vector<double>(nb_grid_points, 0.0));
  scattering_coeff.assign(nb_spectral_points, std::vector<double>(nb_grid_points, 0.0));

  for (size_t i=0; i<nb_grid_points; ++i)
  {
    std::vector<double> absorption_coeff_level(nb_spectral_points, 0.0);
    std::vector<double> scattering_coeff_level(nb_spectral_points, 0.0);
    std::vector<double> band_correction_level;
    std::vector<double> band_peak_level;

    transport_coeff.calculate(
      atmosphere->temperature[i],
      atmosphere->pressure[i],
      atmosphere->number_densities[i],
      absorption_coeff_level,
      scattering_coeff_level,
      nb_bands > 0 ? &band_spec : nullptr,
      nb_bands > 0 ? &band_correction_level : nullptr,
      nb_bands > 0 ? &band_peak_level : nullptr);

    for (size_t j=0; j<nb_spectral_points; ++j)
    {
      absorption_coeff[j][i] = absorption_coeff_level[j];
      scattering_coeff[j][i] = scattering_coeff_level[j];
    }

    for (size_t b=0; b<nb_bands; ++b)
    {
      band_correction[b][i] = band_correction_level[b];
      band_peak_coeff[b][i] = band_peak_level[b];
    }
  }

  cloud_optical_depths.assign(nb_spectral_points, std::vector<double>(nb_grid_points-1, 0.0));
  cloud_single_scattering.assign(nb_spectral_points, std::vector<double>(nb_grid_points-1, 0.0));
  cloud_asym_param.assign(nb_spectral_points, std::vector<double>(nb_grid_points-1, 0.0));
}


}

#endif
