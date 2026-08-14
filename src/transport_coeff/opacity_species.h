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


#ifndef OPACITY_SPECIES_H
#define OPACITY_SPECIES_H


#include <vector>
#include <string>
#include <iostream>

#include "../chemistry/chem_species.h"
#include "../spectral_grid/spectral_grid.h"
#include "sampled_data.h"


namespace ngam{


//inputs for the band-closure correction of the local energy balance (clima_rce_correction):
//trapezoid weights of the sampled point list and the correction band each point falls in.
//The correction per band is n_s * (int_band sigma dnu - sum_{k in band} w_k sigma_k): the
//native line opacity the sampled representation misses (or overcounts).
struct BandCorrectionSpec {
  const std::vector<double>* point_weights = nullptr;
  const std::vector<int>* band_of_point = nullptr;
  size_t nb_bands = 0;
};


class OpacitySpecies {
  public:
    OpacitySpecies(
      const unsigned int index,
      const std::string name,
      const std::string folder) 
        : species_index(index), species_name(name), species_folder(folder) 
        {}
    virtual ~OpacitySpecies() {}
   
    bool dataAvailable() {
      if (rayleigh_available || continuum_available)
        return true;

      return cross_section_available;}

    virtual void calcTransportCoefficients(
      const double temperature,
      const double pressure,
      const std::vector<double>& number_densities,
      std::vector<double>& absorption_coeff,
      std::vector<double>& scattering_coeff,
      const BandCorrectionSpec* band_spec = nullptr,
      std::vector<double>* band_correction = nullptr,
      std::vector<double>* band_peak_coeff = nullptr);
    
    const size_t species_index = 0;
    const std::string species_name = "";
    const std::string species_folder = "";
  protected:
    std::string cross_section_file_path;
    SpectralGrid* spectral_grid;

    double species_mass = 0;
    
    size_t pressure_reference_species = _TOTAL;
    std::vector<size_t> cia_collision_partner;

    bool cross_section_available = false;
    bool rayleigh_available = false;
    bool continuum_available = false;

    std::vector<SampledData> sampled_cross_sections;
    std::vector<std::vector<SampledData*>> ordered_data_list;
    std::vector<double> rayleigh_cross_sections;

    void init();
    void orderDataList();

    virtual bool calcContinuumAbsorption(
      const double temperature,
      const std::vector<double>& number_densities,
      std::vector<double>& absorption_coeff) {
        return false;};
    
    virtual bool calcRayleighCrossSections(
      std::vector<double>& cross_sections) {
      return false;};
    
    void readFileList(const std::string file_path);
    
    std::vector<SampledData*> findClosestDataPoints(
      const double sampling_pressure,
      const double sampling_temperature);
    void checkDataAvailability(std::vector<SampledData*>& data_points);

    void calcAbsorptionCrossSections(
      const double local_pressure,
      const double local_temperature,
      std::vector<double>& cross_sections);
    bool interpolateBandData(
      const double local_pressure,
      const double local_temperature,
      std::vector<double> SampledData::* member,
      std::vector<double>& band_data);
    bool calcAbsorptionBandIntegrals(
      const double local_pressure,
      const double local_temperature,
      std::vector<double>& band_integrals);
    bool calcAbsorptionBandPeaks(
      const double local_pressure,
      const double local_temperature,
      std::vector<double>& band_peaks);
    bool calcScatteringCrossSections(std::vector<double>& cross_sections);
    
    double generalRayleighCrossSection(
      double reference_density,
      double refractive_index,
      double king_correction_factor,
      double wavenumber);
};


}

#endif
