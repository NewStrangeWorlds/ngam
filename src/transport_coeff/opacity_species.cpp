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

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <omp.h>
#include <sstream>
#include <algorithm> 
#include <assert.h>
#include <cmath>

#include "opacity_species.h"

#include "../additional/physical_const.h"
#include "../spectral_grid/spectral_grid.h"


namespace ngam{


void OpacitySpecies::init()
{
  std::string file_path = cross_section_file_path;
  
  //if we use the built-in Rayleigh scattering
  //or continuum absorption, we don't need to read any data
  if (rayleigh_available == true || continuum_available == true)
    return;

  readFileList(file_path);
  orderDataList();
}



void OpacitySpecies::readFileList(const std::string file_path)
{
  std::fstream file;
  std::string filelist_name = file_path+species_folder+"/filelist.dat";

  file.open(filelist_name.c_str(), std::ios::in);

  if (file.fail())
  {
    std::cout << "No line absorption data for species " 
      << species_name 
      << " found! Ignoring line absorption from here on.\n";
    cross_section_available = false;

    return;
  }
  else
    cross_section_available = true;


  std::string line;

  //the first line contains the species mass and information on log
  std::getline(file, line);

  std::istringstream input(line);
  std::string storage_string;

  input >> species_mass >> storage_string;
  
  bool log_storage = false;
  if (storage_string == "log") log_storage = true;


  sampled_cross_sections.reserve(10000);


  while (std::getline(file, line))
  {
    double temperature, pressure;
    std::string name;

    std::istringstream input(line);

    input >> pressure >> temperature >> name;

    std::string file_name = file_path+species_folder+"/"+name;

    sampled_cross_sections.push_back(
      SampledData (
        temperature, pressure, file_name, log_storage) );
  }

  file.close();

  sampled_cross_sections.shrink_to_fit();
}


//Orders the T-p points of the cross-section data into an ordered 2D array
//first index refers to the temperatures
//second index refers to the pressure
//both dimensions will be in ascending order
void OpacitySpecies::orderDataList()
{
  if (sampled_cross_sections.size() == 0) return;

  for (auto & i : sampled_cross_sections)
  {
    auto it = std::find_if(ordered_data_list.begin(), ordered_data_list.end(), 
                        [&](const std::vector<SampledData*>& a)
                        {return a[0]->temperature == i.temperature;});

    if (it == ordered_data_list.end()) 
      ordered_data_list.push_back(std::vector<SampledData*> {&i});
    else
      (*it).push_back(&i);
  }

  //first, we order the temperatures
  std::sort(ordered_data_list.begin(), ordered_data_list.end(), 
            [&](std::vector<SampledData*>& a, std::vector<SampledData*>& b)
            {return a[0]->temperature < b[0]->temperature;});


  //then for each temperatures, we order the pressures
  for (auto & i : ordered_data_list)
    std::sort(i.begin(), i.end(), 
             [&](SampledData* a, SampledData* b)
             {return a->pressure < b->pressure;});
}



std::vector<SampledData*> OpacitySpecies::findClosestDataPoints(
  const double sampling_pressure, const double sampling_temperature)
{
  std::vector<SampledData*> interpol_points(4, nullptr);
  
  if (ordered_data_list.size() == 0) return interpol_points;

  //first we find the two temperature data points
  std::vector<SampledData*>* lower_temperature = &ordered_data_list.front();
  std::vector<SampledData*>* upper_temperature = &ordered_data_list.front();

  if (sampling_temperature < ordered_data_list.front().front()->temperature)
  {
    lower_temperature = &ordered_data_list.front();
    upper_temperature = &ordered_data_list.front();
  }
  else if (sampling_temperature > ordered_data_list.back().front()->temperature)
  {
    lower_temperature = &ordered_data_list.back();
    upper_temperature = &ordered_data_list.back();
  }
  else
  {
    for (size_t i=0; i<ordered_data_list.size(); ++i)
    {
      if (ordered_data_list[i].front()->temperature == sampling_temperature)
      {
        lower_temperature = &ordered_data_list[i];
        upper_temperature = &ordered_data_list[i];

        break;
      }
      else if (ordered_data_list[i].front()->temperature < sampling_temperature 
        && ordered_data_list[i+1].front()->temperature > sampling_temperature)
      {
        lower_temperature = &ordered_data_list[i];
        upper_temperature = &ordered_data_list[i+1];

        break;
      }
    }
  }


  //now we find the pressures for the first temperature
  if (sampling_pressure < lower_temperature->front()->pressure)
  {
    interpol_points[0] = lower_temperature->front();
    interpol_points[1] = lower_temperature->front();
  }
  else if (sampling_pressure > lower_temperature->back()->pressure)
  {
    interpol_points[0] = lower_temperature->back();
    interpol_points[1] = lower_temperature->back();
  }
  else
  {
    for (size_t i=0; i<lower_temperature->size(); ++i)
    {
      if (lower_temperature->at(i)->pressure == sampling_pressure)
      {
        interpol_points[0] = lower_temperature->at(i);
        interpol_points[1] = lower_temperature->at(i);

        break;
      }
      else if (lower_temperature->at(i)->pressure < sampling_pressure 
        && lower_temperature->at(i+1)->pressure > sampling_pressure)
      {
        interpol_points[0] = lower_temperature->at(i);
        interpol_points[1] = lower_temperature->at(i+1);

        break;
      }
    }
  }


  //now we find the pressures for the second temperature
  if (sampling_pressure < upper_temperature->front()->pressure)
  {
    interpol_points[2] = upper_temperature->front();
    interpol_points[3] = upper_temperature->front();
  }
  else if (sampling_pressure > upper_temperature->back()->pressure)
  {
    interpol_points[2] = upper_temperature->back();
    interpol_points[3] = upper_temperature->back();
  }
  else
  {
    for (size_t i=0; i<upper_temperature->size(); ++i)
    {
      if (upper_temperature->at(i)->pressure == sampling_pressure)
      {
        interpol_points[2] = upper_temperature->at(i);
        interpol_points[3] = upper_temperature->at(i);

        break;
      }
      else if (upper_temperature->at(i)->pressure < sampling_pressure 
        && upper_temperature->at(i+1)->pressure > sampling_pressure)
      {
        interpol_points[2] = upper_temperature->at(i);
        interpol_points[3] = upper_temperature->at(i+1);

        break;
      }
    }
  }
  
  
  return interpol_points;
}



void OpacitySpecies::calcAbsorptionCrossSections(
  const double pressure, const double temperature, std::vector<double>& cross_sections)
{ 
  std::vector<SampledData*> data_points = findClosestDataPoints(pressure, temperature);

  checkDataAvailability(data_points);


  if (data_points[0] == data_points[1] && data_points[0] == data_points[2] && data_points[0] == data_points[3])
  {
    #pragma omp parallel for
    for (size_t i=0; i<cross_sections.size(); ++i)
      cross_sections[i] = std::pow(10.0, data_points[0]->cross_sections[i]); 

    return;
  }
  
  
  auto linearInterpolation = [](
    const double x1, const double x2, const double y1, const double y2, const double x) {
      return y1 + (y2 - y1) * (x - x1)/(x2 - x1);};

  
  //we first interpolate the cross section in pressure for the two different temperatures
  //after that, the result is interpolated in temperature
  std::vector<double> cross_sections_lower_t = data_points[0]->cross_sections;
  std::vector<double> cross_sections_upper_t = data_points[2]->cross_sections;


  if (data_points[0] != data_points[1])
    #pragma omp parallel for
    for (size_t i=0; i<cross_sections_upper_t.size(); ++i)
      cross_sections_lower_t[i] = linearInterpolation(
        data_points[0]->log_pressure,
        data_points[1]->log_pressure,
        data_points[0]->cross_sections[i], 
        data_points[1]->cross_sections[i],
        std::log10(pressure));

  if (data_points[2] != data_points[3])
    #pragma omp parallel for
    for (size_t i=0; i<cross_sections_upper_t.size(); ++i)
      cross_sections_upper_t[i] = linearInterpolation(
        data_points[2]->log_pressure,
        data_points[3]->log_pressure,
        data_points[2]->cross_sections[i], 
        data_points[3]->cross_sections[i],
        std::log10(pressure));

  if (data_points[0] != data_points[2])
    #pragma omp parallel for
    for (size_t i=0; i<cross_sections_lower_t.size(); ++i)
      cross_sections[i] = linearInterpolation(
        data_points[0]->temperature,
        data_points[2]->temperature,
        cross_sections_lower_t[i], 
        cross_sections_upper_t[i],
        temperature);
  else
    cross_sections = cross_sections_lower_t;
  

  //convert the cross sections back to linear space
  #pragma omp parallel for
  for (size_t i=0; i<cross_sections.size(); ++i)
    cross_sections[i] = std::pow(10.0, cross_sections[i]);
}


//bilinear (log P, T) interpolation of per-band native data (integrals or peaks), the exact
//mirror of calcAbsorptionCrossSections above operating on the given SampledData member.
//Returns false when no line data (or no band data) exist for this species.
bool OpacitySpecies::interpolateBandData(
  const double pressure, const double temperature,
  std::vector<double> SampledData::* member, std::vector<double>& band_data)
{
  if (cross_section_available == false) return false;

  std::vector<SampledData*> data_points = findClosestDataPoints(pressure, temperature);

  checkDataAvailability(data_points);

  if ((data_points[0]->*member).size() != band_data.size()) return false;


  if (data_points[0] == data_points[1] && data_points[0] == data_points[2] && data_points[0] == data_points[3])
  {
    for (size_t i=0; i<band_data.size(); ++i)
      band_data[i] = std::pow(10.0, (data_points[0]->*member)[i]);

    return true;
  }


  auto linearInterpolation = [](
    const double x1, const double x2, const double y1, const double y2, const double x) {
      return y1 + (y2 - y1) * (x - x1)/(x2 - x1);};


  std::vector<double> data_lower_t = data_points[0]->*member;
  std::vector<double> data_upper_t = data_points[2]->*member;


  if (data_points[0] != data_points[1])
    for (size_t i=0; i<data_lower_t.size(); ++i)
      data_lower_t[i] = linearInterpolation(
        data_points[0]->log_pressure,
        data_points[1]->log_pressure,
        (data_points[0]->*member)[i],
        (data_points[1]->*member)[i],
        std::log10(pressure));

  if (data_points[2] != data_points[3])
    for (size_t i=0; i<data_upper_t.size(); ++i)
      data_upper_t[i] = linearInterpolation(
        data_points[2]->log_pressure,
        data_points[3]->log_pressure,
        (data_points[2]->*member)[i],
        (data_points[3]->*member)[i],
        std::log10(pressure));

  if (data_points[0] != data_points[2])
    for (size_t i=0; i<band_data.size(); ++i)
      band_data[i] = linearInterpolation(
        data_points[0]->temperature,
        data_points[2]->temperature,
        data_lower_t[i],
        data_upper_t[i],
        temperature);
  else
    band_data = data_lower_t;


  for (size_t i=0; i<band_data.size(); ++i)
    band_data[i] = std::pow(10.0, band_data[i]);

  return true;
}


bool OpacitySpecies::calcAbsorptionBandIntegrals(
  const double pressure, const double temperature, std::vector<double>& band_integrals)
{
  return interpolateBandData(pressure, temperature, &SampledData::band_integrals, band_integrals);
}


bool OpacitySpecies::calcAbsorptionBandPeaks(
  const double pressure, const double temperature, std::vector<double>& band_peaks)
{
  return interpolateBandData(pressure, temperature, &SampledData::band_peaks, band_peaks);
}


//checks if all sampled cross sections needed later are available and samples them if not...
void OpacitySpecies::checkDataAvailability(std::vector<SampledData*>& data_points)
{
  std::vector<size_t> sampling_points;


  for (auto & i : data_points)
  {
    if (i->is_sampled == false)
    {
      if (sampling_points.size() == 0) sampling_points = spectral_grid->spectralIndexList();

      //native wavenumbers + band layout let the sampling pass also record the per-band
      //native integrals for the band-closure correction (one pass over the loaded file)
      i->sampleCrossSections(
        sampling_points,
        species_mass,
        &spectral_grid->nativeWavenumbers(),
        SpectralGrid::correction_band_width,
        spectral_grid->nbCorrectionBands());
    }
  }

}


void OpacitySpecies::calcTransportCoefficients(
  const double temperature,
  const double pressure,
  const std::vector<double>& number_densities,
  std::vector<double>& absorption_coeff,
  std::vector<double>& scattering_coeff,
  const BandCorrectionSpec* band_spec,
  std::vector<double>* band_correction,
  std::vector<double>* band_peak_coeff)
{
  double number_density = number_densities[species_index];

  for (const auto & i : cia_collision_partner)
    number_density *= number_densities[i];

  if (number_density == 0) return;


  double reference_pressure = pressure;

  //if the opacity species uses a different species for its tabulated pressure
  //use the partial pressure of that reference species here
  if (pressure_reference_species != _TOTAL)
    reference_pressure *= number_densities[pressure_reference_species]/number_densities[_TOTAL];


  std::vector<double> cross_sections(spectral_grid->nbSpectralPoints(), 0.0);


  if (cross_section_available == true)
  {
    calcAbsorptionCrossSections(reference_pressure, temperature, cross_sections);

    #pragma omp parallel for
    for (size_t i=0; i<spectral_grid->nbSpectralPoints(); ++i)
      absorption_coeff[i] += cross_sections[i] * number_density;

    //band-closure correction: what the sampled representation of THIS species' line opacity
    //misses per band, n_s * (int_band sigma dnu - sum_{k in band} w_k sigma_k). Only line
    //absorption enters -- Rayleigh/continuum/CIA are spectrally smooth and already exact on
    //the sampled grid. Where the sampling is complete the two terms cancel and the
    //correction vanishes on its own.
    if (band_spec != nullptr && band_correction != nullptr
        && band_spec->nb_bands > 0 && band_correction->size() == band_spec->nb_bands)
    {
      std::vector<double> band_integrals(band_spec->nb_bands, 0.0);

      if (calcAbsorptionBandIntegrals(reference_pressure, temperature, band_integrals))
      {
        for (size_t i=0; i<spectral_grid->nbSpectralPoints(); ++i)
        {
          const int b = (*band_spec->band_of_point)[i];
          if (b >= 0 && static_cast<size_t>(b) < band_spec->nb_bands)
            band_integrals[b] -= (*band_spec->point_weights)[i] * cross_sections[i];
        }

        for (size_t b=0; b<band_spec->nb_bands; ++b)
          (*band_correction)[b] += band_integrals[b] * number_density;
      }

      //peak absorption coefficient per band (escape-probability weighting): max over species,
      //since different species' line cores rarely coincide within a band
      if (band_peak_coeff != nullptr && band_peak_coeff->size() == band_spec->nb_bands)
      {
        std::vector<double> band_peaks(band_spec->nb_bands, 0.0);

        if (calcAbsorptionBandPeaks(reference_pressure, temperature, band_peaks))
          for (size_t b=0; b<band_spec->nb_bands; ++b)
            (*band_peak_coeff)[b] = std::max((*band_peak_coeff)[b], band_peaks[b] * number_density);
      }
    }
  }


  cross_sections.assign(spectral_grid->nbSpectralPoints(), 0.0);

  if (calcScatteringCrossSections(cross_sections) == true)
    #pragma omp parallel for
    for (size_t i=0; i<spectral_grid->nbSpectralPoints(); ++i)
      scattering_coeff[i] += cross_sections[i] * number_densities[species_index];


  cross_sections.assign(spectral_grid->nbSpectralPoints(), 0.0);

  if (calcContinuumAbsorption(temperature, number_densities, cross_sections) == true)
    #pragma omp parallel for
    for (size_t i=0; i<spectral_grid->nbSpectralPoints(); ++i)
      absorption_coeff[i] += cross_sections[i];
}



bool OpacitySpecies::calcScatteringCrossSections(std::vector<double>& cross_sections)
{

  return calcRayleighCrossSections(cross_sections);

}



double OpacitySpecies::generalRayleighCrossSection(
  double reference_density,
  double refractive_index,
  double king_correction_factor,
  double wavenumber)
{
  return 24. * pow(constants::pi, 3) * pow(wavenumber, 4) / (reference_density*reference_density)
         * pow((refractive_index*refractive_index - 1) 
         / (refractive_index*refractive_index + 2),2) * king_correction_factor;
}


}
