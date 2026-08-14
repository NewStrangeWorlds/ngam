
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
#include <iterator>
#include <fstream>
#include <iomanip>
#include <cmath>

#include "sampled_data.h"


namespace ngam{


void SampledData::deleteSampledData()
{
  std::vector<double>().swap (cross_sections);

  is_sampled = false;
}



void SampledData::sampleCrossSections(
  const std::vector<size_t>& sampling_list_indices, const double species_mass,
  const std::vector<double>* native_wavenumbers,
  const double band_width,
  const size_t nb_bands)
{
  if (!data_file.is_loaded) data_file.loadFile();


  cross_sections.assign(sampling_list_indices.size(), 0.0);


  for(size_t i=0; i<sampling_list_indices.size(); ++i)
  {
    if (sampling_list_indices[i] > data_file.cross_sections.size()-1) break;

    cross_sections[i] = data_file.cross_sections[sampling_list_indices[i]];
  }


  //band-closure correction: per-band integral over the COMPLETE native grid, computed here
  //because this is the only place the full file is ever in memory. Same unit conversion,
  //floor and log10 storage as the sampled points below.
  if (native_wavenumbers != nullptr && nb_bands > 0 && band_width > 0.0)
  {
    band_integrals.assign(nb_bands, 0.0);
    band_peaks.assign(nb_bands, 0.0);

    const std::vector<double>& wn = *native_wavenumbers;
    const size_t nb_data = std::min(data_file.cross_sections.size(), wn.size());

    for (size_t j=0; nb_data >= 2 && j<nb_data; ++j)
    {
      const double dnu = (j == 0) ? (wn[1] - wn[0])
                       : (j+1 == nb_data) ? (wn[j] - wn[j-1])
                       : 0.5*(wn[j+1] - wn[j-1]);
      const size_t b = static_cast<size_t>(wn[j]/band_width);

      if (b < nb_bands)
      {
        band_integrals[b] += data_file.cross_sections[j] * dnu;
        if (data_file.cross_sections[j] > band_peaks[b])
          band_peaks[b] = data_file.cross_sections[j];
      }
    }

    for (auto & i : band_integrals)
    {
      if (species_mass > 0) i *= species_mass/6.022140857e23;

      if (i < 1e-200) i = 1e-200;

      i = std::log10(i);
    }

    for (auto & i : band_peaks)
    {
      if (species_mass > 0) i *= species_mass/6.022140857e23;

      if (i < 1e-200) i = 1e-200;

      i = std::log10(i);
    }
  }


  //convert from opacity in cm2/g to cm2 if neccessary
  //also, apply a small minimum value to allow for interpolation in log later
  for (auto & i : cross_sections)
  {
    if (species_mass > 0) i *= species_mass/6.022140857e23;
    
    if (i < 1e-200) i = 1e-200;
  }

  //convert the cross-section to log for performance reasons
  for (auto & i : cross_sections)
    i = std::log10(i);

  is_sampled = true;

  data_file.unloadData();
}



SampledData::~SampledData()
{
  
}


}


