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

#ifndef SAMPLED_DATA_H
#define SAMPLED_DATA_H


#include <vector>
#include <string>
#include <cmath>


namespace ngam{


class CrossSectionFile {
  public:
    CrossSectionFile(const std::string filename_in, const bool log_data)
        : filename(filename_in)
        , is_data_log(log_data) 
        {}
    void loadFile();
    void unloadData();
    
    const std::string filename = "";

    const bool is_data_log = false;
    bool is_loaded = false;

    std::vector<double> cross_sections;
};



class SampledData{
  public:
    SampledData(
      const double temperature_data, 
      const double pressure_data, 
      const std::string file_name, 
      const bool log_data)
        : pressure(pressure_data)
        , log_pressure(std::log10(pressure_data))
        , temperature(temperature_data)
        , data_file(file_name, log_data)
        {}
    ~SampledData();
    void sampleCrossSections(
      const std::vector<size_t>& sampling_list, const double species_mass,
      const std::vector<double>* native_wavenumbers = nullptr,
      const double band_width = 0.0,
      const size_t nb_bands = 0);
    void deleteSampledData();

    const double pressure = 0.0;
    const double log_pressure = 0.0;
    const double temperature = 0.0;

    bool is_sampled = false;

    std::vector<double> cross_sections;
    //per-band integral int_band sigma dnu over the FULL native grid (band-closure correction),
    //same unit convention and log10 storage as cross_sections; computed during sampling, the
    //only moment the complete file is in memory
    std::vector<double> band_integrals;
    //per-band PEAK sigma over the native grid (escape-probability weighting of the closure:
    //the core optical depth to space decides whether the missing cores are thermalised);
    //same unit convention and log10 storage
    std::vector<double> band_peaks;
  private:
    CrossSectionFile data_file;
};


}

#endif
