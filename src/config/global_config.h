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


#ifndef GLOBAL_CONFIG_H
#define GLOBAL_CONFIG_H

#include <string>
#include <vector>
#include <string>


namespace bear {

struct GlobalConfig {
  GlobalConfig() {};
  GlobalConfig(const std::string retrieval_folder) {
    loadConfigFile(retrieval_folder);};

  GlobalConfig(
    const std::string forward_model_type_,
    const std::string cross_section_file_path_,
    const std::string spectral_disecretisation_,
    const double resolution_);

  bool loadConfigFile(std::string retrieval_folder);
  
  std::string forward_model_type = "";

  std::string cross_section_file_path = "";
  std::string wavenumber_file_path = "";
  std::string retrieval_folder_path = "";
  
  unsigned int spectral_disecretisation = 0;
  double spectral_resolution = 1000;

  unsigned int nb_mpi_processes = 1;
  unsigned int nb_omp_processes = 0;
  
  unsigned int nb_disort_streams = 4;

  unsigned int nb_grid_points = 100;

  double surface_gravity = 100.0; //in cm/s2
  double bottom_radius = 7.1492e9; //in cm
  bool use_variable_gravity = false;

  std::vector<double> atmos_boundary_pressures{1e2, 1e-6}; //in bar

  std::vector<std::string> chemistry_model{"iso"};
  std::vector< std::vector<std::string> > chemistry_parameters{{"H2O", "CO2"}};

  std::string temperature_profile_type = "milne";
  std::vector<std::string> temperature_profile_parameters{};

  std::vector<std::string> cloud_model{};

  std::string radiative_transfer_type = "disort";
  std::vector<std::string> radiative_transfer_parameters{"4"};

  std::vector<std::string> opacity_species_symbol{"H2O", "CO2"};
  std::vector<std::string> opacity_species_folder{"Molecules/1H2-16O__POKAZATEL_e2b", "Molecules/12C-16O2__CDSD_4000_e2b"};
};



}


#endif
