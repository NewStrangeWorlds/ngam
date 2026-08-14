
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


#include <iostream>
#include <omp.h>
#include <fstream>
#include <iomanip>
#include <cmath>

#include "transport_coeff.h"

#include "species_definition.h"
#include "../spectral_grid/spectral_grid.h"
#include "../chemistry/chem_species.h"
#include "../additional/exceptions.h"


namespace ngam{


TransportCoefficients::TransportCoefficients(
  const std::string& cross_section_file_path_,
  SpectralGrid* grid_ptr,
  const std::vector<std::string>& opacity_species_symbol,
  const std::vector<std::string>& opacity_species_folder)
{
  cross_section_file_path = cross_section_file_path_;
  spectral_grid = grid_ptr;

  std::vector<size_t> spectral_indices = spectral_grid->spectralIndexList();
  
  gas_species.reserve(opacity_species_symbol.size());

  bool all_species_added = true;

  for (size_t i=0; i<opacity_species_symbol.size(); ++i)
  {
    bool added = addOpacitySpecies(opacity_species_symbol[i], opacity_species_folder[i]);

    if (!added) all_species_added = false;
  }

  
  std::cout << "\nOpacity specied added:\n";
  for (auto & i : gas_species)
  {
    std::string data_available = "data available: No";
    
    if (i->dataAvailable())
      data_available = "data available: Yes";

    std::cout  << std::setw(8) << std::left << i->species_name << "\t" 
               << std::setw(40) << std::left << i->species_folder << "\t" 
               << std::setw(10) << std::left << data_available << "\n\n";
  }
  
  
  for (auto & i : gas_species)
  {
    if (i->dataAvailable() == false)
    {
      std::string error_message = "Unable to locate all opacity data.\n";
      throw InvalidInput(std::string ("TransportCoefficients::TransportCoefficients"), error_message);
    }
  }

  if (!all_species_added)
  {
    std::string error_message = "Not all opacities species from the model config file could be added!\nThis could be caused by a species unknown to BeAR.\n";
    throw InvalidInput(std::string ("TransportCoefficients::TransportCoefficients"), error_message);
  }

}




bool TransportCoefficients::addOpacitySpecies(
  const std::string& species_symbol, const std::string& species_folder)
{
  //first, the species cases for which separate classes are available
  if (species_symbol == "CIA-H2-H2")
  {
    gas_species.push_back(
      std::make_unique<GasGeneric>(
        cross_section_file_path,
        spectral_grid,
        _H2,
        "CIA H2-H2",
        species_folder,
        std::vector<size_t>{_H2}));

    return true;
  }

  if (species_symbol == "CIA-H2-He")
  {
    gas_species.push_back(
      std::make_unique<GasGeneric>(
        cross_section_file_path,
        spectral_grid,
        _H2,
        "CIA H2-He",
        species_folder,
        std::vector<size_t>{_He}));

    return true;
  }

  if (species_symbol == "CIA-H-He")
  {
    gas_species.push_back(
      std::make_unique<GasGeneric>(
        cross_section_file_path,
        spectral_grid,
        _H,
        "CIA H-He",
        species_folder,
        std::vector<size_t>{_He}));

    return true;
  }

  //H- free-free and bound-free continuum
  if (species_symbol == "H-")
  {
    gas_species.push_back(std::make_unique<GasHm>(cross_section_file_path, spectral_grid));

    return true;
  }

  //CO2-CO2 CIA free-free and bound-free continuum
  if (species_symbol == "CO2-CIA")
  {
    gas_species.push_back(std::make_unique<GasCO2CIA>(cross_section_file_path, spectral_grid));

    return true;
  }

  //H2O self and foreign CIA
  if (species_symbol == "H2O-CIA")
  {
    gas_species.push_back(std::make_unique<GasH2OCIA>(cross_section_file_path, spectral_grid));

    return true;
  }

  //H2 Rayleigh scattering
  if (species_symbol == "H2" && species_folder == "Rayleigh")
  {
    gas_species.push_back(std::make_unique<GasH2Rayleigh>(cross_section_file_path, spectral_grid, ""));

    return true;
  }

  //He Rayleigh scattering
  if (species_symbol == "He" && species_folder == "Rayleigh")
  {
    gas_species.push_back(std::make_unique<GasHeRayleigh>(cross_section_file_path, spectral_grid, ""));

    return true;
  }

  //H Rayleigh scattering
  if (species_symbol == "H" && species_folder == "Rayleigh")
  {
    gas_species.push_back(std::make_unique<GasHRayleigh>(cross_section_file_path, spectral_grid, ""));

    return true;
  }

  //CO Rayleigh
  if (species_symbol == "CO" && species_folder == "Rayleigh")
  {
    gas_species.push_back(std::make_unique<GasCORayleigh>(cross_section_file_path, spectral_grid, ""));

    return true;
  }

  //CO2 Rayleigh
  if (species_symbol == "CO2" && species_folder == "Rayleigh")
  {
    gas_species.push_back(std::make_unique<GasCO2Rayleigh>(cross_section_file_path, spectral_grid, ""));

    return true;
  }

  //CH4 Rayleigh
  if (species_symbol == "CH4" && species_folder == "Rayleigh")
  {
    gas_species.push_back(std::make_unique<GasCH4Rayleigh>(cross_section_file_path, spectral_grid, ""));

    return true;
  }

  //H2O Rayleigh
  if (species_symbol == "H2O" && species_folder == "Rayleigh")
  {
    gas_species.push_back(std::make_unique<GasH2ORayleigh>(cross_section_file_path, spectral_grid, ""));

    return true;
  }

  //N2 Rayleigh
  if (species_symbol == "N2" && species_folder == "Rayleigh")
  {
    gas_species.push_back(std::make_unique<GasN2Rayleigh>(cross_section_file_path, spectral_grid, ""));

    return true;
  }

  //O2 Rayleigh
  if (species_symbol == "O2" && species_folder == "Rayleigh")
  {
    gas_species.push_back(std::make_unique<GasO2Rayleigh>(cross_section_file_path, spectral_grid, ""));

    return true;
  }

  //now we try the generic ones
  for (size_t i=0; i<constants::species_data.size(); ++i)
  {
    if (constants::species_data[i].symbol == species_symbol)
    {
      gas_species.push_back(
        std::make_unique<GasGeneric>(
          cross_section_file_path,
          spectral_grid,
          constants::species_data[i].id,
          constants::species_data[i].symbol,
          species_folder));

      return true;
    }
  }

  //we haven't found the corresponding species
  std::cout << "Opacity species " 
    << species_symbol 
    << " has not been found in the internal list located in chem_species.h!\n";


  return false;
}


//calculates the transport coefficients on the CPU
//calls the calculation method of the individual opacity species
void TransportCoefficients::calculate(
  const double temperature,
  const double pressure,
  const std::vector<double>& number_densities,
  std::vector<double>& absorption_coeff,
  std::vector<double>& scattering_coeff,
  const BandCorrectionSpec* band_spec,
  std::vector<double>* band_correction,
  std::vector<double>* band_peak_coeff)
{
  absorption_coeff.assign(spectral_grid->nbSpectralPoints(), 0);
  scattering_coeff.assign(spectral_grid->nbSpectralPoints(), 0);

  if (band_spec != nullptr && band_correction != nullptr)
    band_correction->assign(band_spec->nb_bands, 0.0);
  if (band_spec != nullptr && band_peak_coeff != nullptr)
    band_peak_coeff->assign(band_spec->nb_bands, 0.0);

  for (unsigned int i=0; i<gas_species.size(); i++)
  {
    //std::cout << gas_species[i]->species_name << "\n";
    gas_species[i]->calcTransportCoefficients(
      temperature,
      pressure,
      number_densities,
      absorption_coeff,
      scattering_coeff,
      band_spec,
      band_correction,
      band_peak_coeff);
    }
}



}
