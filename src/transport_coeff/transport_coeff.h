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


#ifndef TRANSPORT_COEFF_H
#define TRANSPORT_COEFF_H

#include <vector>
#include <memory>

#include "opacity_species.h"
#include "../spectral_grid/spectral_grid.h"


namespace ngam{


class TransportCoefficients {
  public:
    TransportCoefficients(
      const std::string& cross_section_file_path,
      SpectralGrid* grid_ptr,
      const std::vector<std::string>& opacity_species_symbol,
      const std::vector<std::string>& opacity_species_folder);
    void calculate(
      const double temperature,
      const double pressure,
      const std::vector<double>& number_densities,
      std::vector<double>& absorption_coeff,
      std::vector<double>& scattering_coeff);

  private:
    std::string cross_section_file_path;
    SpectralGrid* spectral_grid = nullptr;

    std::vector<std::unique_ptr<OpacitySpecies>> gas_species;

    bool addOpacitySpecies(
      const std::string& species_symbol, const std::string& species_folder);
};


}

#endif
