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


#ifndef BROWN_DWARF_H
#define BROWN_DWARF_H


#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

#include "generic_object.h"


namespace bear{

class BrownDwarf : public GenericObject {
  public:
    BrownDwarf(
      GlobalConfig* config,
      SpectralGrid* spectral_grid) 
      : GenericObject(config, spectral_grid)
    {}
    virtual ~BrownDwarf() {}

    bool computeAtmosphericStructure() override
    {
      std::cout << "Computing brown dwarf atmospheric structure...\n";

      
      std::vector<double> chem_parameters{0.5, 0.5};
      std::vector<double> temp_parameters{1e-4, 3000.0};
      
      atmosphere.calcAtmosphereStructure(
        config->surface_gravity,
        config->bottom_radius,
        config->use_variable_gravity,
        temperature_profile,
        temp_parameters,
        chemistry,
        chem_parameters);
      
      opacity.calculate();

      radiative_transfer->calcSpectrum(
        atmosphere,
        opacity,
        radiation_field);
      


      std::string file_name = "spectrum.dat";
  
      std::fstream file(file_name.c_str(), std::ios::out);

      for (size_t i=0; i<spectral_grid->nb_spectral_points; ++i)
        file << std::setprecision(10) << std::scientific << spectral_grid->wavelength_list[i] << " " << radiation_field.spectrum[i] << "\n";

      file.close();
      
      return true;
    }
};

} // namespace bear

#endif // BROWN_DWARF_H