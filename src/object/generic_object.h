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


#ifndef GENERIC_OBJECT_H
#define GENERIC_OBJECT_H

#include "../config/global_config.h"
#include "../spectral_grid/spectral_grid.h"
#include "../transport_coeff/transport_coeff.h"


namespace bear{


class GenericObject {
  public:
    GenericObject(
      GlobalConfig* config_,
      SpectralGrid* spectral_grid_) 
      : config(config_), 
        spectral_grid(spectral_grid_),
        transport_coeff(
          config, 
          spectral_grid, 
          config->opacity_species_symbol,
          config->opacity_species_folder)
    {}
    virtual ~GenericObject() {}
  private:
    GlobalConfig* config;
    SpectralGrid* spectral_grid;
    
    TransportCoefficients transport_coeff;
};


} // namespace bear

#endif // GENERIC_OBJECT_H