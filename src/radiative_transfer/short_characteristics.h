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


#ifndef _short_char_h
#define _short_char_h


#include <vector>
#include <iostream>
#include <cmath>

#include "radiative_transfer.h"

#include "../atmosphere/atmosphere.h"
#include "../spectral_grid/spectral_grid.h"


namespace ngam {


class ShortCharacteristics : public RadiativeTransfer{
  public:
    ShortCharacteristics(SpectralGrid* spectral_grid_ptr)
    : RadiativeTransfer(spectral_grid_ptr) {};
    virtual ~ShortCharacteristics() {}
    
    virtual void calculate(
      const Atmosphere& atmosphere,
      const OpacityCalculation& opacity,
      RadiativeTransferOutput& output,
      const RadiativeBoundaryConditions& bc = RadiativeBoundaryConditions{}) override;
  
  private:
    const std::vector<double> gauss_nodes{0.211324865405187, 0.788675134594813};
    const std::vector<double> gauss_weights{0.5, 0.5};
    const size_t nb_angles = gauss_nodes.size();

    double calculate(
      const std::vector<double>& absorption_coeff,
      const std::vector<double>& cloud_optical_depth,
      const std::vector<double>& temperature,
      const std::vector<double>& vertical_grid,
      const size_t nu_index);
};


}
#endif


