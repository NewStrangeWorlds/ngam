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

#ifndef SIMPLE_SURFACE_H
#define SIMPLE_SURFACE_H

#include <memory>
#include <vector>

#include "generic_surface.h"
#include "../spectral_grid/spectral_grid.h"


namespace ngam{

//A simple surface model with a constant albedo in the shortwave and zero albedo in the longwave.
//The switch between the two regimes is defined by wavelength_switch (in micron).
class SimpleSurface : public GenericSurface {
  public:
    SimpleSurface(
      SpectralGrid* spectral_grid_,
      const double albedo_shortwave,
      const double wavelength_switch_)
      : GenericSurface(spectral_grid_)
      {
        for (size_t i = 0; i < spectral_grid->nbSpectralPoints(); ++i)
        {
          if (spectral_grid->wavelength_list[i] < wavelength_switch_)
            albedo[i] = albedo_shortwave;
          else
            albedo[i] = 0.0;
        }
      }
    virtual ~SimpleSurface() {}
  protected:
};


} // namespace ngam

#endif // SIMPLE_SURFACE_H
