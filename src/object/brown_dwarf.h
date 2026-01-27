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

#include "generic_object.h"


namespace bear{

class BrownDwarf : public GenericObject {
  public:
    BrownDwarf(
      GlobalConfig* config,
      SpectralGrid* spectral_grid) 
      : GenericObject(config, spectral_grid)
    {std::cout << spectral_grid->nbSpectralPoints() << " points in brown dwarf spectral grid.\n";}
    virtual ~BrownDwarf() {}
};

} // namespace bear

#endif // BROWN_DWARF_H