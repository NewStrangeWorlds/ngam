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


#ifndef _temperature_correction_h
#define _temperature_correction_h

#include "../atmosphere/atmosphere.h"

namespace ngam {

struct RadiativeTransferOutput;
class OpacityCalculation;


class TemperatureCorrection{
  public:
    virtual ~TemperatureCorrection() {}
    virtual void calcCorrection(
      const double surface_gravity,
      Atmosphere& atmosphere,
      const RadiativeTransferOutput& radiation_field,
      const OpacityCalculation& opacity) = 0;
};


}
#endif
