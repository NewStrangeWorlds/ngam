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


#ifndef _constant_temperature_h
#define _constant_temperature_h

#include "temperature.h"
#include "../atmosphere/atmosphere.h"

#include <vector>


namespace ngam {


class ConstantTemperature : public Temperature{
  public:
    ConstantTemperature() {nb_parameters = 1;}
    virtual ~ConstantTemperature() {}
    virtual void calcProfile(
      const std::vector<double>& parameters,
      const double surface_gravity,
      Atmosphere& atmosphere,
      const RadiativeTransferOutput& radiation_field) {
        atmosphere.temperature.assign(atmosphere.pressure.size(), parameters[0]);}
};


}
#endif 
