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


#include "time_stepping_temperature.h"
#include "../additional/exceptions.h"
#include "../additional/physical_const.h"
#include "../atmosphere/atmosphere.h"

#include <algorithm>
#include <vector>
#include <cmath>


namespace ngam {



//calculate the temperature based on the analytical Milne solution
bool TimeSteppingTemperature::calcProfile(
  const std::vector<double>& parameters, 
  const double surface_gravity,
  Atmosphere& atmosphere)
{
  temperature.assign(pressure.size(), 0);

  const double kappa_ross = parameters[0];
  const double t_eff = parameters[1];



  return neglect_model;
}



}

