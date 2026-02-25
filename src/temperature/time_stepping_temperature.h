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


#ifndef _time_stepping_temperature_h
#define _time_stepping_temperature_h

#include <iostream>

#include "temperature_correction.h"


namespace ngam {


class TimeSteppingTemperature : public TemperatureCorrection{
  public:
    TimeSteppingTemperature(
      double dt_fixed_, double alpha_, double target_flux_)
      : dt_fixed(dt_fixed_), alpha(alpha_), target_flux(target_flux_)
    {
      std::cout << "\n- Temperature correction: time stepping\n\n";
    }
    virtual ~TimeSteppingTemperature() {}
    virtual void calcCorrection(
      const double surface_gravity,
      Atmosphere& atmosphere,
      const RadiativeTransferOutput& radiation_field,
      const OpacityCalculation& opacity);
  protected:
    double dt_fixed;
    double alpha;
    double target_flux;
};


}
#endif
