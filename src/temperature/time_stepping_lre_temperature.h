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


#ifndef _time_stepping_lre_temperature_h
#define _time_stepping_lre_temperature_h

#include <iostream>

#include "time_stepping_temperature.h"


namespace ngam {


class TimeSteppingLRETemperature : public TimeSteppingTemperature{
  public:
    TimeSteppingLRETemperature(
      double dt_fixed_, double alpha_, double target_flux_,
      double lre_fraction_)
      : TimeSteppingTemperature(dt_fixed_, alpha_, target_flux_),
        lre_fraction(lre_fraction_)
    {
      std::cout << "    with opacity-weighted LRE correction"
                << " (fraction: " << lre_fraction << ")\n\n";
    }
    virtual ~TimeSteppingLRETemperature() {}
    virtual void calcCorrection(
      const double surface_gravity,
      Atmosphere& atmosphere,
      const RadiativeTransferOutput& radiation_field,
      const OpacityCalculation& opacity);
  private:
    double lre_fraction;
};


}
#endif
