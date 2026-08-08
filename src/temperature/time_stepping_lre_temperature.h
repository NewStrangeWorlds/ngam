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
