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
