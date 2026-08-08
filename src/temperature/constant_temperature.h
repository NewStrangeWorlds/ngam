#ifndef _constant_temperature_h
#define _constant_temperature_h

#include "temperature.h"
#include "../atmosphere/atmosphere.h"

#include <vector>


namespace ngam {


class ConstantTemperature : public TemperatureProfile{
  public:
    ConstantTemperature() {nb_parameters = 1;}
    virtual ~ConstantTemperature() {}
    virtual void calcProfile(
      const std::vector<double>& parameters,
      const double surface_gravity,
      Atmosphere& atmosphere) {
        atmosphere.temperature.assign(atmosphere.pressure.size(), parameters[0]);}
};


}
#endif
