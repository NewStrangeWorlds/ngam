#ifndef _temperature_profile_h
#define _temperature_profile_h

#include <vector>
#include <cstddef>

#include "../atmosphere/atmosphere.h"

namespace ngam {


class TemperatureProfile{
  public:
    virtual ~TemperatureProfile() {}
    virtual void calcProfile(
      const std::vector<double>& parameters,
      const double surface_gravity,
      Atmosphere& atmosphere) = 0;
    size_t nbParameters() {return nb_parameters;}
  protected:
    size_t nb_parameters {};
};


}
#endif
