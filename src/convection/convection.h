#ifndef _convection_h
#define _convection_h

#include <vector>

#include "../atmosphere/atmosphere.h"


namespace ngam {


class Convection {
  public:
    virtual ~Convection() {}

    // Adjust the temperature profile towards this scheme's convective stratification.
    virtual void adjust(Atmosphere& atmosphere) const = 0;

    // The convective temperature gradient nabla = d ln(T) / d ln(P) this scheme enforces,
    // evaluated at a single level from its local state. This lets the linearised temperature
    // correction enforce exactly the gradient of the active scheme (dry, moist, ...) without
    // knowing which one it is. pressure is in bar (ignored by schemes that don't need it).
    virtual double convectiveGradient(
      const std::vector<double>& number_densities,
      double temperature,
      double pressure) const = 0;
};


}

#endif
