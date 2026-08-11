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

    // Smooth (mixing-length) convection: the scheme supplies a convective FLUX law instead of a
    // hard adjustment, and the Newton correctors then treat every level as a free unknown -- no
    // convective mask, no adiabatic slaving, no boundary controller (see
    // doc/mlt_convection_design.md). Adjustment schemes return false (the default).
    virtual bool providesFlux() const { return false; }

    // Mixing-length parameter alpha (lambda = alpha * H_p away from the surface); only meaningful
    // when providesFlux() is true.
    virtual double fluxAlpha() const { return 0.0; }

    // Pressure ceiling [bar] above which the convective flux is zero -- the same guard the
    // adjustment schemes apply (min_convection_pressure). Load-bearing for moist convection: the
    // saturated-parcel moist gradient collapses near the e_s(T) ~ P crossover aloft, which would
    // otherwise read the subsaturated radiative stratosphere as moist-unstable (measured: the whole
    // stratosphere flattened onto the saturated pseudo-adiabat at 214 K).
    virtual double fluxMinPressure() const { return 0.0; }

    // Blackadar wall law for the mixing length near the domain bottom (lambda = k z for z << H_p).
    // Correct for a SURFACE (terrestrial); wrong for a self-luminous object, whose bottom boundary
    // is an artificial domain cut, not a wall -- those use lambda = alpha H_p throughout.
    virtual bool fluxBlackadarSurface() const { return true; }
};


}

#endif
