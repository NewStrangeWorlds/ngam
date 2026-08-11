#ifndef _mixing_length_h
#define _mixing_length_h

#include "convection.h"


namespace ngam {


// Mixing-length convection (AGNI-style, doc/mlt_convection_design.md): supplies the neutrality
// gradient (dry or moist) and the mixing-length parameter alpha; the Newton corrector builds the
// convective flux F_c = 1/2 rho c_p T (lambda^2/H_p) sqrt(g/H_p) x^{3/2} on level links from them
// (x = max(nabla - nabla_ad, 0)). No hard adjustment is performed: adjust() only sets the
// convective flags diagnostically (super-adiabatic links), leaving the temperatures untouched.
// The converged T(P) is insensitive to alpha (the flux law acts as a stiff penalty, see the design
// note) -- alpha is exposed for the alpha-sensitivity acceptance test.
class MixingLengthConvection : public Convection {
  public:
    MixingLengthConvection(
      const bool moist_, const double alpha_ = 1.0, const double min_pressure_ = 0.0,
      const bool blackadar_surface_ = true)
      : moist(moist_), alpha(alpha_), min_pressure(min_pressure_)
      , blackadar_surface(blackadar_surface_) {}

    double convectiveGradient(
      const std::vector<double>& number_densities,
      double temperature,
      double pressure) const override;

    void adjust(Atmosphere& atmosphere) const override;   // diagnostic flags only, T untouched

    bool providesFlux() const override { return true; }
    double fluxAlpha() const override { return alpha; }
    double fluxMinPressure() const override { return min_pressure; }
    bool fluxBlackadarSurface() const override { return blackadar_surface; }

  private:
    const bool moist = false;
    const double alpha = 1.0;
    const double min_pressure = 0.0;      // bar; no convective flux above this level
    const bool blackadar_surface = true;  // wall law at the bottom (surface objects only)
};


}

#endif
