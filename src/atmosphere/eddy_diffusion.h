/*
* This file is part of the ngam code.
* Copyright (C) 2026 Daniel Kitzmann
*
* ngam is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

#ifndef _eddy_diffusion_h
#define _eddy_diffusion_h

#include <memory>
#include <string>
#include <vector>

#include "atmosphere.h"
#include "../convection/convection.h"
#include "../config/module_params.h"


namespace ngam {


// Vertical eddy diffusion coefficient Kzz(p) [cm^2/s] -- an atmosphere property consumed by the
// disequilibrium chemistry (quench module, later the kinetics coupling) and, eventually, clouds.
// Selected by the model's `kzz` module spec:
//
//   mlt (default)   self-consistent: in convective layers from the mixing-length convection
//                   scheme's own flux law (the one the ratio_ul/MLT corrector solves), extended
//                   into the radiative zones by a prescription
//                     scaling     "velocity" (default): Kzz = lambda * w_conv, w_conv the MLT
//                                 convective velocity sqrt(g lambda^2 (nabla - nabla_ad) / H_p)
//                                 (AGNI's default)
//                                 "flux": Ackerman & Marley (2001) / Charnay et al. (2015)
//                                 Kzz = H_p/3 (lambda/H_p)^{4/3} (R F_conv / (mu rho c_p))^{1/3}
//                                 with F_conv from the same flux law (picaso's recipe)
//                     radiative   how Kzz continues ABOVE a convective zone (p_rcb = the zone's
//                                 top pressure, Kzz_rcb = its maximum Kzz within one pressure
//                                 decade below the top):
//                                 "constant" (default): Kzz_rcb
//                                 "power_law": Kzz_rcb (p/p_rcb)^slope, floored at `min`. slope
//                                 is then required: NEGATIVE = mixing grows upward, the
//                                 GCM-fitted form for irradiated planets (Lee 2024, AGNI uses
//                                 -0.4); POSITIVE = decays upward towards `min`, appropriate for
//                                 self-luminous objects whose radiative zone is stirred only by
//                                 overshoot and gravity waves (Freytag et al. 2010: ~1e5-1e7)
//                                 "fixed": a prescribed value above every zone, parameter `value`
//                                 (e.g. 1e6 for a brown dwarf; retrieved photospheric Kzz of
//                                 T dwarfs are ~1e4-1e8, Miles et al. 2020, Mukherjee et al. 2022)
//                     min         floor [cm^2/s] everywhere and the value when nothing is
//                                 convective, default 1e4
//                     relax       initial weight of the Aitken dynamic relaxation between
//                                 successive updates (the weight then adapts itself from the
//                                 residual history, clamped to [0.05, 1]); default 0.5
//                     tolerance   dead band [dex]: a new profile is committed only if it differs
//                                 from the current one by more than this anywhere, so Kzz
//                                 freezes once it is settled and the Newton can finish; 0
//                                 disables it; default 0.05
//                   The radiative extension is anchored at the MAXIMUM Kzz of the zone below (at
//                   the zone's top pressure), not at its top level: the flux law's
//                   super-adiabaticity -- and with it Kzz -- goes to zero continuously at the
//                   radiative-convective boundary (the C1 handover), so the top-level value is
//                   both tiny and numerically erratic while the solver iterates. Radiative layers
//                   BELOW the deepest convective zone keep that zone's bottom value.
//                   The profile is refreshed once per outer iteration from the committed T(P),
//                   whose super-adiabaticity is the stiff quantity the Newton is still solving
//                   for; the raw MLT value therefore swings by orders of magnitude between
//                   iterations early on (measured: the quench point flipped 16 <-> 4 bar and the
//                   BD Newton stalled). Two guards: log-space under-relaxation towards the new
//                   profile, and a HOLD of the previous profile when the new one finds no
//                   convective zone at all. Both leave the converged fixed point untouched.
//   constant        Kzz = value          {value}
//   power_law       Kzz = value (p/pressure)^slope   {value, pressure=1 bar, slope}
class EddyDiffusion {
  public:
    virtual ~EddyDiffusion() {}

    // Kzz per level. `convection` may be null (no convection scheme). Implementations must cope
    // with an atmosphere whose structure (scale height, density) has not been computed yet.
    virtual std::vector<double> profile(
      const Atmosphere& atmosphere,
      const Convection* convection,
      const double surface_gravity) = 0;
};


class ConstantEddyDiffusion : public EddyDiffusion {
  public:
    ConstantEddyDiffusion(const double value_) : value(value_) {}
    std::vector<double> profile(
      const Atmosphere& atmosphere, const Convection*, const double) override
    { return std::vector<double>(atmosphere.pressure.size(), value); }
  private:
    const double value;
};


class PowerLawEddyDiffusion : public EddyDiffusion {
  public:
    PowerLawEddyDiffusion(const double value_, const double pressure_, const double slope_)
      : value(value_), reference_pressure(pressure_), slope(slope_) {}
    std::vector<double> profile(
      const Atmosphere& atmosphere, const Convection*, const double) override;
  private:
    const double value, reference_pressure, slope;
};


class MltEddyDiffusion : public EddyDiffusion {
  public:
    enum class Radiative {constant, power_law, fixed};

    MltEddyDiffusion(
      const bool flux_scaling_, const Radiative radiative_, const double slope_,
      const double fixed_value_, const double min_value_, const double relax_,
      const double tolerance_)
      : flux_scaling(flux_scaling_), radiative(radiative_), slope(slope_)
      , fixed_value(fixed_value_), min_value(min_value_), relax(relax_), tolerance(tolerance_)
      , weight(relax_) {}
    std::vector<double> profile(
      const Atmosphere& atmosphere, const Convection* convection, const double surface_gravity) override;
  private:
    const bool flux_scaling;
    const Radiative radiative;
    const double slope;         // power_law exponent
    const double fixed_value;   // fixed: Kzz above every zone
    const double min_value;
    const double relax;
    const double tolerance;   // dex; smaller changes are not committed

    std::vector<double> previous;            // last committed profile (relaxation / hold state)
    std::vector<double> previous_residual;   // log(raw) - log(committed) of the last update (Aitken)
    double weight = 1.0;                     // current Aitken relaxation weight

    // the raw, unrelaxed MLT profile; convective = true if any link is super-adiabatic
    std::vector<double> rawProfile(
      const Atmosphere& atmosphere, const Convection* convection, const double surface_gravity,
      bool& convective) const;
};


std::unique_ptr<EddyDiffusion> selectEddyDiffusion(const ModuleSpec& spec);


}
#endif
