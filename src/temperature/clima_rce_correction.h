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


#ifndef _clima_rce_correction_h
#define _clima_rce_correction_h

#include "temperature_correction.h"

#include <vector>

namespace ngam {

struct RadiativeTransferOutput;
class OpacityCalculation;
class Atmosphere;
class Convection;


// Faithful re-implementation of Wogan's clima (AdiabatClimate) radiative-convective-equilibrium
// solver, as a self-contained reference. Per calcCorrection it runs ONE full inner trust-region
// Newton (Powell dogleg, MINPACK hybrj-style) at a FIXED convective mask; the driver's outer loop
// owns the RC-iteration / mask re-detection.
//
//  * Residual: clima's heat-capacity-weighted flux divergence (a heating rate). For a radiative level
//    i: g = net_heating[i]/c_eff_i (DISORT's own divergence). Each contiguous CONVECTIVE zone is one
//    DOF (interior slaved to the adiabat) whose residual is the net flux into the whole zone,
//    (F_top - F_bottom)/c_eff_zone; c_eff = (dP/g)*c_p.
//  * Jacobian: finite differences with opacity+composition FROZEN (Planck-only response, clima's
//    operator split); recomputed each Newton step. Built directly on the reduced (slaved) DOFs.
//  * Globalisation: Powell dogleg trust region with MINPACK mode-1 column scaling (the 1/c_eff
//    weighting spans many orders of magnitude down the column).
//
// All RT re-evaluations go through the ForwardEvalFull callback the driver installs.
class ClimaRCECorrection : public TemperatureCorrection{
  public:
    ClimaRCECorrection(
      const double target_flux_,
      const Convection* convection_,
      const double surface_gravity_unused_ = 0.0)
      : target_flux(target_flux_)
      , convection(convection_) {}
    virtual ~ClimaRCECorrection() {}

    // Uses DISORT's analytic Planck-only net-flux temperature Jacobian (= clima's frozen-opacity
    // Jacobian, in one RT solve instead of m finite differences), so the driver must compute it.
    bool requiresRadiationJacobian() const override { return true; }
    bool managesOwnStepSize() const override { return true; }              // trust region damps itself
    void setForwardEvalFull(ForwardEvalFull f) override { forward_eval_full_ = std::move(f); }
    double lastConvergenceResidual() const override { return last_residual_; }

    virtual void calcCorrection(
      const double surface_gravity,
      Atmosphere& atmosphere,
      const RadiativeTransferOutput& radiation_field,
      const OpacityCalculation& opacity) override;

  private:
    const double target_flux = 0.0;
    const Convection* convection = nullptr;

    ForwardEvalFull forward_eval_full_ = nullptr;

    double last_residual_ = -1.0;       // max|weighted g| at the committed point (drives convergence)
    std::vector<int> prev_mask_;        // convective mask from the previous call (for the +-1 limiter)
    double last_inner_change_ = 1e300;  // max|dT/T| the last inner solve applied (settle gate for the mask)
    int rcb_cap_ = -1;                  // anti-overshoot cap on the convective top (clima Mode-3; -1 = none)
    int rcb_lock_ = 0;                  // lockout countdown after a cold-inversion shrink (prevents ABAB toggle)

    // CLIMA_PTC unified mode: pseudo-transient continuation state, persisting across calcCorrection
    // calls (one PTC step per call; the driver's outer loop grows dt). Global multiplier on top of
    // the per-level dt_i ~ radiative time; ramped by Deuflhard's ZIB-02-14 second-order rule.
    double ptc_dt_ = -1.0;              // global pseudo-timestep multiplier (<0 = uninitialised)
    double ptc_prev_fnorm_ = -1.0;      // previous residual norm (for the residual-fell test)
    std::vector<double> ptc_glin_;      // last step's linear-model prediction G_lin = G + J s (Deuflhard ZIB-02-14)
    double ptc_glin_norm_ = -1.0;       // ||G_lin|| from the last step
    double ptc_num_ = 0.0;              // |<G_lin, G - G_lin>| from the last step (ramp numerator)
};


}
#endif
