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
    // mask_band_: dead-band half-width (in levels) for the convective-boundary limiter. The default 2
    // is the terrestrial choice, where the ratio residual is measurably insensitive to a +-1-level RCB
    // placement (Delta tau <~ 1 through the photosphere). SELF-LUMINOUS objects must pass 0: their
    // detached radiative band sits at Delta tau >> 1 per layer, where the collocated residual is
    // Nyquist-degenerate and a one-level placement error locks in a large-amplitude checkerboard
    // member of the root family (measured: flux error bar 3.6e-3 -> 1.1e-4 and band |d2T| 39 -> 13 K
    // when the mask reaches the detected top). Env CLIMA_MASKBAND still overrides for experiments.
    // use_ratio_: true  -> collocated ratio residual + full Uns\"old-Lucy (the recommended scheme)
    //             false -> heat-capacity-weighted flux-divergence residual (exact per-level flux
    //                      conservation, at the cost of a grid-scale checkerboard in the root)
    ClimaRCECorrection(
      const double target_flux_,
      const Convection* convection_,
      const int mask_band_ = 2,
      const bool use_ratio_ = true)
      : target_flux(target_flux_)
      , convection(convection_)
      , mask_band_default(mask_band_)
      , use_ratio(use_ratio_) {}
    virtual ~ClimaRCECorrection() {}

    // Uses DISORT's analytic Planck-only net-flux temperature Jacobian (= clima's frozen-opacity
    // Jacobian, in one RT solve instead of m finite differences), so the driver must compute it.
    bool requiresRadiationJacobian() const override { return true; }
    bool managesOwnStepSize() const override { return true; }              // trust region damps itself
    bool handlesConvectionInternally() const override { return true; }
    bool solvesSurfaceTemperature() const override { return true; }
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
    const int mask_band_default = 2;    // RCB dead-band half-width (see constructor note)
    const bool use_ratio = true;        // residual family (see constructor note)

    ForwardEvalFull forward_eval_full_ = nullptr;

    double last_residual_ = -1.0;       // max|weighted g| at the committed point (drives convergence)
    std::vector<int> prev_mask_;        // convective mask from the previous call (for the +-1 limiter)
    double last_inner_change_ = 1e300;  // max|dT/T| the last inner solve applied (settle gate for the mask)
    int rcb_cap_ = -1;                  // anti-overshoot cap on the convective top (clima Mode-3; -1 = none)
    int rcb_lock_ = 0;                  // lockout countdown after a cold-inversion shrink (prevents ABAB toggle)
    int rcb_grow_floor_ = -1;           // floor on the convective top after a raw-link promotion (-1 = none)
    int rcb_grow_lock_ = 0;             // lockout countdown after a promotion (holds the level against demotion)
    int rcb_last_promoted_ = -1;        // topmost level the raw-link growth check promoted (persists)
    int rcb_no_promote_ = -1;           // grow/retract cycle breaker: this level may not be promoted again

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
