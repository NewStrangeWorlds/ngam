#ifndef _linearised_temperature_correction_h
#define _linearised_temperature_correction_h

#include "temperature_correction.h"

namespace ngam {

struct RadiativeTransferOutput;
class OpacityCalculation;
class Atmosphere;
class Convection;


// Full-linearisation radiative-convective-equilibrium temperature correction (PICASO t_start
// formulation). The Newton unknowns are the radiative levels; the contiguous deep convective zone
// [0, rcb_] is slaved to the adiabat and folded into the reduced flux-constancy system using
// DISORT's analytic net-flux temperature Jacobian (RadiativeTransferOutput::net_flux_jacobian).
// Per outer DISORT solve an inner Newton loop iterates the linearised flux F + J*dT to convergence
// with a backtracking line search; the optically-thin skin is set by direct local-RE inversion.
// See linearised_temperature_correction.cpp for the full description.
class LinearisedTemperatureCorrection : public TemperatureCorrection{
  public:
    LinearisedTemperatureCorrection(
      const double target_flux_,
      const double relaxation_,
      const double tau_scale_,
      const double max_change_fraction_,
      const Convection* convection_ = nullptr,
      const double flux_scale_ = 0.0,
      const bool surface_anchored_ = false)
      : target_flux(target_flux_)
      , relaxation(relaxation_)
      , tau_scale(tau_scale_)
      , max_change_fraction(max_change_fraction_)
      , convection(convection_)
      , flux_scale(flux_scale_)
      , surface_anchored(surface_anchored_) {}
    virtual ~LinearisedTemperatureCorrection() {}

    bool requiresRadiationJacobian() const override { return true; }
    void setForwardFluxEval(ForwardFluxEval f) override { forward_flux_eval_ = std::move(f); }
    // trust/NLEQ mode controls its own step size internally, so the loop must NOT apply its outer
    // per-iteration temperature cap (it would break the adiabat slaving). The PTC path still wants it.
    bool managesOwnStepSize() const override { return trust_active_; }
    bool handlesConvectionInternally() const override { return true; }
    bool solvesSurfaceTemperature() const override { return surface_anchored; }
    // self-luminous: the radiative flux-conservation error. Surface-anchored (terrestrial): the
    // troposphere can never reach radiative flux balance (it carries a finite radiative imbalance
    // by construction), so converge instead on the temperature change once the zone has settled.
    // trust mode is the exception: the troposphere is convective (slaved out of the Newton), so the
    // remaining radiative layers DO reach flux balance -- converge on that energy residual, which (
    // unlike max|dT/T|) a static checkerboard cannot fool into a false-converged, flux-imbalanced state.
    double lastConvergenceResidual() const override
    { return trust_active_ ? last_flux_residual_
           : ((surface_anchored && rcb_ >= 0) ? last_max_dt_frac_ : last_flux_residual_); }

    virtual void calcCorrection(
      const double surface_gravity,
      Atmosphere& atmosphere,
      const RadiativeTransferOutput& radiation_field,
      const OpacityCalculation& opacity) override;
  private:
    const double target_flux = 0.0;          // internal flux sigma*T_int^4 (>0 anchors the deep T)
    const double relaxation = 1.0;           // relaxation of the skin local-RE step
    const double tau_scale = 1.0;            // zeta = tau/(tau + tau_scale)
    const double max_change_fraction = 0.0;  // cap |dT|/T per step (<=0: no cap), no-flux path only

    // active convection scheme (dry/moist/...); supplies the convective gradient the corrector
    // enforces, so it stays agnostic to the scheme. nullptr -> no convection (radiative only).
    const Convection* convection = nullptr;

    // flux normalisation for the residual G = F_net / flux_scale (so target_flux = 0 works, e.g.
    // a terrestrial planet driving F_net -> 0). <=0 -> normalise by target_flux (gas planet / BD).
    const double flux_scale = 0.0;

    // surface-anchored convective zone (terrestrial): the troposphere is slaved to the deepest
    // level (lv 0 = the surface), which stays a flux unknown so its energy balance F_net[0]=0 is
    // enforced. Default (false) anchors the deep zone from ABOVE (gas planet / BD: entropy set by
    // radiative equilibrium above the photosphere).
    const bool surface_anchored = false;

    // max radiative flux-conservation error from the last call (excludes convective and
    // optically-thin-skin layers); the driver converges on this.
    double last_flux_residual_ = -1.0;

    // convection: the contiguous deep zone [0, rcb_] (-1 = none), grown only (never shrunk or
    // fragmented) and only once the inner Newton has settled at near-radiative-equilibrium, so the
    // boundary lands at the true radiative-convective boundary.
    std::vector<int> conv_set_;
    int rcb_ = -1;
    double last_max_dt_frac_ = 1e300;   // max |dT/T| applied last call (excl. skin); gates growth
    bool zones_grew_ = false;           // a zone grew this call -> not yet converged (hold residual)

    // pseudo-transient continuation: the pseudo-time dt (small -> implicit time step, robust; large
    // -> Newton, fast). Adapted by Deuflhard's second-order rule (ZIB report 02-14): the previous
    // step's linearly-predicted residual G_lin (per level) is compared to the now-measured residual
    // to estimate the local nonlinearity and set the optimal next dt; a perfect linear model drives
    // dt -> infinity (automatic switchover to Newton). ptc_rcb_at_step_ invalidates the history when
    // the convective zone has moved. Persists across the outer DISORT iterations.
    double ptc_dt_ = -1.0;
    double prev_fnorm_ = -1.0;
    std::vector<double> ptc_Glin_;     // predicted post-step residual G_lin = G + J*s (by level)
    double ptc_Glin_norm_ = 0.0;       // ||G_lin|| from the last step
    double ptc_num_ = 0.0;             // |(G_lin, G - G_lin)| from the last step
    double ptc_tau_used_ = -1.0;       // dt used in the last step
    int    ptc_rcb_at_step_ = -2;      // rcb_ at the last step (history valid only if unchanged)

    // trust-region (Levenberg-Marquardt) globalisation for the coupled surface-anchored solve
    // (env LIN_TRUST, prototype): solve (A + lambda*D) s = -G with lambda adapted by the measured
    // actual/predicted reduction ratio across DISORT solves (More' hybrj-style). A residual increase
    // raises lambda hard (a soft reject: heavy damping self-corrects without an explicit rollback),
    // so the target=0 near-null mode cannot diverge the way a plain Newton / undamped PTC does.
    // Persists across the outer DISORT iterations; history dropped when the convective zone moves.
    double tr_lambda_ = -1.0;          // LM damping (1/dt-like); large -> steepest descent, ->0 Newton
    double tr_fprev_ = -1.0;           // measured ||G|| (RMS) at the previous step
    double tr_pred_prev_ = 0.0;        // linearly-predicted reduction ||G|| - ||G+J*s|| last step
    int    tr_rcb_at_step_ = -2;       // rcb_ at the last step (history valid only if unchanged)
    bool   trust_active_ = false;      // set per call: trust (LIN_TRUST) path is in use

    // forward-model flux evaluation supplied by the iteration loop; lets the affine-covariant
    // (NLEQ-ERR) damped Newton evaluate the true residual at trial temperatures for its natural
    // monotonicity test. nullptr -> fall back to the single damped step (no trial re-evaluation).
    ForwardFluxEval forward_flux_eval_ = nullptr;
    // NLEQ-ERR damping-prediction history (affine-covariant): previous ordinary and simplified
    // Newton-correction scaled norms and the accepted damping factor.
    double nleq_norm_dx_prev_ = -1.0;
    double nleq_norm_dxbar_prev_ = -1.0;
    double nleq_lambda_prev_ = 1.0;

    // Anderson acceleration of the outer fixed-point map G: T_curr -> T_work (speeds the slow linear
    // tail of the NLEQ iteration). History of inputs x_k and outputs g_k=G(x_k); reset when the
    // convective mask moves (the map changes, so old history is invalid).
    std::vector<std::vector<double>> aa_x_, aa_g_;
    int aa_rcb_ = -2;
};


}
#endif
