#ifndef _helios_temperature_h
#define _helios_temperature_h

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#include "temperature_correction.h"


namespace ngam {


// HELIOS-style pseudo-time relaxation (Malik et al. 2017/2019, source/kernels.cu rad_temp_iter).
//
// This is NOT a physical time step. Each level carries its own step size s_i (in Kelvin) and is moved
// by
//     dT_i = s_i * sign(r_i) * |r_i|^q,     q = step_exponent (HELIOS: 0.1)
// where r_i is the level's dimensionless flux imbalance (the net-flux difference across the level,
// normalised by the transported flux). With q ~ 0.1 the step is nearly residual-blind: a deep layer
// with a tiny imbalance moves by about as many Kelvin per iteration as a thin one, which is what
// beats the stiffness between the optically thin skin and the diffusive deep atmosphere that
// throttles a physical (or local-timescale) explicit step.
//
// The per-level step adapts every adapt_interval iterations from the temperature drift over the
// interval (HELIOS' test): if |T_now - T_stored| < interval/2 * |dT| the level is oscillating and
// s_i /= step_shrink (HELIOS 1.5), otherwise it is progressing monotonically and s_i *= step_grow
// (HELIOS 1.1). Steps are capped at max_step Kelvin.
//
// stencil selects how the level's net-flux difference is formed. All three share the same root (a
// constant net flux) but only one converges (gas planet, 2026-09-05):
//   "backward" (F[i-1]-F[i], flux in from below minus flux out above: the level owns the slab
//              beneath it, HELIOS' own layer residual) -- converged in 307 iterations, deep profile
//              within 5 K of the ratio_ul Newton. DEFAULT.
//   "centered" (F[i+1]-F[i-1], the driver's flux_divergence) -- blind to the odd-even mode of the
//              Delta-tau >> 1 band; floors at ~1.3e-4 with a growing temperature checkerboard.
//   "forward"  (F[i]-F[i+1]) -- diverges: the first radiative level above a convective zone never
//              sees the flux jump out of the zone, and the near-sign-only update amplifies the
//              odd-even mode.
//
// residual selects the imbalance itself: "flux" (the net-flux difference across the level with the
// chosen stencil) or "heating" (HELIOS' actual layer residual transplanted to the collocated grid:
// the local radiative heating 4 pi Delta z_i int kappa (J - B) dnu, which contains the level's own
// Planck term and therefore sees the odd-even mode at any Delta tau). The convergence criterion is
// the flux criterion in both cases, so the "heating" root is reported with its true flux error.
//
// For objects with an internal flux (target_flux > 0) the bottom level is HELIOS' BOA ghost layer:
// its imbalance is (F_int - F_net[0]) / F_scale, the deep-flux anchoring rule.
//
// Convergence residual (lastConvergenceResidual): HELIOS' local radiative-equilibrium criterion,
// max over the non-convective levels of |F_net - F_int| / F_scale. The driver converges on this,
// never on the temperature change -- a small dT per iteration is a property of the step, not of
// the distance to equilibrium.
class HeliosTemperature : public TemperatureCorrection{
  public:
    HeliosTemperature(
      const double target_flux_,
      const double flux_scale_,
      const double step_init_,
      const double step_grow_,
      const double step_shrink_,
      const size_t adapt_interval_,
      const double step_exponent_,
      const double max_step_,
      const std::string& stencil_,
      const std::string& residual_type_)
      : target_flux(target_flux_)
      , flux_scale(flux_scale_ > 0 ? flux_scale_ : (target_flux_ > 0 ? target_flux_ : 1.0))
      , step_init(step_init_)
      , step_grow(step_grow_)
      , step_shrink(step_shrink_)
      , adapt_interval(adapt_interval_ < 2 ? 2 : adapt_interval_)
      , step_exponent(step_exponent_)
      , max_step(max_step_)
      , stencil(stencil_)
      , residual_type(residual_type_)
    {
      if (residual_type != "flux" && residual_type != "heating")
        throw std::invalid_argument("helios residual must be flux or heating, got " + residual_type);
      if (stencil != "centered" && stencil != "forward" && stencil != "backward")
        throw std::invalid_argument("helios stencil must be centered, forward or backward, got " + stencil);

      std::cout << "\n- Temperature correction: HELIOS pseudo-time stepping"
                << " (step_init " << step_init << " K, grow " << step_grow
                << ", shrink " << step_shrink << ", adapt_interval " << adapt_interval
                << ", exponent " << step_exponent << ", max_step " << max_step << " K, "
                << residual_type << " residual, " << stencil << " stencil)\n\n";
    }
    virtual ~HeliosTemperature() {}

    bool managesOwnStepSize() const override { return true; }
    double lastConvergenceResidual() const override { return residual; }

    void calcCorrection(
      const double surface_gravity,
      Atmosphere& atmosphere,
      const RadiativeTransferOutput& radiation_field,
      const OpacityCalculation& opacity) override;

  protected:
    const double target_flux;
    const double flux_scale;
    const double step_init;
    const double step_grow;
    const double step_shrink;
    const size_t adapt_interval;
    const double step_exponent;
    const double max_step;
    const std::string stencil;
    const std::string residual_type;

    std::vector<double> step;      // per-level step size [K]
    std::vector<double> T_store;   // profile at the start of the current adaptation interval
    size_t iteration = 0;
    double residual = -1.0;
};


}
#endif
