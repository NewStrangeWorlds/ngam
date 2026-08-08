#ifndef _temperature_correction_h
#define _temperature_correction_h

#include <functional>
#include <vector>

#include "../atmosphere/atmosphere.h"

namespace ngam {

struct RadiativeTransferOutput;
class OpacityCalculation;


// Forward-model flux evaluation: given a full trial temperature profile, run the forward model
// (chemistry, structure, opacity, radiative transfer WITHOUT the Jacobian) and return the per-level
// net flux. Supplied by the iteration loop so an affine-covariant (error-oriented) damped-Newton
// corrector can evaluate the TRUE residual at trial points for its natural monotonicity test.
using ForwardFluxEval =
  std::function<void(const std::vector<double>& temperature, std::vector<double>& net_flux)>;


// Full forward-model evaluation for the clima-style RCE corrector. Given a trial temperature profile,
// run the forward model and return BOTH the per-level net flux (flux_total) and DISORT's own flux
// divergence (net_heating). `recompute_opacity` selects clima's operator split: TRUE recomputes
// chemistry+structure+opacity (the true/base/trial residual, matching clima recomputing composition and
// opacity), FALSE freezes all of them (the finite-difference Jacobian columns = Planck-only response).
// No radiative-transfer Jacobian is computed (the corrector builds its Jacobian by finite differences).
using ForwardEvalFull =
  std::function<void(const std::vector<double>& temperature, bool recompute_opacity,
                     bool compute_jacobian,
                     std::vector<double>& flux_total, std::vector<double>& net_heating)>;


class TemperatureCorrection{
  public:
    virtual ~TemperatureCorrection() {}

    // Install the forward flux evaluation (see above). No-op for correctors that don't re-evaluate.
    virtual void setForwardFluxEval(ForwardFluxEval) {}

    // Install the full forward-model evaluation (see ForwardEvalFull). No-op unless the corrector uses it.
    virtual void setForwardEvalFull(ForwardEvalFull) {}

    // True if the corrector sets its own step size (so the loop must skip its outer per-iteration cap).
    virtual bool managesOwnStepSize() const { return false; }

    // True for correctors that slave convective zones INTERNALLY and self-converge. The driver must
    // then NOT run its own convective adjustment or Ng acceleration: both edit the committed profile
    // after the solve, which the corrector never sees -- the two fight and the residual floors.
    // Distinct from managesOwnStepSize(), which is only about per-iteration step limiting.
    virtual bool handlesConvectionInternally() const { return false; }

    // True for correctors that determine the SURFACE temperature as part of their own solve (a
    // surface-anchored Newton: T_surf is the bottom level, set by its F_net[0] row). The driver then
    // only mirrors it, and must not run the explicit surface model or the post-solve Shapiro filter.
    // Deliberately separate from handlesConvectionInternally(): the two happen to coincide today, but
    // they are different properties and conflating such predicates has already caused one regression.
    virtual bool solvesSurfaceTemperature() const { return false; }

    // Whether this corrector needs the radiative-transfer temperature Jacobians
    // (dJ/dT, dF/dT). When true the iteration loop must set
    // RadiativeTransferOutput::compute_jacobian before the RT solve.
    virtual bool requiresRadiationJacobian() const { return false; }

    // Convergence residual the corrector wants the loop to test (>= 0 to use it; < 0 means
    // "use the loop's own temperature-change metric"). The linearised corrector returns the
    // max radiative flux-conservation error excluding convective and optically-thin-skin
    // layers (set during the last calcCorrection call).
    virtual double lastConvergenceResidual() const { return -1.0; }

    virtual void calcCorrection(
      const double surface_gravity,
      Atmosphere& atmosphere,
      const RadiativeTransferOutput& radiation_field,
      const OpacityCalculation& opacity) = 0;
};


}
#endif
