#include "clima_rce_correction.h"

#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <functional>
#include <stdexcept>

#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>

#include "../atmosphere/atmosphere.h"
#include "../radiative_transfer/radiative_transfer.h"
#include "../additional/thermodynamic_data.h"
#include "../convection/convection.h"
#include "../transport_coeff/opacity_calc.h"
#include "../spectral_grid/spectral_grid.h"
#include "../additional/quadrature.h"
#include "../../_deps/disortpp-src/src/Planck.hpp"


namespace ngam {


// Dense linear solve A x = b by Gaussian elimination with partial pivoting (A, b overwritten;
// returns false if singular). Local copy so this corrector is self-contained.
static bool solveDenseLU(std::vector<double>& A, std::vector<double>& b, const size_t n)
{
  for (size_t k = 0; k < n; ++k)
  {
    size_t piv = k;
    double pmax = std::abs(A[k*n+k]);
    for (size_t r = k+1; r < n; ++r)
    {
      const double v = std::abs(A[r*n+k]);
      if (v > pmax) { pmax = v; piv = r; }
    }
    if (pmax == 0.0) return false;
    if (piv != k)
    {
      for (size_t c = k; c < n; ++c) std::swap(A[k*n+c], A[piv*n+c]);
      std::swap(b[k], b[piv]);
    }
    const double akk = A[k*n+k];
    for (size_t r = k+1; r < n; ++r)
    {
      const double f = A[r*n+k] / akk;
      if (f == 0.0) continue;
      for (size_t c = k; c < n; ++c) A[r*n+c] -= f * A[k*n+c];
      b[r] -= f * b[k];
    }
  }
  for (size_t ii = n; ii-- > 0; )
  {
    double s = b[ii];
    for (size_t c = ii+1; c < n; ++c) s -= A[ii*n+c] * b[c];
    b[ii] = s / A[ii*n+ii];
  }
  return true;
}


// Bracketed local radiative-equilibrium inversion at one level: find T with
//   sum_nu w_nu kappa_abs(nu,i) B_nu(T) = num   (= sum_nu w_nu kappa_abs J = absorbed),
// i.e. emitted == absorbed. The left side is STRICTLY MONOTONE in T, so bisection cannot overshoot --
// the key property for the optically-thin SW-heating top, where a Newton step DeltaT = -Q/Q' has an
// O(1) numerator (ozone/stellar heating, fixed by the incident flux) over a kappa->0 denominator
// (the radiative restoring slope ~ -4pi kappa_P dB/dT) and, on the convex Wien-tail Planck, limit-
// cycles. Bracketing replaces the overshooting tangent extrapolation; the kappa-magnitude cancels in
// the root, so the inversion returns a large-but-FINITE T monotonically instead of a divergent stride.
static double solveLocalREBracket(
  const double num,
  const std::vector<double>& w,
  const std::vector<double>& wavenumber,
  const OpacityCalculation& opacity,
  const size_t level,
  const double T_init,
  const double Tmax)
{
  if (num <= 0.0) return T_init;                 // no absorption / no incident -> keep
  constexpr double si_to_cgs = 1e3;
  const size_t nb_nu = wavenumber.size();
  auto emitted = [&](double T) {
    double den = 0.0;
    for (size_t k = 0; k < nb_nu; ++k)
      den += w[k] * opacity.absorption_coeff[k][level]
           * disortpp::planckFunction2(wavenumber[k], wavenumber[k], T) * si_to_cgs;
    return den;
  };
  double Tlo = 1.0, Thi = Tmax;
  if (emitted(Tlo) >= num) return Tlo;           // num below the floor -> 1 K
  if (emitted(Thi) <= num) return Thi;           // num above the cap   -> Tmax (large but FINITE)
  for (int it = 0; it < 80; ++it)                // bisection: cannot overshoot a monotone root
  {
    const double Tm = 0.5 * (Tlo + Thi);
    if (emitted(Tm) < num) Tlo = Tm; else Thi = Tm;
    if (Thi - Tlo < 1e-4 * Tm) break;
  }
  return 0.5 * (Tlo + Thi);
}


// Functor wrapping the reduced residual + analytic Jacobian for Eigen's HybridNonLinearSolver
// (a C++ port of MINPACK hybrj). The residual/Jacobian evaluators are supplied as std::functions
// built inside calcCorrection (they close over the profile build, forward eval, slaving, etc.).
struct RceHybrjFunctor
{
  typedef double Scalar;
  enum { InputsAtCompileTime = Eigen::Dynamic, ValuesAtCompileTime = Eigen::Dynamic };
  typedef Eigen::VectorXd InputType;
  typedef Eigen::VectorXd ValueType;
  typedef Eigen::MatrixXd JacobianType;

  int m_ = 0;
  std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> resfn;
  std::function<void(const Eigen::VectorXd&, Eigen::MatrixXd&)> jacfn;

  int inputs() const { return m_; }
  int values() const { return m_; }
  int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const { resfn(x, fvec); return 0; }
  int df(const Eigen::VectorXd& x, Eigen::MatrixXd& fjac) const { jacfn(x, fjac); return 0; }
};


void ClimaRCECorrection::calcCorrection(
  const double surface_gravity,
  Atmosphere& atmosphere,
  const RadiativeTransferOutput& radiation_field,
  const OpacityCalculation& opacity)
{
  const size_t n = atmosphere.temperature.size();
  if (forward_eval_full_ == nullptr || n == 0) { last_residual_ = -1.0; return; }

  // ---- RAW JACOBIAN VERIFICATION (env CLIMA_JACTEST; run with max_iterations=1). Compares the analytic
  // Planck-only temperature Jacobians from DISORT / adding-doubling (net_flux_jacobian dF_net[i]/dT[j] and
  // net_heating_jacobian dNH[i]/dT[j]) against a CENTRAL, FROZEN-OPACITY finite difference over every level
  // -- raw RT Jacobian, no slaving / ceff / differencing. Opacity + sample wavenumbers are held fixed, so
  // the FD isolates the Planck-source response (the analytic's exact domain) and the opacity-sampling noise
  // cancels common-mode in F(T+dT)-F(T-dT). Any residual discrepancy is therefore a genuine analytic-
  // derivative error, not sampling. Writes /tmp/jac_*.txt + a stderr summary, then returns.

  // ---- mean adiabatic gradient over a level pair (scheme-agnostic) ---------------------------
  auto nablaAd = [&](size_t i, size_t j) -> double {
    if (convection == nullptr) return 0.0;
    return 0.5 * (
      convection->convectiveGradient(atmosphere.number_densities[i], atmosphere.temperature[i], atmosphere.pressure[i]) +
      convection->convectiveGradient(atmosphere.number_densities[j], atmosphere.temperature[j], atmosphere.pressure[j]));
  };

  // ---- MLT smooth convection (Convection::providesFlux(), doc/mlt_convection_design.md): the
  // convective flux F_c = C x^{3/2} on level links replaces the ENTIRE discrete-mask machinery --
  // no probe, no slaving, no boundary controller; every level is a free radiative DOF and
  // convection enters the residual as a smooth heating term (+ F_c in the flux-based rows). ------
  const bool mlt = (convection != nullptr) && convection->providesFlux();

  // CLIMA_TIKH=alpha: Tikhonov 4th-difference regularisation in the objective (tests A1/A2);
  // read at function scope: it unlocks the mlt+flux pairing (see the guard further down), the
  // deep-slaving block reads it, and the convergence gate at the end is mode-aware.
  static const double tikh_alpha = [] {
    const char* e = std::getenv("CLIMA_TIKH");
    return (e != nullptr) ? std::atof(e) : 0.0; }();
  // The historical mlt+flux stall is the diagonally-deficient pure-flux Newton (at x=0 the F_c
  // block vanishes) -- exactly the near-null-mode disease the Tikhonov objective regularisation
  // removes. With CLIMA_TIKH>0 the pairing is unlocked (experimental, "three defects" note
  // test A1: single-scheme configuration) and routed through the NLEQ solver.
  // (use_ratio equals the later `ratio` flag whenever no experimental residual mode is
  // selected; all those modes are constexpr-off.)
  const bool tikh_flux_mlt = mlt && !use_ratio && tikh_alpha > 0.0;

  // SETTLE GATE (used by the boundary logic below and the mask update further down): only move the
  // convective boundary once the inner Newton has SETTLED the profile at the current mask (clima
  // solves hybrj to convergence BEFORE moving the mask).
  constexpr double settle_tol = 2e-3;  // max|dT/T| below which the mask may be re-detected
  const bool have_prev = (prev_mask_.size() == n);
  const bool settled = !have_prev || (last_inner_change_ < settle_tol);

  // ---- convective mask (multi-zone). Detect the "desired" mask on a SMOOTHED probe copy, so a
  // transient checkerboard sawtooth near the RCB cannot masquerade as super-adiabatic and grow the
  // convective zone up into the (stably stratified) stratosphere -- a feedback that otherwise
  // destabilises the boundary. The smoothing is for DETECTION ONLY; the residual/Jacobian use the
  // true profile. ----------------------------------------------------------------------------------
  std::vector<int> desired(n, 0);
  if (convection != nullptr && !mlt)
  {
    Atmosphere probe = atmosphere;   // copy: do not disturb the real profile
    // 1-2-1 detection smoothing (default ON; CLIMA_MASKSMOOTH=0 disables): without it the probe places the
    // RCB too HIGH (the steep near-tropopause radiative profile reads super-adiabatic) -> the convective
    // adiabat overshoots cold and KINKS the tropopause. With it the RCB lands correctly; the only side
    // effect (a converged on-adiabat zone occasionally reading as NO convection -- a marginal flip) is
    // guarded just below. (Always on: the smoothing is what makes the RCB land correctly.)
    {
      std::vector<double>& Tp = probe.temperature;
      const std::vector<double> Tin = Tp;
      for (size_t i = 1; i+1 < n; ++i) Tp[i] = 0.25*Tin[i-1] + 0.5*Tin[i] + 0.25*Tin[i+1];   // 1-2-1 filter
    }
    convection->adjust(probe);
    desired = probe.convective;
    // MARGINAL-FLIP GUARD: a converged zone sits ON the adiabat, which the smoothed probe occasionally reads
    // as NO convection at all (desired all-zero) -- a transient flip that, unguarded, deletes the boundary
    // and drives the +-1 RCB toggle. If the probe finds (almost) nothing but a zone was established, keep
    // the previous mask this step (the flip is not a real loss of convection).
    {
      int des_cnt = 0; for (size_t i = 0; i < n; ++i) des_cnt += desired[i];
      int prev_cnt = 0; if (prev_mask_.size() == n) for (size_t i = 0; i < n; ++i) prev_cnt += prev_mask_[i];
      if (prev_cnt > 2 && des_cnt < prev_cnt/2) desired = prev_mask_;   // transient empty/halved flip -> hold
    }
    // the surface-connected troposphere must be anchored AT the surface (lv 0): if the first
    // atmospheric layer convects, the surface belongs to that zone (its DOF), else the surface and
    // its own troposphere decouple (a large surface/air temperature jump).
    if (n > 1 && desired[1]) desired[0] = 1;
    // FILL interior gaps in the surface-connected zone before any comparison: the convective adjustment
    // leaves marginal sub-adiabatic POCKETS inside the troposphere, but ngam's zones are CONTIGUOUS. An
    // unfilled `desired` makes `disagree` permanently large (every interior pocket counts) -> the dead
    // band never engages -> the boundary retreats +-1 forever and mask_changed resets the convergence
    // metric to 1.0 each step (the RCB toggle that blocks convergence). Extend through gaps <= mask_band
    // and fill solid up to the surface-connected top.
    {
      const int gap_fill = mask_band_default;
      int top = -1, gap = 0;
      for (size_t i = 0; i < n; ++i)
      {
        if (desired[i]) { top = static_cast<int>(i); gap = 0; }
        else if (++gap > gap_fill) break;
      }
      for (int i = 0; i <= top; ++i) desired[i] = 1;     // solid surface-connected zone [0, top]
    }

    // TOP OWNERSHIP (picaso): the probe may neither SHRINK nor EXTEND the deep zone's top -- that
    // boundary is owned exclusively by the raw-link controller below (growth check + anti-overshoot,
    // both acting only near convergence). Both probe failure modes are measured:
    //  * shrink votes: a slaved level sits exactly ON the adiabat, so the smoothed copy reads the
    //    converged zone as (sub-)neutral and flaps (Manabe-Wetherald run: desired_top 37 -> 27..29 ->
    //    37 every settle, disagree spiking past the dead band, the limiter shaving one level off the
    //    top per flap -- with re-promotion a permanent ABAB that blocks convergence). picaso's
    //    boundary is grow-only for exactly this reason (growup only, climate.py:1634).
    //  * extension votes: mid-run transients read super-adiabatic on the smoothed copy one level past
    //    the true boundary; the limiter added the level and the ratchet then kept it (moist N=200 run:
    //    top crept one past the vetted boundary and converged with |d2T| 5.75 vs 2.94, an inversion
    //    dip above the forced-cold top).
    // So: clamp the surface-connected desired run to exactly [0, prev_top]. The growth check re-adds
    // levels after verifying the raw link. Detached zones above a gap are untouched (nucleation stays
    // with the probe).
    if (have_prev && prev_mask_[0])
    {
      int prev_top = -1;
      for (int i = static_cast<int>(n)-1; i >= 0; --i) if (prev_mask_[i]) { prev_top = i; break; }
      for (int i = 0; i <= prev_top; ++i) desired[i] = 1;
      for (int i = prev_top+1; i < static_cast<int>(n) && desired[i]; ++i) desired[i] = 0;
    }

    // RAW-BOUNDARY-LINK GROWTH CHECK (picaso find_strat / clima Mode-3 hard invariant): the mask can
    // lock BELOW the true boundary with a cold spike at the first radiative level -- the smoothed
    // probe smears the spike into an apparent stable inversion (so `desired` never reaches it), and
    // even when the probe DOES vote one level up, a 1-level disagreement sits inside the dead band
    // and is swallowed (both measured on the moist terrestrial runs; the spike is a converged fixed
    // point). picaso is immune because its growth test reads ONLY the single UNSMOOTHED link just
    // above the CURRENT boundary (dtdp >= subad*grad_ad, climate.py:2649) and moves the boundary
    // directly. Port that here: test the raw link above the current MASK top (not the probe's vote)
    // against the adiabat and promote the level if it is clearly super-adiabatic. Successive links
    // are tested with the ADIABAT-CONTINUED top temperature (what the level will be once slaved),
    // not the corrupted raw value, so a promoted spike cannot poison the next link's test. At most
    // 2 promotions per settle (clima's boundary-shift limiter); gated on `settled` so a
    // half-converged radiative wish cannot grow the zone; skipped while the anti-overshoot lockout
    // holds (the two must not fight). The promotion bypasses the dead band via rcb_grow_floor_.
    // NEAR-CONVERGENCE GATE for the boundary controller (growth check below + anti-overshoot): move
    // the top only between (almost) fully CONVERGED solves -- picaso re-runs profile() to convergence
    // after every single-level move, clima solves hybrj to convergence before updating the mask. The
    // looser settle gate (dT/T < 2e-3) is NOT enough: mid-run settles still carry transiently
    // super-adiabatic links above the top, the growth check promoted them, and the ratchet then kept
    // them -- the moist N=200 run over-grew one level and converged with a worse profile (|d2T| 5.75
    // vs 2.94) than the trap it was fixing. The gate must sit well above the driver's convergence
    // threshold (1e-5) so the controller always vets the state before the driver can declare
    // convergence; a mask move resets the residual, so vetting happens at least once per fixed point.
    // 1e-3 balances safety and cost: the measured transient over-vote states sit at own_resid >~ 1e-2
    // (still blocked), while a 1e-4 gate made each promotion wait for a full polish (the
    // Manabe-Wetherald run needed ~50 iterations per boundary move and hit the iteration cap).
    const bool near_conv = last_residual_ >= 0.0 && last_residual_ < 1e-3;
    if (std::getenv("CLIMA_DBG"))
      std::fprintf(stderr, "  [grow?] near_conv=%d (resid=%.2e) have_prev=%d rcb_lock=%d rcb_grow_lock=%d\n",
                   (int)near_conv, last_residual_, (int)have_prev, rcb_lock_, rcb_grow_lock_);
    if (near_conv && have_prev && rcb_lock_ == 0)
    {
      int cur_top = -1;
      for (int i = static_cast<int>(n)-1; i >= 0; --i) if (prev_mask_[i]) { cur_top = i; break; }
      if (cur_top >= 0)
      {
        constexpr double grow_margin = 0.02;   // require 2% super-adiabaticity (picaso's nucleation threshold)
        constexpr int    max_grow    = 2;
        double T_top = atmosphere.temperature[cur_top];
        int grown = 0;
        for (int g = 0; g < max_grow && cur_top+1 < static_cast<int>(n); ++g)
        {
          if (cur_top+1 == rcb_no_promote_) break;   // grow/retract cycle breaker (see anti-overshoot)
          const double dlnP = std::log(atmosphere.pressure[cur_top]/atmosphere.pressure[cur_top+1]);
          if (dlnP <= 0.0) break;
          const double nabla = std::log(T_top/atmosphere.temperature[cur_top+1]) / dlnP;
          const double nabla_ad = nablaAd(cur_top, cur_top+1);
          if (std::getenv("CLIMA_DBG"))
            std::fprintf(stderr, "  [grow?] link L%d->L%d nabla=%.3f nabla_ad=%.3f -> %s\n",
                         cur_top, cur_top+1, nabla, nabla_ad,
                         (nabla_ad > 0.0 && nabla > (1.0 + grow_margin)*nabla_ad) ? "PROMOTE" : "stop");
          if (!(nabla_ad > 0.0 && nabla > (1.0 + grow_margin)*nabla_ad)) break;
          // continue the adiabat onto the promoted level for the next link's test
          T_top *= std::pow(atmosphere.pressure[cur_top+1]/atmosphere.pressure[cur_top], nabla_ad);
          ++cur_top;
          desired[cur_top] = 1;
          ++grown;
        }
        if (grown > 0)
        {
          rcb_grow_floor_ = cur_top;   // bypass the dead band: the promotion must reach the mask
          rcb_grow_lock_  = 5;         // hold against demotion while the Newton re-converges
          rcb_last_promoted_ = cur_top;
          rcb_cap_ = -1; rcb_lock_ = 0;
        }
      }
    }

    // ANTI-OVERSHOOT (clima Mode-3): the convective top is snapped to a grid level, so the slaved adiabat
    // can land one level too HIGH -- forcing the top onto the cold adiabat below the radiative profile that
    // would sit there, a sharp cold-inversion KINK at the tropopause (max-curv blew up to ~14 K at L16).
    // Cure exactly as clima: if the first radiative link ABOVE the convective top is a strong COLD INVERSION
    // (dlnT/dlnP << 0, well beyond a gentle stratospheric warming), the adiabat has overshot -> SHRINK the
    // top by one level. A LOCKOUT (rcb_lock_) then caps the top there for a few iterations so the probe
    // cannot immediately re-grow it (the ABAB toggle clima's lockout counter prevents).
    // NEAR-CONVERGENCE ONLY (clima solve-then-update): the trigger reads the profile only between
    // (almost) converged solves, like the raw-link growth check -- same near_conv gate. Firing on
    // transients is destructive now that the top ratchet holds probe-driven re-growth off: the first
    // Newton mega-steps (dT/T ~ 0.5) read as cold inversions, the trigger marched the top down 4
    // levels in 4 iterations, the perpetually re-armed lockout kept the growth check disabled, and
    // the misplaced mask converged with a 155 K hole at the first radiative level (measured on the
    // Manabe-Wetherald run).
    {
      int des_top = -1; for (int i = static_cast<int>(n)-1; i >= 0; --i) if (desired[i]) { des_top = i; break; }
      if (near_conv && des_top >= 1 && des_top+1 < static_cast<int>(n) && rcb_grow_lock_ == 0)
      {
        const double dlnP = std::log(atmosphere.pressure[des_top]/atmosphere.pressure[des_top+1]);
        if (dlnP > 0.0)
        {
          // SPIKE-PROOF: evaluate the inversion with the ADIABAT-CONTINUED top temperature (what the
          // top will be once slaved), not the current raw value. The continuation must be anchored at
          // the last TRUSTED level -- the CURRENT MASK top, whose T is slaved/converged -- and walked
          // up through every level to des_top: anchoring just one level down is not enough, because
          // when the probe votes the top ABOVE the corrupted level (des_top = spike+1), des_top-1 IS
          // the spike, the test reads a huge false cold inversion, and the lockout re-arms every call
          // -- a permanent standoff that also gates the raw-link growth check off (measured on the
          // Manabe-Wetherald run: rcb_lock frozen at 4 for 50 iterations). For an already-slaved
          // des_top the walk is empty and the raw value is used, as before.
          int t_anchor = des_top;
          if (have_prev)
          {
            int prev_top = -1;
            for (int i = static_cast<int>(n)-1; i >= 0; --i) if (prev_mask_[i]) { prev_top = i; break; }
            if (prev_top >= 0 && prev_top < des_top) t_anchor = prev_top;
          }
          double T_top = atmosphere.temperature[t_anchor];
          for (int mlev = t_anchor+1; mlev <= des_top; ++mlev)
            T_top *= std::pow(atmosphere.pressure[mlev]/atmosphere.pressure[mlev-1],
                              nablaAd(mlev-1, mlev));
          const double nabla = std::log(T_top/atmosphere.temperature[des_top+1])/dlnP;
          const double nabla_ad = 0.5*(
            convection->convectiveGradient(atmosphere.number_densities[des_top],   atmosphere.temperature[des_top],   atmosphere.pressure[des_top]) +
            convection->convectiveGradient(atmosphere.number_densities[des_top+1], atmosphere.temperature[des_top+1], atmosphere.pressure[des_top+1]));
          // Trigger threshold: with mask moves now confined to near-converged states (and the probe
          // barred from the top), this is the ONLY mechanism that can shave an over-extended top --
          // in particular the INITIAL mask, which over-reaches because the adiabat-to-isothermal init
          // profile reads super-adiabatic one or two levels past the true boundary (measured: the
          // moist runs kept a converged top one level high, an inversion dip of -0.06..-0.19*nabla_ad
          // above it, and |d2T| twice the vetted placement). So the threshold must catch mild
          // inversions: 0.2*nabla_ad. The grow (+2%) and retract (-20%) tests read the SAME link with
          // the SAME values at a converged state, so at most one can fire; an ABAB cycle is possible
          // only when no rest state exists between two levels (converged-after-grow inverts < -20%
          // AND converged-after-retract exceeds +102%), which rcb_no_promote_ breaks: a level whose
          // promotion the retract undid is not promoted again -- the retracted (old-quality) state
          // then stands, picaso-style preferences inverted because retract only fires on measured
          // inversion, not on marginality.
          constexpr double off_frac = 0.2;   // clima Mode-3 anti-overshoot trigger
          if (nabla < -off_frac*nabla_ad)             // cold inversion above the top -> overshoot
          {
            rcb_cap_  = des_top - 1;                   // retreat one level
            rcb_lock_ = 5;                   // lockout iterations after a cold-inversion shrink
            if (des_top == rcb_last_promoted_)         // undoing our own promotion -> cycle breaker
              rcb_no_promote_ = des_top;
          }
        }
      }
      if (rcb_lock_ > 0 && rcb_cap_ >= 0)              // hold the retreated boundary against re-growth
      {
        for (size_t i = rcb_cap_+1; i < n; ++i) desired[i] = 0;
        --rcb_lock_;
      }
      else { rcb_cap_ = -1; }
    }
  }

  // (settle gate: `settled` computed above the detection block. Holding the mask frozen while the
  // inner Newton works stops the boundary from jittering on a half-converged profile, which is what
  // feeds the checkerboard sawtooth just above the RCB.)

  // dead band: the true radiative-convective boundary sits between grid levels, so the discrete mask
  // would oscillate by +-1 layer forever (each move swinging the surface). Accept a small boundary
  // ambiguity: once the mask is within `mask_band` layers of the detected mask, stop moving it.
  const int mask_band = mask_band_default;   // set per object via the config/selector
  int disagree = 0;
  if (have_prev) for (size_t i = 0; i < n; ++i) disagree += (prev_mask_[i] != desired[i]);

  std::vector<int> mask(n, 0);
  if (have_prev) mask = prev_mask_;
  if (!have_prev)
  {
    // first call: the adiabat start is already convective in the troposphere, so adopt the FULL
    // detected convective region at once (no need to grow it one layer per iteration from scratch).
    mask = desired;
  }
  else if (settled && disagree > mask_band)
  {
    // +-1 boundary limiter toward the desired mask (clima's convective_max_boundary_shift=1): move
    // each zone boundary by at most one layer, but let a NEW zone NUCLEATE as a one-layer seed.
    std::vector<int> add(n, 0);
    for (size_t i = 0; i < n; ++i)
      if (desired[i] && !mask[i] && ((i > 0 && mask[i-1]) || (i+1 < n && mask[i+1])))
        add[i] = 1;
    for (size_t i = 0; i < n; )
    {
      if (desired[i])
      {
        size_t j = i; bool overlap = mask[i] != 0;
        while (j+1 < n && desired[j+1]) { ++j; if (mask[j]) overlap = true; }
        if (!overlap) add[i] = 1;        // seed the bottom of a new zone
        i = j+1;
      }
      else ++i;
    }
    for (size_t i = 0; i < n; ++i) if (add[i]) mask[i] = 1;
    std::vector<int> rem(n, 0);
    for (size_t i = 0; i < n; ++i)
      if (!desired[i] && mask[i] && ((i == 0 || !mask[i-1]) || (i+1 >= n || !mask[i+1])))
        rem[i] = 1;
    for (size_t i = 0; i < n; ++i) if (rem[i]) mask[i] = 0;
  }
  // ANTI-OVERSHOOT ENFORCEMENT (clima Mode-3): the cold-inversion shrink is a deliberate correction, NOT a
  // marginal flip, so it must bypass the dead band (which otherwise holds the over-extended top). While the
  // lockout is active, force the convective top down to rcb_cap_.
  if (rcb_lock_ > 0 && rcb_cap_ >= 0)
    for (size_t i = rcb_cap_+1; i < n; ++i) mask[i] = 0;
  // RAW-LINK PROMOTION ENFORCEMENT: like the shrink above, a promotion is a deliberate correction that
  // must bypass the dead band (a 1-2 level `desired` change never exceeds mask_band, so the limiter
  // alone would swallow it -- the promoted level has to reach the mask directly). Fill from the floor
  // down to the existing zone top (exactly the promoted levels), and hold for the lockout duration.
  if (rcb_grow_lock_ > 0 && rcb_grow_floor_ >= 0)
  {
    for (int i = std::min(rcb_grow_floor_, static_cast<int>(n)-1); i >= 0 && !mask[i]; --i) mask[i] = 1;
    --rcb_grow_lock_;
  }
  else { rcb_grow_floor_ = -1; }
  // force the surface into its troposphere (essential, not subject to the dead band): a convective
  // first layer means the surface is that zone's anchor, or surface and troposphere decouple.
  if (n > 1 && mask[1]) mask[0] = 1;

  // CLIMA_RCB_FORCE=<k>: DIAGNOSTIC ONLY -- hard-pin the convective top at level k (levels > k forced
  // radiative). The RCB is snapped to a grid level, so when the true boundary lies BETWEEN levels the
  // top level is forced fully convective or fully radiative; the resulting boundary flux mismatch shows
  // up as a residual on the first radiative level above. This knob tests that reading by placing the
  // boundary by hand and watching whether that residual moves.
  if (const char* rf = std::getenv("CLIMA_RCB_FORCE"))
  {
    const int ktop = std::atoi(rf);
    for (size_t i = 0; i < n; ++i) mask[i] = (static_cast<int>(i) <= ktop) ? 1 : 0;
  }

  // ---- FIX A (hybrid MLT / deep slaving; "three defects" follow-up). For a SELF-LUMINOUS
  // flux-scheme MLT run, the deep convective interior is stiff in SHAPE and blind in LEVEL:
  // at the converged state F_c = C x^{3/2} = F* gives x = (F*/C)^{2/3} ~ 1e-6 deep, so the
  // MLT rows pin nabla with dF_c/dnabla ~ (3/2)F*/x while the uniform-lnT level mode is
  // sensed only by the (noise-floor) radiative response -- measured: a +5 K uniform shift on
  // the convective interior changes the emergent flux by 1.8e-4 F*, CONSTANT above the zone,
  // i.e. visible only to a block-SPANNING row. The mask path's zone row was that row. So:
  // slave the deep interior (eps below CLIMA_MLT_DEEPEPS, default 3e-4 -- orders of magnitude
  // above the ~1e-6 bottom and below the ~1e-2 RCB values, so the placement cannot flap) to
  // the adiabat via the EXISTING zone machinery, whose budget row spans the block; keep MLT
  // free above (everything MLT bought -- the C1 handover -- is at the RCB, not at 100 bar).
  // The handover mismatch is O(eps) per scale height: no seam at this threshold. Hysteresis:
  // +-1-level criterion drift keeps the previous top (mask_band = 0 for self-luminous, so an
  // unhysteresed flip would reset the convergence metric every iteration).
  // Scope: the flux+MLT pairing (tikh_flux_mlt) needs this to be solvable at all; the ratio
  // scheme's den-diagonal already pins the deep level mode, so the validated default is left
  // untouched (opt-in via CLIMA_MLT_DEEPSLAVE for experiments).
  if (mlt && target_flux > 0.0 && n >= 5
      && (tikh_flux_mlt || std::getenv("CLIMA_MLT_DEEPSLAVE") != nullptr))
  {
    static const double deep_eps = [] {
      const char* e = std::getenv("CLIMA_MLT_DEEPEPS");
      return (e != nullptr) ? std::atof(e) : 3e-4; }();
    const double alpha_mlt_d = std::max(1e-3, convection->fluxAlpha());
    int k_top = -1;
    for (size_t li = 0; li + 1 < n; ++li)
    {
      // UNRAMPED flux-law prefactor C (mirrors the mlt_C construction below, sf excluded):
      // the criterion asks where the CONVERGED x = (F*/C)^{2/3} is tiny, independent of the
      // easy-start homotopy state.
      const double rho = 0.5*(atmosphere.mass_density[li] + atmosphere.mass_density[li+1]);
      const double Tm  = 0.5*(atmosphere.temperature[li]  + atmosphere.temperature[li+1]);
      const double Hp  = 0.5*(atmosphere.scale_height[li] + atmosphere.scale_height[li+1]);
      const double cp  = 0.5*(
        ThermodynamicData::meanHeatCapacity(atmosphere.number_densities[li],   atmosphere.temperature[li]) +
        ThermodynamicData::meanHeatCapacity(atmosphere.number_densities[li+1], atmosphere.temperature[li+1]));
      if (Hp <= 0.0) break;
      const double lam = alpha_mlt_d*Hp;   // self-luminous: no wall law (fluxBlackadarSurface)
      const double C_full = 0.5 * rho * cp * Tm * (lam*lam/Hp)
                          * std::sqrt(std::max(surface_gravity, 0.0)/Hp);
      if (C_full <= 0.0) break;
      const double eps_conv = std::pow(target_flux / C_full, 2.0/3.0);
      if (eps_conv < deep_eps) k_top = static_cast<int>(li) + 1;
      else break;
    }
    std::vector<int> deep_mask(n, 0);
    if (k_top >= 2)
      for (int i = 0; i <= k_top; ++i) deep_mask[i] = 1;
    // hysteresis: keep the previous top for a +-1 criterion drift
    if (have_prev)
    {
      int prev_top = -1;
      for (int i = static_cast<int>(n)-1; i >= 0; --i) if (prev_mask_[i]) { prev_top = i; break; }
      if (prev_top >= 2 && std::abs(prev_top - k_top) == 1)
      { deep_mask.assign(n, 0); for (int i = 0; i <= prev_top; ++i) deep_mask[i] = 1; }
    }
    mask = deep_mask;
  }

  int mask_shift = 0;
  if (have_prev) for (size_t i = 0; i < n; ++i) mask_shift += (mask[i] != prev_mask_[i]);
  const bool mask_changed = (mask_shift > 0);
  // A SMALL boundary toggle (<= mask_band layers) is the expected +-1 RCB ambiguity (the true boundary
  // sits between grid levels, and a converged on-adiabat zone reads as marginally (un)stable from step to
  // step). It must NOT reset the convergence metric to 1.0 -- otherwise the model never converges even at
  // a tiny, smooth flux residual. Only a LARGE mask change (a real zone move) invalidates the residual.
  const bool mask_big_change = (mask_shift > mask_band);
  (void) disagree;
  prev_mask_ = mask;
  atmosphere.convective = mask;   // expose the frozen mask (driver prints N_conv from it)

  // ---- contiguous convective zones; each is one DOF (its bottom level) -------------------------
  struct Zone { size_t lower, upper, dof; };
  std::vector<Zone> zones;
  for (size_t i = 0; i < n; )
  {
    if (mask[i]) { size_t j = i; while (j+1 < n && mask[j+1]) ++j; zones.push_back({i, j, i}); i = j+1; }
    else ++i;
  }

  // ---- adiabat slaving: each zone's interior follows its DOF level on the (frozen) adiabat -----
  std::vector<char>   slaved(n, 0);
  std::vector<size_t> anchor(n, 0);
  std::vector<double> Cfac(n, 1.0);
  std::vector<int>    zone_of_dof(n, -1);
  for (size_t zi = 0; zi < zones.size(); ++zi)
  {
    const Zone& z = zones[zi];
    zone_of_dof[z.dof] = static_cast<int>(zi);
    anchor[z.dof] = z.dof;
    double C = 1.0;
    for (size_t k = z.lower+1; k <= z.upper; ++k)
    {
      C *= std::pow(atmosphere.pressure[k] / atmosphere.pressure[k-1], nablaAd(k, k-1));  // < 1 (cooler aloft)
      slaved[k] = 1; anchor[k] = z.dof; Cfac[k] = C;
    }
  }
  auto adiabatSnap = [&](std::vector<double>& T) {
    for (size_t k = 0; k < n; ++k)
      if (slaved[k]) T[k] = std::max(1.0, T[anchor[k]] * Cfac[k]);
  };

  // ---- DEEP CARVE-OUT (doc Sec.6.8): the optically-thick radiative deep is set by direct gradient
  // integration, NOT the deficient-Jacobian Newton. Identify it here (Planck-mean layer deepness, contiguous
  // from the surface) and mark it 'deep' so it is excluded from the Newton DOF set; it is reconstructed by the
  // gradient integration after the step. Only the photosphere/top/surface and convective zones remain DOFs.
  constexpr bool carve_deep = false;
  std::vector<char> deep(n, 0);
  int kdeep = -1;                                                  // top level of the carved deep zone (-1 = none)
  std::vector<double> deep_dtau(n, 0.0);                          // Planck-mean optical thickness of layer (i,i+1)

  // ---- active DOFs (every level that is neither slaved nor carved-out deep) ---------------------
  std::vector<size_t> unk;
  for (size_t i = 0; i < n; ++i) if (!slaved[i] && !deep[i]) unk.push_back(i);
  const size_t m = unk.size();
  if (m == 0) { last_residual_ = 0.0; return; }
  if (std::getenv("CLIMA_DBG"))
  {
    int des_top = -1; for (int i = static_cast<int>(n)-1; i >= 0; --i) if (desired[i]) { des_top = i; break; }
    std::fprintf(stderr, "  [zones] m=%zu zone0=[%zu,%zu] desired_top=%d settled=%d disagree=%d band=%d\n",
                 m, zones.empty()?0:zones[0].lower, zones.empty()?0:zones[0].upper,
                 des_top, (int)settled, disagree, mask_band);
  }

  // ---- heat capacity c_eff = (dP/g)*c_p per level (frozen), and per-zone sum -------------------
  std::vector<double> ceff(n, 1.0);
  for (size_t i = 0; i < n; ++i)
  {
    const double p_lo = (i+1 < n) ? 0.5*(atmosphere.pressure[i]+atmosphere.pressure[i+1]) : atmosphere.pressure[i];
    const double p_hi = (i > 0)   ? 0.5*(atmosphere.pressure[i]+atmosphere.pressure[i-1]) : atmosphere.pressure[i];
    const double dP = std::abs(p_lo - p_hi) * 1e6;
    const double cp = ThermodynamicData::meanHeatCapacity(atmosphere.number_densities[i], atmosphere.temperature[i]);
    ceff[i] = std::max(1e-30, dP / std::max(surface_gravity, 1e-30) * cp);
  }
  std::vector<double> ceff_zone(zones.size(), 1.0);
  for (size_t zi = 0; zi < zones.size(); ++zi)
  {
    double s = 0.0;
    for (size_t k = zones[zi].lower; k <= zones[zi].upper; ++k) s += ceff[k];
    ceff_zone[zi] = std::max(1e-30, s);
  }

  // ---- residual formulation. DEFAULT: clima's heat-capacity-weighted flux DIFFERENCES (the heating-rate
  // divergence (F[i]-F[i-1])/c_eff). CLIMA_NETFLUX switches the radiative-layer / surface / zone residual
  // to the per-level NET FLUX  F_net[i] -> 0, scaled by the incident stellar flux -- the SAME formulation
  // the self-luminous brown-dwarf/gas-planet path uses (there F_net -> F_int != 0; here the target is 0,
  // but the streams are O(OLR) so it stays well-scaled). The net flux is a SAME-LEVEL difference of the two
  // streams (F_up - F_down): their opacity-sampling noise is correlated and cancels. The divergence is an
  // ADJACENT-LEVEL difference that cancels the smooth signal and EXPOSES the decorrelated sawtooth noise,
  // through a Jacobian that is near-singular in exactly those modes. Both residuals share the zero set
  // (F_net = const = 0  <=>  div F_net = 0) but the net-flux form is far better conditioned -> no sawtooth,
  // so the Shapiro filter is not needed (defaulted off below when this mode is on).
  constexpr bool netflux = false;
  // Flux scale for the flux-type rows. The incident (TOA downward) flux is the natural scale for an
  // IRRADIATED object, but it is ~0 for a self-luminous one -- there the scale is the internal flux
  // F_star = sigma T_int^4. Taking the max covers both and reduces to the old behaviour whenever an
  // incident flux dominates. Without this the self-luminous flux rows are effectively unnormalised
  // (Fnorm falls back to 1) and the residual is reported in raw erg/cm^2/s.
  const double Fnorm = std::max({1.0, std::abs(target_flux),
    radiation_field.flux_down_total.empty() ? 0.0 : std::abs(radiation_field.flux_down_total.back())});

  // CLIMA_CENTERED: use a SYMMETRIC (control-volume) divergence g_i = 0.5*(F[i+1]-F[i-1])/c_eff for the
  // radiative layers instead of the one-sided backward g_i = (F[i]-F[i-1])/c_eff. This is the proper
  // finite-volume balance for a level-centred T[i] (top face minus bottom face). It also has a sin(k)
  // Fourier response -> ZERO at the Nyquist (sawtooth) frequency, so it does not amplify the sampling-
  // noise sawtooth into the residual. RISK: the Nyquist mode is then in the residual's NULL space (odd-
  // even decoupling), so a seeded sawtooth cannot be corrected. The top radiative level (no i+1) falls
  // back to the one-sided form. Mutually exclusive with CLIMA_NETFLUX.
  constexpr bool centered = false;

  // CLIMA_LOCALRE: use DISORT's NATIVE per-layer heating  g_i = net_heating[i]/c_eff  for the radiative
  // levels (with net_heating_jacobian), instead of any difference of the LEVEL fluxes. net_heating =
  // sum_nu w_nu dF_net/dtau is the LOCAL radiative heating sum_nu w_nu kappa(B-J) -- so net_heating=0 IS
  // the local radiative-equilibrium condition B(T_i)=<J>_kappa. Being local (no adjacent-level flux
  // differencing) it should neither excite nor null the sawtooth, AND it pins the optically-thin top
  // through the local Planck term B(T_i) (unlike the cumulative net flux, which collapsed it). Surface
  // and convective zones keep their genuine flux conditions. Mutually exclusive with NETFLUX/CENTERED.
  constexpr bool localre = false;

  // CLIMA_NEWTONLIKE: Deuflhard "Newton-like" method (book sec 2.1.3): the ACCURATE flux-conservation
  // residual (heat-capacity-weighted flux-moment divergence (F[i]-F[i-1])/c_eff -- the genuine F=const
  // condition, clima's residual) driven by a DELIBERATELY APPROXIMATE, diagonally-DOMINANT step Jacobian
  // M = net_heating_jacobian (NHJ) instead of the true but DEFICIENT differenced-net-flux Jacobian. NHJ
  // carries the local Planck diagonal -4pi(1-omega)dB/dT, so it regularises the checkerboard the deficient
  // F' leaves null -> the iteration stays smooth and converges to the smooth flux-conserving solution
  // (the sawtooth is never excited: g of a smooth profile has ~no Nyquist content and NHJ^-1 doesn't
  // amplify it). This uses DISORT/AD's accurate flux in the RESIDUAL while replacing its deficient dense
  // Jacobian with the dominant NHJ for the STEP -- exactly clima's structure (two-stream dominant Jacobian
  // + flux-divergence residual). Globalised by NLEQ-ERR. CLIMA_NL_NETHEAT switches the residual to the
  // solver's own net_heating (J-B form) for the residual-comparison experiment.
  constexpr bool newtonlike = false;
  constexpr bool nl_netheat = false;

  // CLIMA_HSTEP: the diffusion-limit doc's Sec.7.4 configuration -- the DEFECT CORRECTION done properly.
  // The RESIDUAL stays the accurate multi-stream flux-conservation statement (the DEFAULT differenced form
  // (F[i]-F[i-1])/c_eff -- untouched, so the converged root is exactly the multi-stream RCE); only the STEP
  // operator moves from the DEFICIENT differenced net-flux Jacobian (NFJ[i]-NFJ[i-1], whose odd-moment
  // self-cancellation leaves the checkerboard null mode that floors the RCB residual) to the COLLOCATED
  // HEATING Jacobian NHJ*dtau, which carries the local Planck-cooling diagonal and is diagonally dominant.
  // This is what the doc means by "retire the sawtooth by moving the step to the collocated heating
  // operator, which the multi-stream backend already supplies" -- the tridiagonal two-stream H_2s is only
  // the O(n) cost-reduced form of this same operator, not a prerequisite.
  //   NOTE this is NOT CLIMA_NEWTONLIKE: that one ALSO replaced the residual (NH*dtau), making it a
  //   local-RE solve. Here the residual remains the genuine flux-conservation condition.
  // Sub-flags for bisecting the Sec.7.4 row recipe:
  //   CLIMA_HSTEP_SURF : surface row <- the closed-form emission response 4 eps sigma T0^3 (a pure
  //                      diagonal) instead of the dense NFJ[0] row.
  //   CLIMA_HSTEP_ZONE : convective-zone row <- the telescoped SUM of the heating rows in the block
  //                      (F_upper - F_lower-1 = sum_i heating_i), instead of the NFJ difference.
  constexpr bool hstep = false;
  // Layer/level offset knob. DEFAULT 0 (collocated, no shift): the backend computes
  //   flux_divergence[l] = 4pi(1-omega)*(J[l] - B[l])
  // which is genuinely COLLOCATED at level l -- the "each interface uses the layer above"
  // convention selects only which omega is used (a smooth O(1) prefactor), NOT the collocation.
  // So net_heating[k] describes LEVEL k. Shifting it moves the Planck-cooling term off the
  // diagonal and destroys dominance; -1 was tried and is worse. Kept only as a diagnostic knob.
  constexpr bool hstep_surf = false;
  constexpr bool hstep_zone = false;

  // CLIMA_PTC: the UNIFIED construction (doc "The unified construction"). ONE moment-flux residual
  // everywhere, g_i = (F_net,i - target)/Fnorm + w_i g^RE_i, regularised by pseudo-transient continuation
  // -- the step solves (NFJ/Fnorm + C/dt + w_i d g^RE/dT) s = -g, with the heat-capacity diagonal C/dt
  // (dominant in the deep) sliding to the ratio Planck-slope diagonal w_i(-C/den) (the only T-sensitive
  // term as kappa->0) in the optically-thin top. No tau seam: the flux residual is one physical object top
  // to bottom; the ratio term is a SECOND regularisation of it (and carries the stellar BC via num_i, the
  // direct beam in J), active only where F_net goes flat. One PTC step per call; the outer loop grows dt.
  constexpr bool ptc = false;

  // CLIMA_PTC_BLEND: implement the doc's Eq. 19 SMOOTH sliding diagonal instead of the hard local-RE skin
  // overwrite. The hard split (overwrite for tau<tau_skin, pure flux+PTC below) leaves the handoff level
  // conditioned by NEITHER operator -> a localised seam collapse (the 63 K spike at the tau=1 boundary).
  // Eq. 19 puts the ratio term in BOTH residual and Jacobian with a smooth weight w_i = ptc_w[i] (->1 at
  // the optically-thin top, ->0 in the deep): g_i = (F_net-target)/Fnorm + w_i g^RE_i, with the Jacobian
  // gaining w_i d g^RE/dT (the Planck-slope diagonal that conditions the thin top continuously). No switch
  // surface, no overwrite -> no seam. The skin overwrite is disabled when this is on.
  constexpr bool ptc_blend = false;

  // CLIMA_PTC_DEEPINT (doc Sec.6): in the optically-thick radiative deep, replace the slow PTC relaxation by
  // a DIRECT GRADIENT INTEGRATION. There the net-flux Jacobian is banded (tridiagonal) and the flux
  // constrains the temperature GRADIENT, not T, so a forward substitution that drives F_net,i -> target
  // through the well-conditioned off-diagonal dF_i/dT_{i+1} is O(n) and flux-exact -- no Rosseland mean, no
  // diagonal dominance. Diffusion-limit only, so it is weighted by the local-layer deepness and fades out
  // through the photosphere. Matters for the convection-OFF case / thick radiative stratospheres.
  constexpr bool deep_integration = false;

  // CLIMA_LREANCHOR: with the collocated local-heating residual (CLIMA_LOCALRE), the heating Jacobian
  // is dense and has a near-null UNIFORM-temperature-level mode (local RE J=B is ~insensitive to a
  // uniform shift), so the level is pinned only by the surface row -> ill-conditioned (cond ~1e7) ->
  // huge Newton steps. Adding a weak un-differenced net-flux term eps*F[i]/Fstar to every radiative
  // residual restores a global energy-balance sensitivity that pins the DC mode WITHOUT differencing
  // (so it does not reintroduce the checkerboard). eps small keeps the local heating dominant.
  constexpr double lre_anchor = 0.0;

  // CLIMA_RATIO_RE: the ratio-form local radiative-equilibrium residual for the radiative levels,
  //   g^RE_i = num_i/den_i - 1,   num_i = sum_k w_k kappa_k,i J_k,i,   den_i = sum_k w_k kappa_k,i B(nu_k,T_i),
  // a TRUE Newton through the analytic numerator Jacobian meanint_kappa_jacobian (= d num_i/dT_j, opacity
  // frozen). The SAME kappa in num, den and C_i = d den_i/dT_i = sum_k w_k kappa_k,i dB/dT cancels in the
  // ratio, so the residual diagonal -C_i/den_i = -<dlnB/dT>_kappa ~ O(1/T) survives kappa->0 (no checkerboard,
  // no kappa->0 limit cycle that the bare net_heating residual CLIMA_LOCALRE suffered). Surface and convective
  // zones keep their genuine flux conditions (the global energy tether). The optional zeta-flux blend at depth
  // is a deliberate follow-up; this is the pure local-RE form (zeta=0). Mutually exclusive with the others.
  // DEFAULT MODE (2026-08-06). The ratio local-RE residual + the full Unsoeld-Lucy correction is now the
  // default for CLIMA_RCE, because it dominates the old flux-difference default on every measured axis at
  // n=200 (convection on, dry AND moist): 5x smoother (A_Nyq 0.021 vs 0.106 K), same surface temperature,
  // 11-15 iterations, TOA balanced to 0.01 W/m^2, and the committed profile is a genuine root of its own
  // system. The flux residual remains available (CLIMA_FLUX_RESID) and is still preferable when EXACT
  // per-level flux conservation matters more than smoothness: it reaches 1e-7 where this reaches 6e-5
  // (Sec.4's local-RE-vs-conservation gap, reduced but not removed by the Lucy term).
  // Selecting any other experimental mode (PTC/netflux/centered/localre/newtonlike) opts out automatically.
  const bool ratio_other_mode = (netflux || centered || localre || newtonlike || ptc);
  const bool ratio = use_ratio && !ratio_other_mode;

  // MLT convection is implemented for the ratio scheme only. The flux-difference plumbing below
  // (mltFcLvl in the rows, Jacobian and metric) EXISTS but the pairing is guarded off: measured
  // (2026-08-11, moist n=100), flux_divergence with all levels free stalls from a stable-side init
  // (hybrj info=5, zero steps -- with x=0 everywhere the F_c block vanishes and the system is the
  // documented diagonally-deficient pure-flux Newton), stalls from a marginally super-adiabatic
  // init, and WANDERS from a warm start at the converged ratio root (28 K drift, no convergence):
  // the F_c conditioning contribution scales as dF_c/dx ~ sqrt(eps) ~ 1e-2 of its bulk scale at
  // the root, far too weak to replace the mask path's zone rows and slaving, which were
  // load-bearing for this residual. flux_divergence therefore keeps requiring convection_type
  // dry/moist. The F_c terms in the flux-conservation METRIC remain active for ratio+mlt runs
  // (total-flux conservation is the physically meaningful diagnostic there).
  if (mlt && !ratio && !tikh_flux_mlt)
    throw std::runtime_error(
      "MixingLengthConvection (convection mlt_dry/mlt_moist) requires the ratio_ul temperature "
      "correction (measured: the flux_divergence pairing does not converge; use dry/moist "
      "there, or set CLIMA_TIKH>0 for the experimental regularised pairing)");

  // CLIMA_RATIO_LUCY: the FULL Unsoeld-Lucy correction, in the residual. What the corrector previously
  // called "UL" is only Lucy's rank-one uniform LEVEL shift, applied POST-solve -- which moves the profile
  // off the root the Newton just found (measured: solver ||g||inf 1.3e-7 -> 5.7e-4 after the shift). The
  // classical correction also carries a depth-dependent GRADIENT term, the cumulative flux-error integral.
  // From the first moment dJ/dtau = 3F/(4pi) (Eddington), a flux error propagates into a Planck error
  //     dB_i = (F_TOA - F*)/(2pi)            [level: the J(0)=2H(0) top boundary]
  //          + (3/4pi) * int_0^tau_i (F - F*) dtau'   [gradient: one-sided cumulative integral]
  // Putting this IN the residual (not applied afterwards) means the converged profile is a root of the
  // system that includes it -- nothing is applied post-hoc to break. And the flux enters as a ONE-SIDED
  // INTEGRAL, which has only the constant null mode and SUPPRESSES Nyquist content (~1/k), where the
  // per-level flux difference AMPLIFIES it (~k) -- the same "integrate, don't difference" move that fixed
  // the optically-thick deep. Parameter-free: the coefficients are the Eddington closure, not tuning.
  const bool lucy = ratio;        // the full Uns\"old-Lucy correction is intrinsic to the ratio scheme
  // Sign of the Uns"old-Lucy correction: the correction is MINUS the cumulative integral (too much
  // flux => too steep a gradient). Measured: +1 diverges (the convective zone runs away). Not a knob.
  constexpr double lucy_sign = -1.0;

  // CLIMA_THICK_SMOOTH: 4th-difference Nyquist penalty on the ratio rows of OPTICALLY-THICK radiative
  // levels (per-layer Planck-mean dtau > 1). Self-luminous objects have a detached radiative band with
  // dtau >> 1 per layer (measured 15-125 at P ~ 2-19 bar for the T_eff=1000 K brown dwarf at n=100),
  // where BOTH terms of the collocated residual lose their Nyquist response: the ratio term goes blind
  // (J -> B locally for ANY profile once the layer is thick -- H owns neither boundary) and the Lucy
  // cumulative integral suppresses period-2 content by 1/k. The checkerboard is then a NULL direction
  // of the residual: own_resid converges to 1e-6 with a +-1.8e-3 alternating flux error and a 39 K
  // |d2T| still present, and WHICH member of the degenerate family is reached is path-dependent
  // (fresh start vs restart land on different profiles). In the diffusion limit the true solution is
  // smooth on the grid scale, so a Nyquist-selective penalty there encodes real physics: the 4th
  // difference is ~0 for any smooth profile (unlike a 2nd difference it does not fight the real
  // adiabat-like curvature) and O(16a) for a checkerboard of amplitude a. Gating on dtau > 1 keeps it
  // OFF the photosphere and the optically-thin top, where the ratio term itself is well conditioned
  // and the terrestrial no-penalty result must be preserved (this term is inactive for the
  // terrestrial photosphere, whose layers sit at dtau <~ 1).
  // IN the residual AND the Jacobian (not post-hoc), so the committed profile stays a root of its own
  // system. Default ON in ratio mode; CLIMA_THICK_SMOOTH=0 disables, or sets the strength.
  // DEFAULT OFF (opt-in experiment) -- measured on the T_eff=1000 K brown dwarf, n=100:
  //  * lambda=0.3, unguarded stencil: destabilises the radiative-convective coupling (the penalty
  //    leaks into the zone anchor through the junction stencils, the mask detection flips, N_conv
  //    collapses 9 -> 1, no convergence).
  //  * lambda=0.05 with the full-stencil free-radiative guard below: stable and converged, but NO
  //    band improvement -- the guard excludes exactly the junction rows where the checkerboard
  //    peaks -- and the HARD dtau>1 gate books a new 41 K |d2T| seam at the gate boundary
  //    (P~0.07 bar). A future attempt needs a SMOOTH gate weight (e.g. dtau^2/(1+dtau^2)) and a
  //    junction-safe stencil; until then the dtau<~1 resolution rule (tau-adaptive grid) is the
  //    honest lever for the thick-band Nyquist degeneracy.
  constexpr double thick_smooth = 0.0;  // 4th-diff penalty on thick rows: tried, rejected (see doc)
  // full stencil must be free radiative (not surface, not slaved, not a zone DOF)
  auto thickSmoothOK = [&](size_t i) {
    if (!(thick_smooth > 0.0 && i >= 2 && i+2 < n)) return false;
    for (int dl = -2; dl <= 2; ++dl)
    { const size_t l = i + dl; if (l == 0 || slaved[l] || zone_of_dof[l] >= 0) return false; }
    return true;
  };

  // CLIMA_TRIDIAG: restrict the assembled radiative-level Jacobian rows to TRIDIAGONAL form (drop the
  // off-band global radiative coupling). The dense flux/divergence operator carries the sawtooth as a
  // NEAR-NULL (Nyquist) eigenvector -- global +-delta cancellation makes the flux blind to a period-2 T
  // oscillation, so the Newton step amplifies it (J^-1 has a huge eigenvalue there) and opacity-sampling
  // noise blows up into a checkerboard. A tridiagonal (discrete 2nd-difference-like) operator has the
  // Nyquist mode as its MOST-DAMPED eigenvector (eigenvalue ~ -4), so the step DAMPS the sawtooth instead.
  // This is the Olson-Auer-Buchler tridiagonal approximate-Lambda operator (ALI): an APPROXIMATE Jacobian,
  // so the ROOT (g=0 = true RCE) is unchanged -- only the iteration path avoids sawtooth space. The lost
  // global coupling = the absolute flux LEVEL, which is anchored separately (surface row + Unsoeld-Lucy /
  // ratio diagonal). Surface (i==0) and convective-zone rows stay DENSE (they ARE the global tether).
  constexpr bool tridiag = false;

  // per-level spectral sums for the ratio residual + its Jacobian, evaluated at a profile T against the
  // CURRENT radiation field (frozen opacity, so kappa = absorption_coeff matches the numerator Jacobian):
  //   num = sum_k w_k kappa_k,i J_k,i (cgs), den = sum_k w_k kappa_k,i B(nu_k,T_i) (cgs),
  //   C   = sum_k w_k kappa_k,i dB/dT(nu_k,T_i) (cgs) = d den/dT_i.
  const std::vector<double>& ratio_wn = radiation_field.spectral_grid->wavenumber_list;
  const std::vector<double> ratio_w = (ratio || ptc) ? aux::trapezoidalWeights(ratio_wn) : std::vector<double>();
  const size_t ratio_nnu = ratio_wn.size();
  constexpr double ratio_si_to_cgs = 1e3;

  // ---- BAND-CLOSURE CORRECTION (2026-08-12). The sampled points miss the Doppler-narrow line
  // cores that carry the Planck-mean emissivity aloft (measured: sampled den deficient by factors
  // 6-17 at p <= 1e-3 bar, grid-dependent -- the "lottery" that made the local-RE row demand
  // +100..400 K and blew up the high-resolution runs). opacity.band_correction[b][i] holds the
  // per-band NATIVE line-opacity integral minus its sampled representation; that missing opacity
  // exchanges with the BAND-MEAN radiation field:
  //   num += dk_b * Jbar_b,   den += dk_b * B(nu_b, T),   C += dk_b * dB/dT(nu_b, T).
  // Self-gating (dk -> 0 where sampling is complete, e.g. the troposphere at +-3%) and
  // self-cancelling in LTE (deep down Jbar -> B, so num and den corrections cancel exactly where
  // the sampled scheme was already right). 30 cm^-1 bands keep the kappa-B covariance error
  // below ~1 K equivalent (measured against the direct native integral, T = 150..500 K).
  // v1 SCOPE: thermally relevant bands only, x = hc nu/kT < 10 -- kappa B peaks at x ~ 3-6, so
  // this keeps ~98% of the emissivity correction while excluding every band with meaningful
  // STELLAR flux (diluted solar reaches parity with the local Planck function only at x >~ 10).
  // That exclusion is essential: in a sunlit near-IR band the mean radiation field is
  // window-dominated (the beam survives between the lines) while the true core J is
  // thermalised, so the Jbar closure would inject spurious beam heating -- measured with a
  // x < 25 cut as own_resid 85 at the DEEP levels (iteration 1), where the old scheme sat at
  // 26. The SW num therefore stays sampled (benign status quo). The num-side Jbar term is NOT
  // in the analytic Jacobian (MK stays sampled-only): an approximate Jacobian the NLEQ
  // monotonicity test tolerates; the dominant den/C corrections DO enter through these sums.
  // Kill switch: CLIMA_BANDCORR=0; cut override: CLIMA_BANDCORR_XCUT.
  static const bool bandcorr_env = [] {
    const char* e = std::getenv("CLIMA_BANDCORR");
    return !(e != nullptr && std::string(e) == "0"); }();
  static const double bandcorr_xcut = [] {
    const char* e = std::getenv("CLIMA_BANDCORR_XCUT");
    return (e != nullptr) ? std::atof(e) : 10.0; }();
  const bool bandcorr = (ratio || ptc) && bandcorr_env && !opacity.band_correction.empty();
  const size_t nb_cb = bandcorr ? opacity.band_correction.size() : 0;
  std::vector<int> cb_of_k;
  std::vector<double> cb_sumw;
  std::vector<std::vector<double>> cb_Jbar;   // FROZEN band-mean radiation field [level][band]
  std::vector<std::vector<double>> cb_eps;    // FROZEN core escape probability [level][band]
  if (bandcorr)
  {
    constexpr double cb_width = SpectralGrid::correction_band_width;
    cb_of_k.assign(ratio_nnu, -1);
    cb_sumw.assign(nb_cb, 0.0);
    for (size_t k = 0; k < ratio_nnu; ++k)
    {
      const size_t b = static_cast<size_t>(ratio_wn[k]/cb_width);
      if (b < nb_cb) { cb_of_k[k] = static_cast<int>(b); cb_sumw[b] += ratio_w[k]; }
    }
  }
  // ---- ESCAPE-PROBABILITY WEIGHT (v3, "three defects" note item 2). The missing cores'
  // thermalisation state cannot be inferred from any weighting of the SAMPLED points (v2's
  // kappa-weighted Jbar measured: inversion only partially restored, ~208 K vs the 255 K fd
  // target, because the dk>=0 clamp confines the closure to exactly the bands whose cores were
  // never sampled). It IS available from the native tables: the per-band PEAK opacity, stored at
  // sampling time (SampledData::band_peaks), gives the core optical depth to SPACE above each
  // level, tau_b,i = D * sum_{layers above} k_peak dz (D = 1.66 diffusivity). The closure is
  // then scaled per band by eps = exp(-tau): algebraically identical to the cooling-to-space
  // form Jbar_eff = (1-eps) B + eps Jbar (the (1-eps) B part cancels exactly between num and
  // den), but with den/C scaled consistently so the Jacobian stays exact. eps -> 0 where the
  // cores are thick to space: the closure self-cancels, the cancellation-protected IR-exchange
  // balance (and the ozone inversion) survives. eps -> 1 at the thin SW-heated top: the full
  // stabilising correction, exactly where the divergence disease lives. FROZEN per opacity
  // refresh (k_peak and altitudes at the committed state), like every other closure input.
  if (bandcorr && !opacity.band_peak_coeff.empty())
  {
    constexpr double diffusivity = 1.66;
    cb_eps.assign(n, std::vector<double>(nb_cb, 1.0));
    for (size_t b = 0; b < nb_cb; ++b)
    {
      double tau = 0.0;
      cb_eps[n-1][b] = 1.0;
      for (int i = static_cast<int>(n)-2; i >= 0; --i)
      {
        tau += diffusivity
             * 0.5*(opacity.band_peak_coeff[b][i] + opacity.band_peak_coeff[b][i+1])
             * std::abs(atmosphere.altitude[i+1] - atmosphere.altitude[i]);
        cb_eps[i][b] = (tau < 700.0) ? std::exp(-tau) : 0.0;
      }
    }
  }
  // The closure's band-mean radiation field Jbar is FROZEN over the inner solve -- the same
  // operator split as frozen opacity, one level up: with a live Jbar the closure's num term
  // responds to every trial's radiation field while the analytic Jacobian (MK = sampled-J
  // response only) knows nothing about it, and that inconsistency pinned the inner Newton at
  // lambda_min (measured on the covering-20000 grid: same entry residual 130, lambda 1e-4 vs
  // 0.1-0.8 uncorrected). Frozen, the closure is a CONSTANT during the solve -- its Jacobian is
  // exactly zero and residual/Jacobian stay a consistent pair by construction; the snapshot is
  // refreshed with the radiation field (below, before the final metric), so the lag vanishes at
  // the fixed point exactly like the opacity lag.
  //
  // KAPPA-WEIGHTED band mean (v2, "three defects" note): Jbar_b = sum w kappa J / sum w kappa,
  // not the flat mean. The missing cores share the thermalisation state of the band's STRONG
  // sampled points: where those are optically thick the kappa-weighted J -> B(T_i) and the
  // closure self-cancels per band (the cancellation-protected IR-exchange region -- ozone
  // inversion -- survives, where the flat window-dominated mean injected artificial
  // cooling-to-space and moved the 255 K stratopause to 170 K); at the SW-heated thin top the
  // strong points are thin-to-space and the closure keeps its full stabilising effect. A side
  // benefit: with Jbar ~ B in well-sampled bands the dk>=0 clamp inflates num and den EQUALLY,
  // so the clamp's den bias (measured 1.7x at 1 mbar, covering-20000) becomes inert for the
  // root. Bands whose sampling caught no line structure fall back to the flat mean -- the
  // estimator degrades exactly where the deficit is largest, which bounds what v2 can fix
  // (a genuine escape probability from per-band core statistics is the v3 completion).
  auto snapshotBandJ = [&]() {
    if (!bandcorr) return;
    cb_Jbar.assign(n, std::vector<double>(nb_cb, 0.0));
    for (size_t i = 0; i < n; ++i)
    {
      std::vector<double>& row = cb_Jbar[i];
      std::vector<double> sum_wk(nb_cb, 0.0);   // sum w kappa per band (weight normalisation)
      std::vector<double> sum_wj(nb_cb, 0.0);   // flat-mean fallback numerator
      for (size_t k = 0; k < ratio_nnu; ++k)
        if (cb_of_k[k] >= 0)
        {
          const int b = cb_of_k[k];
          const double wk = ratio_w[k] * opacity.absorption_coeff[k][i];
          row[b]    += wk * radiation_field.mean_intensity[i][k];
          sum_wk[b] += wk;
          sum_wj[b] += ratio_w[k] * radiation_field.mean_intensity[i][k];
        }
      for (size_t b = 0; b < nb_cb; ++b)
      {
        if (sum_wk[b] > 0.0)          row[b] /= sum_wk[b];
        else if (cb_sumw[b] > 0.0)    row[b] = sum_wj[b] / cb_sumw[b];
        else                          row[b] = 0.0;
      }
    }
  };
  snapshotBandJ();

  // CLIMA_BANDDUMP: one-shot anatomy of the band closure at the entry profile -- per-level
  // sampled vs correction contributions to num/den, plus the dominant bands at a deep level.
  if (bandcorr && std::getenv("CLIMA_BANDDUMP"))
  {
    static bool band_dumped = false;
    if (!band_dumped)
    {
      band_dumped = true;
      const std::vector<double>& T = atmosphere.temperature;
      std::fprintf(stderr, "  [banddump] lv P T | num_s dnum | den_s dden | gre_s gre_c\n");
      for (size_t i : {0ul, 1ul, 2ul, 10ul, 30ul, 50ul, 70ul, 90ul, 96ul, 99ul})
      {
        if (i >= n) continue;
        double num_s = 0.0, den_s = 0.0, dnum = 0.0, dden = 0.0;
        std::vector<double> sumJ(nb_cb, 0.0);
        for (size_t k = 0; k < ratio_nnu; ++k)
        {
          const double wk = ratio_w[k] * opacity.absorption_coeff[k][i];
          num_s += wk * radiation_field.mean_intensity[i][k];
          den_s += wk * disortpp::planckFunction2(ratio_wn[k], ratio_wn[k], T[i]) * ratio_si_to_cgs;
          if (cb_of_k[k] >= 0) sumJ[cb_of_k[k]] += ratio_w[k] * radiation_field.mean_intensity[i][k];
        }
        double worst = 0.0; size_t worst_b = 0;
        for (size_t b = 0; b < nb_cb; ++b)
        {
          const double dk = std::max(0.0, opacity.band_correction[b][i])
                            * (cb_eps.empty() ? 1.0 : cb_eps[i][b]);
          if (dk == 0.0 || cb_sumw[b] <= 0.0) continue;
          const double nu_b = opacity.band_wavenumber[b];
          if (1.4387769 * nu_b > bandcorr_xcut * std::max(T[i], 1.0)) continue;
          const double dn = dk * cb_Jbar[i][b];
          const double dd = dk * disortpp::planckFunction2(nu_b, nu_b, T[i]) * ratio_si_to_cgs;
          dnum += dn; dden += dd;
          if (std::abs(dn - dd) > worst) { worst = std::abs(dn - dd); worst_b = b; }
        }
        std::fprintf(stderr,
          "  [banddump] %zu %.3e %.1f | %.4e %+.4e | %.4e %+.4e | %+.4e %+.4e  worst_band nu=%.0f dk=%.3e\n",
          i, atmosphere.pressure[i], T[i], num_s, dnum, den_s, dden,
          num_s/den_s - 1.0, (num_s + dnum)/(den_s + dden) - 1.0,
          opacity.band_wavenumber[worst_b], opacity.band_correction[worst_b][i]);
      }
      // full-level pathology scan: corrected den or Planck slope C non-positive (a flipped
      // restoring diagonal poisons the whole affine-covariant inner solve), or extreme gre
      for (size_t i = 0; i < n; ++i)
      {
        double num_c = 0.0, den_c = 0.0, C_c = 0.0, den_s = 0.0, C_s = 0.0;
        std::vector<double> sumJ(nb_cb, 0.0);
        for (size_t k = 0; k < ratio_nnu; ++k)
        {
          const double wk = ratio_w[k] * opacity.absorption_coeff[k][i];
          num_c += wk * radiation_field.mean_intensity[i][k];
          den_c += wk * disortpp::planckFunction2(ratio_wn[k], ratio_wn[k], T[i]) * ratio_si_to_cgs;
          C_c   += wk * disortpp::planckFunctionDeriv2(ratio_wn[k], ratio_wn[k], T[i]) * ratio_si_to_cgs;
          if (cb_of_k[k] >= 0) sumJ[cb_of_k[k]] += ratio_w[k] * radiation_field.mean_intensity[i][k];
        }
        den_s = den_c; C_s = C_c;
        for (size_t b = 0; b < nb_cb; ++b)
        {
          const double dk = std::max(0.0, opacity.band_correction[b][i])
                            * (cb_eps.empty() ? 1.0 : cb_eps[i][b]);
          if (dk == 0.0 || cb_sumw[b] <= 0.0) continue;
          const double nu_b = opacity.band_wavenumber[b];
          if (1.4387769 * nu_b > bandcorr_xcut * std::max(T[i], 1.0)) continue;
          num_c += dk * cb_Jbar[i][b];
          den_c += dk * disortpp::planckFunction2(nu_b, nu_b, T[i]) * ratio_si_to_cgs;
          C_c   += dk * disortpp::planckFunctionDeriv2(nu_b, nu_b, T[i]) * ratio_si_to_cgs;
        }
        const double gre_c = (den_c > 0.0) ? num_c/den_c - 1.0 : -999.0;
        if (den_c <= 0.0 || C_c <= 0.0 || std::abs(gre_c) > 5.0 || den_c < 0.2*den_s || C_c < 0.2*C_s)
          std::fprintf(stderr,
            "  [bandscan] PATHOLOGY lv %zu P=%.3e T=%.1f den_s=%.3e den_c=%.3e C_s=%.3e C_c=%.3e gre_c=%+.3e\n",
            i, atmosphere.pressure[i], T[i], den_s, den_c, C_s, C_c, gre_c);
      }
      std::fprintf(stderr, "  [bandscan] scan complete\n");
    }
  }

  auto ratioSums = [&](const std::vector<double>& T, const size_t i, double& num, double& den, double& C,
                       double* num_sampled = nullptr)
  {
    num = den = C = 0.0;
    for (size_t k = 0; k < ratio_nnu; ++k)
    {
      const double wk = ratio_w[k] * opacity.absorption_coeff[k][i];
      const double wn = ratio_wn[k];
      num += wk * radiation_field.mean_intensity[i][k];
      den += wk * disortpp::planckFunction2(wn, wn, T[i])      * ratio_si_to_cgs;
      C   += wk * disortpp::planckFunctionDeriv2(wn, wn, T[i]) * ratio_si_to_cgs;
    }
    if (num_sampled != nullptr) *num_sampled = num;
    if (bandcorr)
      for (size_t b = 0; b < nb_cb; ++b)
      {
        // dk CLAMPED to >= 0 (v1, OPT-IN): only MISSING opacity is closed. Negative dk (sampled
        // overcounting) is benign by num/den cancellation, and fed into the closure it becomes an
        // ANTI-restoring den/C term that drove a row diagonal through zero (+407 K Newton
        // component @lv55, trial residuals 1e74, covering-20000). A signed residual with a
        // dk>0-only C (dominant-Jacobian pattern) was ALSO tried: it pins lambda at the floor on
        // the covering grid -- some levels' signed correction is genuinely anti-restoring.
        // KNOWN v1 LIMITATION (why this is opt-in): the flat band-mean Jbar is window-dominated,
        // but the missing CORES are thick-to-space through most of the stratosphere and should
        // see ~B_local -- the closure therefore over-cools the IR-exchange region (measured:
        // the cancellation-protected ~255 K at 1 mbar moves to ~170 K on both test grids). The
        // closure is honest only where cores are thin-to-space (the SW-heated top -- exactly
        // where the divergence disease lives). v2 needs an escape-probability-weighted Jbar,
        // Jbar_eff = (1-eps)B + eps*Jbar_flat (cooling-to-space closure).
        const double dk = std::max(0.0, opacity.band_correction[b][i])
                          * (cb_eps.empty() ? 1.0 : cb_eps[i][b]);   // escape-probability weight (v3)
        if (dk == 0.0 || cb_sumw[b] <= 0.0) continue;
        const double nu_b = opacity.band_wavenumber[b];
        if (1.4387769 * nu_b > bandcorr_xcut * std::max(T[i], 1.0)) continue;   // thermal bands only (v1)
        num += dk * cb_Jbar[i][b];                   // FROZEN band-mean J (see snapshotBandJ)
        den += dk * disortpp::planckFunction2(nu_b, nu_b, T[i])      * ratio_si_to_cgs;
        C   += dk * disortpp::planckFunctionDeriv2(nu_b, nu_b, T[i]) * ratio_si_to_cgs;
      }
  };

  // cumulative flux proxy cumF_i = int_0^i (num-den) dz (trapezoidal in altitude from the surface i=0):
  // a reconstruction of the net-flux DEVIATION whose Jacobian is the cumulative heating Jacobian (M-C),
  // self-consistent with the ratio term (Section: AGB construction). Used by BOTH the residual and the
  // Jacobian so the Newton is exact (the earlier actual-flux residual + cumulative Jacobian was
  // inconsistent -> oscillation). den<=0 levels contribute 0.
  auto cumFvec = [&](const std::vector<double>& T, std::vector<double>& cumF) {
    cumF.assign(n, 0.0);
    std::vector<double> S(n, 0.0);
    for (size_t k = 0; k < n; ++k) { double nu, de, C; ratioSums(T, k, nu, de, C); S[k] = nu - de; }
    for (size_t i = 1; i < n; ++i)
      cumF[i] = cumF[i-1] + 0.5*(S[i] + S[i-1])*std::abs(atmosphere.altitude[i]-atmosphere.altitude[i-1]);
  };

  // ---- zeta-flux blend (the composite residual): g_i = xi * g^RE_i + zeta_i * F_i/Fstar. The ratio
  // local-RE term g^RE drives dF/dz -> 0 (constant flux, smooth profile) but determines F only up to an
  // ADDITIVE CONSTANT (the homogeneous solution dF/dz=0 => F=const) -- it cannot see the absolute flux
  // level. The flux-constancy term g^F = F/Fstar DOES, so it removes that constant and enforces global
  // energy balance (target F=0). The depth weight zeta(tau)=tau/(tau+tau_scale) hands control over: ratio
  // owns the optically-thin top (zeta->0, where F is insensitive to local T and the ratio diagonal is the
  // only conditioning), flux owns the thick deep (zeta->1, where g^RE=heating/den goes blind to the level).
  // tau is the Planck-mean cumulative ABSORPTION optical depth from the top, frozen over the inner solve.
  constexpr bool ratio_flux = false;
  // default flux anchor = low-rank DEFLATION (driver); CLIMA_RATIO_PERLEVEL selects the old per-level
  // zeta-blend (kept for comparison; it fights the diagonal because the flux constraint is nonlocal).
  constexpr bool ratio_perlevel = false;
  constexpr double ratio_xi = 1.0;      // weight of the local-RE term; 1 = the scheme as documented
  constexpr double ratio_tauscale = 1.0;
  constexpr bool ratio_fluxdense = false;
  std::vector<double> ratio_zeta(n, 0.0), ratio_tau(n, 0.0);

  // FROZEN least-squares scale calibrating the cumulative proxy to the actual net-flux deviation:
  // F_i - F_0 ~ ratio_flux_scale * cumF_i (auto-handles the 4*pi and unit factors). Computed once at the
  // committed point so the residual (xi g^RE + zeta*flux_scale*cumF/Fnorm) and the cumulative-heating
  // Jacobian stay a consistent Newton through the inner relaxation.
  double ratio_flux_scale = 0.0;

  // Per-layer Planck-mean optical depth, to rescale the Newton-like step Jacobian. The accurate
  // conservation residual is the per-LAYER flux change F[i]-F[i-1] = net_heating[i]*dtau[i], but
  // net_heating_jacobian (NHJ) is the per-TAU Jacobian d(dF/dtau)/dT; multiplying NHJ by dtau[i] gives the
  // per-layer dominant surrogate of d(F[i]-F[i-1])/dT (matching the residual's scale, so the NLEQ-ERR step
  // is consistent -- without it the deep large-dtau layers are mis-scaled and the solver grinds).
  // CLIMA_DUMPTAU: dump the per-layer Planck-mean optical thickness once, then keep going. Used to test
  // the diffusion-limit note's Sec.2.4 open item -- whether the sawtooth amplitude tracks the ABSOLUTE
  // local layer thickness dtau_i (a numerator/truncation effect, ~dtau^3) rather than the thickness
  // contrast eps. Grid refinement varies both at once, so only a direct dtau measurement separates them.
  constexpr bool dumptau = false;
  std::vector<double> nl_dtau(n, 1.0);
  if ((newtonlike && !nl_netheat) || hstep || dumptau || lucy || thick_smooth > 0.0)   // HSTEP/LUCY/THICK_SMOOTH need the per-layer dtau
  {
    const std::vector<double> nl_w = aux::trapezoidalWeights(ratio_wn);
    std::vector<double> kP(n, 0.0);
    for (size_t i = 0; i < n; ++i)
    {
      double sB = 0.0, skB = 0.0;
      for (size_t kk = 0; kk < ratio_nnu; ++kk)
      { const double B = disortpp::planckFunction2(ratio_wn[kk], ratio_wn[kk], atmosphere.temperature[i]);
        sB += nl_w[kk]*B; skB += nl_w[kk]*opacity.absorption_coeff[kk][i]*B; }
      kP[i] = (sB > 0.0) ? skB/sB : 0.0;
    }
    for (size_t i = 1; i < n; ++i)
      nl_dtau[i] = std::max(1e-30, 0.5*(kP[i]+kP[i-1])*std::abs(atmosphere.altitude[i]-atmosphere.altitude[i-1]));
    if (dumptau)
    {
      static bool tau_dumped = false;
      if (!tau_dumped)
      {
        tau_dumped = true;
        std::fprintf(stderr, "  [dtau] n=%zu  per-layer Planck-mean dtau (i, P[bar], T, dtau):\n", n);
        for (size_t i = 1; i < n; ++i)
          std::fprintf(stderr, "  [dtau] %zu %.4e %.2f %.4e\n",
                       i, atmosphere.pressure[i], atmosphere.temperature[i], nl_dtau[i]);
      }
    }
  }

  // PTC top-anchor weight w_i = w0 * tau_w/(tau_i + tau_w): the ratio (Planck-slope) regularisation slides
  // ON in the optically-thin top (small cumulative tau -> w->w0) and OFF in the deep (large tau -> w->0),
  // where PTC's C/dt diagonal does the conditioning. tau = Planck-mean cumulative ABSORPTION optical depth
  // from the top, frozen over the call.
  std::vector<double> ptc_w(n, 0.0), ptc_tau(n, 0.0);

  // Per-level radiative time tau_rad ~ c_p/(kappa_P sigma T^3) (the local Newtonian-cooling time), used to
  // make the pseudo-timestep PER-LEVEL: dt_i = rho * s_i with s_i = tau_rad_i/<tau_rad>. The DEEP has short
  // tau_rad (high kappa, high T) -> small dt -> LARGE C/dt -> strong PTC regularisation of the deficient
  // net-flux Jacobian; the thin top has long tau_rad -> weak PTC, but it is the ratio anchor that conditions
  // the top there. rho is the single global pseudo-time, ramped by the Deuflhard rule in the driver.
  std::vector<double> ptc_srad(n, 1.0);

  // ---- Rosseland/Eddington DEEP preconditioner (CLIMA_PTC_ROSS; implied by the blend). The deep flux is
  // diffusive, F = -(4/3 kappa) d(sigma T^4)/dz, whose ONE-SIDED (interface) discretisation has a GENUINE,
  // diagonally-dominant self term  dF_i/dT_i = 16 sigma T_i^3 / (3 dtau_i) > |dF_i/dT_{i+1}|  (since the
  // deeper level is hotter) -- the very diagonal the multi-stream odd-moment NFJ cancels away. Used as the
  // STEP operator in the optically-thick deep (weight ross_d = dtau/(dtau+1) -> 1), smoothly blended to the
  // multi-stream NFJ in the thin mid/top (where the ratio anchor and the Planck inversion take over). The
  // RESIDUAL stays the accurate multi-stream F_net, so the fixed point is UNCHANGED; only the deep step
  // turns from a slow PTC relaxation into a real Newton contraction (the diffusion-limit doc's "only escape"
  // from the deficient K -- a two-stream/Eddington closure built explicitly as a preconditioner).
  // OPT-IN (CLIMA_PTC_ROSS), NOT default: in a faithful multi-stream, sampled-opacity solver the deep
  // net-flux Rosseland operator is (a) swamped by the enormous per-level C/dt, (b) not diagonally dominant
  // (the down-coupling to the hotter layer beneath exceeds the diagonal, T[i-1]^3 > T[i]^3), and (c) needs
  // the Rosseland-MEAN opacity, which sampled line-gaps (kappa~0 windows) make ill-defined. Left as an
  // experiment; the deep is the diffusion-limit doc's genuinely-hard regime -- in practice the terrestrial
  // deep is the CONVECTIVE troposphere (slaved), so this is moot there.
  std::vector<double> ross_diag(n, 0.0), ross_up(n, 0.0), ross_d(n, 0.0);

  // ---- MLT convective flux on level links (doc/mlt_convection_design.md) -----------------------
  // Link li sits between levels li and li+1 (the same layer as nl_dtau[li+1]). The flux law is
  //   F_c = C * x^{3/2},  x = max(nabla - nabla_ad, 0),  C = 1/2 rho c_p T (lambda^2/H_p) sqrt(g/H_p)
  // with lambda the Blackadar mixing length lambda = k z / (1 + k z/(alpha H_p)) (k = 0.4; z =
  // altitude above the bottom boundary), all thermodynamic prefactors and nabla_ad FROZEN at the
  // committed profile (the same operator split as frozen opacity: residual and Jacobian stay an
  // exactly consistent pair through the inner solve, and the lag vanishes at the fixed point).
  // Only nabla(T) is live, so the analytic link derivatives below are the EXACT residual gradient.
  std::vector<double> mlt_C, mlt_nad, mlt_idlnP;
  if (mlt)
  {
    // easy-start homotopy: scale the flux law by mlt_sf_, doubling once the previous stage settled
    // (raw residual + step size below the stage tolerance). While ramping, last_residual_ is floored
    // (below) so the driver cannot declare convergence on a weakened convection.
    // stage tolerance 0.3: the post-doubling residual decays ~5x per outer iteration (measured
    // 2.6 -> 0.57 -> 0.10 -> 0.017), so waiting for 1e-2 costs ~5 iterations per stage and the
    // 12-stage ramp exceeds the driver's iteration budget; doubling once the transient has decayed
    // below O(0.3) is stable (the per-doubling imbalance is bounded by the physical F_c itself)
    // and cuts the ramp to ~2 iterations per stage.
    constexpr double mlt_sf_ini = 3e-4, mlt_stage_tol = 0.3;
    if (mlt_sf_ < 0.0)
    {
      // AUTO-SKIP: the homotopy exists only to rescue a strongly super-adiabatic start (a dry
      // adiabat on a moist atmosphere: x ~ 0.1 on every link -> unscaled flux ~1e3 x stellar).
      // From a stable-side init (max link x small) the full-strength system converges directly --
      // measured: moist n=100 in 15 iterations (= mask scheme) vs 36 with the ramp, same root.
      double max_x0 = 0.0;
      for (size_t li = 0; li + 1 < n; ++li)
      {
        // links above the flux ceiling carry no convective flux (mlt_pmin, mirrored from the
        // mlt_C construction below) -- they must not vote here either. A converged terrestrial
        // profile is legitimately super-adiabatic above the ceiling (measured x ~ 0.6 at the
        // steep upper-stratosphere lapse), and counting those links forced the slow homotopy
        // ramp on every warm restart from a converged profile.
        if (atmosphere.pressure[li+1] < convection->fluxMinPressure()) continue;
        const double dlnP = std::log(atmosphere.pressure[li] / atmosphere.pressure[li+1]);
        if (dlnP <= 0.0) continue;
        const double x = std::log(atmosphere.temperature[li]/atmosphere.temperature[li+1]) / dlnP
                         - nablaAd(li, li+1);
        max_x0 = std::max(max_x0, x);
      }
      // Stable-side is necessary but NOT sufficient: the skip also needs the init to be CLOSE to
      // the converged zone. A surface-driven troposphere (target_flux = 0) is initialised on the
      // adiabat, so stable-side means near the answer; a self-luminous deep (target_flux > 0)
      // starts far from equilibrium and must WARM into its zone -- at full strength the Newton
      // bangs into the flux knee from below (measured: BD unconverged at 100 it with sf=1, vs 18
      // with the ramp). Ramp whenever target_flux > 0.
      constexpr double ramp_free_x = 0.02;
      mlt_sf_ = std::getenv("MLT_SF_INI") ? std::atof(std::getenv("MLT_SF_INI"))
              : ((max_x0 < ramp_free_x && target_flux <= 0.0) ? 1.0 : mlt_sf_ini);
      if (std::getenv("CLIMA_DBG"))
        std::fprintf(stderr, "  [mlt] init max_x=%.4f -> sf_ini=%.3e\n", max_x0, mlt_sf_);
    }
    else if (mlt_sf_ < 1.0 && mlt_prev_resid_ >= 0.0
             && std::max(mlt_prev_resid_, last_inner_change_) < mlt_stage_tol)
      mlt_sf_ = std::min(1.0, 2.0*mlt_sf_);
    if (std::getenv("CLIMA_DBG"))
      std::fprintf(stderr, "  [mlt] sf=%.3e prev_resid=%.3e\n", mlt_sf_, mlt_prev_resid_);

    const double alpha_mlt = std::max(1e-3, convection->fluxAlpha());
    mlt_C.assign(n > 0 ? n-1 : 0, 0.0);
    mlt_nad.assign(mlt_C.size(), 0.0);
    mlt_idlnP.assign(mlt_C.size(), 0.0);
    const double mlt_pmin = convection->fluxMinPressure();
    for (size_t li = 0; li + 1 < n; ++li)
    {
      // pressure ceiling (see Convection::fluxMinPressure): no convective flux above it
      if (atmosphere.pressure[li+1] < mlt_pmin) { mlt_C[li] = 0.0; continue; }
      const double dlnP = std::log(atmosphere.pressure[li] / atmosphere.pressure[li+1]);
      mlt_idlnP[li] = (dlnP > 0.0) ? 1.0/dlnP : 0.0;
      mlt_nad[li]   = nablaAd(li, li+1);
      const double rho = 0.5*(atmosphere.mass_density[li] + atmosphere.mass_density[li+1]);
      const double Tm  = 0.5*(atmosphere.temperature[li]  + atmosphere.temperature[li+1]);
      const double Hp  = 0.5*(atmosphere.scale_height[li] + atmosphere.scale_height[li+1]);
      const double cp  = 0.5*(
        ThermodynamicData::meanHeatCapacity(atmosphere.number_densities[li],   atmosphere.temperature[li]) +
        ThermodynamicData::meanHeatCapacity(atmosphere.number_densities[li+1], atmosphere.temperature[li+1]));
      const double z   = std::max(0.0,
        0.5*(atmosphere.altitude[li] + atmosphere.altitude[li+1]) - atmosphere.altitude[0]);
      constexpr double k_vk = 0.4;
      // wall law only for objects with a genuine surface; a self-luminous domain bottom is an
      // artificial cut, lambda = alpha H_p throughout (Convection::fluxBlackadarSurface)
      const double lam = !convection->fluxBlackadarSurface()
        ? ((Hp > 0.0) ? alpha_mlt*Hp : 0.0)
        : ((Hp > 0.0 && z > 0.0) ? k_vk*z / (1.0 + k_vk*z/(alpha_mlt*Hp)) : 0.0);
      mlt_C[li] = (Hp > 0.0)
        ? mlt_sf_ * 0.5 * rho * cp * Tm * (lam*lam/Hp) * std::sqrt(std::max(surface_gravity, 0.0)/Hp) : 0.0;
    }
  }
  // convective flux on link li for a trial profile T (0 when stably stratified)
  auto mltLink = [&](const std::vector<double>& T, size_t li) -> double
  {
    const double x = std::log(T[li]/T[li+1]) * mlt_idlnP[li] - mlt_nad[li];
    return (x > 0.0) ? mlt_C[li] * x * std::sqrt(x) : 0.0;
  };
  // d F_c(link li)/dT_li (dlo) and /dT_{li+1} (dhi); both -> 0 as x -> 0+ (the C1 handover)
  auto mltLinkDeriv = [&](const std::vector<double>& T, size_t li, double& dlo, double& dhi)
  {
    dlo = dhi = 0.0;
    const double x = std::log(T[li]/T[li+1]) * mlt_idlnP[li] - mlt_nad[li];
    if (x <= 0.0) return;
    const double f = 1.5 * mlt_C[li] * std::sqrt(x) * mlt_idlnP[li];
    dlo =  f / std::max(T[li],   1.0);
    dhi = -f / std::max(T[li+1], 1.0);
  };
  // volumetric convective heating at level i in the ratio numerator's units (per volume per
  // steradian): q_i = -(dF_c/dz)_i / (4 pi), conservative difference of the two adjacent links
  // (one-sided at the top; the surface level i=0 keeps its flux-anchor row, which carries F_c
  // explicitly instead).
  auto mltQ = [&](const std::vector<double>& T, size_t i) -> double
  {
    if (!mlt || i == 0 || n < 2) return 0.0;
    const double f_below = mltLink(T, i-1);
    const double f_above = (i+1 < n) ? mltLink(T, i) : 0.0;
    const double dz = (i+1 < n)
      ? 0.5*std::abs(atmosphere.altitude[i+1] - atmosphere.altitude[i-1])
      : std::abs(atmosphere.altitude[i] - atmosphere.altitude[i-1]);
    return (dz > 0.0) ? (f_below - f_above) / dz / (4.0*M_PI) : 0.0;
  };
  // LEVEL-collocated convective flux, for the difference-form (flux_divergence) rows and the
  // flux-conservation metric: centered link average in the interior, one-sided at the ends (level
  // 0 = the surface link, matching the surface row; the top level = the topmost link).
  auto mltFcLvl = [&](const std::vector<double>& T, size_t i) -> double
  {
    if (!mlt || n < 2) return 0.0;
    if (i == 0)     return mltLink(T, 0);
    if (i == n-1)   return mltLink(T, n-2);
    return 0.5*(mltLink(T, i-1) + mltLink(T, i));
  };

  // MLT_DUMP: per-call anatomy of the convective term (first 40 calls): committed-profile link
  // superadiabaticity, flux, heating-to-emission ratio q/den. Diagnoses stalls (which row, and
  // whether the term or its normalisation is the problem).
  if (mlt && std::getenv("MLT_DUMP"))
  {
    static int mlt_dump_calls = 0;
    if (mlt_dump_calls < 40)
    {
      ++mlt_dump_calls;
      std::fprintf(stderr, "  [mltdump] call=%d sf=%.3e  (lv P T x Fc q/den nad)\n", mlt_dump_calls, mlt_sf_);
      for (size_t i = 1; i + 1 < n && i < 30; ++i)
      {
        const double x = std::log(atmosphere.temperature[i]/atmosphere.temperature[i+1]) * mlt_idlnP[i] - mlt_nad[i];
        double num, den, C; ratioSums(atmosphere.temperature, i, num, den, C);
        const double q = mltQ(atmosphere.temperature, i);
        std::fprintf(stderr, "  [mltdump] %zu %.3e %.2f %+.4f %.3e %+.3e %.4f\n",
          i, atmosphere.pressure[i], atmosphere.temperature[i], x,
          mltLink(atmosphere.temperature, i), (den > 0.0) ? q/den : 0.0, mlt_nad[i]);
      }
    }
  }

  // ---- residual assembler: clima's heat-capacity-weighted flux DIFFERENCES over the reduced DOFs.
  // clima forms fluxes(i)=f_total(i)-f_total(i-1) (and fluxes(0)=f_total(0) at the surface), so we use
  // exactly that. Because the Jacobian below is a finite difference of THIS residual, the two are
  // self-consistent (no DISORT-divergence/flux stencil mismatch). Only flux_total is needed. ---------
  // Full Unsoeld-Lucy correction dB_i (level + cumulative flux-error integral), from the measured flux.
  // tau increases DOWNWARD from TOA; ngam's index increases UPWARD, so integrate from i=n-1 down to 0.
  // nl_dtau[k] is the optical thickness of the layer between levels k-1 and k.
  // MLT: the conserved column quantity is F_rad + F_c, so the Lucy level+gradient terms must use the
  // TOTAL flux deviation -- anchoring F_rad alone would re-introduce the seam a convective zone
  // carries (design note Sec.2b). The layer-mean convective flux in the [i,i+1] layer is exactly the
  // link flux mltLink(T,i).
  auto lucyVec = [&](const std::vector<double>& F, const std::vector<double>& T, std::vector<double>& dB) {
    dB.assign(n, 0.0);
    if (n < 2) return;
    const double fc_top = mlt ? mltLink(T, n-2) : 0.0;
    dB[n-1] = (F[n-1] + fc_top - target_flux) / (2.0*M_PI);  // level: Eddington top boundary J(0)=2H(0)
    for (int i = static_cast<int>(n)-2; i >= 0; --i)
      dB[i] = dB[i+1] + (3.0/(4.0*M_PI))
        * (0.5*((F[i]-target_flux) + (F[i+1]-target_flux)) + (mlt ? mltLink(T, i) : 0.0))
        * nl_dtau[i+1];
  };

  auto assembleG = [&](const std::vector<double>& F, const std::vector<double>& NH, const std::vector<double>& T, std::vector<double>& g)
  {
    g.assign(m, 0.0);
    std::vector<double> cumF;
    if (ratio && ratio_flux && ratio_perlevel) cumFvec(T, cumF);   // per-level blend only
    std::vector<double> lucyB;
    if (lucy) lucyVec(F, T, lucyB);                                // full Unsoeld-Lucy correction
    for (size_t r = 0; r < m; ++r)
    {
      const size_t i = unk[r];
      const int zi = zone_of_dof[i];
      if (zi >= 0)                                   // convective-zone DOF: net flux into the whole zone
      {
        const Zone& z = zones[zi];
        // Flux entering the zone from below. At the DOMAIN BOTTOM this is the internal flux
        // F_star = sigma T_int^4 (self-luminous); it is 0 for a terrestrial planet, so the
        // terrestrial behaviour is unchanged. Without this the zone row would drive F_u -> 0
        // instead of F_u -> F_star and the whole column would be pushed to the wrong level.
        const double f_lower = (z.lower == 0) ? target_flux : F[z.lower-1];
        // NORMALISATION: this row is the whole zone's energy budget. Dividing by c_eff_zone (the entire
        // troposphere's heat capacity, ~1e10) makes a 0.2 W/m^2 imbalance look like ~1e-8 to the Newton,
        // so it stops there -- while conv_resid divides the SAME quantity by Fnorm and reports ~6e-4.
        // That mismatch, not a solver failure, is the residual floor. Use the flux scale (as `netflux`
        // mode already does) so the solver drives what the metric measures. CLIMA_ZONE_CEFF restores the
        // old weighting.
        const bool zone_fluxnorm = (netflux || ratio);
        // MLT deep block (Fix A): the conserved quantity crossing the block top is the TOTAL
        // flux F_rad + F_c (the boundary link is MLT-active), so the budget row carries it.
        const double fc_top = (mlt && z.upper + 1 < n) ? mltLink(T, z.upper) : 0.0;
        g[r] = (F[z.upper] + fc_top - f_lower) / (zone_fluxnorm ? Fnorm : ceff_zone[zi]);
      }
      else if (i == 0)                               // bottom DOF: F_net[0] -> F_star (0 = terrestrial
        // MLT: the surface loses F_c through the lowest link (sensible heat drawn into the
        // convective column) -- the surface balance must include it (clima's surface_heat_flow).
        g[r] = (F[0] + (mlt && n >= 2 ? mltLink(T, 0) : 0.0) - target_flux) / ceff[0];
      else                                           // radiative level
      {
        // RCB HANDOVER. The collocated (ratio) residual is purely LOCAL, so for the first radiative level
        // above a convective zone NOTHING ties the radiative solution to the top of the slaved adiabat:
        // the zone DOF residual only constrains the zone's TOTAL budget (F[upper]-F[lower-1]), not how the
        // profile joins at its top. The two regions are then free to disagree -> the kink at the RCB and a
        // leftover flux divergence on the junction level (which is exactly what floors conv_resid). The
        // differenced-flux residual gets this coupling for free, because its stencil (F[i]-F[i-1]) SPANS
        // the boundary. So give just that one level the flux-difference form: continuity is restored while
        // local RE (and its smoothness) is kept everywhere else.
        //
        // CLIMA_NORCBFLUX=1: DIAGNOSTIC -- drop the handover and use the plain ratio row at RCB+1.
        // The handover is a FLUX-DIVERGENCE row (a difference of two net-flux Jacobian rows) embedded
        // in an otherwise ratio system, so it inherits exactly the diagonal that collapses as
        // 1/(2*dtau) and vanishes as kappa->0. At RCB+1 in a dry, optically thin stratosphere that
        // single row is near-singular and stops pinning T at that level.
        static const bool no_rcb_flux = std::getenv("CLIMA_NORCBFLUX") != nullptr;
        const bool above_rcb = (i > 0) && (slaved[i-1] || zone_of_dof[i-1] >= 0) && !no_rcb_flux;
        if (ratio && above_rcb)
                                      g[r] = (F[i] - F[i-1]) / ceff[i];   // flux continuity across the RCB
        else if (ratio)             { double num, den, C; ratioSums(T, i, num, den, C);
                                      // MLT: local RE becomes radiative + convective balance,
                                      // 4pi(num - den) - dF_c/dz = 0 -> (num + q)/den = 1.
                                      const double q = mltQ(T, i);
                                      const double gre = (den > 0.0) ? ((num + q) / den - 1.0) : 0.0;
                                      const double gflux = 0.0;
                                      // LEVEL ANCHOR: the ratio system is square (zone budget + one local-RE
                                      // row per radiative level), so nothing is left over to pin the absolute
                                      // flux LEVEL -- and Sec.4 guarantees the local-RE root is not flux
                                      // balanced (measured: F_net(TOA) = 0.198 W/m^2). A uniform pre- or
                                      // post-solve shift cannot fix this: applied after, it breaks the zone
                                      // budget; applied before, the Newton undoes it. The cure is a weak
                                      // UN-differenced flux term in the residual itself, which restores
                                      // sensitivity to the uniform mode without differencing (so it cannot
                                      // reintroduce the checkerboard). Same device as CLIMA_LREANCHOR.
                                      double glucy = 0.0;
                                      if (lucy) {                       // full UL: level + gradient, in the residual
                                        const double Btot = 5.670374e-5*std::pow(std::max(T[i],1.0),4)/M_PI;
                                        glucy = lucy_sign * lucyB[i] / std::max(Btot, 1e-30); }
                                      // optically-thick Nyquist penalty (see thick_smooth above): uses the
                                      // BUILT profile T, so slaved stencil neighbours fold consistently
                                      // through Cfac in the Jacobian.
                                      double gsm = 0.0;
                                      if (nl_dtau[i] > 1.0 && thickSmoothOK(i))
                                        gsm = thick_smooth *
                                          (T[i-2] - 4.0*T[i-1] + 6.0*T[i] - 4.0*T[i+1] + T[i+2]) / T[i];
                                      g[r] = ratio_xi*gre + gflux + glucy + gsm; }
        else                          // one-sided backward (clima); MLT: difference the TOTAL flux
          g[r] = (F[i] - F[i-1]
                  + (mlt ? (mltFcLvl(T, i) - mltFcLvl(T, i-1)) : 0.0)) / ceff[i];
      }
    }
  };

  // start from the committed profile and the radiation field already solved at it by the driver
  const std::vector<double> T_base = atmosphere.temperature;
  std::vector<double> x(m);
  for (size_t r = 0; r < m; ++r) x[r] = T_base[unk[r]];

  // build the full profile for a reduced DOF vector (overwrite unknowns, snap the slaved zones)
  auto buildProfile = [&](const std::vector<double>& xv) {
    std::vector<double> T = T_base;
    for (size_t r = 0; r < m; ++r) T[unk[r]] = std::max(1.0, xv[r]);
    adiabatSnap(T);
    return T;
  };

  // ---- CLIMA_LREJ: bracketed LAMBDA-ITERATION update for the radiative levels (the kappa->0 cure).
  // Instead of a coupled net-heating/flux Newton (which limit-cycles in the optically-thin SW-heating
  // top, see the implementation notes), update each radiative level INDEPENDENTLY by the monotone,
  // bracketed local-RE inversion B(T_i)=<J>_kappa,i using the CURRENT radiation field's mean intensity
  // (a Lambda iteration: J frozen this outer step, the driver re-solves RT between outer iterations).
  // The surface keeps the flux anchor F_net[0]=0 (the budget tether); convective zones stay slaved.
  // A per-iteration log-temperature cap is the blunt trust region against any residual convex overshoot.
  // evaluate the weighted residual g at xv. Opacity (and composition/structure) are FROZEN during the
  // inner solve (recompute_opacity=false), so the trial residuals are CONSISTENT with the Planck-only
  // Jacobian -- otherwise the omitted dsigma/dT term flips the residual gradient at optically-thin
  // levels and the trust region rejects every step (a stall). This is the operator split: the driver
  // recomputes opacity between outer iterations, so the lag vanishes at the fixed point (exact there).
  // With want_jac, also compute DISORT's Planck-only net-flux Jacobian at xv (read by rebuildJ).
  auto evalG = [&](const std::vector<double>& xv, bool want_jac, std::vector<double>& g) {
    const std::vector<double> T = buildProfile(xv);
    std::vector<double> F, NH;
    forward_eval_full_(T, /*recompute_opacity=*/false, want_jac, F, NH);
    assembleG(F, NH, T, g);
  };

  // ---- reduced Jacobian J (m x m) + column scaling D from DISORT's Planck-only net-flux temperature
  // Jacobian (= clima's frozen-opacity FD Jacobian, in ONE RT solve). For a radiative level i:
  // dg_i/dT_l = (NFJ[i][l]-NFJ[i-1][l])/ceff_i; surface: NFJ[0][l]/ceff_0; a convective-zone DOF:
  // (NFJ[upper][l]-NFJ[lower-1][l])/ceff_zone. Slaved layers fold into their anchor column with Cfac.
  // REBUILT at each accepted Newton step (clima's njev=1/step), reading radiation_field after an eval
  // with want_jac=true.
  std::vector<int> col_of_level(n, -1);
  for (size_t c = 0; c < m; ++c) col_of_level[unk[c]] = static_cast<int>(c);
  std::vector<double> J(m*m, 0.0), D(m, 1.0);
  auto rebuildJ = [&]() {
    const std::vector<std::vector<double>>& NFJ = radiation_field.net_flux_jacobian;
    const std::vector<std::vector<double>>& NHJ = radiation_field.net_heating_jacobian;
    const std::vector<std::vector<double>>& MK  = radiation_field.meanint_kappa_jacobian;

    // MLT: per-link flux derivatives at the linearisation point (= atmosphere.temperature, set by
    // the want_jac eval that preceded this rebuild). Exact for the frozen-prefactor flux law, so
    // residual and Jacobian stay a consistent pair (NLEQ-ERR rejects inconsistent operators).
    std::vector<double> mj_dlo, mj_dhi;
    if (mlt && n >= 2)
    {
      mj_dlo.assign(n-1, 0.0); mj_dhi.assign(n-1, 0.0);
      for (size_t li = 0; li + 1 < n; ++li)
        mltLinkDeriv(atmosphere.temperature, li, mj_dlo[li], mj_dhi[li]);
    }
    // d(F_c at LEVEL i)/dT scattered into `row` with sign/scale -- must mirror mltFcLvl exactly
    auto addFcLvlDeriv = [&](size_t i, double s, std::vector<double>& row)
    {
      if (i == 0)          { row[0]   += s*mj_dlo[0];       row[1] += s*mj_dhi[0]; }
      else if (i == n-1)   { row[n-2] += s*mj_dlo[n-2];     row[n-1] += s*mj_dhi[n-2]; }
      else                 { row[i-1] += 0.5*s*mj_dlo[i-1];
                             row[i]   += 0.5*s*(mj_dhi[i-1] + mj_dlo[i]);
                             row[i+1] += 0.5*s*mj_dhi[i]; }
    };

    // ---- AGB cumulative-heating flux Jacobian (clima_..._linearisation.cpp / thesis 3.64). The flux
    // deviation is the depth integral of the local heating S = num - den, so d(F)/dT is the CUMULATIVE
    // integral of the heating Jacobian H = M - C (M = meanint_kappa_jacobian, C = sum w kappa dB/dT) --
    // built from the SAME pieces as the ratio term, self-consistent, and far better conditioned than the
    // foreign DENSE net-flux Jacobian (which swamps the ratio diagonal at depth). A single least-squares
    // scale calibrates the proxy to the actual net flux (auto-handles the 4*pi and unit factors). The
    // RESIDUAL still uses the actual flux F[i] (assembleG); only this Jacobian term changes.
    // d(lucyB_i)/dT_j : the same one-sided integral applied to the flux Jacobian. Must match lucyVec, or
    // NLEQ-ERR rejects the steps (an inconsistent Jacobian is what stalled CLIMA_HSTEP).
    std::vector<std::vector<double>> lucyJ;
    if (lucy)
    {
      lucyJ.assign(n, std::vector<double>(n, 0.0));
      for (size_t j = 0; j < n; ++j) lucyJ[n-1][j] = NFJ[n-1][j] / (2.0*M_PI);
      if (mlt && n >= 2)   // top-boundary convective term (matches lucyVec's fc_top)
      {
        lucyJ[n-1][n-2] += mj_dlo[n-2] / (2.0*M_PI);
        lucyJ[n-1][n-1] += mj_dhi[n-2] / (2.0*M_PI);
      }
      for (int i = static_cast<int>(n)-2; i >= 0; --i)
      {
        for (size_t j = 0; j < n; ++j)
          lucyJ[i][j] = lucyJ[i+1][j] + (3.0/(4.0*M_PI))*0.5*(NFJ[i][j] + NFJ[i+1][j])*nl_dtau[i+1];
        if (mlt)             // layer convective flux mltLink(T,i): d/dT_i, d/dT_{i+1}
        {
          lucyJ[i][i]   += (3.0/(4.0*M_PI)) * mj_dlo[i] * nl_dtau[i+1];
          lucyJ[i][i+1] += (3.0/(4.0*M_PI)) * mj_dhi[i] * nl_dtau[i+1];
        }
      }
    }

    std::vector<std::vector<double>> cumH;   // [i][j] cumulative heating Jacobian from the surface (i=0)
    if (ratio && ratio_flux && ratio_perlevel && !ratio_fluxdense)
    {
      std::vector<double> Call(n, 0.0);
      for (size_t k = 0; k < n; ++k)
      { double nu, de, C; ratioSums(atmosphere.temperature, k, nu, de, C); Call[k] = C; }
      cumH.assign(n, std::vector<double>(n, 0.0));
      for (size_t i = 1; i < n; ++i)
      {
        const double dz = std::abs(atmosphere.altitude[i] - atmosphere.altitude[i-1]);
        for (size_t j = 0; j < n; ++j)
        {
          const double Hij  = MK[i][j]   - (i   == j ? Call[i]   : 0.0);
          const double Hi1j = MK[i-1][j] - (i-1 == j ? Call[i-1] : 0.0);
          cumH[i][j] = cumH[i-1][j] + 0.5*(Hij + Hi1j)*dz;
        }
      }
    }
    const double flux_scale = ratio_flux_scale;   // frozen calibration (consistent with the residual)

    std::vector<double> row(n);
    for (size_t r = 0; r < m; ++r)
    {
      const size_t i = unk[r];
      const int zi = zone_of_dof[i];
      // ratio local-RE radiative row needs the per-level spectral sums at the current linearisation
      // point (= atmosphere.temperature, set by the want_jac eval that preceded this rebuild).
      double ratio_inv = 0.0, ratio_diag_corr = 0.0, planck_slope = 0.0;
      if (((ratio && i != 0) || ptc) && zi < 0)
      {
        double num, den, C;
        ratioSums(atmosphere.temperature, i, num, den, C);
        ratio_inv = (den > 0.0) ? 1.0/den : 0.0;
        // MLT: the residual numerator is (num + q), so the den-derivative term carries it too
        const double q_committed = mlt ? mltQ(atmosphere.temperature, i) : 0.0;
        ratio_diag_corr = (num + q_committed) * ratio_inv * ratio_inv * C;   // -(num+q)/den^2 * C
        // STABLE Planck-slope diagonal C/den = <dln B/dT>_kappa (a kappa-weighted average, FINITE as
        // kappa->0, unlike num*C/den^2). This is the doc's -C_i/den_i; used by the Eq.19 blend instead of
        // the unstable exact ratio Jacobian terms (which carry 1/den, 1/den^2 and blow up at the thin top).
        planck_slope = (den > 0.0) ? C / den : 0.0;
        // (band closure: with the FROZEN band-mean Jbar the closure's num term is a constant
        //  during the inner solve, so MK -- the sampled-J response -- IS the exact num Jacobian;
        //  den/C corrections enter through the sums above. No extra terms needed.)
      }
      const double rd = (zi < 0) ? ross_d[i] : 0.0;       // Rosseland deep-preconditioner weight (0 = pure NFJ)
      for (size_t l = 0; l < n; ++l)
      {
        if (zi >= 0 && hstep_zone)
        {                                            // Sec.7.4: a net flux across a contiguous block
          const Zone& z = zones[zi];                 // telescopes into the SUM of the heating rows it
          double s = 0.0;                            // contains -- no net-flux Jacobian needed.
          for (size_t k = (z.lower == 0 ? 1 : z.lower); k <= z.upper; ++k) s += NHJ[k][l]*nl_dtau[k];
          row[l] = s / ceff_zone[zi];
        }
        else if (zi >= 0) { const Zone& z = zones[zi]; const double low = (z.lower==0)?0.0:NFJ[z.lower-1][l];
                            const bool zfn = (netflux || ratio);
                            row[l] = (NFJ[z.upper][l]-low)/(zfn ? Fnorm : ceff_zone[zi]); }  // must match assembleG
        else if (i == 0 && hstep_surf) row[l] = 0.0;   // Sec.7.4: closed-form surface row, filled below
        else if (i == 0) row[l] = NFJ[0][l]/ceff[0];
        // xi*(1/den)d num/dT  +  zeta * d(F/Fstar)/dT. Flux Jacobian = the AGB cumulative-heating cumH
        // (calibrated, diagonally-dominant); CLIMA_RATIO_FLUXDENSE falls back to the raw dense NFJ.
        // RCB handover row: must MATCH the residual chosen in assembleG for this level, else the Newton
        // stalls (an inconsistent Jacobian is rejected by the trust region -- measured elsewhere).
        else if (ratio && i > 0 && (slaved[i-1] || zone_of_dof[i-1] >= 0)
                 && std::getenv("CLIMA_NORCBFLUX") == nullptr)
                        row[l] = (NFJ[i][l] - NFJ[i-1][l]) / ceff[i];
        else if (ratio) row[l] = ratio_xi * (MK[i][l] * ratio_inv)
                                                              + (lucy ? lucy_sign * lucyJ[i][l] / std::max(5.670374e-5*std::pow(std::max(atmosphere.temperature[i],1.0),4)/M_PI, 1e-30) : 0.0)
                               ; // (level is handled by the Lucy term)
        // CLIMA_HSTEP (doc Sec.7.4): defect correction -- the accurate multi-stream FLUX residual is kept
        // (assembleG untouched), only the STEP moves to the collocated heating operator. NHJ is the per-tau
        // Jacobian d(dF/dtau)/dT, so NHJ*dtau[i] is the dominant surrogate of d(F[i]-F[i-1])/dT, matching
        // the residual's scale. Carries the Planck-cooling diagonal -> no checkerboard null mode.
        // LAYER-INDEX OFFSET: the backend's flux divergence at an interface uses the layer ABOVE it (its own
        // TOA-first ordering), so after ngam's 0=BOA reversal net_heating[k] describes the layer between ngam
        // levels k and k+1. The residual row here is (F[i]-F[i-1]) -- the layer between i-1 and i -- so the
        // matching heating row is k = i-1 (and nl_dtau[i], the same layer, is already correct). Offset is a
        // knob (CLIMA_HSTEP_OFF, default -1) because a mismatched pairing makes M a poor approximation of J
        // and the trust region stalls (hybrj info=4/5) rather than failing loudly.
        // SIGN: net_heating = 4pi(1-omega)(J-B) is POSITIVE for heating, and energy conservation gives
        // heating ~ -dF/dz. ngam's index increases UPWARD, so F[i]-F[i-1] ~ -heating[i]*dtau: the Jacobian
        // of this residual is -NHJ*dtau/ceff. Using +NHJ flips the step exactly backwards -> M^-1 g is an
        // ASCENT direction at every step length (measured: rejected for lambda from 1 down to 1e-9 with a
        // sane ||s||~3e2), which is precisely how a trust region reports info=4/5 with zero accepted steps.
        else row[l] = (NFJ[i][l]-NFJ[i-1][l])/ceff[i];
      }
      if (i == 0 && hstep_surf)                     // Sec.7.4: the surface row is the closed-form emission
      {                                             // response 4 eps sigma T0^3 (the GENUINE surface diagonal
        const double sig_cgs = 5.670374e-5;         // -- no odd-moment cancellation at a boundary).
        row[0] = 4.0*sig_cgs*std::pow(std::max(atmosphere.temperature[0], 1.0), 3) / ceff[0];
      }
      if (mlt && i == 0 && n >= 2)                  // MLT surface balance: d F_c(link 0)/dT_{0,1}
      {
        row[0] += mj_dlo[0] / ceff[0];
        row[1] += mj_dhi[0] / ceff[0];
      }
      if (mlt && zi >= 0)                           // MLT deep block: d F_c(top link)/dT (Fix A)
      {
        const Zone& z = zones[zi];
        if (z.upper + 1 < n)
        {
          const bool zfn = (netflux || ratio);
          const double s = 1.0 / (zfn ? Fnorm : ceff_zone[zi]);
          row[z.upper]   += s * mj_dlo[z.upper];
          row[z.upper+1] += s * mj_dhi[z.upper];
        }
      }
      // (skip for the RCB-handover level: its row is the flux difference, not the ratio form, so the
      //  ratio Planck-cooling diagonal must not be added on top of it)
      const bool rcb_handover_row = ratio && i > 0 && (slaved[i-1] || zone_of_dof[i-1] >= 0)
                                    && std::getenv("CLIMA_NORCBFLUX") == nullptr;
      if (ratio && zi < 0 && i != 0 && !rcb_handover_row)
      {
        row[i] -= ratio_xi * ratio_diag_corr;      // local Planck-cooling diagonal -xi*C_i/den_i
        if (mlt && i >= 1)                          // d q_i/dT_l of the convective heating (matches mltQ)
        {
          const double dz = (i+1 < n)
            ? 0.5*std::abs(atmosphere.altitude[i+1] - atmosphere.altitude[i-1])
            : std::abs(atmosphere.altitude[i] - atmosphere.altitude[i-1]);
          if (dz > 0.0)
          {
            const double s = ratio_xi * ratio_inv / (4.0*M_PI) / dz;
            row[i-1] += s * mj_dlo[i-1];            // link below (i-1,i)
            row[i]   += s * mj_dhi[i-1];
            if (i+1 < n)
            {
              row[i]   -= s * mj_dlo[i];            // link above (i,i+1), entering with -F_c
              row[i+1] -= s * mj_dhi[i];
            }
          }
        }
        // optically-thick Nyquist penalty stencil (must match assembleG's gsm; the full-stencil
        // free-radiative guard means no slaved/zone folding is involved). The d(1/T_i) term of the
        // normalisation is O(s4/T^2), ~1e-6 of the stencil weights -- omitted.
        if (nl_dtau[i] > 1.0 && thickSmoothOK(i))
        {
          const double w = thick_smooth / std::max(atmosphere.temperature[i], 1.0);
          row[i-2] += w; row[i-1] -= 4.0*w; row[i] += 6.0*w; row[i+1] -= 4.0*w; row[i+2] += w;
        }
        // fully transparent (den=0) AND no flux anchor (zeta~0, the thin top): the row is all-zero ->
        // give it a unit restoring diagonal so the LU stays nonsingular (g_i=0 there -> dx_i=0 anyway).
        if (ratio_inv == 0.0 && ratio_zeta[i] < 1e-12) { for (size_t l = 0; l < n; ++l) row[l] = 0.0; row[i] = -1.0; }
      }
      else if (ptc_blend && zi < 0 && i != 0)        // Eq.19 Jacobian: + w_i d g^RE/dT_i ~ -w_i C/den
        row[i] -= ptc_w[i] * planck_slope;           // STABLE Planck-slope diagonal -> conditions the thin top (NFJ + C/dt deep)
      if (mlt && !ratio && zi < 0 && i != 0)          // flux_divergence rows: d(Fc_lvl[i]-Fc_lvl[i-1])/dT
      {
        addFcLvlDeriv(i,   +1.0/ceff[i], row);
        addFcLvlDeriv(i-1, -1.0/ceff[i], row);
      }
      if (rd > 0.0 && i >= 1)                         // Rosseland deep preconditioner: the genuine diffusion
      {                                              // self + DOWN-coupling terms (couples level i to i-1)
        row[i]   += rd * ross_diag[i] / Fnorm;       // dF[i]/dT[i]   < 0
        row[i-1] += rd * ross_up[i]   / Fnorm;       // dF[i]/dT[i-1] > 0 (toward the surface)
      }
      for (size_t c = 0; c < m; ++c) J[r*m+c] = row[unk[c]];
      for (size_t k = 0; k < n; ++k)
        if (slaved[k]) { const int c = col_of_level[anchor[k]]; if (c >= 0) J[r*m+c] += row[k]*Cfac[k]; }
    }
    // ---- CLIMA_TRIDIAG: drop the off-band coupling on the RADIATIVE-level rows (keep |r-c|<=1). Surface
    // and zone-anchor rows stay dense (the global energy tether). The diagonal is always retained, so the
    // ratio's strong restoring diagonal (-xi C_i/den_i) and the local 2nd-difference coupling survive --
    // exactly the structure that makes the Nyquist sawtooth the best-conditioned (not the null) direction.
    double Dmax = 0.0;
    for (size_t c = 0; c < m; ++c)
    {
      double cn = 0.0; for (size_t r = 0; r < m; ++r) cn += J[r*m+c]*J[r*m+c];
      D[c] = std::sqrt(cn); Dmax = std::max(Dmax, D[c]);
    }
    // floor the column scaling so (near-)SINGULAR columns -- degenerate optically-thin levels whose
    // flux divergence is ~insensitive to their own temperature -- still get LM regularisation
    // (lambda*Dfloor^2). Without this the LM step runs free in the null direction, hits the cap, and
    // does nothing (a null-space stall).
    const double Dfloor = (Dmax > 0.0) ? 1e-2*Dmax : 1.0;
    for (size_t c = 0; c < m; ++c) D[c] = std::max(D[c], Dfloor);
  };
  auto dnorm = [&](const std::vector<double>& p) { double s = 0.0; for (size_t c = 0; c < m; ++c) { const double e = D[c]*p[c]; s += e*e; } return std::sqrt(s); };

  // Establish the baseline residual AND Jacobian from a SINGLE forward eval at the snapped profile, so
  // the trust-region trials (which evaluate the same buildProfile->forward map) are CONSISTENT with the
  // base. Using the driver's radiation_field instead leaves a baseline offset (the driver's committed
  // convective layers are not exactly on the adiabat that buildProfile snaps to), which makes even a
  // zero step show a spurious residual increase -> every step rejected -> the dogleg stalls.
  std::vector<double> gbase;
  evalG(x, /*want_jac=*/true, gbase);
  snapshotBandJ();   // freeze the closure's Jbar at the snapped-base radiation field
  rebuildJ();
  const bool dbg = std::getenv("CLIMA_DBG") != nullptr;
  // CLIMA_PATHCHECK (temporary diagnostic): evaluate g at the IDENTICAL point with want_jac=false
  // (the trial path) and report the row-wise difference to gbase. Any nonzero difference is a
  // jac/no-jac evaluation-path inconsistency, which puts a hard floor under theta in the NLEQ
  // monotonicity test (theta(lambda->0) = ||J^-1 g_nojac|| / ||J^-1 g_jac|| instead of 1).
  if (std::getenv("CLIMA_PATHCHECK"))
  {
    std::vector<double> g_nojac;
    evalG(x, /*want_jac=*/false, g_nojac);
    double dmax = 0.0; size_t rmax = 0;
    for (size_t r = 0; r < m; ++r)
      if (std::abs(g_nojac[r] - gbase[r]) > dmax)
      { dmax = std::abs(g_nojac[r] - gbase[r]); rmax = r; }
    std::fprintf(stderr,
      "  [pathcheck] max|g_nojac - g_jac| = %.6e @lv%zu  (g_jac=%.6e g_nojac=%.6e)\n",
      dmax, unk[rmax], gbase[rmax], g_nojac[rmax]);
    // continuity probe ALONG THE NEWTON DIRECTION (a uniform relative perturbation leaves the
    // link gradients dlnT/dlnP exactly unchanged, so gradient-keyed branches are blind to it).
    // Replicates the failing NLEQ trial: for each lambda, is |g(x+lam*dx) - g(x)| ~ lam (smooth)
    // or lambda-independent (a jump)?
    {
      std::vector<double> Jc = J, dxp(m);
      for (size_t r = 0; r < m; ++r) dxp[r] = -gbase[r];
      if (solveDenseLU(Jc, dxp, m))
      {
        for (double eps : {1e-4, 1e-2, 1.0})
        {
          std::vector<double> xp = x, gp;
          for (size_t r = 0; r < m; ++r) xp[r] = std::max(1.0, x[r] + eps*dxp[r]);
          evalG(xp, /*want_jac=*/false, gp);
          double d1 = 0.0, d2 = 0.0, d3 = 0.0; size_t r1 = 0, r2 = 0, r3 = 0;
          for (size_t r = 0; r < m; ++r)
          {
            const double d = std::abs(gp[r] - gbase[r]);
            if (d > d1)      { d3=d2; r3=r2; d2=d1; r2=r1; d1=d; r1=r; }
            else if (d > d2) { d3=d2; r3=r2; d2=d; r2=r; }
            else if (d > d3) { d3=d; r3=r; }
          }
          std::fprintf(stderr,
            "  [dirprobe] eps=%.0e  top|dg|: lv%zu %.3e (g=%.3e)  lv%zu %.3e  lv%zu %.3e\n",
            eps, unk[r1], d1, gbase[r1], unk[r2], d2, unk[r3], d3);
        }
      }
    }
    // restore the base radiation field + Jacobian state for the solve below
    evalG(x, /*want_jac=*/true, gbase);
    rebuildJ();
  }
  (void) dnorm;

  // ---- smoothness regularisation: a small penalty on the discrete FOURTH difference of T, added to the
  // radiative-layer residual. The flux-divergence residual is near-NULL for high-frequency (sawtooth) T
  // modes (the radiative flux is a non-local smoothing integral of T), so opacity-sampling noise in the
  // fluxes is amplified into a checkerboard. The 4th difference is ZERO for any smooth profile (even a
  // steep physical gradient -- unlike the 2nd difference, which fights real curvature) but LARGE for a
  // sawtooth, so it damps ONLY the checkerboard and leaves the converged physics unchanged. ngam needs
  // this where clima leans on smooth band-averaged (k-distribution) fluxes. CLIMA_SMOOTH sets strength.
  // NOTE: default OFF. The 4th-difference penalty cleanly removes the sawtooth in the PURE-RADIATIVE
  // case, but in the radiative-CONVECTIVE case it feeds back through the convective mask detection
  // (penalty -> profile change -> mask over-grows -> surface collapses). Integrating it with convection
  // needs more care, so it is opt-in until then.
  constexpr double lam_s = 0.0;
  double jscale = 0.0;
  {
    std::vector<double> dg;
    for (size_t r = 0; r < m; ++r) { const size_t i = unk[r]; if (zone_of_dof[i] < 0 && i != 0) dg.push_back(std::abs(J[r*m+r])); }
    if (!dg.empty()) { std::sort(dg.begin(), dg.end()); jscale = dg[dg.size()/2]; }   // median |diag| = local residual scale
  }
  auto isSmoothDOF = [&](size_t r) {
    const size_t i = unk[r];
    if (!(lam_s > 0.0 && jscale > 0.0 && i >= 2 && i+2 < n)) return false;
    // the WHOLE 5-point stencil must be free radiative levels (not the surface, not slaved/convective),
    // else the penalty folds a convective neighbour into the surface column and destabilises the solve.
    for (int dl = -2; dl <= 2; ++dl) { const size_t l = i + dl; if (l == 0 || slaved[l] || zone_of_dof[l] >= 0) return false; }
    return true;
  };
  // 4th-difference stencil weights: [+1, -4, +6, -4, +1] at levels i-2..i+2
  auto addSmoothG = [&](const std::vector<double>& T, std::vector<double>& g) {
    for (size_t r = 0; r < m; ++r) if (isSmoothDOF(r))
    { const size_t i = unk[r]; g[r] += lam_s*jscale*(T[i-2] - 4.0*T[i-1] + 6.0*T[i] - 4.0*T[i+1] + T[i+2]); }
  };
  auto addSmoothJac = [&](Eigen::MatrixXd& fjac) {
    const double w = lam_s*jscale;
    auto col = [&](size_t l, double v, int r) {   // fold a neighbour level into the reduced columns (slaving)
      if (l >= n) return;
      if (slaved[l] && anchor[l] < n) { const int c = col_of_level[anchor[l]]; if (c >= 0) fjac(r,c) += v*Cfac[l]; }
      else { const int c = col_of_level[l]; if (c >= 0) fjac(r,c) += v; }
    };
    for (size_t r = 0; r < m; ++r) if (isSmoothDOF(r))
    { const size_t i = unk[r]; col(i-2, w, (int)r); col(i-1, -4.0*w, (int)r); fjac((int)r,(int)r) += 6.0*w; col(i+1, -4.0*w, (int)r); col(i+2, w, (int)r); }
  };
  if (std::getenv("CLIMA_DBG"))
  { int ns = 0; for (size_t r=0;r<m;++r) if (isSmoothDOF(r)) ns++;
    std::fprintf(stderr, "  [smooth] jscale=%.3e lam=%.2f n_smooth=%d/%zu  ||gbase||=%.3e\n",
      jscale, lam_s, ns, m, [&]{double s=0;for(double e:gbase)s+=e*e;return std::sqrt(s);}()); }

  // ---- full first-iteration dump for analysis (env CLIMA_DUMP; run with max_iterations=1) ----------

  (void) gbase;

  // ---- inner solve: MINPACK hybrj (Eigen HybridNonLinearSolver). The residual is the heat-capacity-
  // weighted flux divergence; the Jacobian is the analytic Planck-only reduced Jacobian. hybrj's
  // QR-based dogleg + rank-1 Broyden updates are numerically stable for the near-singular flux-
  // divergence Jacobian (it weights the large singular values and does not excite the small-singular-
  // value SAWTOOTH modes a weakly-damped LM amplifies). Opacity/composition are FROZEN within the
  // solve (operator split; the driver updates them between outer iterations -> exact at the fixed point).
  //
  // CLIMA_RATIO_RE uses instead NLEQ-ERR (Deuflhard): an affine-covariant damped Newton that monitors
  // the Newton CORRECTION norm in T-space, not the residual norm. Two reasons it is the right driver for
  // the ratio residual: (1) affine covariance makes the globalisation invariant to the huge scale gap
  // between the ratio rows (~1e-2) and the surface/zone flux rows (~1e3) -- a residual trust region would
  // be dominated by the flux rows; (2) the natural monotonicity test rejects any trial step that inflates
  // the residual, including an overshoot that drives den = sum w kappa B(T) -> 0 (the unbounded ratio
  // residual's pole), so the convex-Wien-tail overshoot that blew up stock hybrj cannot occur. Same scheme
  // already proven on the flux residual; here the matrix is the strictly diagonally-dominant ratio Jacobian
  // so NO Laplacian regularisation is needed (the near-null flux-divergence J required it; this one does not).
  if (ratio || newtonlike || tikh_flux_mlt)
  {
    // NLEQ-ERR (Deuflhard) affine-covariant damped Newton -- used both for the pure ratio residual and for
    // the Newton-like flux-conservation mode (accurate flux divergence residual + dominant NHJ step Jacobian).
    const int    maxit = std::getenv("CLIMA_RATIO_MAXIT") ? std::atoi(std::getenv("CLIMA_RATIO_MAXIT")) : 50;
    const double xtol  = std::getenv("CLIMA_RATIO_XTOL")  ? std::atof(std::getenv("CLIMA_RATIO_XTOL"))  : 1e-7;
    constexpr double lambda_min = 1e-4;

    // affine-covariant scaled correction norm: RMS of dx[r] / T_scale[r].
    auto scaledNorm = [&](const std::vector<double>& v, const std::vector<double>& xc) {
      double s2 = 0.0;
      for (size_t r = 0; r < m; ++r) { const double sc = std::max(xc[r], 1.0); s2 += (v[r]/sc)*(v[r]/sc); }
      return std::sqrt(s2 / static_cast<double>(m));
    };
    auto allFinite = [&](const std::vector<double>& v) { for (double e : v) if (!std::isfinite(e)) return false; return true; };

    // ---- CLIMA_TIKH=alpha (test A2, "three defects" note): Tikhonov regularisation IN THE
    // OBJECTIVE, min ||g||^2 + alpha^2 ||D4 T||^2, D4 the UNCONDITIONAL fourth difference of the
    // full profile (no Delta-tau gate, no junction guard -- both killed the earlier
    // CLIMA_THICK_SMOOTH attempt; unconditional is viable now because the MLT C1 handover left
    // no corner for a smoothness term to fight). Rationale: in the Delta-tau >> 1 band the
    // Nyquist mode is a NULL direction of the collocated residual (degenerate root family,
    // path-dependent member selection); D4 responds with 16 at Nyquist and (k Delta)^4 -> 0 on
    // resolved modes, so any alpha inside the wide singular-value gap pins the grid-scale mode
    // while perturbing the physics by O(alpha^2) -- the alpha-plateau is the validation test.
    // IN the objective, so the g=0 root family is intersected rather than shifted, and NOT
    // post-hoc, so the committed profile is a stationary point of the system actually solved.
    // The step becomes the Gauss--Newton correction of the augmented least-squares system,
    //   (J^T J + a^2 S^T S) dx = -(J^T g + a^2 S^T S x),
    // with S = D4 folded onto the reduced DOFs (slaved levels are LINEAR in their anchor via
    // Cfac, so S is a constant matrix per call and S*T_full = S_red*x exactly).
    const bool tikh = tikh_alpha > 0.0;
    const double tikh_a2 = tikh_alpha*tikh_alpha;
    size_t n_pen = 0;
    std::vector<double> S_pen;                       // n_pen x m, row-major
    if (tikh && n >= 5)
    {
      n_pen = n - 4;
      S_pen.assign(n_pen*m, 0.0);
      auto fold = [&](size_t row, size_t l, double w) {
        if (l >= n) return;
        if (slaved[l]) { const int c = col_of_level[anchor[l]]; if (c >= 0) S_pen[row*m+c] += w*Cfac[l]; }
        else           { const int c = col_of_level[l];         if (c >= 0) S_pen[row*m+c] += w; }
      };
      for (size_t i = 2; i+2 < n; ++i)
      {
        const size_t row = i - 2;
        fold(row, i-2,  1.0); fold(row, i-1, -4.0); fold(row, i,  6.0);
        fold(row, i+1, -4.0); fold(row, i+2,  1.0);
      }
    }
    // augmented Gauss-Newton correction at (xv, gv); returns false on a singular system
    auto tikhStep = [&](const std::vector<double>& gv, const std::vector<double>& xv,
                        std::vector<double>& dxout) -> bool {
      std::vector<double> A(m*m, 0.0), b(m, 0.0);
      for (size_t k = 0; k < m; ++k)
        for (size_t r = 0; r < m; ++r)
        {
          const double Jkr = J[k*m+r];
          if (Jkr == 0.0) continue;
          for (size_t c = 0; c < m; ++c) A[r*m+c] += Jkr * J[k*m+c];
          b[r] -= Jkr * gv[k];
        }
      std::vector<double> Sx(n_pen, 0.0);
      for (size_t p = 0; p < n_pen; ++p)
        for (size_t c = 0; c < m; ++c) Sx[p] += S_pen[p*m+c]*xv[c];
      for (size_t p = 0; p < n_pen; ++p)
        for (size_t r = 0; r < m; ++r)
        {
          const double Spr = S_pen[p*m+r];
          if (Spr == 0.0) continue;
          for (size_t c = 0; c < m; ++c) A[r*m+c] += tikh_a2 * Spr * S_pen[p*m+c];
          b[r] -= tikh_a2 * Spr * Sx[p];
        }
      if (!solveDenseLU(A, b, m)) return false;
      dxout = b;
      return true;
    };
    // augmented objective norm ||g||^2 + a^2 ||S x||^2 (floor test)
    auto augNorm2 = [&](const std::vector<double>& gv, const std::vector<double>& xv) {
      double s = 0.0;
      for (size_t r = 0; r < m; ++r) s += gv[r]*gv[r];
      if (tikh)
        for (size_t p = 0; p < n_pen; ++p)
        {
          double sx = 0.0;
          for (size_t c = 0; c < m; ++c) sx += S_pen[p*m+c]*xv[c];
          s += tikh_a2*sx*sx;
        }
      return s;
    };

    std::vector<double> g;
    double lambda = 1.0;
    int it = 0;
    for (; it < maxit; ++it)
    {
      // an RT failure at the current iterate (e.g. a trial-driven exotic profile making the
      // backend's thermal-emission floor trip) ends the inner solve at the last good state
      // rather than killing the process
      try { evalG(x, /*want_jac=*/true, g); }
      catch (const std::exception& e)
      { if (dbg) std::fprintf(stderr, "  [nleq] base eval failed (%s) -- keeping iterate\n", e.what());
        break; }
      rebuildJ();

      std::vector<double> dx_v;
      if (tikh)
      {
        if (!tikhStep(g, x, dx_v)) break;       // augmented Gauss-Newton correction
      }
      else
      {
        std::vector<double> A = J, b(m);        // ordinary Newton correction  J dx = -g
        for (size_t r = 0; r < m; ++r) b[r] = -g[r];
        if (!solveDenseLU(A, b, m)) break;      // singular (should not happen: J is diagonally dominant)
        dx_v = b;
      }
      const std::vector<double> dx = dx_v;
      const double norm_dx = scaledNorm(dx, x);
      if (dbg && it < 3)
      {
        size_t rmax = 0; double vmax = 0.0, gmax = 0.0; size_t rg = 0;
        for (size_t r = 0; r < m; ++r)
        {
          if (std::abs(dx[r]) > vmax) { vmax = std::abs(dx[r]); rmax = r; }
          if (std::abs(g[r])  > gmax) { gmax = std::abs(g[r]);  rg = r; }
        }
        std::fprintf(stderr, "  [nleq-it] it=%d norm_dx=%.3e max|dx|=%.3e @lv%zu (x=%.1f)  max|g|=%.3e @lv%zu\n",
                     it, norm_dx, vmax, unk[rmax], x[rmax], gmax, unk[rg]);
      }
      if (norm_dx < xtol)                       // converged in the Newton correction
      {
        // COMMIT the final correction before declaring convergence (Deuflhard's termination:
        // x* = x + dx). Breaking without it discards a step that annihilates the current residual
        // (g + J dx = 0), which left own_resid floored at ~5e-5 with dT/T = 0 EXACTLY -- a stall on
        // a stiff row (the Lucy TOA level term, amplified by 1/B(T_top)) whose tiny-in-x correction
        // the loop computed and then threw away every outer iteration.
        for (size_t r = 0; r < m; ++r) x[r] = std::max(1.0, x[r] + dx[r]);
        break;
      }

      // NLEQ-ERR damping: accept lambda when theta = ||dxbar||/||dx|| <= 1 - lambda/4, where the SIMPLIFIED
      // correction dxbar = -J^{-1} g(x + lambda dx) reuses the matrix J(x) but the TRUE residual at the trial.
      double lam = std::min(1.0, 2.0*lambda);   // Deuflhard prediction / warm start
      double lam_acc = -1.0; std::vector<double> x_acc;
      for (int trial = 0; trial < 25; ++trial)
      {
        std::vector<double> xt(m);
        for (size_t r = 0; r < m; ++r) xt[r] = std::max(1.0, x[r] + lam*dx[r]);
        std::vector<double> gt;
        bool trial_ok = true;
        try { evalG(xt, /*want_jac=*/false, gt); }
        catch (const std::exception&) { trial_ok = false; }   // crashing trial = inadmissible step
        double theta = 1e300;
        if (trial_ok && allFinite(gt))
        {
          std::vector<double> b2;
          if (tikh)
          {
            if (tikhStep(gt, xt, b2) && allFinite(b2)) theta = scaledNorm(b2, xt) / norm_dx;
          }
          else
          {
            std::vector<double> A2 = J; b2.assign(m, 0.0);
            for (size_t r = 0; r < m; ++r) b2[r] = -gt[r];
            if (solveDenseLU(A2, b2, m) && allFinite(b2)) theta = scaledNorm(b2, xt) / norm_dx;
          }
        }
        if (dbg && it < 3)
          std::fprintf(stderr, "  [nleq-trial] it=%d lam=%.3e theta=%.3e\n", it, lam, theta);
        if (theta <= 1.0 - 0.25*lam) { lam_acc = lam; x_acc = xt; break; }   // natural monotonicity test
        // regularity floor: the smallest step is taken ONLY if it decreases the RESIDUAL NORM.
        // Unconditionally committing the floor step turned an unsatisfiable row into a runaway:
        // 50 floor-steps per call with the residual GROWING (||g|| 6.6 -> 14) accumulated a
        // +827 K displacement (measured, Earth 30000-pt grid, iteration 4) instead of stalling
        // loudly at the base profile. The test is on ||g|| rather than theta: theta compares
        // affine-covariant correction norms through the (possibly approximate) Jacobian and can
        // exceed 1 on genuinely productive steps -- a theta gate here blocked the early deep
        // Unsoeld-Lucy transient that the floor steps legitimately work through (measured: the
        // R=5000 run stalled at own_resid ~21 instead of passing 26 -> 6.6). A residual-norm
        // decrease is always safe progress; without one the inner solve terminates, keeps the
        // current iterate, and reports the elevated residual honestly.
        if (lam <= lambda_min)
        {
          if (trial_ok && allFinite(gt))
          {
            // with CLIMA_TIKH the objective norm includes the penalty term
            const double n0 = augNorm2(g, x), n1 = augNorm2(gt, xt);
            if (n1 <= n0) { lam_acc = lam; x_acc = xt; }
          }
          break;
        }
        lam = std::max(0.5*lam, lambda_min);                                 // reject -> halve the damping
      }
      if (lam_acc <= 0.0) break;                // no admissible step
      x = x_acc; lambda = lam_acc;
    }
    if (dbg)
    {
      double gmax = 0.0; for (double e : g) gmax = std::max(gmax, std::abs(e));
      std::fprintf(stderr, "  [nleq-ratio] iters=%d lambda=%.3e ||g||inf=%.3e\n", it, lambda, gmax);
    }
  }
  else
  {
    RceHybrjFunctor functor;
    functor.m_ = static_cast<int>(m);
    functor.resfn = [&](const Eigen::VectorXd& xe, Eigen::VectorXd& fvec) {
      std::vector<double> xv(m); for (size_t r=0;r<m;++r) xv[r] = xe[(int)r];
      std::vector<double> gg; evalG(xv, /*want_jac=*/false, gg);
      if (lam_s > 0.0) addSmoothG(buildProfile(xv), gg);   // smoothness penalty (cheap; no RT)
      fvec.resize(m); for (size_t r=0;r<m;++r) fvec[(int)r] = gg[r];
    };
    // Analytic Planck-only Jacobian by default (fast). It is slightly ill-conditioned (a spurious
    // near-null smooth mode) but hybrj handles it, and the finite-difference Jacobian (CLIMA_FDJAC,
    // m extra RT solves/step, clima's choice) gives the SAME converged profile -- so the residual
    // sawtooth is intrinsic to the flux-divergence residual, not the Jacobian.
    const bool analytic_jac = std::getenv("CLIMA_FDJAC") == nullptr;
    functor.jacfn = [&, analytic_jac](const Eigen::VectorXd& xe, Eigen::MatrixXd& fjac) {
      std::vector<double> xv(m); for (size_t r=0;r<m;++r) xv[r] = xe[(int)r];
      fjac.resize(m, m);
      if (analytic_jac)
      {
        std::vector<double> gg; evalG(xv, /*want_jac=*/true, gg);   // analytic Planck-only Jacobian at xv
        rebuildJ();
        for (size_t r=0;r<m;++r) for (size_t c=0;c<m;++c) fjac((int)r,(int)c) = J[r*m+c];
      }
      else
      {
        // FINITE-DIFFERENCE Jacobian, opacity frozen (clima's approach -- the TRUE dg/dx). The analytic
        // net_flux_jacobian carries a spurious near-null SMOOTH mode (cond ~2e5 vs FD ~1e2) that misleads
        // hybrj and leaves the checkerboard; the FD Jacobian is well-conditioned and suppresses it.
        std::vector<double> g0; evalG(xv, /*want_jac=*/false, g0);
        for (size_t c = 0; c < m; ++c)
        {
          std::vector<double> xp(xv); const double dd = 1e-3 * std::max(std::abs(xv[c]), 1.0); xp[c] += dd;
          std::vector<double> gp; evalG(xp, /*want_jac=*/false, gp);
          for (size_t r=0;r<m;++r) fjac((int)r,(int)c) = (gp[r]-g0[r])/dd;
        }
      }
      if (lam_s > 0.0) addSmoothJac(fjac);   // analytic Laplacian of the smoothness penalty (consistent with resfn)
    };

    // ---- CLIMA_FPSTEP: PURE DAMPED FIXED-POINT defect correction, bypassing MINPACK entirely.
    // hybrj is a trust-region dogleg that ASSUMES fjac = dG/dx: it validates each step by an
    // actual-vs-predicted reduction ratio, takes its gradient leg from J^T G, and applies Broyden
    // updates. Defect correction deliberately supplies M != dG/dx, so the ratio test fails
    // systematically, the trust radius collapses and every step is rejected -- exactly the observed
    // info=4/5 with max|dT/T| = 0. Sec.7.1 instead prescribes  T <- T - M^-1 G(T)  with a line
    // search on ||G|| only. This is that iteration, so the stall can be attributed to the OPERATOR
    // rather than to the globalisation. Pair with CLIMA_HSTEP to test M = collocated heating H.
    {
    Eigen::VectorXd xe(m); for (size_t r=0;r<m;++r) xe[(int)r] = x[r];
    Eigen::HybridNonLinearSolver<RceHybrjFunctor> solver(functor);
    solver.parameters.maxfev = std::getenv("CLIMA_MAXFEV") ? std::atoi(std::getenv("CLIMA_MAXFEV")) : 120;
    solver.parameters.xtol   = std::getenv("CLIMA_XTOL")   ? std::atof(std::getenv("CLIMA_XTOL"))   : 1e-8;
    const int info = solver.solve(xe);
    for (size_t r=0;r<m;++r) x[r] = std::max(1.0, xe[(int)r]);
    if (dbg)
      std::fprintf(stderr, "  [hybrj] info=%d nfev=%d njev=%d fnorm=%.3e\n",
        info, (int)solver.nfev, (int)solver.njev, solver.fvec.blueNorm());
    }
  }

  // ---- UNSOELD-LUCY LEVEL ANCHOR for the collocated (ratio) residual, applied AFTER the Newton.
  // The ratio system is SQUARE (zone budget + one local-RE row per radiative level), so there is no free
  // dimension for the TOA constraint, and Sec.4 guarantees the local-RE root is not flux conserving. The
  // consequence is ONE unsatisfiable constraint whose error can only be MOVED, not removed (measured,
  // N=200): with this shift the TOA closes to 1e-5 W/m^2 and the zone budget carries 0.194 W/m^2; without
  // it (or applying it pre-solve, which the Newton simply undoes) the zone closes and the TOA carries
  // 0.198 W/m^2. Default: pin the TOA, since it is the primary energy-balance diagnostic. Use
  // CLIMA_RATIO_NOUL to move the error to the TOA instead. A weak un-differenced flux term in the residual
  // (CLIMA_LREANCHOR) does NOT help here -- its Jacobian row is the deficient dense NFJ and it makes the
  // TOA imbalance worse (0.198 -> 0.236 -> 0.543 W/m^2 at eps = 0, 0.02, 0.10).
  if (ratio && !lucy && !radiation_field.net_flux_jacobian.empty())
  {
    std::vector<double> gtmp;
    evalG(x, /*want_jac=*/true, gtmp);            // refresh the radiation field + Jacobian AT x
    const size_t itoa = n - 1;
    double denom = 0.0;
    for (size_t j = 0; j < n; ++j) denom += radiation_field.net_flux_jacobian[itoa][j] * T_base[j];
    const double Ftoa = radiation_field.flux_total[itoa] - target_flux;
    if (std::abs(denom) > 1e-30)
    {
      double alpha = -Ftoa / denom;
      alpha = std::max(-0.1, std::min(0.1, alpha));
      for (size_t r = 0; r < m; ++r) x[r] = std::max(1.0, x[r] * (1.0 + alpha));
      if (dbg) std::fprintf(stderr, "  [ratio-UL] Ftoa=%.4e alpha=%.4e\n", Ftoa, alpha);
    }
  }

  // ---- commit the accepted profile; final TRUE eval so the model state (T, opacity, radiation
  //      field) is consistent at the committed point for the driver ----------------------------
  std::vector<double> T_final = buildProfile(x);

  // ---- DEEP GRADIENT INTEGRATION (doc Sec.6.5/6.6): reconstruct the carved-out deep [0..kdeep], which
  // buildProfile left frozen, by integrating the thermal-flux-rescaled Planck gradient DOWN from the
  // photosphere anchor at kdeep+1 (a Newton DOF). PURE integration (no Newton/PTC/blend). On ngam's coarse
  // deep grid the Jacobian off-diagonal vanishes (~e^-dtau), so D is not read from it: the Planck increment is
  // rescaled by the carried-flux ratio F_star/F_th (Eq.28), an O(1) factor stationary at flux conservation
  // (F_tot->0). At a FLAT layer that ratio is 0/0 (measured D undefined) and the old linear-extrapolation
  // band-aid fixed T but left the flux seamed (the F_net[1] surface-air spike). The fix (doc Sec.6.6 grey-floor
  // fallback): there switch to the ABSOLUTE flux-carrying gradient dB = F_star*dtau/K, K read from the clean
  // deep above (the solver's own diffusivity), which carries the target flux through the flat layer and closes
  // the seam. The DOWNWARD anchor (cold photosphere) keeps the surface a stiff bottom leaf -- the bulk deep does
  // not follow T0 -- so there is no leaf->root runaway (the SURFACE anchor + upward sweep does run away because
  // its 1-D K00 misses the deep-following term). The historical alternative was a coupled surface+deep BC, held
  // in reserve should a residual non-diffusive surface-jump seam survive the grey-floor fallback.

  // NOTE: a post-hoc Shapiro filter on the committed profile used to live here. It is REMOVED, not
  // merely defaulted off: it mutates the profile AFTER the solve, so the Newton never sees it and the
  // two fight -- measured, it floors the residual at ~6e-4 and the run can never converge (the inner
  // solve reaches 1e-13 and the filter hands the error straight back). The sawtooth is instead cured
  // at its source by the collocated ratio residual, whose Nyquist eigenvalue is O(1) rather than ~0.
  // If cosmetic smoothing of the FINAL output is ever wanted, do it outside the solver loop.

  std::vector<double> Ffin, NHfin;
  forward_eval_full_(T_final, /*recompute_opacity=*/true, /*compute_jacobian=*/false, Ffin, NHfin);
  snapshotBandJ();   // refresh the closure's Jbar with the committed radiation field (metric below)

  // how much this inner solve moved the profile -> gates the mask re-detection on the NEXT call
  double inner_change = 0.0;
  for (size_t i = 0; i < n; ++i)
    inner_change = std::max(inner_change, std::abs(T_final[i] - T_base[i]) / std::max(T_final[i], 1.0));
  last_inner_change_ = inner_change;

  // MLT_RELAX (diagnostic knob): under-relax the commit, T <- T_old + gamma (T_root - T_old).
  // Damps the OUTER operator-split iteration when the composition feedback (Manabe-Wetherald
  // water, ~6%/K) displaces the frozen-opacity root faster than the split contracts -- the mask
  // scheme never sees this because the affected near-surface levels are slaved (no local-RE rows).
  if (mlt)
  {
    static const double mlt_relax = std::getenv("MLT_RELAX") ? std::atof(std::getenv("MLT_RELAX")) : 1.0;
    if (mlt_relax > 0.0 && mlt_relax < 1.0)
      for (size_t i = 0; i < n; ++i)
        T_final[i] = T_base[i] + mlt_relax * (T_final[i] - T_base[i]);
  }
  atmosphere.temperature = T_final;
  // MLT: expose the convective region diagnostically (levels bounding an active-flux link) -- the
  // mask machinery is bypassed, so this is the analogue of AGNI's mask_c, for output/driver display.
  if (mlt)
  {
    std::fill(atmosphere.convective.begin(), atmosphere.convective.end(), 0);
    for (size_t li = 0; li + 1 < n; ++li)
      if (mltLink(T_final, li) > 0.0)
      { atmosphere.convective[li] = 1; atmosphere.convective[li+1] = 1; }
    for (size_t i = 0; i < n; ++i)               // Fix A: the slaved deep block is convective
      if (prev_mask_[i]) atmosphere.convective[i] = 1;
  }

  // ---- convergence metric (clima form, multi-stream-honest tolerance + a settled gate). PRIMARY = the
  //      max per-level net-flux imbalance / incident stellar flux. This is the CREEP-PROOF check: a deep
  //      that is still drifting toward equilibrium carries an elevated per-level flux until it actually
  //      gets there, so it cannot falsely pass (unlike a temperature-settled-only test, which is blind to
  //      a slow drift). The FLOOR of this residual is NOT zero: a faithful multi-stream, sampled-opacity
  //      solver leaves a ~1% (few W/m^2) per-layer flux divergence in the deep (the diffusion-limit doc's
  //      local-RE-vs-flux-conservation inconsistency) that only a two-stream solver like clima drives to
  //      1e-5. So the convergence_threshold must be set just ABOVE that floor (~1e-3, observe it per case),
  //      NOT 1e-5. SECONDARY gate: max|dT/T| this step (inner_change) is folded in via max(), so a profile
  //      sitting near the flux threshold but still moving is not declared converged. Held at 1.0 while the
  //      convective mask is making a real (>band) move. --------------------------------------------------
  double flux_resid = 0.0;
  const double fnorm = std::max({1.0, std::abs(target_flux),
    radiation_field.flux_down_total.empty() ? 0.0 : std::abs(radiation_field.flux_down_total.back())});
  for (size_t r = 0; r < m; ++r)
  {
    const size_t i = unk[r];
    const int zi = zone_of_dof[i];
    double res;
    // same bottom-boundary convention as assembleG: at the domain bottom the inflow is F_star
    if (zi >= 0) { const Zone& z = zones[zi]; const double low = (z.lower==0)?target_flux:Ffin[z.lower-1];
                   res = Ffin[z.upper] + ((mlt && z.upper+1 < n) ? mltLink(T_final, z.upper) : 0.0) - low; }
    // MLT: conservation is a statement about the TOTAL flux F_rad + F_c (in a convective zone the
    // radiative flux alone is legitimately non-constant). Matches the assembleG rows.
    else if (i == 0) res = Ffin[0] + (mlt ? mltFcLvl(T_final, 0) : 0.0) - target_flux;
    else res = Ffin[i] - Ffin[i-1]
             + (mlt ? (mltFcLvl(T_final, i) - mltFcLvl(T_final, i-1)) : 0.0);
    flux_resid = std::max(flux_resid, std::abs(res) / fnorm);
  }
  // the carved-out deep (incl. the surface endpoint when carved) is not in unk, but its flux conservation
  // still gates convergence (creep-proof)
  for (size_t i = 0; i < n; ++i) if (deep[i]) flux_resid = std::max(flux_resid, std::abs(Ffin[i]-target_flux) / fnorm);

  // ---- MODE-AWARE criterion. flux_resid above measures FLUX CONSERVATION, which is the residual the
  // flux/PTC modes actually drive -- but NOT the one a COLLOCATED mode drives. The ratio (local-RE) mode
  // converges to the local-RE root, and the diffusion-limit doc's Sec.4 guarantees that root is not flux
  // conserving at finite resolution: measured at n=200 it sits at flux_resid ~ 5.7e-4 (a smooth, one-signed
  // 0.19 W/m^2 bias above the RCB, zero sign changes) while its OWN residual and max|dT/T| are ~1e-14.
  // Judging it by flux_resid is therefore a metric-residual mismatch, not a stall -- it reports "not
  // converged" for a fully converged profile. So gate each mode on the residual IT drives, by re-assembling
  // the residual vector at the committed point. flux_resid is still computed and reported: it is the honest
  // error bar (the distance between the two roots), just not the convergence test for a collocated mode.
  double own_resid = flux_resid;
  const bool collocated = (ratio || localre || newtonlike);
  size_t own_arg = 0;                     // level index of the dominant residual row (diagnostic)
  if (collocated)
  {
    std::vector<double> gfin;
    assembleG(Ffin, NHfin, T_final, gfin);
    own_resid = 0.0;
    for (size_t r = 0; r < gfin.size(); ++r)
      if (std::abs(gfin[r]) > own_resid) { own_resid = std::abs(gfin[r]); own_arg = unk[r]; }
  }
  if (dbg)
    std::fprintf(stderr, "  [converge] own_resid=%.3e @lv%zu  flux_resid=%.3e (error bar)  dT/T=%.3e  mode=%s\n",
                 own_resid, own_arg, flux_resid, inner_change, collocated ? "collocated" : "flux");
  // CLIMA_ROWDUMP: one-shot per-level anatomy of the committed collocated residual -- per-layer
  // Planck-mean dtau, the ratio term gre, the Lucy term glucy, and the flux error. Separates the
  // optically-thick band (dtau >> 1, gre blind) from the photosphere.
  if (collocated && std::getenv("CLIMA_ROWDUMP"))
  {
    static bool row_dumped = false;
    if (!row_dumped)
    {
      row_dumped = true;
      std::vector<double> lucyB; lucyVec(Ffin, T_final, lucyB);
      std::fprintf(stderr, "  [rowdump] lv P[bar] T dtau gre glucy (F-F*)/F* type\n");
      for (size_t i = 0; i < n; ++i)
      {
        double num, den, C; ratioSums(T_final, i, num, den, C);
        const double gre = (den > 0.0) ? (num/den - 1.0) : 0.0;
        const double Btot = 5.670374e-5*std::pow(std::max(T_final[i],1.0),4)/M_PI;
        const double glucy = lucy_sign * lucyB[i] / std::max(Btot, 1e-30);
        const char* type = slaved[i] ? "slaved" : (zone_of_dof[i] >= 0 ? "zone" :
          (i == 0 ? "bottom" : ((slaved[i-1] || zone_of_dof[i-1] >= 0) ? "handover" : "ratio")));
        std::fprintf(stderr, "  [rowdump] %zu %.4e %.2f %.4e %+.4e %+.4e %+.4e %s\n",
          i, atmosphere.pressure[i], T_final[i], nl_dtau[i], gre, glucy,
          (Ffin[i]-target_flux)/fnorm, type);
      }
    }
  }
  // MLT ramp bookkeeping: gate the next stage on the RAW residual, and hold the reported residual
  // above the driver threshold while convection is still weakened (a converged solution at
  // sf < 1 is not a solution of the full problem).
  mlt_prev_resid_ = own_resid;
  // CLIMA_TIKH mode-aware gate: the scheme drives the AUGMENTED objective, whose stationary
  // point retains an O(alpha) compromise in max|g| by design (measured: floors at 2.5e-3 for
  // alpha=1e-3 with the profile dead stationary, dT/T ~ 1e-14, for 80 wasted iterations).
  // own_resid stays reported as the honest physical error bar; convergence is judged on
  // stationarity (inner_change), which is creep-proof through the outer opacity refresh.
  // Tikhonov gate amended: stationarity ALONE endorsed a Wien cold trap (flux+MLT+Tikh BD:
  // every step rejected at a grossly wrong state -> inner_change = 0 -> "converged" at
  // T_bot 549 K radiating 0.05% of F_int). Stationarity must be paired with a LOOSE sanity
  // ceiling on the physical residual: generous enough for the O(alpha) penalty floor
  // (~2.5e-3 measured), tight enough that a saturated flux residual (|g| -> 1 in the trap)
  // can never pass.
  last_residual_ = mask_big_change ? 1.0
    : (tikh_alpha > 0.0 ? std::max(inner_change, (own_resid < 0.05 ? 0.0 : own_resid))
                        : std::max(own_resid, inner_change));
  if (mlt && mlt_sf_ < 1.0) last_residual_ = std::max(last_residual_, 0.5);
}


}
