#include "linearised_temperature_correction.h"

#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include "../atmosphere/atmosphere.h"
#include "../radiative_transfer/radiative_transfer.h"
#include "../transport_coeff/opacity_calc.h"
#include "../spectral_grid/spectral_grid.h"
#include "../additional/quadrature.h"
#include "../additional/thermodynamic_data.h"
#include "../convection/convection.h"

// DISORT's own monochromatic Planck (and its derivative), so the local-RE inversion
// uses exactly the source DISORT emits.
#include "../../_deps/disortpp-src/src/Planck.hpp"


namespace ngam {


namespace {

// Dense Gaussian elimination with partial pivoting. Solves A x = b in place
// (b <- x); A is row-major n*n. Returns false if singular.
bool solveDense(std::vector<double>& A, std::vector<double>& b, const size_t n)
{
  for (size_t k=0; k<n; ++k)
  {
    size_t piv = k;
    double pmax = std::abs(A[k*n+k]);
    for (size_t i=k+1; i<n; ++i)
    {
      const double v = std::abs(A[i*n+k]);
      if (v > pmax) { pmax = v; piv = i; }
    }
    if (pmax == 0.) return false;

    if (piv != k)
    {
      for (size_t j=k; j<n; ++j) std::swap(A[k*n+j], A[piv*n+j]);
      std::swap(b[k], b[piv]);
    }

    const double akk = A[k*n+k];
    for (size_t i=k+1; i<n; ++i)
    {
      const double f = A[i*n+k]/akk;
      if (f == 0.) continue;
      for (size_t j=k; j<n; ++j) A[i*n+j] -= f*A[k*n+j];
      b[i] -= f*b[k];
    }
  }

  for (int i=static_cast<int>(n)-1; i>=0; --i)
  {
    double s = b[i];
    for (size_t j=i+1; j<n; ++j) s -= A[i*n+j]*b[j];
    b[i] = s/A[i*n+i];
  }

  return true;
}


// Direct local-RE Planck inversion at one level: find T with
//   sum_nu w_nu kappa_abs(nu,i) B_nu(T) = num   (= sum_nu w_nu kappa_abs J = absorbed),
// i.e. emitted == absorbed. The left side is monotonic in T, so a damped 1-D Newton with a
// bound lands on the (always finite) equilibrium temperature with no overshoot, however small
// dB/dT is. Used for the optically-thin skin, where the net flux is ~insensitive to the local
// temperature so the flux Newton cannot pin it. Falls back to T_init if there is no absorption.
double solveLocalRE(
  const double num,
  const std::vector<double>& w,
  const std::vector<double>& wavenumber,
  const OpacityCalculation& opacity,
  const size_t level,
  const double T_init)
{
  constexpr double si_to_cgs = 1e3;
  const size_t nb_nu = wavenumber.size();

  if (num <= 0.0) return T_init;

  double T = T_init;
  for (int it=0; it<40; ++it)
  {
    double den = 0.0, dden = 0.0;
    for (size_t k=0; k<nb_nu; ++k)
    {
      const double wk = w[k] * opacity.absorption_coeff[k][level];
      const double wn = wavenumber[k];
      den  += wk * disortpp::planckFunction2(wn, wn, T)      * si_to_cgs;
      dden += wk * disortpp::planckFunctionDeriv2(wn, wn, T) * si_to_cgs;
    }
    if (dden <= 0.0) break;

    double dT = (num - den) / dden;               // Newton on (den(T) - num) = 0
    if (dT >  0.5 * T) dT =  0.5 * T;              // damp / bound (monotonic -> safe)
    if (dT < -0.5 * T) dT = -0.5 * T;
    T += dT;
    if (T < 1.0) T = 1.0;

    if (std::abs(dT) < 1e-4 * T) break;
  }
  return T;
}

}



// Full-linearisation radiative-convective-equilibrium temperature correction (PICASO t_start
// formulation, using DISORT's analytic temperature Jacobian instead of finite differences):
//
//  * the convective zone is the contiguous deep set [0, rcb_], grown only when the inner Newton
//    has settled at near-radiative-equilibrium, and stopping at the true radiative-convective
//    boundary (no fragmentation, no creep);
//  * the Newton unknowns are ONLY the radiative levels -- each convective layer is slaved to the
//    adiabat from the radiative level just above its zone (the anchor), and its flux response is
//    folded analytically into the anchor's Jacobian column (small, well-conditioned all-flux
//    system; the entropy is set by radiative equilibrium above the zone);
//  * per outer DISORT solve, an inner Newton loop iterates the LINEARISED flux F + J*dT to
//    convergence with a backtracking line search (cheap -- no DISORT re-solve), so the expensive
//    outer iterations advance at the true Newton rate;
//  * the optically-thin skin (where the net flux is insensitive to the local T) is set by direct
//    local-RE Planck inversion rather than the flux Newton.
void LinearisedTemperatureCorrection::calcCorrection(
  const double surface_gravity,
  Atmosphere& atmosphere,
  const RadiativeTransferOutput& radiation_field,
  const OpacityCalculation& opacity)
{
  const size_t n = atmosphere.temperature.size();

  // flux normalisation G = (F_net - target_flux) / fscale. fscale = target_flux for the gas planet
  // / BD; an external scale (e.g. absorbed stellar) when target_flux = 0 (terrestrial). The flux
  // Newton runs whenever we have such a scale.
  const double fscale = (flux_scale > 0.0) ? flux_scale : target_flux;
  const bool use_flux = (fscale > 0.0);

  // trust-region prototype (env LIN_TRUST, surface-anchored only): solve the COUPLED flux-balance
  // root -- all optically-thick radiative levels plus the surface as Newton unknowns, the skin on
  // local RE, the convective zone slaved -- with a Levenberg-Marquardt trust region instead of the
  // default 1-DOF-surface PTC. This is clima's hybrj-on-the-flux-balance approach (Wogan et al.
  // 2025): the residual is the bare net-flux imbalance (no thermal-inertia weighting -> no stiffness)
  // and the trust region globalises the otherwise-divergent target=0 coupled Newton.
  const bool trust = surface_anchored && std::getenv("LIN_TRUST") != nullptr;
  trust_active_ = trust;

  // CLIMA_TIKH active? Read at function scope: the zone-nucleation gate below must not use a
  // FOREIGN metric -- under the Tikhonov objective the flux residual has an irreducible
  // O(alpha) floor by construction (measured 1.44e-3 at alpha=1e-3, permanently above the
  // 1e-3 gate), so gating nucleation on it asks for convergence in a currency the active
  // scheme cannot pay. Gate on settle alone in that mode.
  static const bool tikh_on = [] {
    const char* e = std::getenv("CLIMA_TIKH");
    return e != nullptr && std::atof(e) > 0.0; }();

  // experiment (env LIN_PERLEVEL): solve the surface-anchored case with the SAME scheme the
  // self-luminous (brown dwarf) path uses -- the PER-LEVEL net-flux residual + inner Newton with a
  // backtracking line search on the linear model -- instead of the bespoke divergence/NLEQ-ERR path.
  // The surface (lv 0) stays a flux unknown, so F_net[0]=0 anchors the absolute level (the role
  // sigma*T_int^4 plays for the brown dwarf). Tests whether the working BD solver also works here.
  const bool per_level = trust && std::getenv("LIN_PERLEVEL") != nullptr;

  // clima-style heat-capacity-weighted divergence residual (env LIN_HEATCAP, default ON in trust).
  // clima's Newton residual is the HEATING RATE dT/dt = (F[i]-F[i-1]) / c_eff, c_eff = rho*c_p*dz =
  // (dP/g)*c_p, not the bare flux divergence. The 1/c_eff weighting cancels the density/thickness
  // scaling of the raw-divergence diagonal (~ kappa*dB/dT*dz, which vanishes in thin upper layers and
  // leaves them unpinned -> the flat stratosphere + the local-RE-skin seam); the weighted diagonal
  // (~ sigma*dB/dT/c_p) stays O(1) all the way up, so the WHOLE radiative column is pinned by ONE
  // Newton and the local-RE skin is no longer needed. Set LIN_HEATCAP=0 to fall back to /fscale + skin.
  const bool heatcap_weight = trust && !per_level
    && (!std::getenv("LIN_HEATCAP") || std::atof(std::getenv("LIN_HEATCAP")) != 0.0);

  const std::vector<double>& wavenumber = radiation_field.spectral_grid->wavenumber_list;
  const size_t nb_nu = wavenumber.size();
  const std::vector<double> w = aux::trapezoidalWeights(wavenumber);

  // ---- grey (wavenumber-mean extinction) optical depth from the top -> regime weight zeta -----
  double wsum = 0.0; for (double wk : w) wsum += wk;
  std::vector<double> kappa_grey(n, 0.0);
  for (size_t i=0; i<n; ++i)
  {
    double k = 0.0;
    for (size_t kk=0; kk<nb_nu; ++kk)
      k += w[kk] * (opacity.absorption_coeff[kk][i] + opacity.scattering_coeff[kk][i]);
    kappa_grey[i] = (wsum > 0.0) ? k / wsum : 0.0;
  }
  std::vector<double> tau(n, 0.0);
  for (int i=static_cast<int>(n)-2; i>=0; --i)
    tau[i] = tau[i+1] + 0.5 * (kappa_grey[i] + kappa_grey[i+1])
                            * std::abs(atmosphere.altitude[i+1] - atmosphere.altitude[i]);
  std::vector<double> zeta(n, 0.0);
  for (size_t i=0; i<n; ++i)
    zeta[i] = use_flux ? tau[i] / (tau[i] + tau_scale) : 0.0;

  // zeta below this = optically-thin skin: the flux Newton is too weakly conditioned there (it is
  // set by local RE instead), and convection cannot occur in it.
  const double skin_zeta = 0.1;

  // midpoint convective gradient between two levels, taken from the active convection scheme so the
  // corrector enforces exactly that scheme's stratification (dry, moist, ...). Only ever called
  // when a convection scheme is present (growth is gated on it; the slaving runs only for layers a
  // zone already contains).
  auto nablaAd = [&](size_t i, size_t j) -> double
  {
    return 0.5 * (
      convection->convectiveGradient(
        atmosphere.number_densities[i], atmosphere.temperature[i], atmosphere.pressure[i]) +
      convection->convectiveGradient(
        atmosphere.number_densities[j], atmosphere.temperature[j], atmosphere.pressure[j]));
  };

  // ---- convection: contiguous deep-rooted zone [0, rcb_], grown when settled -----------------
  // The zone is kept CONTIGUOUS from the base (PICASO's nstr boundary, not a per-layer flag set):
  // a radiative pocket trapped inside the deep is optically thick, so its flux Newton is weakly
  // conditioned and oscillates. We grow rcb_ only once the inner Newton has settled (max|dT/T| <
  // grow_tol) AND -- after the first zone exists -- the radiative flux has converged for the
  // present zone (so the boundary gradient is the true sharp one, not a transient that would make
  // the zone creep past the photosphere). Scanning from the base, the zone extends through the
  // sustained super-adiabatic region and stops at the first sub-adiabatic radiative layer. Grow-only.
  constexpr double grow_tol      = 1e-3;
  constexpr double flux_grow_tol = 1e-3;
  if (conv_set_.size() != n) conv_set_.assign(n, 0);
  zones_grew_ = false;
  // grow once the present zone has SETTLED (max|dT/T| < grow_tol). Self-luminous (gas/BD) also
  // requires the radiative flux to have converged, so the boundary gradient is the true sharp one
  // (and a transient hot deep can't lock in). A surface-driven troposphere (terrestrial) can never
  // reach radiative flux balance -- the would-be convective layers keep a finite radiative
  // imbalance -- so there we grow on settle alone (PTC removes the checkerboard, so settle is safe).
  // Tikhonov translation of the self-luminous "flux converged" requirement: the residual can
  // never pass the absolute threshold (O(alpha) floor), but it PLATEAUS there once the present
  // zone's radiative solve is done -- gate growth on that plateau (stationary to 5% across
  // outer iterations). Settle alone is NOT enough: a merely-settled transient is still
  // super-adiabatic above the true RCB and the grow-only scan then ratchets the top upward
  // without limit (measured after the first, settle-only version of this gate).
  const bool tikh_flux_plateau = tikh_on
    && last_flux_residual_ >= 0.0 && tikh_gate_resid_prev_ > 0.0
    && std::abs(last_flux_residual_ - tikh_gate_resid_prev_) < 0.05 * last_flux_residual_;
  tikh_gate_resid_prev_ = last_flux_residual_;
  const bool may_grow = convection != nullptr && use_flux && !trust
    && last_max_dt_frac_ < grow_tol
    && (surface_anchored || tikh_flux_plateau
        || (last_flux_residual_ >= 0.0 && last_flux_residual_ < flux_grow_tol));
  if (may_grow)
  {
    int top = rcb_;
    for (size_t i=0; i+1<n; ++i)
    {
      if (zeta[i] < skin_zeta || zeta[i+1] < skin_zeta) break;   // never into the skin
      const double dlnP = std::log(atmosphere.pressure[i] / atmosphere.pressure[i+1]);  // > 0
      if (dlnP <= 0.0) break;
      const double nabla = std::log(
        atmosphere.temperature[i] / atmosphere.temperature[i+1]) / dlnP;
      const double nabla_ad = nablaAd(i, i+1);
      // include the existing zone (its layers sit on the adiabat, nabla == nabla_ad) and extend
      // through any layer the radiative profile wants steeper than the adiabat
      if (static_cast<int>(i) <= rcb_ || nabla > nabla_ad) top = static_cast<int>(i) + 1;
      else break;
    }
    // boundary-motion limiter (clima): grow the boundary by at most max_shift layers per settle, so
    // the convective perturbation stays small and the PTC step can damp it.
    constexpr int max_shift = 2;
    if ((surface_anchored || tikh_on) && top > rcb_ + max_shift) top = rcb_ + max_shift;
    if (top > rcb_)
    {
      rcb_ = top;
      zones_grew_ = true;
      // re-damp the PTC step after the zone moved: a fresh small dt absorbs the boundary
      // perturbation before dt grows back towards the Newton step.
      if (surface_anchored) ptc_dt_ = -1.0;
    }
    // Tikhonov-mode ANTI-OVERSHOOT RETREAT (ported from the ratio-path boundary controller;
    // this grow-only controller historically never needed one because the converged-flux gate
    // stopped growth at the true boundary). Under the penalty objective the smoothed radiative
    // profile reads marginally super-adiabatic through a wider band, the staircase overshoots,
    // and the signature is a strong COLD INVERSION on the first radiative link above the
    // anchor (measured: nabla = -0.32 at the runaway's end, zone 36 levels vs the true 10).
    // At the same plateaued states the grow scan reads, retreat one level; grow and retreat
    // read the same links at converged states, so at most one fires per settle.
    else if (tikh_flux_plateau && rcb_ >= 1 && rcb_ + 2 < static_cast<int>(n))
    {
      const size_t a = rcb_ + 1;   // the anchor (first radiative level above the zone)
      const double dlnP = std::log(atmosphere.pressure[a] / atmosphere.pressure[a+1]);
      if (dlnP > 0.0)
      {
        const double nab = std::log(
          atmosphere.temperature[a] / atmosphere.temperature[a+1]) / dlnP;
        const double nad = nablaAd(a, a+1);
        if (nad > 0.0 && nab < -0.2*nad)
        {
          rcb_ -= 1;
          zones_grew_ = true;
        }
      }
    }
  }

  // trust mode: determine the convective mask with grow+SHRINK by running the proven convective
  // adjustment on a 1-2-1 SMOOTHED copy of the profile. The smoothing cancels the radiative Newton's
  // period-2 checkerboard so the lapse-rate test is clean; the adjustment retracts over-convection,
  // which a grow-only scan cannot (the radiative-equilibrium profile is super-adiabatic over a huge
  // range, so grow-only claims almost everything). M1 takes the deep contiguous zone [0, rcb_];
  // multiple/detached zones follow in M2.
  // FREEZE the mask while the radiative Newton settles for it; re-detect only between solves (when
  // settled) or when no mask exists yet. Re-detecting every iteration reads the still-moving profile
  // and chatters the boundary into a limit cycle -- this is clima's solve-then-update outer loop.
  if (trust && convection != nullptr && (rcb_ < 0 || last_max_dt_frac_ < grow_tol))
  {
    Atmosphere probe = atmosphere;                 // copy; the convective adjustment's own sweeps
    convection->adjust(probe);                     // smooth residual roughness, so no pre-smoothing
    int new_rcb = -1;                              // (which biased the lapse rate and shrank the zone)
    for (size_t i=0; i<n; ++i) { if (probe.convective[i]) new_rcb = static_cast<int>(i); else break; }
    // boundary-motion limiter (clima): move the radiative-convective boundary by at most 2 layers per
    // re-detection, so a single noisy detection cannot collapse a near-converged zone in one step.
    if (rcb_ >= 0) new_rcb = std::max(rcb_-2, std::min(rcb_+2, new_rcb));
    zones_grew_ = (new_rcb != rcb_);               // mask still moving -> not converged
    rcb_ = new_rcb;
  }

  std::fill(conv_set_.begin(), conv_set_.end(), 0);
  for (int i=0; i<=rcb_ && i<static_cast<int>(n); ++i) conv_set_[i] = 1;
  atmosphere.convective = conv_set_;
  auto convective = [&](size_t i) { return conv_set_[i] != 0; };

  // SLAVED = the convective layers eliminated from the Newton (set onto the adiabat from their
  // anchor). In surface-anchored (terrestrial) mode the deepest level (lv 0 = the surface) is NOT
  // slaved -- it stays a flux unknown so its energy balance F_net[0]=0 is enforced -- and the
  // troposphere is slaved UP from it. Otherwise the whole zone is slaved to the level above.
  auto slaved = [&](size_t i) { return convective(i) && !(surface_anchored && i == 0); };

  // ---- adiabat slaving: anchor and cumulative factor for each slaved layer --------------------
  // T_k = T_anchor * Cfac[k]. Gas/BD: anchor = radiative level just above the zone, Cfac >= 1
  // (deeper hotter, building down). Terrestrial: anchor = lv 0 (the surface), Cfac <= 1 (cooler
  // aloft, building up).
  std::vector<size_t> anchor(n, 0);
  std::vector<double> Cfac(n, 1.0);
  for (size_t k=0; k<n; ++k)
  {
    if (!slaved(k)) continue;
    if (surface_anchored)
    {
      anchor[k] = 0;
      double C = 1.0;
      for (int mlev = 1; mlev <= static_cast<int>(k); ++mlev)
      {
        const double nad = nablaAd(mlev, mlev-1);
        C *= std::pow(atmosphere.pressure[mlev] / atmosphere.pressure[mlev-1], nad);  // < 1
      }
      Cfac[k] = C;
    }
    else
    {
      int top = static_cast<int>(k);
      while (top+1 < static_cast<int>(n) && convective(static_cast<size_t>(top+1))) ++top;
      anchor[k] = static_cast<size_t>(top) + 1;   // skin-gate keeps the zone top below TOA
      double C = 1.0;
      for (int mlev = static_cast<int>(anchor[k]); mlev > static_cast<int>(k); --mlev)
      {
        const double nad = nablaAd(mlev, mlev-1);
        C *= std::pow(atmosphere.pressure[mlev-1] / atmosphere.pressure[mlev], nad);  // > 1
      }
      Cfac[k] = C;
    }
  }

  // ---- local-RE temperature for the optically-thin skin (direct Planck inversion B(T)=<J>_kappa)
  std::vector<double> delta_local(n, 0.0);
  for (size_t i=0; i<n; ++i)
  {
    if (slaved(i)) continue;
    // Surface-anchored (target=0): local RE pins EVERY radiative layer -- flux constancy is
    // degenerate there (the isothermal/collapse mode is a near-null mode of F_net=0), so the deep
    // cannot be left to the flux-Newton or it collapses; only lv 0 (the surface) is excluded, kept
    // as the flux anchor. Self-luminous (target>0): the optically-thick deep is in the diffusion
    // limit (J ~ B, local RE degenerate) and is pinned by the flux-Newton's F_int gradient -- skip.
    if (surface_anchored) { if (i == 0) continue; }
    else if (1.0 - zeta[i] < 1e-3) continue;
    double num = 0.0;                              // absorbed = sum_nu w kappa_abs J
    for (size_t k=0; k<nb_nu; ++k)
      num += w[k] * opacity.absorption_coeff[k][i] * radiation_field.mean_intensity[i][k];
    const double T_local = solveLocalRE(num, w, wavenumber, opacity, i, atmosphere.temperature[i]);
    delta_local[i] = T_local - atmosphere.temperature[i];
  }

  // ---- convergence metric: radiative flux-conservation error (excludes convective layers, where
  //      convection carries the flux, and the optically-thin skin, set by local RE). Held above
  //      the threshold while a zone is still growing, so the driver cannot "converge" on a still-
  //      changing profile. This is what gates convective growth on the next call.
  double max_flux_resid = 0.0;
  if (use_flux)
    for (size_t i=0; i<n; ++i)
      if (!slaved(i) && zeta[i] >= skin_zeta)
        max_flux_resid = std::max(max_flux_resid,
          std::abs((radiation_field.flux_total[i] - target_flux) / fscale));
  if (zones_grew_) max_flux_resid = std::max(max_flux_resid, 1.0);
  last_flux_residual_ = use_flux ? max_flux_resid : -1.0;

  // ---- optional FULL-profile trace (env LIN_PROFILE): T, zeta, flux residual, and the local-RE
  // step dT_localRE (= how far local radiative equilibrium B(T)=<J> wants to move the layer). A
  // large |dT_localRE| flags a layer that is far from local RE -- i.e. where the well-conditioned
  // local-RE constraint should pin T, even if the (degenerate, target=0) flux-Newton has let it
  // drift/collapse. Works regardless of convection (used for the pure-radiative test bed).
  if (std::getenv("LIN_PROFILE") && use_flux)
  {
    std::cout << "  [prof]  lv      P          T        zeta     fluxres   dT_lRE   slaved\n";
    for (size_t i=0; i<n; ++i)
    {
      if (!(i%5==0 || i<6 || i+3>=n)) continue;
      const double fr = (radiation_field.flux_total[i] - target_flux) / fscale;
      std::cout << "  [prof] " << std::setw(3) << i
                << std::scientific << std::setprecision(2)
                << "  " << std::setw(9) << atmosphere.pressure[i]
                << "  " << std::setw(9) << atmosphere.temperature[i]
                << "  " << std::setw(8) << zeta[i]
                << "  " << std::setw(9) << fr
                << "  " << std::setw(9) << delta_local[i]
                << "    " << (slaved(i) ? 1 : 0) << "\n";
    }
  }

  // ---- optional per-layer trace around the radiative-convective boundary (env LIN_CONV) --------
  if (std::getenv("LIN_CONV") && use_flux && convection != nullptr)
  {
    int n_conv = 0, n_super = 0;
    for (int c : conv_set_) n_conv += c;
    for (size_t i=0; i+1<n; ++i)
    {
      const double dlnP = std::log(atmosphere.pressure[i]/atmosphere.pressure[i+1]);
      if (dlnP <= 0) continue;
      const double na = std::log(atmosphere.temperature[i]/atmosphere.temperature[i+1])/dlnP;
      if (na > nablaAd(i, i+1)) ++n_super;
    }
    std::cout << "  [conv] grew=" << zones_grew_ << " last_dT/T=" << last_max_dt_frac_
              << " resid=" << last_flux_residual_
              << " n_super=" << n_super << " n_conv=" << n_conv << "\n";
    size_t top_conv = 0; for (size_t i=0; i<n; ++i) if (conv_set_[i]) top_conv = i;
    const size_t hi = std::min(n-1, top_conv + 6);
    std::cout << "  [conv]  lv  conv     P         T       zeta    nabla   nabla_ad  fluxres\n";
    for (size_t k=0; k<=hi; ++k)
    {
      const size_t i = hi - k;
      double na = 0.0, nad = 0.0;
      if (i+1 < n)
      {
        const double dlnP = std::log(atmosphere.pressure[i]/atmosphere.pressure[i+1]);
        if (dlnP > 0) na = std::log(atmosphere.temperature[i]/atmosphere.temperature[i+1])/dlnP;
        nad = nablaAd(i, i+1);
      }
      const double fr = (radiation_field.flux_total[i] - target_flux) / fscale;
      std::cout << "  [conv] " << std::setw(3) << i << "   " << conv_set_[i]
                << std::scientific << std::setprecision(2)
                << "  " << std::setw(9) << atmosphere.pressure[i]
                << "  " << std::setw(9) << atmosphere.temperature[i]
                << "  " << std::setw(8) << zeta[i]
                << "  " << std::setw(8) << na
                << "  " << std::setw(8) << nad
                << "  " << std::setw(9) << fr << "\n";
    }
  }

  // sets the slaved layers onto the adiabat from their (already-updated) anchor
  auto adiabatSnap = [&](std::vector<double>& T)
  {
    for (size_t k=0; k<n; ++k)
      if (slaved(k) && anchor[k] < n)
        T[k] = std::max(1.0, T[anchor[k]] * Cfac[k]);
  };

  double max_dt_frac = 0.0;   // max |dT/T| applied this call over non-skin layers -> gates growth
  const std::vector<double> T_curr = atmosphere.temperature;   // state the RT (F, J) was solved at

  if (!use_flux)
  {
    // no flux scale at all: pure local-RE relaxation of the radiative layers
    for (size_t i=0; i<n; ++i)
    {
      if (slaved(i)) continue;
      double d = relaxation * delta_local[i];
      if (max_change_fraction > 0.0)
      {
        const double cap = max_change_fraction * atmosphere.temperature[i];
        if (std::abs(d) > cap) d = std::copysign(cap, d);
      }
      atmosphere.temperature[i] = std::max(1.0, atmosphere.temperature[i] + d);
      if (zeta[i] >= skin_zeta)
        max_dt_frac = std::max(max_dt_frac, std::abs(d) / atmosphere.temperature[i]);
    }
    adiabatSnap(atmosphere.temperature);
    last_max_dt_frac_ = max_dt_frac;
    return;
  }

  // ---- pseudo-transient continuation (PTC) Newton step (clima-style) --------------------------
  // One damped Newton step per RT solve: solve (C/dt + J) s = -(F_net - target_flux), where J is
  // the reduced net-flux Jacobian (slaved layers folded into their anchor), C is the per-level
  // column heat capacity, and dt a PSEUDO-time grown as the flux residual falls. Small dt makes the
  // system diagonally dominant -- an implicit time step (= the old time-stepping: robust, no
  // checkerboard); dt -> infinity recovers the pure Newton step (fast). The skin is held on local
  // RE; convective layers are slaved to the adiabat.
  // Which constraint sets each radiative level. Surface-anchored (target=0): local RE pins ALL
  // radiative layers (well-conditioned when target=0); only lv 0 (the surface) is a flux unknown,
  // anchoring the absolute level via the surface balance F_net[0]=0. Self-luminous (target>0): only
  // the optically-thin skin is on local RE; the thick deep is the flux-Newton's job (F_int gradient).
  // trust mode promotes ALL optically-thick radiative levels (incl. lv 0, the surface, whose
  // F_net[0]=0 row anchors the absolute level) into the flux-Newton, leaving only the skin on local
  // RE -- the coupled system clima root-finds. The default surface-anchored path keeps just the
  // surface in the Newton and relaxes the rest by per-layer local RE (slow Gauss-Seidel).
  // heat-capacity weighting lets the Newton reach much deeper into the optically-thin layers (it
  // un-sticks the moderately-thin band that the Tikhonov floor used to swamp), so the local-RE skin
  // is narrowed to only the GENUINELY transparent top (zeta < ~0.03), where net_heating is ~noise and
  // the skin (local RE = skin temperature) is the right physics. This shrinks the old zeta=0.1 seam
  // to the top few near-vacuum levels where the Newton and local-RE solutions nearly coincide.
  const double skin_zeta_use = heatcap_weight
    ? (std::getenv("LIN_SKINZETA") ? std::atof(std::getenv("LIN_SKINZETA")) : 0.03)
    : skin_zeta;
  auto useLocalRE = [&](size_t i) {
    if (slaved(i)) return false;
    if (surface_anchored && !trust) return (i != 0);
    return zeta[i] < skin_zeta_use;
  };
  auto isFluxUnknown = [&](size_t i) {
    if (slaved(i)) return false;
    if (surface_anchored && !trust) return (i == 0);
    return zeta[i] >= skin_zeta_use;
  };

  std::vector<double> T_work = T_curr;
  for (size_t i=0; i<n; ++i)
    if (useLocalRE(i))
    {
      // local-RE skin step. In trust/NLEQ mode (no outer cap) cap it to a fraction of T per call: the
      // optically-thin skin can have no stable local-RE fixed point (a near-isothermal skin chasing
      // the deep field), so an uncapped full step compounds into a runaway. The PTC path keeps the
      // outer per-iteration cap, so it leaves the skin step uncapped here (unchanged behaviour).
      double d = relaxation * delta_local[i];
      if (trust)
      { const double cap = 0.2 * T_curr[i]; if (std::abs(d) > cap) d = std::copysign(cap, d); }
      T_work[i] = std::max(1.0, T_curr[i] + d);
    }

  // Flux-LEVEL (Unsoeld-Lucy) correction (surface-anchored / target=0 only): local RE sets the
  // profile SHAPE (dF/dtau -> 0) but not the absolute flux LEVEL -- it leaves the conserved net flux
  // at some non-zero constant, so the column cools toward F_net=0 only very slowly. Drive the top
  // net flux F_net(top) -> 0 directly by a near-uniform FRACTIONAL shift T -> T*(1+alpha) of the
  // radiative column, with alpha the Newton step of the top flux w.r.t. a uniform log-T change:
  //   alpha = -F_net(top) / sum_j (dF_net(top)/dT_j) * T_j.
  // Self-luminous (target>0) sets its level through the F_int flux-Newton, so this is skipped there.
  // Trust mode also skips it: the coupled LM already carries the surface as an unknown (its F_net[0]=0
  // row sets the level), and a separate uniform pre-shift would confound the trust-region ratio test.
  if (surface_anchored && !trust)
  {
    const size_t itoa = n - 1;
    double denom = 0.0;
    for (size_t j=0; j<n; ++j)
      denom += radiation_field.net_flux_jacobian[itoa][j] * T_curr[j];
    const double F_toa = radiation_field.flux_total[itoa] - target_flux;
    if (std::abs(denom) > 1e-30)
    {
      double alpha = -F_toa / denom;                       // fractional uniform-T Newton step
      alpha = std::max(-0.1, std::min(0.1, alpha));        // bound the per-iteration shift
      // shift the WHOLE radiative column (incl. the surface lv 0), so the local-RE profile shape is
      // preserved while the level moves; excluding the surface opens a cliff at the bottom.
      for (size_t i=0; i<n; ++i)
        if (!slaved(i)) T_work[i] = std::max(1.0, T_work[i] * (1.0 + alpha));
    }
  }
  adiabatSnap(T_work);

  std::vector<size_t> unk;   // flux unknowns: optically-thick radiative levels (self-luminous), or
  for (size_t i=0; i<n; ++i) // just lv 0, the surface anchor (terrestrial)
    if (isFluxUnknown(i)) unk.push_back(i);
  const size_t m = unk.size();

  if (m > 0)
  {
    // ---- shared: reduced Jacobian A = J/fscale + slaved-layer adiabat fold-in + light Laplacian
    // smoothing (suppresses the checkerboard near-null mode of flux constancy at adjacent thick
    // layers, which otherwise creeps the convective boundary; vanishes at convergence) + a tiny
    // Tikhonov floor.
    std::vector<double> A(m*m, 0.0);
    for (size_t r=0; r<m; ++r)
      for (size_t c=0; c<m; ++c)
        A[r*m+c] = radiation_field.net_flux_jacobian[unk[r]][unk[c]] / fscale;
    for (size_t k=0; k<n; ++k)
    {
      if (!slaved(k) || anchor[k] >= n) continue;
      int c = -1;
      for (size_t cc=0; cc<m; ++cc) if (unk[cc] == anchor[k]) { c = static_cast<int>(cc); break; }
      if (c < 0) continue;
      for (size_t r=0; r<m; ++r)
        A[r*m + c] += radiation_field.net_flux_jacobian[unk[r]][k] * Cfac[k] / fscale;
    }
    // reduced flux Jacobian, before any regularisation -- the physical linear model d(F_net)/dT used
    // both for the smoothing row scale and for the PTC linear-model residual prediction G + J*s.
    const std::vector<double> Jred(A);
    {
      constexpr double smooth_factor = 0.3;
      for (size_t r=0; r<m; ++r)
      {
        double row_max = 0.0;
        for (size_t c=0; c<m; ++c) row_max = std::max(row_max, std::abs(Jred[r*m+c]));
        const double lam = smooth_factor * row_max;
        if (r > 0   && unk[r-1] == unk[r]-1) { A[r*m+r] += lam; A[r*m+(r-1)] -= lam; }
        if (r+1 < m && unk[r+1] == unk[r]+1) { A[r*m+r] += lam; A[r*m+(r+1)] -= lam; }
      }
    }
    double diag_mean = 0.0;
    for (size_t k=0; k<m; ++k) diag_mean += std::abs(A[k*m+k]);
    diag_mean /= static_cast<double>(m);
    const double reg = 1e-6 * (diag_mean + 1e-300);
    for (size_t k=0; k<m; ++k) A[k*m+k] += (A[k*m+k] >= 0.0 ? reg : -reg);

    if (surface_anchored && trust && !per_level)
    {
      // ---- inner Newton on the NATIVE flux-DIVERGENCE residual. Drive the net HEATING (net_heating =
      // sum_nu w_nu dF_net/dtau, DISORT's own flux divergence -- a true second-difference operator with
      // a NON-zero diagonal ~ T^3 that pins optically-thick layers, unlike the per-level net flux whose
      // diagonal vanishes in the diffusion limit) to zero on the radiative levels, while the surface
      // (unk[0]) keeps the absolute balance F_net[0]=0 to anchor the level. Backtracking line search on
      // the linear model (no DISORT re-solve); the opacity lag is handled by the outer loop. Convective
      // layers are slaved (their step folds into the anchor column with Cfac).
      // row scale per unknown: clima's heat-capacity weighting (1/c_eff, c_eff = (dP/g)*c_p) when
      // heatcap_weight is on -- this is what conditions the thin upper layers and lets the single
      // Newton pin the whole column; else the legacy flux scale (1/fscale).
      std::vector<double> rscale(n, 1.0 / fscale);
      if (heatcap_weight)
        for (size_t i=0; i<n; ++i)
        {
          const double p_lo = (i+1<n) ? 0.5*(atmosphere.pressure[i]+atmosphere.pressure[i+1]) : atmosphere.pressure[i];
          const double p_hi = (i>0)   ? 0.5*(atmosphere.pressure[i]+atmosphere.pressure[i-1]) : atmosphere.pressure[i];
          const double dP = std::abs(p_lo - p_hi) * 1e6;
          const double cp = ThermodynamicData::meanHeatCapacity(
            atmosphere.number_densities[i], atmosphere.temperature[i]);
          const double c_eff = std::max(1e-30, dP / std::max(surface_gravity, 1e-30) * cp);
          rscale[i] = 1.0 / c_eff;
        }

      std::vector<std::vector<double>> Jrow(m, std::vector<double>(n));   // full Jacobian row per unknown
      std::vector<double> r0(m);                                          // base residual per unknown
      for (size_t r=0; r<m; ++r)
      {
        const size_t i = unk[r];
        const double sc = rscale[i];
        if (i == 0)   // surface: absolute net-flux balance F_net[0]=0 (sets the level)
        {
          for (size_t j=0; j<n; ++j) Jrow[r][j] = radiation_field.net_flux_jacobian[0][j] * sc;
          r0[r] = (radiation_field.flux_total[0] - target_flux) * sc;
        }
        else          // radiative layer: heating rate (weighted flux divergence) -> 0
        {
          for (size_t j=0; j<n; ++j) Jrow[r][j] = radiation_field.net_heating_jacobian[i][j] * sc;
          r0[r] = radiation_field.net_heating[i] * sc;
        }
      }

      // reduced Jacobian A: column c is the unknown unk[c]; slaved layers fold into their anchor (Cfac).
      std::vector<double> A(m*m, 0.0);
      for (size_t r=0; r<m; ++r)
        for (size_t c=0; c<m; ++c) A[r*m+c] = Jrow[r][unk[c]];
      for (size_t k=0; k<n; ++k)
      {
        if (!slaved(k) || anchor[k] >= n) continue;
        int c = -1;
        for (size_t cc=0; cc<m; ++cc) if (unk[cc] == anchor[k]) { c = static_cast<int>(cc); break; }
        if (c < 0) continue;
        for (size_t r=0; r<m; ++r) A[r*m+c] += Jrow[r][k] * Cfac[k];
      }
      // NO curvature smoothing: net_heating_jacobian is ALREADY a second-difference (Laplacian)
      // operator, so adding a Laplacian smoothing band-aid double-counts it and CREATES a checkerboard
      // (more smoothing -> worse). Keep only a tiny Tikhonov floor for the linear solve. (LIN_SMOOTH
      // can re-enable the band-aid for experiments; default 0.)
      {
        const double sf = std::getenv("LIN_SMOOTH") ? std::atof(std::getenv("LIN_SMOOTH")) : 0.0;
        if (sf != 0.0)
        {
          const std::vector<double> Araw(A);
          for (size_t r=0; r<m; ++r)
          {
            double row_max = 0.0;
            for (size_t c=0; c<m; ++c) row_max = std::max(row_max, std::abs(Araw[r*m+c]));
            const double lam = sf * row_max;
            if (r>0   && unk[r-1]==unk[r]-1) { A[r*m+r]+=lam; A[r*m+(r-1)]-=lam; }
            if (r+1<m && unk[r+1]==unk[r]+1) { A[r*m+r]+=lam; A[r*m+(r+1)]-=lam; }
          }
        }
        double dm=0.0; for (size_t k=0;k<m;++k) dm+=std::abs(A[k*m+k]); dm/=static_cast<double>(m);
        for (size_t k=0;k<m;++k) A[k*m+k] += (A[k*m+k]>=0.0?1.0:-1.0)*1e-6*(dm+1e-300);
      }

      auto linResid = [&](const std::vector<double>& T)->double {
        double f=0.0;
        for (size_t r=0;r<m;++r){
          double g=r0[r];
          for (size_t j=0;j<n;++j) g += Jrow[r][j]*(T[j]-T_curr[j]);
          f += g*g;
        }
        return 0.5*f;
      };
      std::vector<double> dunk(m), Asol(m*m), Ttrial(n);
      for (int inner=0; inner<40; ++inner)
      {
        const double f_old = linResid(T_work);
        for (size_t r=0;r<m;++r){
          double g=r0[r];
          for (size_t j=0;j<n;++j) g += Jrow[r][j]*(T_work[j]-T_curr[j]);
          dunk[r] = -g;
        }
        Asol = A;
        if (!solveDense(Asol, dunk, m)) break;
        double scale=1.0;
        for (size_t r=0;r<m;++r){ const double fr=std::abs(dunk[r])/T_work[unk[r]]; if (fr>0.30) scale=std::min(scale,0.30/fr); }
        for (size_t r=0;r<m;++r) dunk[r]*=scale;
        double alam=1.0, step_frac=0.0;
        for (int ls=0;ls<20;++ls){
          Ttrial=T_work;
          for (size_t r=0;r<m;++r) Ttrial[unk[r]]=std::max(1.0,T_work[unk[r]]+alam*dunk[r]);
          adiabatSnap(Ttrial);
          if (linResid(Ttrial) <= f_old*(1.0-1e-4*alam) || alam<1e-3) break;
          alam*=0.5;
        }
        for (size_t r=0;r<m;++r){ const double d=alam*dunk[r]; step_frac=std::max(step_frac,std::abs(d)/T_work[unk[r]]); T_work[unk[r]]=std::max(1.0,T_work[unk[r]]+d); }
        adiabatSnap(T_work);
        if (step_frac<1e-6) break;
      }
    }
    else if (false)   // (old LM/NLEQ-ERR manual-divergence path, retained below but disabled)
    {
      std::vector<double> G(m);                    // per-level net flux residual at T_work
      for (size_t r=0; r<m; ++r)
      {
        const size_t i = unk[r];
        double Flin = radiation_field.flux_total[i];
        for (size_t j=0; j<n; ++j)
          Flin += radiation_field.net_flux_jacobian[i][j] * (T_work[j] - T_curr[j]);
        G[r] = (Flin - target_flux) / fscale;
      }

      // divergence residual Gd and Jacobian Jdiv: row 0 = surface balance (absolute), rows >=1 =
      // backward difference between consecutive unknowns (the layer heating between them).
      std::vector<double> Gd(m), Jdiv(m*m);
      Gd[0] = G[0];
      for (size_t c=0; c<m; ++c) Jdiv[0*m+c] = Jred[0*m+c];
      for (size_t r=1; r<m; ++r)
      {
        Gd[r] = G[r] - G[r-1];
        for (size_t c=0; c<m; ++c) Jdiv[r*m+c] = Jred[r*m+c] - Jred[(r-1)*m+c];
      }

      double fnorm = 0.0;
      for (size_t r=0; r<m; ++r) fnorm += Gd[r]*Gd[r];
      fnorm = std::sqrt(fnorm / static_cast<double>(m));

      std::vector<double> Ddiag(m);
      for (size_t r=0; r<m; ++r) Ddiag[r] = std::max(std::abs(Jdiv[r*m+r]), 1e-300);
      double diag_mean = 0.0;
      for (size_t r=0; r<m; ++r) diag_mean += Ddiag[r];
      diag_mean /= static_cast<double>(m);

      (void) fnorm;

      // ---- regularised divergence-Newton matrix M0: Jdiv + Laplacian curvature smoothing (suppresses
      // the thick-radiative checkerboard near-null mode; vanishes at the solution) + a tiny Tikhonov
      // floor. The SAME M0 gives both the ordinary and the simplified Newton correction, which makes
      // the natural monotonicity test below affine covariant for this (regularised) system.
      std::vector<double> M0 = Jdiv;
      for (size_t r=0; r<m; ++r)
        if (r > 0 && r+1 < m && unk[r-1] == unk[r]-1 && unk[r+1] == unk[r]+1)
        {
          // curvature penalty must be strong enough to lift the checkerboard near-null mode out of M0
          // (otherwise the Newton DIRECTION dx = -M0^{-1}Gd is huge and even lambda_min*dx blows up --
          // NLEQ-ERR damps the step length, not a bad direction). It still vanishes at the solution.
          const double ls = 2.0 * Ddiag[r];
          M0[r*m+r]     += 2.0*ls;
          M0[r*m+(r-1)] -= ls;
          M0[r*m+(r+1)] -= ls;
        }
      const double reg = 1e-3 * (diag_mean + 1e-300);
      for (size_t r=0; r<m; ++r) M0[r*m+r] += (M0[r*m+r] >= 0.0 ? reg : -reg);

      // ordinary Newton correction:  M0 dx = -Gd
      std::vector<double> dx(m, 0.0);
      { std::vector<double> A2 = M0, b = Gd; for (double& v : b) v = -v;
        if (solveDense(A2, b, m)) dx = b; }

      // affine-covariant SCALED correction norm: RMS of dx[r]/scale[r], scale ~ local temperature.
      auto scaledNorm = [&](const std::vector<double>& v) -> double {
        double s2 = 0.0;
        for (size_t r=0; r<m; ++r)
        { const double sc = std::max(T_work[unk[r]], 1.0); s2 += (v[r]/sc)*(v[r]/sc); }
        return std::sqrt(s2 / static_cast<double>(m));
      };
      const double norm_dx = scaledNorm(dx);

      if (forward_flux_eval_ && norm_dx > 1e-300)
      {
        // ===== NLEQ-ERR (Deuflhard): affine-covariant damped Newton with the natural monotonicity
        // test. The damping lambda multiplies the Newton STEP (not the matrix). lambda is accepted
        // when the natural monotonicity test theta = ||dxbar||/||dx|| <= 1 - lambda/4 holds, where the
        // SIMPLIFIED Newton correction dxbar = -M0^{-1} F(x + lambda dx) uses the SAME matrix M0 but
        // the TRUE residual at the trial point (forward_flux_eval_ -> a forward RT solve). Monitoring
        // the correction (error) norm rather than the residual norm keeps the step well behaved at the
        // ill-conditioned radiative-convective boundary, where a residual trust region provably stalls
        // (Deuflhard 3.3.1). Reuses the same factorisation budget as the old step plus 1 RT solve/trial.
        constexpr double lambda_min = 1e-4;
        double lambda = (tr_rcb_at_step_ != rcb_ || nleq_lambda_prev_ <= 0.0)
                      ? 1.0 : std::min(1.0, 2.0*nleq_lambda_prev_);   // warm start, reset on mask move

        std::vector<double> T_trial(n), flux_trial(n), dxbar(m), T_accept;
        double lambda_accept = -1.0;
        auto trialTheta = [&](double lam, double& theta) -> bool {
          T_trial = T_work;
          for (size_t r=0; r<m; ++r)
            T_trial[unk[r]] = std::max(1.0, T_work[unk[r]] + lam*dx[r]);
          adiabatSnap(T_trial);
          forward_flux_eval_(T_trial, flux_trial);          // TRUE forward RT solve at the trial
          std::vector<double> Gt(m), Gdt(m);
          for (size_t r=0; r<m; ++r) Gt[r] = (flux_trial[unk[r]] - target_flux) / fscale;
          Gdt[0] = Gt[0];
          for (size_t r=1; r<m; ++r) Gdt[r] = Gt[r] - Gt[r-1];
          std::vector<double> A2 = M0, b(m);
          for (size_t r=0; r<m; ++r) b[r] = -Gdt[r];
          if (!solveDense(A2, b, m)) return false;
          dxbar = b;
          theta = scaledNorm(dxbar) / norm_dx;
          return true;
        };

        for (int trial=0; trial<25; ++trial)
        {
          double theta;
          if (!trialTheta(lambda, theta)) break;
          if (theta <= 1.0 - 0.25*lambda)                   // restricted natural monotonicity test
          { lambda_accept = lambda; T_accept = T_trial; break; }
          if (lambda <= lambda_min)                         // regularity floor: accept the small step
          { lambda_accept = lambda; T_accept = T_trial; break; }
          lambda = std::max(0.5*lambda, lambda_min);        // reject -> halve the damping, retry
        }

        if (lambda_accept > 0.0)
        {
          T_work = T_accept;
          nleq_lambda_prev_ = lambda_accept;
          tr_rcb_at_step_ = rcb_;
        }
        else
        {
          nleq_lambda_prev_ = -1.0;                          // no good step: reset, leave T_work as is
        }
      }
      else
      {
        // fallback (no forward eval): a single capped Newton step (no affine-covariant globalisation)
        for (size_t r=0; r<m; ++r)
        {
          double d = dx[r];
          if (max_change_fraction > 0.0)
          { const double cap = max_change_fraction * T_work[unk[r]];
            if (std::abs(d) > cap) d = std::copysign(cap, d); }
          T_work[unk[r]] = std::max(1.0, T_work[unk[r]] + d);
        }
        adiabatSnap(T_work);
      }
    }
    else if (surface_anchored && !per_level)
    {
      // ---- PTC step (clima-style): one damped Newton step per RT solve, for the surface-driven
      // (terrestrial) case where the troposphere cannot reach radiative flux balance. Solve
      // (J + C/dt) s = -(F_net - target); C = per-level column heat capacity (dP/g)*c_p, dt a
      // pseudo-time grown as the residual falls (small dt = implicit time step, robust; large dt
      // = Newton). The skin is on local RE; convective layers slaved.
      std::vector<double> heatcap(n, 1.0);
      for (size_t i=0; i<n; ++i)
      {
        const double p_lo = (i+1<n) ? 0.5*(atmosphere.pressure[i]+atmosphere.pressure[i+1]) : atmosphere.pressure[i];
        const double p_hi = (i>0)   ? 0.5*(atmosphere.pressure[i]+atmosphere.pressure[i-1]) : atmosphere.pressure[i];
        const double dP = std::abs(p_lo - p_hi) * 1e6;
        const double cp = ThermodynamicData::meanHeatCapacity(
          atmosphere.number_densities[i], atmosphere.temperature[i]);
        heatcap[i] = std::max(1e-30, dP / std::max(surface_gravity, 1e-30) * cp);
      }
      std::vector<double> G(m);
      double fnorm = 0.0;
      for (size_t r=0; r<m; ++r)
      {
        G[r] = (radiation_field.flux_total[unk[r]] - target_flux) / fscale;
        fnorm += G[r]*G[r];
      }
      fnorm = std::sqrt(fnorm / static_cast<double>(m));

      // ---- adaptive pseudo-timestep: Deuflhard's second-order rule (ZIB 02-14, sec. 3). The last
      // step stored its linearly-predicted residual G_lin; the now-measured residual G reveals the
      // local nonlinearity ||G - G_lin||, giving the optimal next dt:
      //   tau_opt = tau * |(G_lin, G0 - G_lin)| / (2 ||G_lin|| ||G - G_lin||).
      // A near-perfect linear model (||G - G_lin|| -> 0) sends dt -> infinity (switchover to Newton);
      // strong nonlinearity or a residual rise shrinks it (correction mode). The history is dropped
      // when there is none yet, or when the convective zone moved (the unknown set changed).
      if (ptc_dt_ <= 0.0 || ptc_rcb_at_step_ != rcb_)
      {
        double s = 0.0; int cnt = 0;
        for (size_t r=0; r<m; ++r)
        {
          const double jd = std::abs(Jred[r*m+r]);
          if (jd > 0.0) { s += heatcap[unk[r]] / (jd * fscale); ++cnt; }
        }
        ptc_dt_ = (cnt > 0) ? 0.3 * s/cnt : 1.0;
      }
      else if (ptc_Glin_norm_ > 0.0 && ptc_tau_used_ > 0.0)
      {
        double dnorm2 = 0.0;                         // ||G_measured - G_predicted||^2
        for (size_t r=0; r<m; ++r)
        {
          const double dd = G[r] - ptc_Glin_[unk[r]];
          dnorm2 += dd*dd;
        }
        const double dnorm = std::sqrt(dnorm2);
        const bool   fell  = (fnorm < prev_fnorm_);
        double ratio = (dnorm <= 1e-30) ? 1e30
                     : ptc_num_ / (2.0 * ptc_Glin_norm_ * dnorm);
        // terminal phase (settled + zone stable): allow a hard sprint toward the Newton step.
        const double grow_cap = (last_max_dt_frac_ < 1e-3 && !zones_grew_) ? 50.0 : 8.0;
        ratio = std::min(std::max(ratio, 0.1), grow_cap);
        if (!fell) ratio = std::min(ratio, 0.5);     // correction mode: shrink on a residual rise
        ptc_dt_ *= ratio;
      }
      prev_fnorm_ = fnorm;
      for (size_t r=0; r<m; ++r) A[r*m+r] += heatcap[unk[r]] / (ptc_dt_ * fscale);

      std::vector<double> s(m);
      for (size_t r=0; r<m; ++r) s[r] = -G[r];
      if (solveDense(A, s, m))
      {
        // store the linear-model prediction G_lin = G + Jred*s and the scalars the next step needs.
        if (ptc_Glin_.size() != n) ptc_Glin_.assign(n, 0.0);
        double gl_norm2 = 0.0, num = 0.0;
        for (size_t r=0; r<m; ++r)
        {
          double gl = G[r];
          for (size_t c=0; c<m; ++c) gl += Jred[r*m+c] * s[c];
          ptc_Glin_[unk[r]] = gl;
          gl_norm2 += gl*gl;
          num      += gl * (G[r] - gl);
        }
        ptc_Glin_norm_   = std::sqrt(gl_norm2);
        ptc_num_         = std::abs(num);
        ptc_tau_used_    = ptc_dt_;
        ptc_rcb_at_step_ = rcb_;

        for (size_t r=0; r<m; ++r)
        {
          double d = s[r];
          if (max_change_fraction > 0.0)
          {
            const double cap = max_change_fraction * T_work[unk[r]];
            if (std::abs(d) > cap) d = std::copysign(cap, d);
          }
          T_work[unk[r]] = std::max(1.0, T_work[unk[r]] + d);
        }
      }
      adiabatSnap(T_work);
    }
    else
    {
      // ---- inner Newton loop on the linearised flux F + J*dT (self-luminous: gas planet / BD).
      // Iterate the reduced flux Newton to convergence (cheap -- no DISORT re-solve) with a
      // backtracking line search; the smoothing vanishes at the inner fixed point -> a smooth,
      // unbiased F_lin = F_int, so the outer DISORT iterations run at the true Newton rate.
      //
      // CLIMA_TIKH=alpha (test A1 on the flux-residual family): Tikhonov 4th-difference
      // regularisation IN THE OBJECTIVE, min 0.5||g_lin||^2 + 0.5 alpha^2 ||D4 T||^2. The
      // net-flux kernel's odd-moment symmetry makes the Nyquist mode a near-null direction of
      // this residual, so the converged root carries the grid-scale sawtooth; D4 responds with
      // 16 at Nyquist and (k Delta)^4 -> 0 on resolved modes, so any alpha inside the wide
      // singular-value gap pins the sawtooth while leaving the physics untouched (and the
      // sawtooth is flux-neutral, so removing it costs nothing in conservation). With alpha > 0
      // the step becomes the Gauss--Newton correction of the augmented objective and the line
      // search descends the augmented f; alpha = 0 is the unchanged original path. Slaved
      // levels fold linearly into their anchors, local-RE skin levels enter the stencil as
      // constants.
      static const double lin_tikh_alpha = [] {
        const char* e = std::getenv("CLIMA_TIKH");
        return (e != nullptr) ? std::atof(e) : 0.0; }();
      const bool lin_tikh = lin_tikh_alpha > 0.0 && n >= 5;
      const double lin_tikh_a2 = lin_tikh_alpha*lin_tikh_alpha;
      const size_t n_pen = lin_tikh ? n - 4 : 0;
      std::vector<double> S_pen;                     // n_pen x m: d(D4 T)_row / d T_unk
      std::vector<char> pen_active;                  // stencil fully radiative (guard, see below)
      if (lin_tikh)
      {
        S_pen.assign(n_pen*m, 0.0);
        pen_active.assign(n_pen, 1);
        // GUARD: penalty rows only where the FULL stencil is free radiative. The unconditional
        // penalty is viable only with a C1 radiative-convective handover (MLT); this path is
        // locked to the adjustment schemes, whose RCB corner is REAL physics -- unguarded, D4
        // fights the corner until the convective zone collapses (measured: BD deep went
        // all-radiative, T_bot +560 K, with the sawtooth removed but the physics destroyed).
        // Unlike the old terrestrial thick-smooth failure, the BD radiative band is wide, so the
        // guard leaves substantial interior coverage where the sawtooth actually lives.
        for (size_t i=2; i+2<n; ++i)
        {
          const size_t row = i-2;
          for (size_t l=i-2; l<=i+2; ++l)
            if (slaved(l)) { pen_active[row] = 0; break; }
          if (!pen_active[row]) continue;
          auto fold = [&](size_t l, double w) {
            for (size_t c=0; c<m; ++c) if (unk[c] == l) { S_pen[row*m+c] += w; break; }
          };
          fold(i-2, 1.0); fold(i-1, -4.0); fold(i, 6.0); fold(i+1, -4.0); fold(i+2, 1.0);
        }
      }
      auto penRow = [&](const std::vector<double>& T, size_t row)->double {
        const size_t i = row + 2;
        return T[i-2] - 4.0*T[i-1] + 6.0*T[i] - 4.0*T[i+1] + T[i+2];
      };
      auto linResid = [&](const std::vector<double>& T)->double
      {
        double f = 0.0;
        for (size_t r=0; r<m; ++r)
        {
          const size_t i = unk[r];
          double Flin = radiation_field.flux_total[i];
          for (size_t j=0; j<n; ++j)
            Flin += radiation_field.net_flux_jacobian[i][j] * (T[j] - T_curr[j]);
          const double g = (Flin - target_flux) / fscale;
          f += g * g;
        }
        if (lin_tikh)
          for (size_t row=0; row<n_pen; ++row)
          { if (!pen_active[row]) continue;
            const double p = penRow(T, row); f += lin_tikh_a2 * p * p; }
        return 0.5 * f;
      };
      std::vector<double> dunk(m), Asolve(m*m), Ttrial(n);
      constexpr int max_inner = 40;
      for (int inner=0; inner<max_inner; ++inner)
      {
        const double f_old = linResid(T_work);
        std::vector<double> gres(m);
        for (size_t r=0; r<m; ++r)
        {
          const size_t i = unk[r];
          double Flin = radiation_field.flux_total[i];
          for (size_t j=0; j<n; ++j)
            Flin += radiation_field.net_flux_jacobian[i][j] * (T_work[j] - T_curr[j]);
          gres[r] = (Flin - target_flux) / fscale;
          dunk[r] = -gres[r];
        }
        if (lin_tikh)
        {
          // Gauss--Newton normal equations of the augmented objective:
          //   (A^T A + a^2 S^T S) d = -(A^T g + a^2 S^T p)
          std::vector<double> N(m*m, 0.0), b(m, 0.0);
          for (size_t k=0; k<m; ++k)
            for (size_t r=0; r<m; ++r)
            {
              const double Akr = A[k*m+r];
              if (Akr == 0.0) continue;
              for (size_t c=0; c<m; ++c) N[r*m+c] += Akr * A[k*m+c];
              b[r] -= Akr * gres[k];
            }
          for (size_t row=0; row<n_pen; ++row)
          {
            if (!pen_active[row]) continue;
            const double p = penRow(T_work, row);
            for (size_t r=0; r<m; ++r)
            {
              const double Spr = S_pen[row*m+r];
              if (Spr == 0.0) continue;
              for (size_t c=0; c<m; ++c) N[r*m+c] += lin_tikh_a2 * Spr * S_pen[row*m+c];
              b[r] -= lin_tikh_a2 * Spr * p;
            }
          }
          if (!solveDense(N, b, m)) break;
          dunk = b;
        }
        else
        {
        Asolve = A;
        if (!solveDense(Asolve, dunk, m)) break;
        }
        double scale = 1.0;
        for (size_t r=0; r<m; ++r)
        {
          const double frac = std::abs(dunk[r]) / T_work[unk[r]];
          if (frac > 0.30) scale = std::min(scale, 0.30 / frac);
        }
        for (size_t r=0; r<m; ++r) dunk[r] *= scale;
        double alam = 1.0, step_frac = 0.0;
        for (int ls=0; ls<20; ++ls)
        {
          Ttrial = T_work;
          for (size_t r=0; r<m; ++r) Ttrial[unk[r]] = std::max(1.0, T_work[unk[r]] + alam*dunk[r]);
          adiabatSnap(Ttrial);
          if (linResid(Ttrial) <= f_old * (1.0 - 1e-4*alam) || alam < 1e-3) break;
          alam *= 0.5;
        }
        for (size_t r=0; r<m; ++r)
        {
          const double d = alam*dunk[r];
          step_frac = std::max(step_frac, std::abs(d) / T_work[unk[r]]);
          T_work[unk[r]] = std::max(1.0, T_work[unk[r]] + d);
        }
        adiabatSnap(T_work);
        if (step_frac < 1e-6) break;
      }
    }
  }

  // commit the converged inner solution
  for (size_t i=0; i<n; ++i)
  {
    const double d = T_work[i] - T_curr[i];
    atmosphere.temperature[i] = std::max(1.0, T_work[i]);
    if (zeta[i] >= skin_zeta)
      max_dt_frac = std::max(max_dt_frac, std::abs(d) / atmosphere.temperature[i]);
  }
  last_max_dt_frac_ = max_dt_frac;
}


}
