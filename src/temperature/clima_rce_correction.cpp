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


#include "clima_rce_correction.h"

#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <functional>

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
  if (std::getenv("CLIMA_JACTEST"))
  {
    const std::vector<double> T0 = atmosphere.temperature;
    std::vector<double> F0, NH0;
    forward_eval_full_(T0, /*recompute_opacity=*/false, /*compute_jacobian=*/true, F0, NH0);
    const std::vector<std::vector<double>> Anf = radiation_field.net_flux_jacobian;    // analytic dF_net[i]/dT[j]
    const std::vector<std::vector<double>> Anh = radiation_field.net_heating_jacobian; // analytic dNH[i]/dT[j]
    const std::vector<std::vector<double>> Amk = radiation_field.meanint_kappa_jacobian; // analytic d(num)[i]/dT[j]

    // FD reference for the kappa-weighted mean-intensity sum num_i = sum_k w_k kappa_k,i J_k,i (cgs),
    // built from the SAME frozen absorption_coeff the analytic contraction uses (recompute_opacity=false
    // keeps kappa fixed), so the test isolates d(num)/dT, not an opacity response.
    const std::vector<double>& wn_jt = radiation_field.spectral_grid->wavenumber_list;
    const std::vector<double> w_jt = aux::trapezoidalWeights(wn_jt);
    const size_t nb_nu_jt = wn_jt.size();
    auto numNow = [&](std::vector<double>& out){
      out.assign(n, 0.0);
      for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < nb_nu_jt; ++k)
          out[i] += w_jt[k] * opacity.absorption_coeff[k][i] * radiation_field.mean_intensity[i][k];
    };

    const double dT = std::getenv("CLIMA_JACDT") ? std::atof(std::getenv("CLIMA_JACDT")) : 0.1;
    std::vector<std::vector<double>> Fnf(n, std::vector<double>(n, 0.0)), Fnh(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> Fmk(n, std::vector<double>(n, 0.0));
    for (size_t j = 0; j < n; ++j)
    {
      std::vector<double> Tp = T0, Tm = T0; Tp[j] += dT; Tm[j] -= dT;
      std::vector<double> Fp, NHp, Fm, NHm, nump, numm;
      forward_eval_full_(Tp, false, false, Fp, NHp); numNow(nump);
      forward_eval_full_(Tm, false, false, Fm, NHm); numNow(numm);
      for (size_t i = 0; i < n; ++i)
      {
        Fnf[i][j] = (Fp[i] - Fm[i]) / (2.0*dT);
        Fnh[i][j] = (NHp[i] - NHm[i]) / (2.0*dT);
        Fmk[i][j] = (nump[i] - numm[i]) / (2.0*dT);
      }
      if (j % 20 == 0) std::fprintf(stderr, "[JACTEST] FD column %zu/%zu\n", j, n);
    }
    auto summarize = [&](const char* name, const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& Fd){
      double maxA=0,maxFd=0,maxAbsDiff=0,frobD=0,frobA=0,worstRel=0; size_t bi=0,bj=0,wi=0,wj=0;
      for (size_t i=0;i<n;++i) for (size_t j=0;j<n;++j){
        const double a=A[i][j], f=Fd[i][j], d=std::abs(a-f);
        maxA=std::max(maxA,std::abs(a)); maxFd=std::max(maxFd,std::abs(f));
        if (d>maxAbsDiff){maxAbsDiff=d; bi=i; bj=j;} frobD+=d*d; frobA+=a*a;
      }
      for (size_t i=0;i<n;++i) for (size_t j=0;j<n;++j){
        const double a=A[i][j], f=Fd[i][j], sc=std::max(std::abs(a),std::abs(f));
        if (sc > 1e-3*maxA){ const double rel=std::abs(a-f)/sc; if (rel>worstRel){worstRel=rel; wi=i; wj=j;} }
      }
      std::fprintf(stderr, "[JACTEST] %-16s max|A|=%.4e max|FD|=%.4e  ||A-FD||/||A||=%.4e\n", name, maxA, maxFd, std::sqrt(frobD/std::max(frobA,1e-300)));
      std::fprintf(stderr, "                   max|A-FD|=%.3e @[%zu][%zu] A=%.3e FD=%.3e | worst rel(sig)=%.3f @[%zu][%zu] A=%.3e FD=%.3e\n",
        maxAbsDiff,bi,bj,A[bi][bj],Fd[bi][bj], worstRel,wi,wj,A[wi][wj],Fd[wi][wj]);
    };
    summarize("net_flux_jac", Anf, Fnf);
    summarize("net_heating_jac", Anh, Fnh);
    summarize("meanint_kappa_jac", Amk, Fmk);
    auto dump=[&](const char* fn, const std::vector<std::vector<double>>& M){ FILE* f=std::fopen(fn,"w"); for(size_t i=0;i<n;++i){for(size_t j=0;j<n;++j)std::fprintf(f,"%.10e ",M[i][j]); std::fprintf(f,"\n");} std::fclose(f); };
    dump("/tmp/jac_Anf.txt",Anf); dump("/tmp/jac_FDnf.txt",Fnf); dump("/tmp/jac_Anh.txt",Anh); dump("/tmp/jac_FDnh.txt",Fnh);
    dump("/tmp/jac_Amk.txt",Amk); dump("/tmp/jac_FDmk.txt",Fmk);
    { FILE* f=std::fopen("/tmp/jac_levels.txt","w"); for(size_t i=0;i<n;++i) std::fprintf(f,"%zu %.8e %.8e\n", i, atmosphere.pressure[i], T0[i]); std::fclose(f); }
    std::fprintf(stderr, "[JACTEST] wrote /tmp/jac_{Anf,FDnf,Anh,FDnh,levels}.txt (n=%zu dT=%g central)\n", n, dT);
    forward_eval_full_(T0, /*recompute_opacity=*/true, false, F0, NH0);   // restore clean state
    atmosphere.temperature = T0; last_residual_ = -1.0; return;
  }

  // ---- mean adiabatic gradient over a level pair (scheme-agnostic) ---------------------------
  auto nablaAd = [&](size_t i, size_t j) -> double {
    if (convection == nullptr) return 0.0;
    return 0.5 * (
      convection->convectiveGradient(atmosphere.number_densities[i], atmosphere.temperature[i], atmosphere.pressure[i]) +
      convection->convectiveGradient(atmosphere.number_densities[j], atmosphere.temperature[j], atmosphere.pressure[j]));
  };

  // ---- convective mask (multi-zone). Detect the "desired" mask on a SMOOTHED probe copy, so a
  // transient checkerboard sawtooth near the RCB cannot masquerade as super-adiabatic and grow the
  // convective zone up into the (stably stratified) stratosphere -- a feedback that otherwise
  // destabilises the boundary. The smoothing is for DETECTION ONLY; the residual/Jacobian use the
  // true profile. ----------------------------------------------------------------------------------
  std::vector<int> desired(n, 0);
  if (convection != nullptr)
  {
    Atmosphere probe = atmosphere;   // copy: do not disturb the real profile
    // 1-2-1 detection smoothing (default ON; CLIMA_MASKSMOOTH=0 disables): without it the probe places the
    // RCB too HIGH (the steep near-tropopause radiative profile reads super-adiabatic) -> the convective
    // adiabat overshoots cold and KINKS the tropopause. With it the RCB lands correctly; the only side
    // effect (a converged on-adiabat zone occasionally reading as NO convection -- a marginal flip) is
    // guarded just below.
    if (!std::getenv("CLIMA_MASKSMOOTH") || std::atof(std::getenv("CLIMA_MASKSMOOTH")) != 0.0)
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
      const int gap_fill = std::getenv("CLIMA_MASKBAND") ? std::atoi(std::getenv("CLIMA_MASKBAND")) : 2;
      int top = -1, gap = 0;
      for (size_t i = 0; i < n; ++i)
      {
        if (desired[i]) { top = static_cast<int>(i); gap = 0; }
        else if (++gap > gap_fill) break;
      }
      for (int i = 0; i <= top; ++i) desired[i] = 1;     // solid surface-connected zone [0, top]
    }

    // ANTI-OVERSHOOT (clima Mode-3): the convective top is snapped to a grid level, so the slaved adiabat
    // can land one level too HIGH -- forcing the top onto the cold adiabat below the radiative profile that
    // would sit there, a sharp cold-inversion KINK at the tropopause (max-curv blew up to ~14 K at L16).
    // Cure exactly as clima: if the first radiative link ABOVE the convective top is a strong COLD INVERSION
    // (dlnT/dlnP << 0, well beyond a gentle stratospheric warming), the adiabat has overshot -> SHRINK the
    // top by one level. A LOCKOUT (rcb_lock_) then caps the top there for a few iterations so the probe
    // cannot immediately re-grow it (the ABAB toggle clima's lockout counter prevents).
    {
      int des_top = -1; for (int i = static_cast<int>(n)-1; i >= 0; --i) if (desired[i]) { des_top = i; break; }
      if (des_top >= 1 && des_top+1 < static_cast<int>(n))
      {
        const double dlnP = std::log(atmosphere.pressure[des_top]/atmosphere.pressure[des_top+1]);
        if (dlnP > 0.0)
        {
          const double nabla = std::log(atmosphere.temperature[des_top]/atmosphere.temperature[des_top+1])/dlnP;
          const double nabla_ad = 0.5*(
            convection->convectiveGradient(atmosphere.number_densities[des_top],   atmosphere.temperature[des_top],   atmosphere.pressure[des_top]) +
            convection->convectiveGradient(atmosphere.number_densities[des_top+1], atmosphere.temperature[des_top+1], atmosphere.pressure[des_top+1]));
          const double off_frac = std::getenv("CLIMA_RCB_OFFFRAC") ? std::atof(std::getenv("CLIMA_RCB_OFFFRAC")) : 0.5;
          if (nabla < -off_frac*nabla_ad)             // sharp cold inversion above the top -> overshoot
          {
            rcb_cap_  = des_top - 1;                   // retreat one level
            rcb_lock_ = std::getenv("CLIMA_RCB_LOCK") ? std::atoi(std::getenv("CLIMA_RCB_LOCK")) : 5;
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

  // SETTLE GATE: only move the convective boundary once the inner Newton has SETTLED the profile at
  // the current mask (clima solves hybrj to convergence BEFORE moving the mask). Otherwise hold the
  // mask frozen and keep iterating -- this stops the boundary from jittering on a half-converged
  // profile, which is what feeds the checkerboard sawtooth just above the RCB.
  const double settle_tol = std::getenv("CLIMA_SETTLE") ? std::atof(std::getenv("CLIMA_SETTLE")) : 2e-3;
  const bool have_prev = (prev_mask_.size() == n);
  const bool settled = !have_prev || (last_inner_change_ < settle_tol);

  // dead band: the true radiative-convective boundary sits between grid levels, so the discrete mask
  // would oscillate by +-1 layer forever (each move swinging the surface). Accept a small boundary
  // ambiguity: once the mask is within `mask_band` layers of the detected mask, stop moving it.
  const int mask_band = std::getenv("CLIMA_MASKBAND") ? std::atoi(std::getenv("CLIMA_MASKBAND")) : 2;
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
  const bool carve_deep = std::getenv("CLIMA_PTC") && std::getenv("CLIMA_PTC_BLEND") && std::getenv("CLIMA_PTC_DEEPINT");
  std::vector<char> deep(n, 0);
  int kdeep = -1;                                                  // top level of the carved deep zone (-1 = none)
  std::vector<double> deep_dtau(n, 0.0);                          // Planck-mean optical thickness of layer (i,i+1)
  if (carve_deep)
  {
    const std::vector<double>& wl = radiation_field.spectral_grid->wavenumber_list;
    const std::vector<double> ww = aux::trapezoidalWeights(wl);
    std::vector<double> kPl(n, 0.0);
    for (size_t i = 0; i < n; ++i)
    {
      double sB = 0.0, skB = 0.0;
      for (size_t k = 0; k < wl.size(); ++k)
      { const double B = disortpp::planckFunction2(wl[k], wl[k], atmosphere.temperature[i]);
        sB += ww[k]*B; skB += ww[k]*opacity.absorption_coeff[k][i]*B; }
      kPl[i] = (sB > 0.0) ? skB/sB : 0.0;
    }
    const double deep_thr = std::getenv("CLIMA_PTC_DEEPTHR") ? std::atof(std::getenv("CLIMA_PTC_DEEPTHR")) : 0.5;
    for (size_t i = 0; i + 1 < n; ++i)                             // grow the deep from the surface while layers stay thick
    {
      if (slaved[i] || slaved[i+1]) break;                        // hit convection -> end of the radiative deep
      const double kappa = 0.5*(kPl[i+1] + kPl[i]);
      const double dtau  = kappa * std::abs(atmosphere.altitude[i+1] - atmosphere.altitude[i]);
      deep_dtau[i] = dtau;                                        // layer (i,i+1) thickness -> grey-flux fallback below
      if (dtau/(dtau + 1.0) < deep_thr) break;                    // thinned to dtau<~1 -> photosphere edge
      kdeep = static_cast<int>(i + 1);
    }
    if (kdeep >= 2 && kdeep + 1 < static_cast<int>(n))
    {
      for (int i = 1; i <= kdeep; ++i) deep[i] = 1;               // levels 1..kdeep are integration-only
      // The SURFACE (level 0) is the bottom endpoint of the SAME one-anchor downward sweep, not a second
      // anchor: extend the integration through it and drop the standalone surface energy-balance Newton (doc
      // §6.7 "one scalar, not two"). T0 then falls out as the stiff bottom leaf -- the bulk deep is anchored
      // at the cold photosphere above, so T0 does not drive the deep and the surface stays non-degenerate
      // (no leaf->root runaway). F_net[0]->target is satisfied as the i=0 endpoint of flux conservation.
      if (zone_of_dof[0] < 0) deep[0] = 1;
    }
    else
      kdeep = -1;
  }

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
  const bool netflux = std::getenv("CLIMA_NETFLUX") != nullptr;
  const double Fnorm = radiation_field.flux_down_total.empty()
    ? 1.0 : std::max(1.0, std::abs(radiation_field.flux_down_total.back()));

  // CLIMA_CENTERED: use a SYMMETRIC (control-volume) divergence g_i = 0.5*(F[i+1]-F[i-1])/c_eff for the
  // radiative layers instead of the one-sided backward g_i = (F[i]-F[i-1])/c_eff. This is the proper
  // finite-volume balance for a level-centred T[i] (top face minus bottom face). It also has a sin(k)
  // Fourier response -> ZERO at the Nyquist (sawtooth) frequency, so it does not amplify the sampling-
  // noise sawtooth into the residual. RISK: the Nyquist mode is then in the residual's NULL space (odd-
  // even decoupling), so a seeded sawtooth cannot be corrected. The top radiative level (no i+1) falls
  // back to the one-sided form. Mutually exclusive with CLIMA_NETFLUX.
  const bool centered = std::getenv("CLIMA_CENTERED") != nullptr;

  // CLIMA_LOCALRE: use DISORT's NATIVE per-layer heating  g_i = net_heating[i]/c_eff  for the radiative
  // levels (with net_heating_jacobian), instead of any difference of the LEVEL fluxes. net_heating =
  // sum_nu w_nu dF_net/dtau is the LOCAL radiative heating sum_nu w_nu kappa(B-J) -- so net_heating=0 IS
  // the local radiative-equilibrium condition B(T_i)=<J>_kappa. Being local (no adjacent-level flux
  // differencing) it should neither excite nor null the sawtooth, AND it pins the optically-thin top
  // through the local Planck term B(T_i) (unlike the cumulative net flux, which collapsed it). Surface
  // and convective zones keep their genuine flux conditions. Mutually exclusive with NETFLUX/CENTERED.
  const bool localre = std::getenv("CLIMA_LOCALRE") != nullptr;

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
  const bool newtonlike = std::getenv("CLIMA_NEWTONLIKE") != nullptr;
  const bool nl_netheat = std::getenv("CLIMA_NL_NETHEAT") != nullptr;

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
  const bool hstep      = std::getenv("CLIMA_HSTEP") != nullptr;
  // Layer/level offset knob. DEFAULT 0 (collocated, no shift): the backend computes
  //   flux_divergence[l] = 4pi(1-omega)*(J[l] - B[l])
  // which is genuinely COLLOCATED at level l -- the "each interface uses the layer above"
  // convention selects only which omega is used (a smooth O(1) prefactor), NOT the collocation.
  // So net_heating[k] describes LEVEL k. Shifting it moves the Planck-cooling term off the
  // diagonal and destroys dominance; -1 was tried and is worse. Kept only as a diagnostic knob.
  const int  hstep_off  = std::getenv("CLIMA_HSTEP_OFF") ? std::atoi(std::getenv("CLIMA_HSTEP_OFF")) : 0;
  const double hstep_sign = std::getenv("CLIMA_HSTEP_SIGN") ? std::atof(std::getenv("CLIMA_HSTEP_SIGN")) : -1.0;
  const bool hstep_surf = hstep && std::getenv("CLIMA_HSTEP_SURF") != nullptr;
  const bool hstep_zone = hstep && std::getenv("CLIMA_HSTEP_ZONE") != nullptr;

  // CLIMA_PTC: the UNIFIED construction (doc "The unified construction"). ONE moment-flux residual
  // everywhere, g_i = (F_net,i - target)/Fnorm + w_i g^RE_i, regularised by pseudo-transient continuation
  // -- the step solves (NFJ/Fnorm + C/dt + w_i d g^RE/dT) s = -g, with the heat-capacity diagonal C/dt
  // (dominant in the deep) sliding to the ratio Planck-slope diagonal w_i(-C/den) (the only T-sensitive
  // term as kappa->0) in the optically-thin top. No tau seam: the flux residual is one physical object top
  // to bottom; the ratio term is a SECOND regularisation of it (and carries the stellar BC via num_i, the
  // direct beam in J), active only where F_net goes flat. One PTC step per call; the outer loop grows dt.
  const bool ptc = std::getenv("CLIMA_PTC") != nullptr;

  // CLIMA_PTC_BLEND: implement the doc's Eq. 19 SMOOTH sliding diagonal instead of the hard local-RE skin
  // overwrite. The hard split (overwrite for tau<tau_skin, pure flux+PTC below) leaves the handoff level
  // conditioned by NEITHER operator -> a localised seam collapse (the 63 K spike at the tau=1 boundary).
  // Eq. 19 puts the ratio term in BOTH residual and Jacobian with a smooth weight w_i = ptc_w[i] (->1 at
  // the optically-thin top, ->0 in the deep): g_i = (F_net-target)/Fnorm + w_i g^RE_i, with the Jacobian
  // gaining w_i d g^RE/dT (the Planck-slope diagonal that conditions the thin top continuously). No switch
  // surface, no overwrite -> no seam. The skin overwrite is disabled when this is on.
  const bool ptc_blend = ptc && std::getenv("CLIMA_PTC_BLEND") != nullptr;

  // CLIMA_PTC_DEEPINT (doc Sec.6): in the optically-thick radiative deep, replace the slow PTC relaxation by
  // a DIRECT GRADIENT INTEGRATION. There the net-flux Jacobian is banded (tridiagonal) and the flux
  // constrains the temperature GRADIENT, not T, so a forward substitution that drives F_net,i -> target
  // through the well-conditioned off-diagonal dF_i/dT_{i+1} is O(n) and flux-exact -- no Rosseland mean, no
  // diagonal dominance. Diffusion-limit only, so it is weighted by the local-layer deepness and fades out
  // through the photosphere. Matters for the convection-OFF case / thick radiative stratospheres.
  const bool deep_integration = ptc_blend && std::getenv("CLIMA_PTC_DEEPINT") != nullptr;

  // CLIMA_LREANCHOR: with the collocated local-heating residual (CLIMA_LOCALRE), the heating Jacobian
  // is dense and has a near-null UNIFORM-temperature-level mode (local RE J=B is ~insensitive to a
  // uniform shift), so the level is pinned only by the surface row -> ill-conditioned (cond ~1e7) ->
  // huge Newton steps. Adding a weak un-differenced net-flux term eps*F[i]/Fstar to every radiative
  // residual restores a global energy-balance sensitivity that pins the DC mode WITHOUT differencing
  // (so it does not reintroduce the checkerboard). eps small keeps the local heating dominant.
  const double lre_anchor = std::getenv("CLIMA_LREANCHOR") ? std::atof(std::getenv("CLIMA_LREANCHOR")) : 0.0;

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
  const bool ratio = (std::getenv("CLIMA_RATIO_RE") != nullptr)
                  || (!ratio_other_mode && std::getenv("CLIMA_FLUX_RESID") == nullptr);

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
  const bool lucy = ratio && std::getenv("CLIMA_NO_LUCY") == nullptr;   // default ON with the ratio residual
  const double lucy_sign = std::getenv("CLIMA_LUCY_SIGN") ? std::atof(std::getenv("CLIMA_LUCY_SIGN")) : -1.0;

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
  const bool tridiag = std::getenv("CLIMA_TRIDIAG") != nullptr;

  // per-level spectral sums for the ratio residual + its Jacobian, evaluated at a profile T against the
  // CURRENT radiation field (frozen opacity, so kappa = absorption_coeff matches the numerator Jacobian):
  //   num = sum_k w_k kappa_k,i J_k,i (cgs), den = sum_k w_k kappa_k,i B(nu_k,T_i) (cgs),
  //   C   = sum_k w_k kappa_k,i dB/dT(nu_k,T_i) (cgs) = d den/dT_i.
  const std::vector<double>& ratio_wn = radiation_field.spectral_grid->wavenumber_list;
  const std::vector<double> ratio_w = (ratio || ptc) ? aux::trapezoidalWeights(ratio_wn) : std::vector<double>();
  const size_t ratio_nnu = ratio_wn.size();
  constexpr double ratio_si_to_cgs = 1e3;
  auto ratioSums = [&](const std::vector<double>& T, const size_t i, double& num, double& den, double& C)
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
  const bool ratio_flux = std::getenv("CLIMA_RATIO_FLUX") != nullptr;   // opt-in flux anchor
  // default flux anchor = low-rank DEFLATION (driver); CLIMA_RATIO_PERLEVEL selects the old per-level
  // zeta-blend (kept for comparison; it fights the diagonal because the flux constraint is nonlocal).
  const bool ratio_perlevel = std::getenv("CLIMA_RATIO_PERLEVEL") != nullptr;
  const double ratio_xi = std::getenv("CLIMA_RATIO_XI") ? std::atof(std::getenv("CLIMA_RATIO_XI")) : 1.0;
  const double ratio_tauscale = std::getenv("CLIMA_RATIO_ZETA_TAUSCALE") ? std::atof(std::getenv("CLIMA_RATIO_ZETA_TAUSCALE")) : 1.0;
  const bool ratio_fluxdense = std::getenv("CLIMA_RATIO_FLUXDENSE") != nullptr;
  std::vector<double> ratio_zeta(n, 0.0), ratio_tau(n, 0.0);
  if (ratio && ratio_flux && ratio_perlevel)
  {
    std::vector<double> kP(n, 0.0);                   // Planck-mean absorption coeff per level
    for (size_t i = 0; i < n; ++i)
    {
      double sB = 0.0, skB = 0.0;
      for (size_t k = 0; k < ratio_nnu; ++k)
      { const double B = disortpp::planckFunction2(ratio_wn[k], ratio_wn[k], atmosphere.temperature[i]);
        sB += ratio_w[k]*B; skB += ratio_w[k]*opacity.absorption_coeff[k][i]*B; }
      kP[i] = (sB > 0.0) ? skB/sB : 0.0;
    }
    for (size_t i = n-1; i > 0; --i)                  // cumulative from TOA (i=n-1) downward (altitude grows with i)
      ratio_tau[i-1] = ratio_tau[i] + 0.5*(kP[i]+kP[i-1])*std::abs(atmosphere.altitude[i]-atmosphere.altitude[i-1]);
    for (size_t i = 0; i < n; ++i) ratio_zeta[i] = ratio_tau[i] / (ratio_tau[i] + ratio_tauscale);
    if (std::getenv("CLIMA_DBG"))
      std::fprintf(stderr, "  [ratio-zeta] tau: top=%.2e bot=%.2e  zeta: top=%.3f bot=%.3f (tau_scale=%.2g xi=%.2g)\n",
        ratio_tau[n-1], ratio_tau[0], ratio_zeta[n-1], ratio_zeta[0], ratio_tauscale, ratio_xi);
  }

  // FROZEN least-squares scale calibrating the cumulative proxy to the actual net-flux deviation:
  // F_i - F_0 ~ ratio_flux_scale * cumF_i (auto-handles the 4*pi and unit factors). Computed once at the
  // committed point so the residual (xi g^RE + zeta*flux_scale*cumF/Fnorm) and the cumulative-heating
  // Jacobian stay a consistent Newton through the inner relaxation.
  double ratio_flux_scale = 0.0;
  if (ratio && ratio_flux && ratio_perlevel)
  {
    std::vector<double> cumF0; cumFvec(atmosphere.temperature, cumF0);
    double sxy = 0.0, sxx = 0.0;
    for (size_t i = 0; i < n; ++i)
    { const double dF = radiation_field.flux_total[i] - radiation_field.flux_total[0];
      sxy += dF*cumF0[i]; sxx += cumF0[i]*cumF0[i]; }
    ratio_flux_scale = (sxx > 0.0) ? sxy/sxx : 0.0;
  }

  // Per-layer Planck-mean optical depth, to rescale the Newton-like step Jacobian. The accurate
  // conservation residual is the per-LAYER flux change F[i]-F[i-1] = net_heating[i]*dtau[i], but
  // net_heating_jacobian (NHJ) is the per-TAU Jacobian d(dF/dtau)/dT; multiplying NHJ by dtau[i] gives the
  // per-layer dominant surrogate of d(F[i]-F[i-1])/dT (matching the residual's scale, so the NLEQ-ERR step
  // is consistent -- without it the deep large-dtau layers are mis-scaled and the solver grinds).
  // CLIMA_DUMPTAU: dump the per-layer Planck-mean optical thickness once, then keep going. Used to test
  // the diffusion-limit note's Sec.2.4 open item -- whether the sawtooth amplitude tracks the ABSOLUTE
  // local layer thickness dtau_i (a numerator/truncation effect, ~dtau^3) rather than the thickness
  // contrast eps. Grid refinement varies both at once, so only a direct dtau measurement separates them.
  const bool dumptau = std::getenv("CLIMA_DUMPTAU") != nullptr;
  std::vector<double> nl_dtau(n, 1.0);
  if ((newtonlike && !nl_netheat) || hstep || dumptau || lucy)   // CLIMA_HSTEP/LUCY need the per-layer dtau
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
  if (ptc)
  {
    const double w0 = std::getenv("CLIMA_PTC_W0") ? std::atof(std::getenv("CLIMA_PTC_W0")) : 1.0;
    const double tau_w = std::getenv("CLIMA_PTC_TAUW") ? std::atof(std::getenv("CLIMA_PTC_TAUW")) : 1.0;
    const std::vector<double> w_int = aux::trapezoidalWeights(ratio_wn);
    std::vector<double> kP(n, 0.0);
    for (size_t i = 0; i < n; ++i)
    {
      double sB = 0.0, skB = 0.0;
      for (size_t kk = 0; kk < ratio_nnu; ++kk)
      { const double B = disortpp::planckFunction2(ratio_wn[kk], ratio_wn[kk], atmosphere.temperature[i]);
        sB += w_int[kk]*B; skB += w_int[kk]*opacity.absorption_coeff[kk][i]*B; }
      kP[i] = (sB > 0.0) ? skB/sB : 0.0;
    }
    for (size_t i = n-1; i > 0; --i)
      ptc_tau[i-1] = ptc_tau[i] + 0.5*(kP[i]+kP[i-1])*std::abs(atmosphere.altitude[i]-atmosphere.altitude[i-1]);
    for (size_t i = 0; i < n; ++i) ptc_w[i] = w0 * tau_w / (ptc_tau[i] + tau_w);
  }

  // Per-level radiative time tau_rad ~ c_p/(kappa_P sigma T^3) (the local Newtonian-cooling time), used to
  // make the pseudo-timestep PER-LEVEL: dt_i = rho * s_i with s_i = tau_rad_i/<tau_rad>. The DEEP has short
  // tau_rad (high kappa, high T) -> small dt -> LARGE C/dt -> strong PTC regularisation of the deficient
  // net-flux Jacobian; the thin top has long tau_rad -> weak PTC, but it is the ratio anchor that conditions
  // the top there. rho is the single global pseudo-time, ramped by the Deuflhard rule in the driver.
  std::vector<double> ptc_srad(n, 1.0);
  if (ptc && (ptc_blend || std::getenv("CLIMA_PTC_PERLEVEL")))   // per-level dt~tau_rad (Eq.20). IMPLIED by the
  {                                               // blend (the ratio diagonal conditions the weak-reg top); else opt-in
    const std::vector<double> w_int = aux::trapezoidalWeights(ratio_wn);
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
      double sB = 0.0, skB = 0.0;
      for (size_t kk = 0; kk < ratio_nnu; ++kk)
      { const double B = disortpp::planckFunction2(ratio_wn[kk], ratio_wn[kk], atmosphere.temperature[i]);
        sB += w_int[kk]*B; skB += w_int[kk]*opacity.absorption_coeff[kk][i]*B; }
      const double kP_i = (sB > 0.0) ? skB/sB : 0.0;
      const double cp = ThermodynamicData::meanHeatCapacity(atmosphere.number_densities[i], atmosphere.temperature[i]);
      const double T3 = std::pow(std::max(atmosphere.temperature[i], 1.0), 3);
      ptc_srad[i] = cp / (std::max(kP_i, 1e-30) * T3);   // proportional to tau_rad (constant cancels in the mean)
      mean += ptc_srad[i];
    }
    mean = std::max(mean / static_cast<double>(n), 1e-300);
    for (size_t i = 0; i < n; ++i) ptc_srad[i] = std::max(1e-6, ptc_srad[i] / mean);   // normalised to mean 1
  }

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
  if (ptc && std::getenv("CLIMA_PTC_ROSS"))     // Rosseland preconditioner experiment only; the deep carve-out
  {                                             // (CLIMA_PTC_DEEPINT) integrates the gradient instead (below)
    const std::vector<double> w_int = aux::trapezoidalWeights(ratio_wn);
    const double sigma_cgs = 5.670374e-5;                 // erg cm^-2 s^-1 K^-4
    std::vector<double> kP(n, 0.0);
    for (size_t i = 0; i < n; ++i)
    {
      double sB = 0.0, skB = 0.0;
      for (size_t kk = 0; kk < ratio_nnu; ++kk)
      { const double B = disortpp::planckFunction2(ratio_wn[kk], ratio_wn[kk], atmosphere.temperature[i]);
        sB += w_int[kk]*B; skB += w_int[kk]*opacity.absorption_coeff[kk][i]*B; }
      kP[i] = (sB > 0.0) ? skB/sB : 0.0;
    }
    // F_net[i] ~ -(4/3 kappa) (sigma T[i]^4 - sigma T[i-1]^4)/dz  (interface BELOW level i, toward the
    // surface). Coupling DOWNWARD (to i-1) is essential: it puts each level's dependence on the warmer
    // layer beneath it -- in particular L1's dependence on the hot surface L0 -- INTO that level's own row,
    // so it can take a real warming step. (Coupling upward to i+1 instead buries L1's surface coupling in
    // the overwritten surface row.) dF[i]/dT[i] = -pref T[i]^3 (NEGATIVE), dF[i]/dT[i-1] = +pref T[i-1]^3.
    for (size_t i = 1; i < n; ++i)
    {
      const double kappa = 0.5*(kP[i]+kP[i-1]);
      const double dz    = std::abs(atmosphere.altitude[i]-atmosphere.altitude[i-1]);
      const double dtau  = std::max(kappa*dz, 1e-30);     // local layer optical thickness
      const double pref  = 16.0*sigma_cgs/(3.0*dtau);
      ross_diag[i] = -pref*std::pow(std::max(atmosphere.temperature[i],   1.0), 3);   // dF[i]/dT[i]   < 0
      ross_up[i]   =  pref*std::pow(std::max(atmosphere.temperature[i-1], 1.0), 3);   // dF[i]/dT[i-1] > 0 (down-coupling)
      ross_d[i]    = dtau/(dtau + 1.0);                   // -> 1 optically thick (diffusion valid), -> 0 thin
    }
    if (std::getenv("CLIMA_DBG"))
      std::fprintf(stderr, "  [ross] ross_d: L0=%.3f L5=%.3f L20=%.3f L50=%.3f  diag0=%.2e nfj00=%.2e\n",
                   ross_d[0], ross_d[std::min<size_t>(5,n-1)], ross_d[std::min<size_t>(20,n-1)],
                   ross_d[std::min<size_t>(50,n-1)], ross_diag[0], radiation_field.net_flux_jacobian[0][0]);
  }

  // ---- residual assembler: clima's heat-capacity-weighted flux DIFFERENCES over the reduced DOFs.
  // clima forms fluxes(i)=f_total(i)-f_total(i-1) (and fluxes(0)=f_total(0) at the surface), so we use
  // exactly that. Because the Jacobian below is a finite difference of THIS residual, the two are
  // self-consistent (no DISORT-divergence/flux stencil mismatch). Only flux_total is needed. ---------
  // Full Unsoeld-Lucy correction dB_i (level + cumulative flux-error integral), from the measured flux.
  // tau increases DOWNWARD from TOA; ngam's index increases UPWARD, so integrate from i=n-1 down to 0.
  // nl_dtau[k] is the optical thickness of the layer between levels k-1 and k.
  auto lucyVec = [&](const std::vector<double>& F, std::vector<double>& dB) {
    dB.assign(n, 0.0);
    if (n < 2) return;
    dB[n-1] = (F[n-1] - target_flux) / (2.0*M_PI);          // level: Eddington top boundary J(0)=2H(0)
    for (int i = static_cast<int>(n)-2; i >= 0; --i)
      dB[i] = dB[i+1] + (3.0/(4.0*M_PI)) * 0.5*((F[i]-target_flux) + (F[i+1]-target_flux)) * nl_dtau[i+1];
  };

  auto assembleG = [&](const std::vector<double>& F, const std::vector<double>& NH, const std::vector<double>& T, std::vector<double>& g)
  {
    g.assign(m, 0.0);
    std::vector<double> cumF;
    if (ratio && ratio_flux && ratio_perlevel) cumFvec(T, cumF);   // per-level blend only
    std::vector<double> lucyB;
    if (lucy) lucyVec(F, lucyB);                                   // full Unsoeld-Lucy correction
    for (size_t r = 0; r < m; ++r)
    {
      const size_t i = unk[r];
      const int zi = zone_of_dof[i];
      if (zi >= 0)                                   // convective-zone DOF: net flux into the whole zone
      {
        const Zone& z = zones[zi];
        const double f_lower = (z.lower == 0) ? 0.0 : F[z.lower-1];
        // NORMALISATION: this row is the whole zone's energy budget. Dividing by c_eff_zone (the entire
        // troposphere's heat capacity, ~1e10) makes a 0.2 W/m^2 imbalance look like ~1e-8 to the Newton,
        // so it stops there -- while conv_resid divides the SAME quantity by Fnorm and reports ~6e-4.
        // That mismatch, not a solver failure, is the residual floor. Use the flux scale (as `netflux`
        // mode already does) so the solver drives what the metric measures. CLIMA_ZONE_CEFF restores the
        // old weighting.
        const bool zone_fluxnorm = (netflux || ratio) && !std::getenv("CLIMA_ZONE_CEFF");
        g[r] = (F[z.upper] - f_lower) / (zone_fluxnorm ? Fnorm : ceff_zone[zi]);
      }
      else if (ptc)                                  // moment-flux conservation; with CLIMA_PTC_BLEND the
      {                                              // Eq.19 sliding ratio term is ADDED (smooth, no seam),
        g[r] = (F[i] - target_flux) / Fnorm;         // else the optically-thin top is handled by the hard
        if (ptc_blend)                               // local-RE SKIN overwrite in the driver.
        {
          double num, den, C; ratioSums(T, i, num, den, C);
          // g^RE = num/den - 1 is numerically unstable as den -> 0 (the kappa->0 sampled line-gap top:
          // num,den both -> 0, the ratio is noise). CLAMP the residual term: the conditioning diagonal
          // -C/den = -dln(den)/dT stays finite there (added in rebuildJ), but the residual must be bounded
          // so it cannot blow up the step. |num/den-1|<=1 (J within 2x of B); beyond that it is top noise.
          double gre = (den > 0.0) ? (num / den - 1.0) : 0.0;
          gre = std::max(-1.0, std::min(1.0, gre));
          g[r] += ptc_w[i] * gre;                    // + w_i g^RE_i  (w_i -> 1 thin top, -> 0 deep)
        }
      }
      else if (i == 0)                               // surface DOF: F_net[0]=0 balance
        g[r] = F[0] / (netflux ? Fnorm : ceff[0]);
      else                                           // radiative level
      {
        // RCB HANDOVER. The collocated (ratio) residual is purely LOCAL, so for the first radiative level
        // above a convective zone NOTHING ties the radiative solution to the top of the slaved adiabat:
        // the zone DOF residual only constrains the zone's TOTAL budget (F[upper]-F[lower-1]), not how the
        // profile joins at its top. The two regions are then free to disagree -> the kink at the RCB and a
        // leftover flux divergence on the junction level (which is exactly what floors conv_resid). The
        // differenced-flux residual gets this coupling for free, because its stencil (F[i]-F[i-1]) SPANS
        // the boundary. So give just that one level the flux-difference form: continuity is restored while
        // local RE (and its smoothness) is kept everywhere else. Disable with CLIMA_RATIO_NORCBFLUX.
        const bool above_rcb = (i > 0) && (slaved[i-1] || zone_of_dof[i-1] >= 0);
        if (ratio && above_rcb && !std::getenv("CLIMA_RATIO_NORCBFLUX"))
                                      g[r] = (F[i] - F[i-1]) / ceff[i];   // flux continuity across the RCB
        else if (ratio)             { double num, den, C; ratioSums(T, i, num, den, C);
                                      const double gre = (den > 0.0) ? (num / den - 1.0) : 0.0;
                                      const double gflux = (ratio_flux && ratio_perlevel) ? ratio_zeta[i]*ratio_flux_scale*cumF[i]/Fnorm : 0.0;
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
                                      g[r] = ratio_xi*gre + gflux + glucy + lre_anchor * F[i] / Fnorm; }
        else if (newtonlike)          g[r] = (nl_netheat ? NH[i] : NH[i]*nl_dtau[i]) / ceff[i]; // per-layer flux-change residual (NH*dtau), consistent with NHJ*dtau
        else if (localre)             g[r] = NH[i] / ceff[i] + lre_anchor * F[i] / Fnorm;  // local heating + weak TOA-flux anchor
        else if (netflux)             g[r] = F[i] / Fnorm;                          // per-level NET FLUX -> 0
        else if (centered && i+1 < n) g[r] = 0.5*(F[i+1] - F[i-1]) / ceff[i];        // symmetric divergence
        else                          g[r] = (F[i] - F[i-1]) / ceff[i];             // one-sided backward (clima)
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
  if (std::getenv("CLIMA_LREJ"))
  {
    const std::vector<double>& wavenumber = radiation_field.spectral_grid->wavenumber_list;
    const std::vector<double> w = aux::trapezoidalWeights(wavenumber);
    const size_t nb_nu = wavenumber.size();
    const double cap = std::getenv("CLIMA_LREJ_CAP") ? std::atof(std::getenv("CLIMA_LREJ_CAP")) : 0.1; // |dln T|
    const double Tmax = 2.0 * (*std::max_element(T_base.begin(), T_base.end()));

    std::vector<double> Tn = T_base;
    for (size_t r = 0; r < m; ++r)
    {
      const size_t i = unk[r];
      if (zone_of_dof[i] >= 0) continue;                 // convective-zone DOFs: handled by slaving below
      double Ti;
      if (i == 0)                                        // surface: 1-D flux balance F_net[0]=0
      {
        const double K00 = radiation_field.net_flux_jacobian[0][0];
        const double dT = (std::abs(K00) > 1e-30) ? -radiation_field.flux_total[0] / K00 : 0.0;
        Ti = T_base[0] + dT;
      }
      else                                               // radiative level: bracketed local RE B=<J>_kappa
      {
        double num = 0.0;
        for (size_t k = 0; k < nb_nu; ++k)
          num += w[k] * opacity.absorption_coeff[k][i] * radiation_field.mean_intensity[i][k];
        Ti = solveLocalREBracket(num, w, wavenumber, opacity, i, T_base[i], Tmax);
      }
      // per-iteration log-temperature trust region (cap the convex-overshoot stride)
      double dln = std::log(std::max(Ti, 1.0)) - std::log(std::max(T_base[i], 1.0));
      dln = std::max(-cap, std::min(cap, dln));
      Tn[i] = std::max(1.0, T_base[i] * std::exp(dln));
    }
    adiabatSnap(Tn);                                     // slave the convective interiors

    std::vector<double> Ffin, NHfin;
    forward_eval_full_(Tn, /*recompute_opacity=*/true, /*compute_jacobian=*/false, Ffin, NHfin);
    double inner_change = 0.0;
    for (size_t i = 0; i < n; ++i)
      inner_change = std::max(inner_change, std::abs(Tn[i] - T_base[i]) / std::max(Tn[i], 1.0));
    last_inner_change_ = inner_change;
    atmosphere.temperature = Tn;

    const double fnorm = radiation_field.flux_down_total.empty()
      ? 1.0 : std::max(1.0, std::abs(radiation_field.flux_down_total.back()));
    double fr = 0.0;
    for (size_t r = 0; r < m; ++r)
    {
      const size_t i = unk[r];
      const int zi = zone_of_dof[i];
      double res;
      if (zi >= 0) { const Zone& z = zones[zi]; const double low = (z.lower==0)?0.0:Ffin[z.lower-1]; res = Ffin[z.upper]-low; }
      else if (i == 0) res = Ffin[0];
      else res = Ffin[i] - Ffin[i-1];
      fr = std::max(fr, std::abs(res) / fnorm);
    }
    last_residual_ = mask_big_change ? 1.0 : fr;
    return;
  }
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
      for (int i = static_cast<int>(n)-2; i >= 0; --i)
        for (size_t j = 0; j < n; ++j)
          lucyJ[i][j] = lucyJ[i+1][j] + (3.0/(4.0*M_PI))*0.5*(NFJ[i][j] + NFJ[i+1][j])*nl_dtau[i+1];
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
        double num, den, C; ratioSums(atmosphere.temperature, i, num, den, C);
        ratio_inv = (den > 0.0) ? 1.0/den : 0.0;
        ratio_diag_corr = num * ratio_inv * ratio_inv * C;   // d(num/den)/dT_i den-term: -(num/den^2) C
        // STABLE Planck-slope diagonal C/den = <dln B/dT>_kappa (a kappa-weighted average, FINITE as
        // kappa->0, unlike num*C/den^2). This is the doc's -C_i/den_i; used by the Eq.19 blend instead of
        // the unstable exact ratio Jacobian terms (which carry 1/den, 1/den^2 and blow up at the thin top).
        planck_slope = (den > 0.0) ? C / den : 0.0;
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
                            const bool zfn = (netflux || ratio) && !std::getenv("CLIMA_ZONE_CEFF");
                            row[l] = (NFJ[z.upper][l]-low)/(zfn ? Fnorm : ceff_zone[zi]); }  // must match assembleG
        else if (ptc) row[l] = (1.0 - rd)*NFJ[i][l]/Fnorm;  // multi-stream NFJ faded out in the deep (rd->1); C/dt added in driver
        else if (i == 0 && hstep_surf) row[l] = 0.0;   // Sec.7.4: closed-form surface row, filled below
        else if (i == 0) row[l] = NFJ[0][l]/(netflux ? Fnorm : ceff[0]);
        // xi*(1/den)d num/dT  +  zeta * d(F/Fstar)/dT. Flux Jacobian = the AGB cumulative-heating cumH
        // (calibrated, diagonally-dominant); CLIMA_RATIO_FLUXDENSE falls back to the raw dense NFJ.
        // RCB handover row: must MATCH the residual chosen in assembleG for this level, else the Newton
        // stalls (an inconsistent Jacobian is rejected by the trust region -- measured elsewhere).
        else if (ratio && i > 0 && (slaved[i-1] || zone_of_dof[i-1] >= 0) && !std::getenv("CLIMA_RATIO_NORCBFLUX"))
                        row[l] = (NFJ[i][l] - NFJ[i-1][l]) / ceff[i];
        else if (ratio) row[l] = ratio_xi * (MK[i][l] * ratio_inv)
                               + ((ratio_flux && ratio_perlevel) ? ratio_zeta[i] * (ratio_fluxdense ? NFJ[i][l] : flux_scale*cumH[i][l]) / Fnorm : 0.0)
                               + (lucy ? lucy_sign * lucyJ[i][l] / std::max(5.670374e-5*std::pow(std::max(atmosphere.temperature[i],1.0),4)/M_PI, 1e-30) : 0.0)
                               + lre_anchor * NFJ[i][l] / Fnorm;   // level anchor (must match assembleG)
        else if (newtonlike) row[l] = NHJ[i][l]*nl_dtau[i]/ceff[i];   // Newton-like: dominant heating Jac (*dtau matches the per-layer F-diff residual)
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
        else if (hstep) { const int k = std::max(1, std::min<int>(n-1, (int)i + hstep_off));
                          row[l] = hstep_sign*NHJ[k][l]*nl_dtau[i]/ceff[i]; }
        else if (localre) row[l] = NHJ[i][l]/ceff[i] + lre_anchor * NFJ[i][l]/Fnorm;   // heating Jac + weak flux anchor
        else if (netflux) row[l] = NFJ[i][l]/Fnorm;
        else if (centered && i+1 < n) row[l] = 0.5*(NFJ[i+1][l]-NFJ[i-1][l])/ceff[i];
        else row[l] = (NFJ[i][l]-NFJ[i-1][l])/ceff[i];
      }
      if (i == 0 && hstep_surf)                     // Sec.7.4: the surface row is the closed-form emission
      {                                             // response 4 eps sigma T0^3 (the GENUINE surface diagonal
        const double sig_cgs = 5.670374e-5;         // -- no odd-moment cancellation at a boundary).
        row[0] = 4.0*sig_cgs*std::pow(std::max(atmosphere.temperature[0], 1.0), 3) / ceff[0];
      }
      // (skip for the RCB-handover level: its row is the flux difference, not the ratio form, so the
      //  ratio Planck-cooling diagonal must not be added on top of it)
      const bool rcb_handover_row = ratio && i > 0 && (slaved[i-1] || zone_of_dof[i-1] >= 0)
                                    && !std::getenv("CLIMA_RATIO_NORCBFLUX");
      if (ratio && zi < 0 && i != 0 && !rcb_handover_row)
      {
        row[i] -= ratio_xi * ratio_diag_corr;      // local Planck-cooling diagonal -xi*C_i/den_i
        // fully transparent (den=0) AND no flux anchor (zeta~0, the thin top): the row is all-zero ->
        // give it a unit restoring diagonal so the LU stays nonsingular (g_i=0 there -> dx_i=0 anyway).
        if (ratio_inv == 0.0 && ratio_zeta[i] < 1e-12) { for (size_t l = 0; l < n; ++l) row[l] = 0.0; row[i] = -1.0; }
      }
      else if (ptc_blend && zi < 0 && i != 0)        // Eq.19 Jacobian: + w_i d g^RE/dT_i ~ -w_i C/den
        row[i] -= ptc_w[i] * planck_slope;           // STABLE Planck-slope diagonal -> conditions the thin top (NFJ + C/dt deep)
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
    if (tridiag)
      for (size_t r = 0; r < m; ++r)
      {
        const size_t i = unk[r];
        if (i == 0 || zone_of_dof[i] >= 0) continue;                 // keep surface + zone rows dense
        for (size_t c = 0; c < m; ++c)
          if (c + 1 < r || c > r + 1) J[r*m+c] = 0.0;                // zero everything off the tridiagonal band
      }
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
  rebuildJ();
  const bool dbg = std::getenv("CLIMA_DBG") != nullptr;
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
  const double lam_s = std::getenv("CLIMA_SMOOTH") ? std::atof(std::getenv("CLIMA_SMOOTH")) : 0.0;
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
  if (std::getenv("CLIMA_DUMP"))
  {
    const std::vector<std::vector<double>>& NFJ = radiation_field.net_flux_jacobian;
    FILE* f;
    f = std::fopen("/tmp/dump_meta.txt", "w");  std::fprintf(f, "%zu %zu\n", n, m); std::fclose(f);
    f = std::fopen("/tmp/dump_levels.txt", "w");
    for (size_t i=0;i<n;++i) std::fprintf(f, "%zu %.10e %.10e %.10e %.10e %.10e\n",
      i, atmosphere.pressure[i], atmosphere.temperature[i], ceff[i],
      radiation_field.flux_total[i], radiation_field.flux_down_total[i]);
    std::fclose(f);
    f = std::fopen("/tmp/dump_unk.txt", "w");  for (size_t r=0;r<m;++r) std::fprintf(f, "%zu\n", unk[r]); std::fclose(f);
    f = std::fopen("/tmp/dump_g.txt", "w");    for (size_t r=0;r<m;++r) std::fprintf(f, "%.10e\n", gbase[r]); std::fclose(f);
    f = std::fopen("/tmp/dump_D.txt", "w");    for (size_t c=0;c<m;++c) std::fprintf(f, "%.10e\n", D[c]); std::fclose(f);
    f = std::fopen("/tmp/dump_NFJ.txt", "w");
    for (size_t i=0;i<n;++i){ for (size_t j=0;j<n;++j) std::fprintf(f, "%.10e ", NFJ[i][j]); std::fprintf(f, "\n"); }
    std::fclose(f);
    f = std::fopen("/tmp/dump_J.txt", "w");
    for (size_t r=0;r<m;++r){ for (size_t c=0;c<m;++c) std::fprintf(f, "%.10e ", J[r*m+c]); std::fprintf(f, "\n"); }
    std::fclose(f);
    // FINITE-DIFFERENCE Jacobian (frozen opacity) -- the TRUE dg/dx, for comparison with the analytic J.
    {
      std::vector<double> g0fd; evalG(x, /*want_jac=*/false, g0fd);
      std::vector<double> Jfd(m*m, 0.0);
      for (size_t c=0;c<m;++c)
      {
        std::vector<double> xp(x);
        const double dd = 1e-3 * std::max(std::abs(x[c]), 1.0);
        xp[c] += dd;
        std::vector<double> gp; evalG(xp, /*want_jac=*/false, gp);
        for (size_t r=0;r<m;++r) Jfd[r*m+c] = (gp[r]-g0fd[r])/dd;
      }
      f = std::fopen("/tmp/dump_Jfd.txt", "w");
      for (size_t r=0;r<m;++r){ for (size_t c=0;c<m;++c) std::fprintf(f, "%.10e ", Jfd[r*m+c]); std::fprintf(f, "\n"); }
      std::fclose(f);
    }
    std::fprintf(stderr, "  [dump] wrote /tmp/dump_*.txt incl. Jfd (n=%zu m=%zu)\n", n, m);
  }

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
  if (ptc)
  {
    // ---- ONE pseudo-transient continuation step per call (BD/gas surface_anchored machinery, ported to
    // target_flux=0). Solve (J + C/(rho s_i)) s = -g, J = (1-w)NFJ/Fnorm + w*(ratio Jac), C = ceff. The
    // PER-LEVEL pseudo-timestep dt_i = rho*s_i (s_i = tau_rad/<tau_rad>) makes the deep strongly regularised
    // (short tau_rad) and lets the ratio anchor condition the thin top. The single global pseudo-time rho is
    // ramped by the DEUFLHARD ZIB-02-14 second-order rule: grow toward the Newton step when the linear model
    // is accurate, shrink on nonlinearity or a residual rise. The driver's outer loop re-solves the RT.
    const double lncap = std::getenv("CLIMA_PTC_LNCAP") ? std::atof(std::getenv("CLIMA_PTC_LNCAP")) : 0.5;
    const double grow  = std::getenv("CLIMA_PTC_GROW")  ? std::atof(std::getenv("CLIMA_PTC_GROW"))  : 4.0;
    // local RE is applied to the ENTIRE radiative column (every non-zone, non-surface level), not just
    // the optically-thin skin. The bulk radiative zone (RCB..skin) must reach radiative equilibrium
    // (F=const) for F[RCB] to equal F[TOA]=0; the coupled flux Newton cannot drive it (diffusion-limit
    // null space -> dead ramp), but the monotone local-RE (Lambda) inversion B(T)=<J>_kappa does, exactly
    // as the working LinearisedTemperatureCorrection does. tau_skin (default huge) keeps the old thin-skin
    // -only behaviour available for A/B via CLIMA_PTC_TAUSKIN.
    // under the blend the skin is the genuinely optically-thin top (tau<1 default); the blend handles the
    // mid smoothly below it (no seam), the skin Planck-inverts the top robustly above it.
    const double tau_skin = std::getenv("CLIMA_PTC_TAUSKIN") ? std::atof(std::getenv("CLIMA_PTC_TAUSKIN")) : (ptc_blend ? 1.0 : 1e30);
    auto is_skin = [&](size_t r){ return ptc_tau[unk[r]] < tau_skin && zone_of_dof[unk[r]] < 0; };  // optically-thin top (Planck-inverted)
    // cap the heat capacity in the PTC regularisation: the massive bottom layer has ceff ~1e9, which
    // over-damps the surface (C/(rho Fnorm) huge -> frozen) and stops it pinning the flux constant. The
    // cap lets it move at moderate rho while keeping the relative weighting through the bulk.
    std::vector<double> sorted_ceff(ceff); std::sort(sorted_ceff.begin(), sorted_ceff.end());
    const double ceff_cap = sorted_ceff[n/2] * (std::getenv("CLIMA_PTC_CEFFCAP") ? std::atof(std::getenv("CLIMA_PTC_CEFFCAP")) : 3.0);
    auto Cof = [&](size_t i){ return std::min(ceff[i], ceff_cap); };
    auto rms = [&](const std::vector<double>& v){ double s=0; for (double e:v) s+=e*e; return std::sqrt(s/static_cast<double>(std::max<size_t>(1,m))); };

    std::vector<double> g;
    evalG(x, /*want_jac=*/true, g); rebuildJ();
    const double fnorm = rms(g);

    // reference pseudo-time: c0 * MEDIAN per-level diagonal scale C_i/(s_i Fnorm). The median (not the mean)
    // is essential with Eq.20 per-level s_i ~ tau_rad: the deep s_i is tiny, so the MEAN is dominated by the
    // deep -> huge rho -> weak reg everywhere -> overshoot/runaway. The median is robust to that spread.
    const double c0 = std::getenv("CLIMA_PTC_RHO0") ? std::atof(std::getenv("CLIMA_PTC_RHO0")) : 0.3;
    double rho_ref;
    { std::vector<double> dscale(m);
      for (size_t r = 0; r < m; ++r) dscale[r] = Cof(unk[r])/(ptc_srad[unk[r]]*Fnorm);
      std::sort(dscale.begin(), dscale.end());
      rho_ref = c0 * dscale[m/2]; }

    if (ptc_dt_ <= 0.0 || mask_changed)            // init rho so the MEDIAN diagonal starts DOMINANT (~1/c0)
      ptc_dt_ = rho_ref;
    else if ((ptc_blend || std::getenv("CLIMA_PTC_ZIB")) && ptc_glin_norm_ > 0.0)  // Deuflhard 2nd-order ramp
    {                                              // (ZIB 02-14): the global multiplier on the per-level tau_rad,
      // exactly as Eq.20 + the doc prescribe. DEFAULT under CLIMA_PTC_BLEND -- the blend has no skin/UL
      // overwrite, so the linear-model prediction g_lin = g + J s is clean (not poisoned), and the second-
      // order rule grows rho toward the Newton step when the model is accurate, shrinks it on nonlinearity.
      double dnorm2 = 0.0;                          // ||G_measured - G_predicted||
      for (size_t r = 0; r < m; ++r) { if (is_skin(r)) continue;
        const double d = g[r] - ptc_glin_[unk[r]]; dnorm2 += d*d; }
      const double dnorm = std::sqrt(dnorm2);
      double ratio = (dnorm <= 1e-30) ? grow : ptc_num_ / (2.0*ptc_glin_norm_*dnorm);
      ratio = std::min(std::max(ratio, 0.1), grow);
      if (fnorm >= ptc_prev_fnorm_) ratio = std::min(ratio, 0.5);   // residual rose -> shrink (correction mode)
      ptc_dt_ *= ratio;
    }
    else if (ptc_prev_fnorm_ > 0.0)                 // SER (switched evolution relaxation), DEFAULT.
    {
      // The ZIB second-order ramp compares the measured residual to the PTC step's LINEAR prediction
      // g_lin; but the Unsoeld-Lucy uniform shift and the local-RE skin ALSO move x afterwards, so
      // ||g - g_lin|| reads as wild nonlinearity and collapses rho -> 0 (the coupled step goes inert and
      // the deep/RCB imbalance never clears). SER only looks at the residual NORM, immune to those
      // overwrites: grow the pseudo-time in proportion to the residual drop, shrink on a rise. This is
      // the classic Mulder-Van Leer continuation that drives steady-state CFD; it is what lets rho climb
      // toward the Newton step so the surface/zone DOF can finally drive F[RCB] -> 0.
      double ratio = (fnorm < ptc_prev_fnorm_)
        ? std::min(grow, ptc_prev_fnorm_ / std::max(fnorm, 1e-30))
        : 0.5;                                       // residual rose -> halve (correction mode)
      ratio = std::min(std::max(ratio, 0.1), grow);
      ptc_dt_ *= ratio;
    }
    // BAND-CLAMP rho around the dominant-diagonal reference. rho must never collapse below rho_ref/band
    // (the recurring death-spiral: C/(rho s) -> inf -> steps freeze -> residual floors -> ramp shrinks more
    // -> rho -> 0) nor run away above rho_ref*band (un-regularises the deficient deep K). The ZIB/SER rule
    // only modulates rho WITHIN this band.
    const double band = std::getenv("CLIMA_PTC_BAND") ? std::atof(std::getenv("CLIMA_PTC_BAND")) : 1e2;
    ptc_dt_ = std::min(std::max(ptc_dt_, rho_ref/band), rho_ref*band);
    ptc_prev_fnorm_ = fnorm;

    // residual breakdown: WHERE does ||g|| live? report the largest |g[r]| and its level + tau (so we can
    // tell a top-ratio floor from a deep-flux floor).
    if (dbg) {
      double mx=0; size_t lmx=0;
      for (size_t r=0;r<m;++r) if (std::abs(g[r])>mx){ mx=std::abs(g[r]); lmx=unk[r]; }
      std::fprintf(stderr,"  [ptc-g] max|g|=%.3e @L%zu (tau=%.2e w=%.2f zone=%d)\n",
                   mx,lmx,ptc_tau[lmx],ptc_w[lmx],zone_of_dof[lmx]);
    }

    std::vector<double> A = J, s(m);
    for (size_t r = 0; r < m; ++r)
    {
      // PTC heat-capacity diagonal, FADED OUT where the Rosseland deep preconditioner is active (rd->1):
      // the strong C/dt was a crutch for the DEFICIENT K; with a genuine diagonally-dominant diffusion
      // operator the deep no longer needs it and can take a real Newton step. (1-rd) keeps full PTC in the
      // thin mid/top (where NFJ is still deficient and the ratio anchor conditions the very top).
      const double rd = ross_d[unk[r]];
      A[r*m+r] += (1.0 - rd) * Cof(unk[r])/(ptc_dt_*ptc_srad[unk[r]]*Fnorm);
      s[r] = -g[r];
    }
    if (solveDenseLU(A, s, m))
    {
      // store the linear-model prediction G_lin = g + J s and the ramp scalars for the next call
      if (ptc_glin_.size() != n) ptc_glin_.assign(n, 0.0);
      double gl2 = 0.0, num = 0.0;
      for (size_t r = 0; r < m; ++r)
      { double gl = g[r]; for (size_t c = 0; c < m; ++c) gl += J[r*m+c]*s[c];
        ptc_glin_[unk[r]] = gl; if (is_skin(r)) continue; gl2 += gl*gl; num += gl*(g[r]-gl); }
      ptc_glin_norm_ = std::sqrt(gl2); ptc_num_ = std::abs(num);

      double maxdln = 0.0;
      for (size_t r = 0; r < m; ++r) maxdln = std::max(maxdln, std::abs(s[r])/std::max(x[r], 1.0));
      const double sc = (maxdln > lncap) ? lncap/maxdln : 1.0;       // log-temperature safety cap (rarely binds)
      for (size_t r = 0; r < m; ++r) x[r] = std::max(1.0, x[r] + sc*s[r]);

      // ---- UNSOELD-LUCY uniform level shift (THE missing piece). The coupled flux-PTC step above drives
      // the column SHAPE to radiative equilibrium (dF/dtau -> 0, so F_net -> const down the column), but the
      // absolute LEVEL of that constant is pinned only by the single surface row and converges agonisingly
      // slowly -- the exact observed stall: "F_net essentially constant throughout, just not zero, surface
      // frozen". Drive the conserved net flux to zero DIRECTLY by a near-uniform fractional shift
      // T -> T*(1+alpha) of the whole column, alpha the Newton step of the TOA net flux w.r.t. a uniform
      // log-T change:  alpha = -(F_net(top) - target) / sum_j (dF_net(top)/dT_j) * T_j.  This is the classic
      // Unsoeld-Lucy / Lambda-iteration level correction, and is precisely what the working
      // LinearisedTemperatureCorrection PTC uses to set the level (it keeps only the surface in its Newton
      // and relaxes the rest by local RE + this shift). Bounded per step; applied before the local-RE skin
      // so the genuinely-thin top still ends on local RE. Disable with CLIMA_PTC_NOUL for A/B.
      // RESTORED under the blend: for target=0 the deficient flux-K leaves the absolute LEVEL (the uniform-T
      // null mode) unconstrained -> F_net settles to a constant != 0 and the surface freezes. UL drives that
      // constant to zero. It does perturb the ZIB linear-model prediction, but the rho band-clamp prevents
      // any runaway/freeze from that, and alpha -> 0 as F_net(TOA) -> 0 so the perturbation vanishes at
      // convergence.
      if (!std::getenv("CLIMA_PTC_NOUL"))
      {
        const size_t itoa = n - 1;
        double denom = 0.0;
        for (size_t j = 0; j < n; ++j) denom += radiation_field.net_flux_jacobian[itoa][j] * T_base[j];
        const double Ftoa = radiation_field.flux_total[itoa] - target_flux;
        if (std::abs(denom) > 1e-30)
        {
          double alpha = -Ftoa / denom;
          alpha = std::max(-0.1, std::min(0.1, alpha));               // bound the per-iteration level move
          for (size_t r = 0; r < m; ++r) x[r] = std::max(1.0, x[r] * (1.0 + alpha));
          if (dbg) std::fprintf(stderr, "  [ptc-UL] Ftoa=%.4e alpha=%.4e\n", Ftoa, alpha);
        }
      }

      // ---- LOCAL-RE SKIN: overwrite the optically-thin top (tau < tau_skin) by DIRECTLY converting the
      // absorbed mean intensity to a temperature via the Planck function: solve B(T_i)=<J>_kappa (emitted =
      // absorbed). kappa there is SMALL but NONZERO, so this inversion is well-defined and robust -- unlike
      // the ratio g^RE=num/den-1 (fragile as den->0). The insensitive net-flux residual cannot determine T
      // there; the stellar beam is in num_i (the top BC). RUNS UNDER THE BLEND TOO: the blend handles the mid
      // smoothly below tau_skin (no hard seam), the skin Planck-inverts the genuinely-thin top above it.
      {
      const std::vector<double>& sk_wn = radiation_field.spectral_grid->wavenumber_list;
      const std::vector<double> sk_w = aux::trapezoidalWeights(sk_wn);
      const size_t sk_nnu = sk_wn.size();
      double Tmax = 1.0; for (size_t r = 0; r < m; ++r) Tmax = std::max(Tmax, x[r]); Tmax *= 2.0;
      for (size_t r = 0; r < m; ++r)
      {
        const size_t i = unk[r];
        if (i == 0 || zone_of_dof[i] >= 0) continue;            // surface stays the flux anchor; zones slaved
        // SMOOTH skin handoff: blend the Planck-inverted T into the stepped T with the SAME smooth weight
        // sig = ptc_w[i] (->1 thin top, ->0 deep) instead of a HARD overwrite below tau_skin. A hard switch
        // leaves the boundary level conditioned by neither side -> the residual ~0.1 bar (tau~1) kink. With
        // the smooth weight the skin fades in continuously: deep (sig~0) = pure blend step, top (sig~1) =
        // pure Planck inversion, no switch surface. Deep levels are near LTE (J~B) so T_skin~T anyway.
        const double sig = ptc_w[i];
        if (sig < 1e-3) continue;                               // negligible skin weight -> skip the root-find
        double num = 0.0;
        for (size_t k = 0; k < sk_nnu; ++k) num += sk_w[k]*opacity.absorption_coeff[k][i]*radiation_field.mean_intensity[i][k];
        const double Tskin = solveLocalREBracket(num, sk_w, sk_wn, opacity, i, x[r], Tmax);
        x[r] = std::max(1.0, (1.0 - sig)*x[r] + sig*Tskin);
      }
      }

      // ---- DIRECT SURFACE ENERGY BALANCE (the surface analog of the top Planck inversion). The surface
      // flux Jacobian NFJ[0][0] = 4 eps sigma T[0]^3 is the GENUINE surface-emission response (not the
      // deficient interior odd-moment diagonal), so drive F_net[0]=0 by a direct 1-D Newton on T[0] from
      // the committed point: dT = -F_net[0]/NFJ[0][0]. This uses the surface's own (effectively large) time
      // step, as its high heat capacity demands, instead of the atmospheric tau_rad that froze it. Bounded
      // per call. Replaces the (frozen) PTC step for the surface DOF only. Disable with CLIMA_PTC_NOSURFBAL.
      if (ptc_blend && !std::getenv("CLIMA_PTC_NOSURFBAL"))
        for (size_t r = 0; r < m; ++r)
        {
          if (unk[r] != 0 || zone_of_dof[0] >= 0) continue;          // only the BOA-tied surface (no conv zone)
          const double K00 = radiation_field.net_flux_jacobian[0][0];
          if (std::abs(K00) <= 1e-30) continue;
          double dT = -radiation_field.flux_total[0] / K00;          // 1-D Newton on the surface balance
          const double cap = 0.1 * std::max(T_base[0], 1.0);         // bound the per-call move
          dT = std::max(-cap, std::min(cap, dT));
          x[r] = std::max(1.0, T_base[0] + dT);
          if (dbg) std::fprintf(stderr, "  [ptc-surf] F[0]=%.3e K00=%.3e dT=%.3f T0=%.2f\n",
                                radiation_field.flux_total[0], K00, dT, x[r]);
        }

      // The optically-thick deep is carved out of the Newton (deep[]/kdeep) and reconstructed by gradient
      // integration AFTER buildProfile -- see the "DEEP GRADIENT INTEGRATION" block near the commit. Nothing
      // to do here: those DOFs are not in unk.
    }
    if (dbg) std::fprintf(stderr, "  [ptc%s] rho=%.3e ||g||=%.3e\n", ptc_blend?"-blend":"", ptc_dt_, fnorm);
  }
  else if (ratio && ratio_flux && ratio_perlevel)
  {
    // ---- AGB-style fixed-omega damped Newton (matching the AGB linearised corrector). The
    // zeta-flux term makes the residual depend NONLOCALLY on T (the flux at i integrates all heating
    // below), so the inner Jacobian cannot be made diagonally dominant and an inner Newton-to-convergence
    // (NLEQ-ERR) over-damps to lambda_min and stalls. The AGB instead takes ONE under-relaxed step per RT
    // solve and lets the outer loop re-solve the opacity/RT; we do the same -- a few omega-relaxed steps
    // with a small sign-preserving Tikhonov floor (keeps empty rows non-singular) and a log-temperature
    // cap (bounds the stride on the unbounded ratio pole), no monotonicity gate.
    const int    maxit = std::getenv("CLIMA_RATIO_MAXIT") ? std::atoi(std::getenv("CLIMA_RATIO_MAXIT")) : 8;
    const double omega = std::getenv("CLIMA_RATIO_OMEGA") ? std::atof(std::getenv("CLIMA_RATIO_OMEGA")) : 0.3;
    const double lncap = std::getenv("CLIMA_RATIO_LNCAP") ? std::atof(std::getenv("CLIMA_RATIO_LNCAP")) : 0.1;
    const double gtol  = std::getenv("CLIMA_RATIO_GTOL")  ? std::atof(std::getenv("CLIMA_RATIO_GTOL"))  : 1e-6;
    std::vector<double> g;
    int it = 0; double gmax = 0.0;
    for (; it < maxit; ++it)
    {
      evalG(x, /*want_jac=*/true, g); rebuildJ();
      gmax = 0.0; for (double e : g) gmax = std::max(gmax, std::abs(e));
      if (gmax < gtol) break;

      std::vector<double> A = J, b(m);
      double dmean = 0.0; for (size_t r = 0; r < m; ++r) dmean += std::abs(A[r*m+r]);
      dmean /= static_cast<double>(m);
      const double reg = 1e-6 * (dmean + 1e-300);                       // sign-preserving Tikhonov floor
      for (size_t r = 0; r < m; ++r) A[r*m+r] += (A[r*m+r] >= 0.0 ? reg : -reg);
      for (size_t r = 0; r < m; ++r) b[r] = -g[r];
      if (!solveDenseLU(A, b, m)) break;

      double maxdln = 0.0;                                              // log-temperature trust cap
      for (size_t r = 0; r < m; ++r) maxdln = std::max(maxdln, std::abs(b[r])/std::max(x[r], 1.0));
      const double sc = (maxdln > lncap) ? lncap/maxdln : 1.0;
      for (size_t r = 0; r < m; ++r) x[r] = std::max(1.0, x[r] + omega*sc*b[r]);
    }
    if (dbg) std::fprintf(stderr, "  [omega-ratio] iters=%d omega=%.2f ||g||inf=%.3e\n", it, omega, gmax);
  }
  else if (ratio && ratio_flux)
  {
    // ---- ALTERNATING deflation (ratio-converge, then a flux-only step in the soft subspace). The fully
    // coupled deflated Newton g_def = g^RE + P_Vk(F) grinds because pushing the flux re-violates local RE
    // inside one residual. Instead alternate two phases that DON'T fight:
    //   (A) converge the pure ratio g^RE (NLEQ-ERR) -> smooth profile, F = const (the photosphere shape);
    //   (B) take ONE flux-correcting step RESTRICTED to span{V_k}, the k softest right-singular vectors of
    //       J_ratio. Since J_ratio V_k ~ 0, a step along V_k is nearly INVISIBLE to g^RE -- it moves the
    //       flux without re-violating the ratio. The step solves the k-dim least squares min||(NFJ/Fnorm)
    //       V_k y + F/Fnorm|| (project the flux deviation onto what the soft modes can reach), dT = V_k y.
    // The driver's outer loop supplies the A/B alternation and the opacity update between calls.
    const int    kdef   = std::getenv("CLIMA_RATIO_DEFLATE_K") ? std::atoi(std::getenv("CLIMA_RATIO_DEFLATE_K")) : 8;
    const int    nratio = std::getenv("CLIMA_RATIO_NRATIO") ? std::atoi(std::getenv("CLIMA_RATIO_NRATIO")) : 40;
    const double xtol   = std::getenv("CLIMA_RATIO_XTOL")  ? std::atof(std::getenv("CLIMA_RATIO_XTOL"))  : 1e-7;
    const double lncap  = std::getenv("CLIMA_RATIO_LNCAP") ? std::atof(std::getenv("CLIMA_RATIO_LNCAP")) : 0.1;
    constexpr double lambda_min = 1e-4;
    const int k = std::min(kdef, static_cast<int>(m));
    std::vector<double> g;

    auto scaledNorm = [&](const std::vector<double>& v, const std::vector<double>& xc) {
      double s2 = 0.0; for (size_t r = 0; r < m; ++r) { const double sc = std::max(xc[r], 1.0); s2 += (v[r]/sc)*(v[r]/sc); }
      return std::sqrt(s2 / static_cast<double>(m)); };
    auto allFinite = [&](const std::vector<double>& v) { for (double e : v) if (!std::isfinite(e)) return false; return true; };

    // ---- Phase A: NLEQ-ERR on the pure ratio residual ----
    double lambda = 1.0; int ai = 0; double gmax = 0.0;
    for (; ai < nratio; ++ai)
    {
      evalG(x, /*want_jac=*/true, g); rebuildJ();
      gmax = 0.0; for (double e : g) gmax = std::max(gmax, std::abs(e));
      std::vector<double> A = J, b(m); for (size_t r = 0; r < m; ++r) b[r] = -g[r];
      if (!solveDenseLU(A, b, m)) break;
      const std::vector<double> dx = b; const double ndx = scaledNorm(dx, x);
      if (ndx < xtol) break;
      double lam = std::min(1.0, 2.0*lambda); double lam_acc = -1.0; std::vector<double> x_acc;
      for (int trial = 0; trial < 25; ++trial)
      {
        std::vector<double> xt(m), gt; for (size_t r = 0; r < m; ++r) xt[r] = std::max(1.0, x[r] + lam*dx[r]);
        evalG(xt, false, gt); double theta = 1e300;
        if (allFinite(gt)) { std::vector<double> A2 = J, b2(m); for (size_t r = 0; r < m; ++r) b2[r] = -gt[r];
          if (solveDenseLU(A2, b2, m) && allFinite(b2)) theta = scaledNorm(b2, xt)/ndx; }
        if (theta <= 1.0 - 0.25*lam) { lam_acc = lam; x_acc = xt; break; }
        if (lam <= lambda_min)       { lam_acc = lam; x_acc = xt; break; }
        lam = std::max(0.5*lam, lambda_min);
      }
      if (lam_acc <= 0.0) break; x = x_acc; lambda = lam_acc;
    }

    // ---- Phase B: one flux-only step confined to span{V_k} ----
    evalG(x, /*want_jac=*/true, g); rebuildJ();              // ratio converged; read J, NFJ, F at x
    Eigen::MatrixXd Jm(m, m);
    for (size_t r = 0; r < m; ++r) for (size_t c = 0; c < m; ++c) Jm(r,c) = J[r*m+c];
    Eigen::BDCSVD<Eigen::MatrixXd> svd(Jm, Eigen::ComputeThinV);
    Eigen::MatrixXd Vk = svd.matrixV().rightCols(k);

    // (1) SMOOTH the deflation subspace. The ratio's deep soft modes are ~50% Nyquist energy (rough): the
    // flux least-squares then SELECTS those rough directions and deposits a deep sawtooth (the diagonally-
    // deficient flux operator NFJ has been handed authority over the level-to-level structure, which it
    // populates in the checkerboard null space). We deny it that subspace: low-pass each column of V_k and
    // re-orthonormalise, so the flux step dT = V_k y is smooth BY CONSTRUCTION. A flux constant/offset
    // correction is intrinsically smooth, so this loses ~nothing of the flux reduction.
    if (std::getenv("CLIMA_RATIO_ROUGHVK") == nullptr)
    {
      const int npass = std::getenv("CLIMA_RATIO_VKSMOOTH") ? std::atoi(std::getenv("CLIMA_RATIO_VKSMOOTH")) : 2;
      for (int p = 0; p < npass; ++p)
      {
        Eigen::MatrixXd Vs = Vk;
        for (size_t r = 1; r+1 < m; ++r)
        {
          const size_t i = unk[r];
          if (i == 0 || zone_of_dof[i] >= 0) continue;
          if (unk[r-1] != i-1 || unk[r+1] != i+1) continue;       // adjacent free levels only
          if (slaved[i] || slaved[i-1] || slaved[i+1]) continue;
          Vs.row(r) = 0.25*Vk.row(r-1) + 0.5*Vk.row(r) + 0.25*Vk.row(r+1);   // 1-2-1 low-pass
        }
        Vk = Vs;
      }
      Eigen::HouseholderQR<Eigen::MatrixXd> qr(Vk);                // re-orthonormalise (thin Q)
      Vk = qr.householderQ() * Eigen::MatrixXd::Identity(m, k);
    }

    Eigen::VectorXd Fres(m);
    for (size_t r = 0; r < m; ++r) Fres(r) = radiation_field.flux_total[unk[r]]/Fnorm;
    Eigen::VectorXd dT(m);
    if (std::getenv("CLIMA_RATIO_HPREC"))
    {
      // OPTION 2: set the flux step direction with the diagonally-DOMINANT heating Jacobian H = M - C
      // (M = meanint_kappa_jacobian, C = sum w kappa dB/dT), not the diagonally-deficient NFJ. NFJ has no
      // local diagonal, so it populates the checkerboard null space (the sawtooth); H carries the -C Planck
      // diagonal, so the step structure it sets is smooth by the operator itself. NFJ is NOWHERE in the
      // step -- only in the residual (exact F), so it still converges to F=0. Confined to span{V_k} for
      // conditioning (H is dominant but den-scaled -> singular at the kappa->0 top). Solve (V_k^T H V_k) y =
      // V_k^T Fres; sign handled by the +/- line search below.
      const std::vector<std::vector<double>>& MK2 = radiation_field.meanint_kappa_jacobian;
      std::vector<double> Cred(m, 0.0);
      for (size_t r = 0; r < m; ++r) { double nu, de, C; ratioSums(atmosphere.temperature, unk[r], nu, de, C); Cred[r] = C; }
      Eigen::MatrixXd Hred(m, m);
      for (size_t r = 0; r < m; ++r) for (size_t c = 0; c < m; ++c)
        Hred(r,c) = MK2[unk[r]][unk[c]] - (r==c ? Cred[r] : 0.0);
      Eigen::MatrixXd HV = Vk.transpose()*Hred*Vk;          // k x k: dominant operator in the subspace
      Eigen::VectorXd y = HV.fullPivLu().solve(Vk.transpose()*Fres);
      dT = Vk*y;
    }
    else
    {
      // OPTION 1 (default): least-squares flux reduction via NFJ projected onto the (smoothed) soft subspace.
      Eigen::MatrixXd NFJm(m, m);
      for (size_t r = 0; r < m; ++r) for (size_t c = 0; c < m; ++c)
        NFJm(r,c) = radiation_field.net_flux_jacobian[unk[r]][unk[c]]/Fnorm;
      Eigen::MatrixXd Aflux = NFJm*Vk;                       // m x k : flux response to soft-mode moves
      Eigen::VectorXd y = (Aflux.transpose()*Aflux).ldlt().solve(-(Aflux.transpose()*Fres));
      dT = Vk*y;                                             // flux-only step (ratio-invisible)
    }

    // The ratio's DEEP soft modes (optically-thick, local RE degenerate) include grid-scale/Nyquist
    // directions, so the unconstrained flux step injects a sawtooth in the lower atmosphere. One Shapiro
    // pass on the STEP annihilates the Nyquist component exactly (alpha=1) while keeping the smooth
    // flux-reducing part -- the flux deviation is a smooth quantity, so its correction should be too.
    // Interior free radiative levels only (both neighbours adjacent free levels; not surface/zone/slaved).
    // Default OFF now -- the smooth-V_k subspace (fix 1) makes the step smooth by construction; this is the
    // option-3 guard rail, kept available via CLIMA_RATIO_STEPSMOOTH.
    if (std::getenv("CLIMA_RATIO_STEPSMOOTH") != nullptr)
    {
      Eigen::VectorXd dTs = dT;
      for (size_t r = 1; r+1 < m; ++r)
      {
        const size_t i = unk[r];
        if (i == 0 || zone_of_dof[i] >= 0) continue;
        if (unk[r-1] != i-1 || unk[r+1] != i+1) continue;     // require adjacent free levels
        if (slaved[i] || slaved[i-1] || slaved[i+1]) continue;
        dTs(r) = dT(r) + 0.25*(dT(r-1) - 2.0*dT(r) + dT(r+1));
      }
      dT = dTs;
    }
    double maxdln = 0.0; for (size_t r = 0; r < m; ++r) maxdln = std::max(maxdln, std::abs(dT(r))/std::max(x[r], 1.0));
    const double base = (maxdln > lncap) ? lncap/maxdln : 1.0;
    const double f0 = Fres.norm(); double t = base; bool acc = false;   // backtrack on ||F/Fnorm||, both signs
    for (int ls = 0; ls < 14 && !acc; ++ls)
    {
      for (double s : {1.0, -1.0})
      {
        std::vector<double> xt(m), gt; for (size_t r = 0; r < m; ++r) xt[r] = std::max(1.0, x[r] + s*t*dT(r));
        evalG(xt, false, gt);
        Eigen::VectorXd Frt(m); for (size_t r = 0; r < m; ++r) Frt(r) = radiation_field.flux_total[unk[r]]/Fnorm;
        if (std::isfinite(Frt.norm()) && Frt.norm() < f0) { x = xt; acc = true; break; }
      }
      t *= 0.5;
    }
    if (dbg) std::fprintf(stderr, "  [defl-alt] ratioIters=%d k=%d ||gRE||inf=%.2e ||F/Fn||=%.3e step=%.2f\n", ai, k, gmax, f0, t);
  }
  else if (ratio || newtonlike)
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

    std::vector<double> g;
    double lambda = 1.0;
    int it = 0;
    for (; it < maxit; ++it)
    {
      evalG(x, /*want_jac=*/true, g);   // residual + analytic ratio Jacobian at the current point
      rebuildJ();

      std::vector<double> A = J, b(m);          // ordinary Newton correction  J dx = -g
      for (size_t r = 0; r < m; ++r) b[r] = -g[r];
      if (!solveDenseLU(A, b, m)) break;        // singular (should not happen: J is diagonally dominant)
      const std::vector<double> dx = b;
      const double norm_dx = scaledNorm(dx, x);
      if (norm_dx < xtol) break;                // converged in the Newton correction

      // NLEQ-ERR damping: accept lambda when theta = ||dxbar||/||dx|| <= 1 - lambda/4, where the SIMPLIFIED
      // correction dxbar = -J^{-1} g(x + lambda dx) reuses the matrix J(x) but the TRUE residual at the trial.
      double lam = std::min(1.0, 2.0*lambda);   // Deuflhard prediction / warm start
      double lam_acc = -1.0; std::vector<double> x_acc;
      for (int trial = 0; trial < 25; ++trial)
      {
        std::vector<double> xt(m);
        for (size_t r = 0; r < m; ++r) xt[r] = std::max(1.0, x[r] + lam*dx[r]);
        std::vector<double> gt; evalG(xt, /*want_jac=*/false, gt);
        double theta = 1e300;
        if (allFinite(gt))
        {
          std::vector<double> A2 = J, b2(m);
          for (size_t r = 0; r < m; ++r) b2[r] = -gt[r];
          if (solveDenseLU(A2, b2, m) && allFinite(b2)) theta = scaledNorm(b2, xt) / norm_dx;
        }
        if (theta <= 1.0 - 0.25*lam) { lam_acc = lam; x_acc = xt; break; }   // natural monotonicity test
        if (lam <= lambda_min)       { lam_acc = lam; x_acc = xt; break; }   // regularity floor: take the small step
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
    if (std::getenv("CLIMA_FPSTEP"))
    {
      const int    fp_it  = std::getenv("CLIMA_FP_ITER") ? std::atoi(std::getenv("CLIMA_FP_ITER")) : 20;
      const double fp_tol = std::getenv("CLIMA_FP_TOL")  ? std::atof(std::getenv("CLIMA_FP_TOL"))  : 1e-12;
      auto l2 = [](const std::vector<double>& v){ double s=0.0; for (double e : v) s += e*e; return std::sqrt(s); };
      std::vector<double> g;
      evalG(x, /*want_jac=*/true, g);
      double gn = l2(g);
      const double gn_first = gn;
      int it = 0, nrej = 0;
      for (; it < fp_it && gn > fp_tol; ++it)
      {
        rebuildJ();                                        // M at the current point
        std::vector<double> A(J), rhs(m);
        for (size_t r = 0; r < m; ++r) rhs[r] = -g[r];
        if (!solveDenseLU(A, rhs, m)) { if (dbg) std::fprintf(stderr, "  [fp] LU fail at it=%d\n", it); break; }
        // DEEP backtracking: H has a NEAR-NULL DC (flux-level) mode -- its eigenvalue is -E2 ~ e^-dtau
        // (doc Sec.7.4) -- so M^-1 hugely amplifies the level component of a flux-conservation residual
        // and ||s|| can be astronomical. Halving only ~8 times cannot distinguish "ascent direction"
        // from "descent direction, absurd step length", so allow lambda down to ~1e-9.
        const int ls_max = std::getenv("CLIMA_FP_LS") ? std::atoi(std::getenv("CLIMA_FP_LS")) : 30;
        double sn = 0.0, smax = 0.0;
        for (size_t r = 0; r < m; ++r) { sn += rhs[r]*rhs[r]; smax = std::max(smax, std::abs(rhs[r])); }
        sn = std::sqrt(sn);
        double lam = 1.0; bool accepted = false; double lam_acc = 0.0;  // backtrack on ||G|| ONLY
        for (int ls = 0; ls < ls_max; ++ls)
        {
          std::vector<double> xt(m);
          for (size_t r = 0; r < m; ++r) xt[r] = std::max(1.0, x[r] + lam*rhs[r]);
          std::vector<double> gt; evalG(xt, /*want_jac=*/false, gt);
          const double gtn = l2(gt);
          if (gtn < gn) { x = xt; gn = gtn; accepted = true; lam_acc = lam; break; }
          lam *= 0.5;
        }
        if (dbg && it < 3)
          std::fprintf(stderr, "  [fp:step] it=%d ||s||=%.3e max|s|=%.3e  accepted=%d lambda=%.3e ||G||=%.3e\n",
                       it, sn, smax, (int)accepted, lam_acc, gn);
        if (!accepted) { ++nrej; break; }                  // no descent even at lambda~1e-9 -> ascent dir
        evalG(x, /*want_jac=*/true, g);                    // refresh radiation field for the next M
        gn = l2(g);
      }
      if (dbg)
        std::fprintf(stderr, "  [fp] iters=%d ||G||: %.3e -> %.3e  (ratio %.3f/it)  rejected=%d\n",
                     it, gn_first, gn, (it > 0 && gn_first > 0.0) ? std::pow(gn/gn_first, 1.0/it) : 1.0, nrej);
    }
    else {
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
  if (ratio && !lucy && !std::getenv("CLIMA_RATIO_NOUL") && !radiation_field.net_flux_jacobian.empty())
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
  if (carve_deep && kdeep >= 2 && !radiation_field.flux_net_thermal_total.empty())
  {
    const std::vector<double>& Fth  = radiation_field.flux_net_thermal_total;
    const std::vector<double>& Ftot = radiation_field.flux_total;
    const double sig = 5.670374e-5;
    auto Bof = [&](double T){ return sig*T*T*T*T/M_PI; };
    auto Tof = [&](double B){ return std::pow(std::max(B,1e-30)*M_PI/sig, 0.25); };
    const double rmin = 0.2, rmax = 5.0, ffloor = 1.0;
    constexpr double Kgrey = 4.0*M_PI/3.0;                        // grey diffusivity F = -(4pi/3) dB/dtau (B = sigma T^4/pi)
    std::vector<double> Bn(n, 0.0);
    Bn[kdeep+1] = Bof(T_final[kdeep+1]);                          // anchor: the photosphere DOF (Newton value)
    const int ibot = (deep[0] ? 0 : 1);                          // sweep through the surface when it is carved
    double Keff = 0.0;                                            // diffusivity Fth*dtau/dB_down, calibrated from clean layers above
    for (int i = kdeep; i >= ibot; --i)
    {
      const double dB_old = Bof(T_base[i+1]) - Bof(T_base[i]);    // current Planck increment, layer i
      const double Ftar   = Fth[i] - Ftot[i];                    // target thermal flux = -F_stellar (Eq.30)
      // Where the carried-flux ratio is well conditioned (non-flat layer carrying real flux), rescale the
      // current Planck increment multiplicatively by F_star/F_th -- this IS the measured-D form. At a FLAT spot
      // the measured D = -dtau*Fth/dB_old is 0/0, so the multiplicative form collapses (ratio*0=0) and cannot
      // build the flux-carrying gradient -- the old linear-extrapolation/freeze band-aids patched T but left the
      // flux seamed (notably F1 at the surface-air jump). Instead apply the doc's grey-floor fallback: the
      // ABSOLUTE gradient dB_down = F_star*dtau/K that carries the target flux regardless of the current
      // increment. K is read from the last clean layer above (unit-safe, the solver's own diffusivity), falling
      // back to the grey 4pi/3 until one has calibrated it.
      const bool flat = (i + 2 <= kdeep + 1) && std::abs(dB_old) < 0.7 * std::abs(Bn[i+1] - Bn[i+2]);
      double dB_down;                                            // Planck increment going down across layer i (Bn[i]-Bn[i+1] > 0)
      if (!flat && std::abs(Fth[i]) > ffloor)
      {
        const double ratio = std::max(rmin, std::min(rmax, 1.0 - Ftot[i]/Fth[i]));   // F_star/F_th, clamped O(1)
        dB_down = -dB_old * ratio;                               // measured-D multiplicative rescaling
        if (std::abs(dB_down) > 1e-30 && Fth[i]*dB_down > 0.0)    // calibrate K from this clean, consistent-sign layer
          Keff = Fth[i] * deep_dtau[i] / dB_down;
      }
      else
      {
        const double K = (Keff > 1e-30) ? Keff : Kgrey;          // grey-floor absolute flux-carrying gradient
        dB_down = Ftar * deep_dtau[i] / K;
      }
      Bn[i] = Bn[i+1] + dB_down;                                 // one-sided downward integration
      T_final[i] = std::max(1.0, Tof(Bn[i]));
    }
    if (std::getenv("CLIMA_DBG"))
    {
      const double Ft0 = Fth[0]-Ftot[0], Ft1 = Fth[1]-Ftot[1], Ft2 = Fth[2]-Ftot[2];
      std::fprintf(stderr, "  [ptc-deep] carve kdeep=%d T0=%.1f T1=%.1f T(kdeep)=%.1f anchor=%.1f  F0=%.2e F1=%.2e\n",
                   kdeep, T_final[0], T_final[1], T_final[kdeep], T_final[kdeep+1],
                   radiation_field.flux_total[0], radiation_field.flux_total[1]);
      std::fprintf(stderr, "  [ptc-deep] dtau(0,1)=%.2e dtau(1,2)=%.2e  Fstar(thermal target) L0=%.2e L1=%.2e L2=%.2e  Keff=%.3e\n",
                   deep_dtau[0], deep_dtau[1], Ft0, Ft1, Ft2, Keff);
    }
  }

  // Shapiro filter on the committed radiative PROFILE: removes the flux-neutral Nyquist (2*dz) sawtooth
  // that opacity-sampling noise excites through the radiative-equilibrium Jacobian's near-null grid mode.
  // Applied each iteration; the solver re-establishes the real (long-wavelength) structure while the
  // filter keeps suppressing the grid mode, so it does NOT bias the fixed point (the Nyquist is flux-
  // neutral) the way a residual curvature penalty does (which fed the convective-mask runaway). alpha=1
  // annihilates the Nyquist exactly; alpha~0.3-0.5 barely touches real inversions. Interior RADIATIVE
  // levels only (both neighbours radiative, away from the RCB and surface). CLIMA_SHAPIRO=0 disables it.
  {
    // Shapiro default OFF for every flux-residual / ratio / ptc mode: although it damps the radiative
    // sawtooth, it MUTATES the committed profile, and the next iteration's convective-mask detection reads
    // that profile -> smoothing the lower-stratosphere levels shifts the RCB off its true radiative-
    // equilibrium position (mask feedback -> wrong troposphere extent / surface T). The sawtooth is cured
    // at the source by CLIMA_TRIDIAG (an approximate Jacobian that damps the Nyquist step) WITHOUT touching
    // the profile or the mask. Re-enable per-run with CLIMA_SHAPIRO only for the pure-radiative stress test.
    // CLIMA_HSTEP defaults the Shapiro filter OFF: the whole point is that the dominant heating step
    // operator should retire the sawtooth on its own, so the filter must not mask the result.
    const double alpha_s = std::getenv("CLIMA_SHAPIRO") ? std::atof(std::getenv("CLIMA_SHAPIRO")) : ((netflux || centered || localre || ratio || newtonlike || ptc || hstep) ? 0.0 : 1.0);
    if (alpha_s > 0.0)
    {
      std::vector<double> Tf = T_final;
      for (size_t i = 1; i+1 < n; ++i)
      {
        if (slaved[i] || slaved[i-1] || slaved[i+1]) continue;                                 // skip convective/RCB
        if (zone_of_dof[i] >= 0 || zone_of_dof[i-1] >= 0 || zone_of_dof[i+1] >= 0) continue;    // skip zone DOFs
        Tf[i] = std::max(1.0, T_final[i] + 0.25*alpha_s*(T_final[i-1] - 2.0*T_final[i] + T_final[i+1]));
      }
      T_final = Tf;
    }
  }

  std::vector<double> Ffin, NHfin;
  forward_eval_full_(T_final, /*recompute_opacity=*/true, /*compute_jacobian=*/false, Ffin, NHfin);

  // how much this inner solve moved the profile -> gates the mask re-detection on the NEXT call
  double inner_change = 0.0;
  for (size_t i = 0; i < n; ++i)
    inner_change = std::max(inner_change, std::abs(T_final[i] - T_base[i]) / std::max(T_final[i], 1.0));
  last_inner_change_ = inner_change;

  atmosphere.temperature = T_final;

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
  const double fnorm = radiation_field.flux_down_total.empty()
    ? 1.0 : std::max(1.0, std::abs(radiation_field.flux_down_total.back()));
  for (size_t r = 0; r < m; ++r)
  {
    const size_t i = unk[r];
    const int zi = zone_of_dof[i];
    double res;
    if (zi >= 0) { const Zone& z = zones[zi]; const double low = (z.lower==0)?0.0:Ffin[z.lower-1]; res = Ffin[z.upper]-low; }
    else if (i == 0) res = Ffin[0];
    else res = (netflux || ptc) ? Ffin[i] : (Ffin[i] - Ffin[i-1]);   // ptc: actual net-flux imbalance (conservation)
    flux_resid = std::max(flux_resid, std::abs(res) / fnorm);
  }
  // the carved-out deep (incl. the surface endpoint when carved) is not in unk, but its flux conservation
  // still gates convergence (creep-proof)
  for (size_t i = 0; i < n; ++i) if (deep[i]) flux_resid = std::max(flux_resid, std::abs(Ffin[i]) / fnorm);

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
  if (collocated)
  {
    std::vector<double> gfin;
    assembleG(Ffin, NHfin, T_final, gfin);
    own_resid = 0.0;
    for (double e : gfin) own_resid = std::max(own_resid, std::abs(e));
  }
  if (dbg)
    std::fprintf(stderr, "  [converge] own_resid=%.3e  flux_resid=%.3e (error bar)  dT/T=%.3e  mode=%s\n",
                 own_resid, flux_resid, inner_change, collocated ? "collocated" : "flux");
  last_residual_ = mask_big_change ? 1.0 : std::max(own_resid, inner_change);
}


}
