/*
* This file is part of the ngam code.
* Copyright (C) 2026 Daniel Kitzmann
*
* ngam is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

#ifndef _quench_chemistry_h
#define _quench_chemistry_h

#include <string>
#include <vector>

#include "chemistry.h"
#include "chem_species.h"


namespace ngam {


// Quenching approximation for disequilibrium chemistry in H2-dominated atmospheres, following
// Zahnle & Marley (2014, ApJ 797, 41; ZM14). It is a POST-PROCESSING step on an equilibrium
// composition and must therefore be listed AFTER a module that provides one (normally the
// FastChem `equilibrium` module): chemistry=[("equilibrium", {...}), ("quench", {...})].
//
// For every quenched family the chemical timescale t_chem(T, p) from ZM14 is compared with the
// mixing timescale t_mix = H^2/Kzz (ZM14 Sec. 4, with the mixing length L = H). Walking up from
// the bottom of the atmosphere, the quench point is the first level where t_chem > t_mix; it is
// located CONTINUOUSLY by interpolating the two timescales in log p between the bracketing levels,
// and the frozen mixing ratio is the equilibrium value interpolated at that pressure. Above the
// quench point the mixing ratios of the family are held at the frozen value. The continuous
// placement matters because this module is re-evaluated at every Newton trial temperature: a
// quench point that jumped between levels would make the composition (and the flux residual)
// discontinuous in T.
//
// Families and timescales (p in bar, T in K, m = metallicity relative to solar):
//   CO/CH4/H2O   t_CO  = (1/t_q1 + 1/t_q2)^-1                                     ZM14 Eq. 14
//                t_q1  = 1.5e-6  p^-1 m^-0.7 exp(42000/T) s                        ZM14 Eq. 12
//                t_q2  = 40      p^-2        exp(25000/T) s                        ZM14 Eq. 13
//   NH3/N2       t_NH3 = 1.0e-7  p^-1        exp(52000/T) s                        ZM14 Eq. 32
//   HCN          t_HCN = 1.5e-4  p^-1 m^-0.7 exp(36000/T) s                        ZM14 Eq. 40
//   CO2          t_CO2 = 1.0e-10 p^-0.5      exp(38000/T) s                        ZM14 Eq. 44
// CO2 is special (ZM14 Sec. 6): below its own quench point it stays in quasi-equilibrium with the
// (quenched) CO, H2O and H2 through pCO*pH2O/(pCO2*pH2) = K(T) (ZM14 Eq. 43); above it, it is
// frozen. K(T) is not taken from the ZM14 fit but implied by the equilibrium composition itself,
// so the quasi-equilibrium CO2 is f_CO2,eq * [f_CO f_H2O/f_H2] / [f_CO f_H2O/f_H2]_eq. This
// reduces exactly to the equilibrium CO2 wherever CO and H2O are unquenched.
// H2O is frozen together with CO and CH4 because oxygen conservation ties it to the CO/CH4
// quench (f_O = f_H2O + f_CO in ZM14 Eq. 8).
//
// Mass balance: whatever the quenched species gain or lose is taken from / given to H2, as in the
// picaso and Exo-REM implementations. The mean molecular weight is recomputed afterwards.
//
// Kzz comes from the model's eddy diffusion profile (the `kzz` model option, see
// atmosphere/eddy_diffusion.h): self-consistent from the mixing-length convection by default, or a
// prescribed constant / power law. It is refreshed once per outer iteration (lagged inside the
// Newton trial evaluations).
//
// Validity: the ZM14 fits are for H2-dominated atmospheres of roughly solar to a few times solar
// metallicity, without photochemistry. The module refuses to run if there is no H2.
class QuenchChemistry : public Chemistry{
  public:
    QuenchChemistry(const double metallicity_, const bool relax_quench_points_ = true);
    virtual ~QuenchChemistry() {}

    virtual bool calcChemicalComposition(
      const std::vector<double>& parameters,
      const std::vector<double>& temperature,
      const std::vector<double>& pressure,
      std::vector<std::vector<double>>& number_densities,
      std::vector<double>& mean_molecular_weight);

    virtual void setSurfaceGravity(const double gravity) { surface_gravity = gravity; }

    // The model's Kzz profile (cm^2/s, one value per level; see EddyDiffusion). Required.
    virtual void setKzz(const std::vector<double>& kzz_profile) { kzz = kzz_profile; }

    // Inside the corrector's trial evaluations the quench points and frozen mixing ratios of the
    // last committed evaluation are reapplied instead of being re-derived from the trial T, so the
    // quench point cannot move between the residual and the Jacobian that linearises it. NOTE:
    // the ratio_ul corrector does not recompute chemistry in its trial evaluations at all, so this
    // only matters for correctors that do (the true-residual evaluations of the affine-covariant
    // schemes); it did NOT cure the warm-Jupiter stall it was written for (2026-09-08), whose
    // cause is the first Newton step from a cold Guillot start with the quench composition.
    virtual void setLagged(const bool lagged_) { lagged = lagged_; }

    // Quench pressures (bar) of the last evaluation, one per family in the order
    // CO/CH4/H2O, NH3/N2, HCN, CO2; NaN = the family did not quench inside the grid.
    const std::vector<double>& quenchPressures() const { return quench_pressures; }

  private:
    enum family {co_family, nh3_family, hcn_family, co2_family, nb_families};
    static const std::vector<std::string> family_names;

    double metallicity;     // relative to solar
    bool relax_quench_points;   // Aitken relaxation of the quench pressures between evaluations

    double surface_gravity = 0;                 // cm/s^2, set by the owning object
    std::vector<double> kzz;                    // cm^2/s per level, set by the owning object

    std::vector<double> quench_pressures;
    std::vector<double> printed_quench_pressures;   // for rate-limited diagnostics
    bool warned_below_grid = false;

    // committed quench state, reapplied while lagged
    struct FrozenSpecies { chemical_species_id id; double value; };
    struct FamilyState { bool quenched = false; size_t first_level = 0; std::vector<FrozenSpecies> frozen; };
    std::vector<FamilyState> committed;   // one per family; empty until the first full evaluation
    bool lagged = false;

    // Relaxation of the quench pressures between outer evaluations (Aitken on ln p_q per family,
    // with a dead band). The outer loop "solve T for a frozen composition, then move the quench
    // points" is a Picard iteration whose gain exceeds one where the quench point sits at the
    // CO/CH4 transition of the photosphere: measured on the warm Jupiter, the CO quench pressure
    // flipped 2 <-> 11 bar and one level swung +-184 K every outer iteration, indefinitely. Same
    // remedy as for the self-consistent Kzz (atmosphere/eddy_diffusion.h).
    struct Relaxation { double log_p = 0.0; double residual_prev = 0.0; double weight = 0.5; bool valid = false; };
    std::vector<Relaxation> relaxation;   // one per family
    static constexpr double relax_dead_band = 0.02;   // dex in the quench pressure

    // Relax a freshly derived quench pressure against the committed one; returns the pressure to
    // use (and records it). A family that stops quenching resets its state.
    double relaxQuenchPressure(const family f, const double p_new);

    std::vector<double> mixingTimescale(
      const std::vector<double>& temperature,
      const std::vector<double>& pressure,
      const std::vector<double>& mean_molecular_weight) const;

    double logChemicalTimescale(
      const family f, const double temperature, const double pressure) const;

    // Locate the quench point of a family: returns true and the quench pressure (bar) plus the
    // index of the first level ABOVE the quench point if the family quenches inside the grid.
    bool findQuenchPoint(
      const family f,
      const std::vector<double>& temperature,
      const std::vector<double>& pressure,
      const std::vector<double>& log_t_mix,
      double& quench_pressure,
      size_t& first_quenched_level);

    // Freeze a mixing-ratio profile above the quench pressure at its value interpolated there;
    // returns the frozen value.
    static double freezeProfile(
      const std::vector<double>& pressure,
      const double quench_pressure,
      const size_t first_quenched_level,
      std::vector<double>& mixing_ratio);

    void printDiagnostics();
};


}
#endif
