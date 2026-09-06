/*
* This file is part of the ngam code.
* Copyright (C) 2026 Daniel Kitzmann
*
* ngam is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

#include "quench_chemistry.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "chem_species.h"
#include "../additional/exceptions.h"
#include "../additional/physical_const.h"


namespace ngam {


const std::vector<std::string> QuenchChemistry::family_names {
  "CO/CH4/H2O", "NH3/N2", "HCN", "CO2"};


QuenchChemistry::QuenchChemistry(const double metallicity_)
  : metallicity(metallicity_)
{
  if (metallicity <= 0)
    throw InvalidInput("QuenchChemistry", "metallicity must be positive\n");

  std::cout << "- Chemistry model: " << "quench (Zahnle & Marley 2014)" << "\n"
            << "  - Kzz:              from the model's eddy diffusion profile (`kzz` option)\n"
            << "  - metallicity:      " << metallicity << " x solar\n"
            << "  - quenched species: CO, CH4, H2O; NH3, N2; HCN; CO2\n\n";

  nb_parameters = 0;
  quench_pressures.assign(nb_families, std::numeric_limits<double>::quiet_NaN());
  printed_quench_pressures.assign(nb_families, -1.0);
}



std::vector<double> QuenchChemistry::mixingTimescale(
  const std::vector<double>& temperature,
  const std::vector<double>& pressure,
  const std::vector<double>& mean_molecular_weight) const
{
  const size_t nb_levels = pressure.size();
  std::vector<double> log_t_mix(nb_levels);

  for (size_t i = 0; i < nb_levels; ++i)
  {
    // pressure scale height with the surface gravity (the timescale fits are order-of-magnitude
    // estimates; the altitude dependence of g is irrelevant at this level of accuracy)
    const double scale_height = constants::boltzmann_k * temperature[i]
      / (mean_molecular_weight[i] * constants::amu * surface_gravity);

    log_t_mix[i] = 2.0 * std::log10(scale_height) - std::log10(kzz[i]);
  }

  return log_t_mix;
}



// log10 of the ZM14 chemical timescales (p in bar, T in K). Evaluated in log space because the
// Arrhenius factors overflow a double for cold upper layers (exp(52000/T) at T < 75 K).
double QuenchChemistry::logChemicalTimescale(
  const family f, const double temperature, const double pressure) const
{
  const double log_p = std::log10(pressure);
  const double log_e = std::log10(std::exp(1.0));   // ln -> log10
  const double log_m = std::log10(metallicity);

  switch (f)
  {
    case co_family:
    {
      const double log_tq1 = std::log10(1.5e-6) - log_p - 0.7 * log_m + 42000.0 / temperature * log_e;
      const double log_tq2 = std::log10(40.0) - 2.0 * log_p + 25000.0 / temperature * log_e;
      // harmonic combination (Eq. 14): 1/t = 1/tq1 + 1/tq2, done in log space
      const double log_min = std::min(log_tq1, log_tq2);
      return log_min - std::log10(1.0 + std::pow(10.0, log_min - std::max(log_tq1, log_tq2)));
    }
    case nh3_family:
      return std::log10(1.0e-7) - log_p + 52000.0 / temperature * log_e;
    case hcn_family:
      return std::log10(1.5e-4) - log_p - 0.7 * log_m + 36000.0 / temperature * log_e;
    case co2_family:
      return std::log10(1.0e-10) - 0.5 * log_p + 38000.0 / temperature * log_e;
    default:
      return 0.0;
  }
}



bool QuenchChemistry::findQuenchPoint(
  const family f,
  const std::vector<double>& temperature,
  const std::vector<double>& pressure,
  const std::vector<double>& log_t_mix,
  double& quench_pressure,
  size_t& first_quenched_level)
{
  const size_t nb_levels = pressure.size();

  // D(i) = log t_chem - log t_mix: negative where chemistry is faster than mixing (equilibrium),
  // positive where mixing wins (quenched). Search upward from the bottom (index 0).
  double d_below = logChemicalTimescale(f, temperature[0], pressure[0]) - log_t_mix[0];

  if (d_below > 0)
  {
    // already quenched at the bottom of the grid: the true quench point lies deeper. Freeze at the
    // bottom level and say so once -- the frozen abundances are then only as good as the bottom
    // boundary of the grid.
    if (!warned_below_grid)
    {
      std::cout << "  QuenchChemistry: " << family_names[f]
                << " is already quenched at the bottom of the grid (" << pressure[0]
                << " bar); freezing at the bottom-level composition. Extend the grid deeper "
                << "for a proper quench point.\n";
      warned_below_grid = true;
    }
    quench_pressure = pressure[0];
    first_quenched_level = 0;
    return true;
  }

  for (size_t i = 1; i < nb_levels; ++i)
  {
    const double d_here = logChemicalTimescale(f, temperature[i], pressure[i]) - log_t_mix[i];

    if (d_here > 0)
    {
      // root of D in log p between levels i-1 (D <= 0) and i (D > 0)
      const double x_below = std::log10(pressure[i-1]);
      const double x_here  = std::log10(pressure[i]);
      const double x_q = x_below + (x_here - x_below) * d_below / (d_below - d_here);

      quench_pressure = std::pow(10.0, x_q);
      first_quenched_level = i;
      return true;
    }

    d_below = d_here;
  }

  return false;   // chemistry faster than mixing everywhere: stays in equilibrium
}



void QuenchChemistry::freezeProfile(
  const std::vector<double>& pressure,
  const double quench_pressure,
  const size_t first_quenched_level,
  std::vector<double>& mixing_ratio)
{
  double frozen_value = mixing_ratio[first_quenched_level];

  if (first_quenched_level > 0)
  {
    // interpolate the equilibrium profile at the quench pressure: log-linear in log p when both
    // bracketing values are positive, linear otherwise
    const size_t i = first_quenched_level;
    const double x_below = std::log10(pressure[i-1]);
    const double x_here  = std::log10(pressure[i]);
    const double w = (std::log10(quench_pressure) - x_below) / (x_here - x_below);

    const double f_below = mixing_ratio[i-1];
    const double f_here  = mixing_ratio[i];

    if (f_below > 0 && f_here > 0)
      frozen_value = std::exp((1.0 - w) * std::log(f_below) + w * std::log(f_here));
    else
      frozen_value = (1.0 - w) * f_below + w * f_here;
  }

  for (size_t i = first_quenched_level; i < mixing_ratio.size(); ++i)
    mixing_ratio[i] = frozen_value;
}



bool QuenchChemistry::calcChemicalComposition(
  const std::vector<double>& parameters,
  const std::vector<double>& temperature,
  const std::vector<double>& pressure,
  std::vector<std::vector<double>>& number_densities,
  std::vector<double>& mean_molecular_weight)
{
  (void) parameters;

  if (surface_gravity <= 0)
    throw InvalidInput("QuenchChemistry",
      "surface gravity has not been set (the owning object must call setSurfaceGravity)\n");

  const size_t nb_levels = pressure.size();

  if (kzz.size() != nb_levels || *std::min_element(kzz.begin(), kzz.end()) <= 0)
    throw InvalidInput("QuenchChemistry",
      "no valid Kzz profile (the owning object must call setKzz with positive values)\n");

  // the quenched species, grouped by family; every family shares one quench point
  const std::vector<std::vector<chemical_species_id>> family_species {
    {_CO, _CH4, _H2O}, {_NH3, _N2}, {_HCN}, {_CO2}};

  // mixing ratios of everything we touch (H2 is the mass-balance reservoir)
  auto mixingRatio = [&](const chemical_species_id id) {
    std::vector<double> f(nb_levels);
    for (size_t i = 0; i < nb_levels; ++i)
      f[i] = number_densities[i][id] / number_densities[i][_TOTAL];
    return f;
  };

  std::vector<double> f_h2 = mixingRatio(_H2);

  if (*std::max_element(f_h2.begin(), f_h2.end()) <= 0)
    throw InvalidInput("QuenchChemistry",
      "no H2 in the composition. The quench module post-processes an H2-dominated EQUILIBRIUM "
      "composition and must be listed after the module that provides it, e.g. "
      "chemistry=[(\"equilibrium\", {...}), (\"quench\", {...})]\n");

  const std::vector<double> log_t_mix = mixingTimescale(temperature, pressure, mean_molecular_weight);

  // running mass-balance correction per level: sum over species of (old - new) goes into H2;
  // and the matching change of the mean molecular weight
  std::vector<double> h2_correction(nb_levels, 0.0);
  std::vector<double> mu_correction(nb_levels, 0.0);

  auto commit = [&](const chemical_species_id id, const std::vector<double>& f_new) {
    const double mass = constants::species_data[id].molecular_weight;
    for (size_t i = 0; i < nb_levels; ++i)
    {
      const double f_old = number_densities[i][id] / number_densities[i][_TOTAL];
      h2_correction[i] += f_old - f_new[i];
      mu_correction[i] += (f_new[i] - f_old) * mass;
      number_densities[i][id] = f_new[i] * number_densities[i][_TOTAL];
    }
  };

  // --- CO/CH4/H2O, NH3/N2, HCN: freeze the equilibrium mixing ratios above the quench point ---
  // (CO2 comes last: its quasi-equilibrium needs the QUENCHED CO and H2O)
  const std::vector<double> f_co_eq  = mixingRatio(_CO);
  const std::vector<double> f_h2o_eq = mixingRatio(_H2O);
  const std::vector<double> f_h2_eq  = f_h2;

  for (const family f : {co_family, nh3_family, hcn_family})
  {
    double p_q;
    size_t level_q;

    if (!findQuenchPoint(f, temperature, pressure, log_t_mix, p_q, level_q))
    {
      quench_pressures[f] = std::numeric_limits<double>::quiet_NaN();
      continue;
    }

    quench_pressures[f] = p_q;

    for (const auto id : family_species[f])
    {
      std::vector<double> f_species = mixingRatio(id);
      freezeProfile(pressure, p_q, level_q, f_species);
      commit(id, f_species);
    }
  }

  // --- CO2 (ZM14 Sec. 6): quasi-equilibrium with the quenched CO, H2O, H2 below its own quench
  // point (Eq. 43 with K implied by the equilibrium composition), frozen above it ---
  {
    const std::vector<double> f_co  = mixingRatio(_CO);
    const std::vector<double> f_h2o = mixingRatio(_H2O);
    std::vector<double> f_co2 = mixingRatio(_CO2);

    for (size_t i = 0; i < nb_levels; ++i)
    {
      const double product_eq = f_co_eq[i] * f_h2o_eq[i] / f_h2_eq[i];
      const double product    = f_co[i] * f_h2o[i] / f_h2_eq[i];

      if (product_eq > 0 && std::isfinite(product / product_eq))
        f_co2[i] *= product / product_eq;
    }

    double p_q;
    size_t level_q;

    if (findQuenchPoint(co2_family, temperature, pressure, log_t_mix, p_q, level_q))
    {
      quench_pressures[co2_family] = p_q;
      freezeProfile(pressure, p_q, level_q, f_co2);
    }
    else
      quench_pressures[co2_family] = std::numeric_limits<double>::quiet_NaN();

    commit(_CO2, f_co2);
  }

  // --- mass balance into H2, then the mean molecular weight ---
  // The weight is updated DIFFERENTIALLY, mu += sum_j (f_new - f_old) m_j over the species touched
  // here, rather than recomputed from ngam's species list: that list lacks the heavy trace species
  // (Mg, Si, Fe, ...) the equilibrium solver includes in its own mu, so a recompute would shift mu
  // by ~0.3% even where nothing is quenched.
  for (size_t i = 0; i < nb_levels; ++i)
  {
    const double f_h2_new = std::max(0.0, f_h2[i] + h2_correction[i]);
    number_densities[i][_H2] = f_h2_new * number_densities[i][_TOTAL];
    mu_correction[i] += (f_h2_new - f_h2[i]) * constants::species_data[_H2].molecular_weight;
    mean_molecular_weight[i] += mu_correction[i];
  }

  printDiagnostics();

  return false;
}



// Print the quench pressures, but only when one of them moved by more than 10% since the last
// print -- this module is evaluated at every Newton trial point, so unconditional output would
// swamp the log.
void QuenchChemistry::printDiagnostics()
{
  bool changed = false;

  for (size_t f = 0; f < nb_families; ++f)
  {
    const double p_now = quench_pressures[f];
    const double p_old = printed_quench_pressures[f];

    const bool now_quenched = std::isfinite(p_now);
    const bool old_quenched = p_old > 0;

    if (now_quenched != old_quenched
        || (now_quenched && std::abs(p_now - p_old) > 0.1 * p_old))
      changed = true;
  }

  if (!changed) return;

  std::cout << "  QuenchChemistry: quench pressures [bar]";

  for (size_t f = 0; f < nb_families; ++f)
  {
    std::cout << "  " << family_names[f] << ": ";

    if (std::isfinite(quench_pressures[f]))
    {
      std::cout << std::scientific << std::setprecision(2) << quench_pressures[f];
      printed_quench_pressures[f] = quench_pressures[f];
    }
    else
    {
      std::cout << "equilibrium";
      printed_quench_pressures[f] = -1.0;
    }
  }

  std::cout << std::defaultfloat << "\n";
}


}
