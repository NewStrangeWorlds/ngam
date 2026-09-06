/*
* This file is part of the ngam code.
* Copyright (C) 2026 Daniel Kitzmann
*
* ngam is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

#include "eddy_diffusion.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "../additional/exceptions.h"
#include "../additional/physical_const.h"
#include "../additional/thermodynamic_data.h"


namespace ngam {


std::vector<double> PowerLawEddyDiffusion::profile(
  const Atmosphere& atmosphere, const Convection*, const double)
{
  std::vector<double> kzz(atmosphere.pressure.size());

  for (size_t i = 0; i < kzz.size(); ++i)
    kzz[i] = value * std::pow(atmosphere.pressure[i] / reference_pressure, slope);

  return kzz;
}



std::vector<double> MltEddyDiffusion::profile(
  const Atmosphere& atmosphere,
  const Convection* convection,
  const double surface_gravity)
{
  bool convective = false;
  std::vector<double> kzz = rawProfile(atmosphere, convection, surface_gravity, convective);

  const bool have_previous = previous.size() == kzz.size();

  // hold: a momentarily stable profile mid-iteration must not collapse Kzz to the floor
  if (!convective && have_previous)
    return previous;

  // Aitken dynamic relaxation (Kuettler & Wall 2008) on the log profile. The outer loop
  // Kzz -> quench composition -> opacity -> T -> Kzz is a Picard iteration with a loop gain
  // measured at ~2.5 on the brown dwarf: undamped it settles into a period-2 cycle (residual
  // 2.2e-2 <-> 2.5e-2, never converging), a fixed weight of 0.5 converges but linearly (ratio
  // 0.74/iteration, 49 instead of 18 iterations). Aitken estimates the optimal weight
  // 1/(1 + gain) from the last two residuals r = log(raw) - log(committed); the user's `relax` is
  // the starting weight, clamped to [weight_min, 1].
  if (have_previous)
  {
    std::vector<double> r(kzz.size());
    double max_change = 0.0;
    for (size_t i = 0; i < kzz.size(); ++i)
    {
      r[i] = std::log(kzz[i]) - std::log(previous[i]);
      max_change = std::max(max_change, std::abs(r[i]));
    }

    // Dead band: below `tolerance` the update is physically irrelevant (0.05 dex in Kzz moves a
    // quench temperature by ~0.3%, far inside the ZM14 fit uncertainty) but numerically harmful --
    // every committed change, however small, shifts the composition and kicks the Newton residual
    // back up by an order of magnitude, so the outer loop converged only as fast as Kzz itself
    // (measured: 51 iterations with Aitken alone vs 18 with a fixed Kzz). Freezing Kzz here lets
    // the Newton finish quadratically.
    if (max_change < tolerance * std::log(10.0))
      return previous;

    if (previous_residual.size() == r.size())
    {
      double num = 0.0, den = 0.0;
      for (size_t i = 0; i < r.size(); ++i)
      {
        const double dr = r[i] - previous_residual[i];
        num += previous_residual[i] * dr;
        den += dr * dr;
      }
      if (den > 0.0) weight = -weight * num / den;
      constexpr double weight_min = 0.05;
      weight = std::min(1.0, std::max(weight_min, weight));
    }
    previous_residual = r;

    for (size_t i = 0; i < kzz.size(); ++i)
      kzz[i] = std::exp(std::log(previous[i]) + weight * r[i]);
  }

  previous = kzz;
  return kzz;
}



std::vector<double> MltEddyDiffusion::rawProfile(
  const Atmosphere& atmosphere,
  const Convection* convection,
  const double surface_gravity,
  bool& convective) const
{
  const size_t n = atmosphere.pressure.size();
  std::vector<double> kzz(n, 0.0);
  convective = false;

  // structure not available yet (e.g. the chemistry pass of an analytic initialisation): nothing
  // to derive from, fall back to the floor
  const bool have_structure = atmosphere.scale_height.size() == n && n > 1
    && *std::min_element(atmosphere.scale_height.begin(), atmosphere.scale_height.end()) > 0
    && atmosphere.mass_density.size() == n && atmosphere.altitude.size() == n;

  if (!have_structure || convection == nullptr || surface_gravity <= 0)
    return std::vector<double>(n, min_value);

  // ---- convective layers: the MLT flux law of the corrector (clima_rce_correction.cpp, mlt_C),
  // with the convergence ramp factor at its final value of 1 ------------------------------------
  const bool mlt = convection->providesFlux();
  const double alpha = mlt ? std::max(1e-3, convection->fluxAlpha()) : 1.0;
  const bool wall_law = mlt && convection->fluxBlackadarSurface();
  const double p_min = mlt ? convection->fluxMinPressure() : 0.0;
  constexpr double k_vk = 0.4;

  std::vector<double> kzz_link(n - 1, 0.0);

  for (size_t li = 0; li + 1 < n; ++li)
  {
    if (atmosphere.pressure[li+1] < p_min) continue;

    const double d_ln_p = std::log(atmosphere.pressure[li] / atmosphere.pressure[li+1]);
    if (d_ln_p <= 0) continue;

    const double nabla = std::log(atmosphere.temperature[li] / atmosphere.temperature[li+1]) / d_ln_p;
    const double nabla_ad = 0.5 * (
      convection->convectiveGradient(
        atmosphere.number_densities[li], atmosphere.temperature[li], atmosphere.pressure[li]) +
      convection->convectiveGradient(
        atmosphere.number_densities[li+1], atmosphere.temperature[li+1], atmosphere.pressure[li+1]));

    const double x = nabla - nabla_ad;   // super-adiabaticity
    if (x <= 0) continue;

    const double h_p = 0.5 * (atmosphere.scale_height[li] + atmosphere.scale_height[li+1]);
    const double z = std::max(0.0,
      0.5 * (atmosphere.altitude[li] + atmosphere.altitude[li+1]) - atmosphere.altitude[0]);
    const double lambda = wall_law
      ? ((z > 0) ? k_vk * z / (1.0 + k_vk * z / (alpha * h_p)) : 0.0)
      : alpha * h_p;
    if (lambda <= 0) continue;

    // convective velocity of the flux law F_c = 1/2 rho c_p T (lambda^2/H_p) sqrt(g/H_p) x^{3/2}
    // = 1/2 rho c_p w_conv (T lambda x / H_p), i.e. w_conv = lambda sqrt(g x / H_p)
    const double w_conv = lambda * std::sqrt(surface_gravity * x / h_p);

    if (!flux_scaling)
      kzz_link[li] = lambda * w_conv;
    else
    {
      const double rho = 0.5 * (atmosphere.mass_density[li] + atmosphere.mass_density[li+1]);
      const double t_m = 0.5 * (atmosphere.temperature[li] + atmosphere.temperature[li+1]);
      const double mu  = 0.5 * (atmosphere.mean_molecular_weight[li] + atmosphere.mean_molecular_weight[li+1]);
      const double c_p = 0.5 * (
        ThermodynamicData::meanHeatCapacity(atmosphere.number_densities[li], atmosphere.temperature[li]) +
        ThermodynamicData::meanHeatCapacity(atmosphere.number_densities[li+1], atmosphere.temperature[li+1]));
      const double r_spec = constants::boltzmann_k / (mu * constants::amu);   // erg/(g K)

      const double f_conv = 0.5 * rho * c_p * w_conv * t_m * lambda * x / h_p;
      kzz_link[li] = h_p / 3.0 * std::pow(lambda / h_p, 4.0/3.0)
                   * std::cbrt(r_spec * f_conv / (rho * c_p));
    }
  }

  // link -> level: average of the adjacent links (one-sided at the domain ends)
  for (size_t i = 0; i < n; ++i)
  {
    const double below = (i > 0) ? kzz_link[i-1] : 0.0;
    const double above = (i + 1 < n) ? kzz_link[i] : 0.0;
    if (i == 0) kzz[i] = above;
    else if (i + 1 == n) kzz[i] = below;
    else kzz[i] = 0.5 * (below + above);
  }

  // ---- radiative layers -------------------------------------------------------------------------
  // Each radiative level takes its value from the convective zone underneath it: the zone's
  // MAXIMUM Kzz, extended from the zone's top pressure as a power law or a constant (see the
  // header for why not the top level's own value). Radiative layers under the deepest zone keep
  // that zone's bottom value.
  // The anchor is the zone's maximum Kzz within ONE PRESSURE DECADE below its top: inside a zone
  // Kzz keeps growing with depth (the scale height does), so the plain maximum would sit at the
  // domain bottom and make the radiative-zone value depend on how deep the grid goes.
  constexpr double anchor_decades = 1.0;
  long zone_top = -1;      // index of the top level of the zone below
  double zone_max = 0.0;   // its anchor Kzz
  for (size_t i = 0; i < n; ++i)
  {
    if (kzz[i] > 0)
    {
      convective = true;
      continue;
    }
    if (i > 0 && kzz[i-1] > 0)
    {
      // the zone below just ended at level i-1: find its anchor
      zone_top = static_cast<long>(i - 1);
      zone_max = 0.0;
      const double p_limit = atmosphere.pressure[i-1] * std::pow(10.0, anchor_decades);
      for (long j = zone_top; j >= 0 && kzz[j] > 0 && atmosphere.pressure[j] <= p_limit; --j)
        zone_max = std::max(zone_max, kzz[j]);
    }
    if (zone_top < 0) continue;   // below the deepest zone: handled after the loop

    const size_t r = static_cast<size_t>(zone_top);
    switch (radiative)
    {
      case Radiative::constant:  kzz[i] = zone_max; break;
      case Radiative::power_law:
        kzz[i] = zone_max * std::pow(atmosphere.pressure[i] / atmosphere.pressure[r], slope); break;
      case Radiative::fixed:     kzz[i] = fixed_value; break;
    }
  }

  long deepest = -1;
  for (size_t i = 0; i < n; ++i)
    if (kzz[i] > 0) { deepest = static_cast<long>(i); break; }

  if (deepest > 0)
    for (long i = 0; i < deepest; ++i) kzz[i] = kzz[deepest];

  for (auto& k : kzz) k = std::max(k, min_value);

  return kzz;
}



namespace eddy_diffusion_modules{
  enum id {mlt, constant, power_law};
  const std::vector<std::string> description {"mlt", "constant", "power_law"};
}


std::unique_ptr<EddyDiffusion> selectEddyDiffusion(const ModuleSpec& spec)
{
  const auto module_id = static_cast<eddy_diffusion_modules::id>(resolveModuleType(
    spec.type, eddy_diffusion_modules::description, {}, "kzz"));

  ParamReader reader(spec, "kzz");
  std::unique_ptr<EddyDiffusion> module;

  switch (module_id)
  {
    case eddy_diffusion_modules::mlt :
    {
      const std::string scaling = reader.getString("scaling", "velocity");
      const std::string radiative = reader.getString("radiative", "constant");
      if (scaling != "velocity" && scaling != "flux")
        throw InvalidInput("kzz", "scaling must be \"velocity\" or \"flux\", got \"" + scaling + "\"\n");

      using Radiative = MltEddyDiffusion::Radiative;
      Radiative radiative_mode = Radiative::constant;
      double slope = 0.0, fixed_value = 0.0;
      if (radiative == "constant")
        radiative_mode = Radiative::constant;
      else if (radiative == "power_law")
      {
        radiative_mode = Radiative::power_law;
        slope = reader.requireDouble("slope");
      }
      else if (radiative == "fixed")
      {
        radiative_mode = Radiative::fixed;
        fixed_value = reader.requireDouble("value");
        if (fixed_value <= 0) throw InvalidInput("kzz", "value must be positive\n");
      }
      else
        throw InvalidInput("kzz",
          "radiative must be \"constant\", \"power_law\" or \"fixed\", got \"" + radiative + "\"\n");

      const double min_value = reader.getDouble("min", 1e4);
      const double relax = reader.getDouble("relax", 0.5);
      const double tolerance = reader.getDouble("tolerance", 0.05);
      if (min_value <= 0)
        throw InvalidInput("kzz", "min must be positive\n");
      if (relax <= 0 || relax > 1)
        throw InvalidInput("kzz", "relax must be in (0, 1]\n");
      if (tolerance < 0)
        throw InvalidInput("kzz", "tolerance must be >= 0\n");

      std::cout << "- Kzz: mixing-length (" << scaling << " scaling), radiative zones: " << radiative;
      if (radiative_mode == Radiative::power_law) std::cout << " with slope " << slope;
      if (radiative_mode == Radiative::fixed) std::cout << " at " << fixed_value << " cm2/s";
      std::cout << ", floor " << min_value << " cm2/s, relaxation " << relax
                << ", update tolerance " << tolerance << " dex\n";
      module = std::make_unique<MltEddyDiffusion>(
        scaling == "flux", radiative_mode, slope, fixed_value, min_value, relax, tolerance);
      break;
    }

    case eddy_diffusion_modules::constant :
    {
      const double value = reader.requireDouble("value");
      if (value <= 0) throw InvalidInput("kzz", "value must be positive\n");
      std::cout << "- Kzz: constant " << value << " cm2/s\n";
      module = std::make_unique<ConstantEddyDiffusion>(value);
      break;
    }

    case eddy_diffusion_modules::power_law :
    {
      const double value = reader.requireDouble("value");
      const double pressure = reader.getDouble("pressure", 1.0);
      const double slope = reader.requireDouble("slope");
      if (value <= 0 || pressure <= 0) throw InvalidInput("kzz", "value and pressure must be positive\n");
      std::cout << "- Kzz: power law " << value << " cm2/s x (p/" << pressure << " bar)^" << slope << "\n";
      module = std::make_unique<PowerLawEddyDiffusion>(value, pressure, slope);
      break;
    }
  }

  reader.finish();
  return module;
}


}
