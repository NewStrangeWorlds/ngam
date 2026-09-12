/*
* This file is part of the ngam code.
* Copyright (C) 2026 Daniel Kitzmann
*
* ngam is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

#include "external_chemistry.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "chem_species.h"
#include "../additional/exceptions.h"


namespace ngam {


ExternalChemistry::ExternalChemistry()
{
  std::cout << "- Chemistry model: external composition (set from the driver; no-op until then)\n\n";
  nb_parameters = 0;
}



void ExternalChemistry::setMixingRatios(
  const std::vector<std::string>& symbols,
  const std::vector<std::vector<double>>& mixing_ratios)
{
  const size_t nb_levels = mixing_ratios.size();

  if (nb_levels == 0)
    throw InvalidInput("ExternalChemistry", "empty mixing ratio table\n");

  for (const auto& row : mixing_ratios)
    if (row.size() != symbols.size())
      throw InvalidInput("ExternalChemistry",
        "every mixing ratio row must have one entry per species symbol\n");

  profiles.clear();
  ignored_symbols.clear();

  for (size_t k = 0; k < symbols.size(); ++k)
  {
    const auto it = std::find_if(
      constants::species_data.begin(), constants::species_data.end(),
      [&](const chemistry_data& s) { return s.symbol == symbols[k]; });

    if (it == constants::species_data.end() || it->id == _TOTAL)
    {
      ignored_symbols.push_back(symbols[k]);
      continue;
    }

    Profile p;
    p.id = it->id;
    p.mixing_ratio.resize(nb_levels);
    for (size_t i = 0; i < nb_levels; ++i)
      p.mixing_ratio[i] = std::max(0.0, mixing_ratios[i][k]);
    profiles.push_back(p);
  }

  if (!ignored_symbols.empty() && !reported_ignored)
  {
    std::cout << "  ExternalChemistry: ignoring species unknown to ngam:";
    for (const auto& s : ignored_symbols) std::cout << " " << s;
    std::cout << "\n";
    reported_ignored = true;
  }

  if (profiles.empty())
    throw InvalidInput("ExternalChemistry", "none of the supplied species is known to ngam\n");
}



bool ExternalChemistry::calcChemicalComposition(
  const std::vector<double>& parameters,
  const std::vector<double>& temperature,
  const std::vector<double>& pressure,
  std::vector<std::vector<double>>& number_densities,
  std::vector<double>& mean_molecular_weight)
{
  (void) parameters; (void) temperature;

  if (profiles.empty()) return false;   // nothing supplied yet: pass the composition through

  const size_t nb_levels = pressure.size();

  if (profiles.front().mixing_ratio.size() != nb_levels)
    throw InvalidInput("ExternalChemistry",
      "the external composition has a different number of levels than the model\n");

  for (size_t i = 0; i < nb_levels; ++i)
  {
    const double n_tot = number_densities[i][_TOTAL];

    // budget of the supplied species as allotted by the preceding modules
    double sum_old = 0.0, sum_new = 0.0;
    for (const auto& p : profiles)
    {
      sum_old += number_densities[i][p.id] / n_tot;
      sum_new += p.mixing_ratio[i];
    }
    const double scale = (sum_new > 0.0) ? sum_old / sum_new : 0.0;

    double mu_correction = 0.0;
    for (const auto& p : profiles)
    {
      const double f_old = number_densities[i][p.id] / n_tot;
      const double f_new = p.mixing_ratio[i] * scale;
      mu_correction += (f_new - f_old) * constants::species_data[p.id].molecular_weight;
      number_densities[i][p.id] = f_new * n_tot;
    }
    mean_molecular_weight[i] += mu_correction;
  }

  return false;
}


}
