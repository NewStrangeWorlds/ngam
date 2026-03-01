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


#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <stdexcept>

#include "fixed_chemistry.h"
#include "chem_species.h"
#include "../additional/exceptions.h"
#include "../additional/physical_const.h"


namespace ngam {


void FixedChemistry::readFile(const std::string& file_path)
{
  std::ifstream file(file_path);

  if (!file)
    throw std::runtime_error(
      "FixedChemistry: could not open file: " + file_path);

  std::cout << "- Chemistry model: fixed composition from " << file_path << "\n";

  // Read header line (starts with #)
  std::string header_line;
  std::getline(file, header_line);

  if (header_line.empty() || header_line[0] != '#')
    throw std::runtime_error(
      "FixedChemistry: first line must be a header starting with #");

  // Remove leading '#' and parse column names
  header_line = header_line.substr(1);
  std::istringstream header_stream(header_line);

  std::vector<std::string> column_names;
  std::string token;

  while (header_stream >> token)
    column_names.push_back(token);

  if (column_names.size() < 2)
    throw std::runtime_error(
      "FixedChemistry: header must contain at least pressure and one species");

  // Map each column (after pressure) to a species ID
  // Columns with unknown species get _TOTAL as a placeholder (will be skipped)
  const size_t nb_columns = column_names.size();

  std::vector<chemical_species_id> column_species(nb_columns, _TOTAL);

  std::cout << "  - Species in file: ";

  for (size_t col = 1; col < nb_columns; ++col)
  {
    bool found = false;

    for (auto& s : constants::species_data)
    {
      if (s.symbol == column_names[col])
      {
        column_species[col] = s.id;
        species.push_back(s.id);
        found = true;
        break;
      }
    }

    if (found)
      std::cout << column_names[col] << "  ";
    else
      std::cout << column_names[col] << "(skipped)  ";
  }

  std::cout << "\n";

  // Read data rows
  std::vector<double> file_pressure;
  std::vector<std::vector<double>> file_mixing_ratios;

  std::string line;

  while (std::getline(file, line))
  {
    if (line.empty() || line[0] == '#')
      continue;

    std::istringstream iss(line);
    double p;

    if (!(iss >> p))
      continue;

    file_pressure.push_back(p);

    std::vector<double> row(nb_columns, 0.0);

    for (size_t col = 1; col < nb_columns; ++col)
      iss >> row[col];

    file_mixing_ratios.push_back(row);
  }

  if (file_pressure.size() < 2)
    throw std::runtime_error(
      "FixedChemistry: need at least 2 data points in " + file_path);

  // Store as log10(pressure) for interpolation
  // Reorganise into per-species vectors indexed by species ID
  pressure.resize(file_pressure.size());

  for (size_t i = 0; i < file_pressure.size(); ++i)
    pressure[i] = std::log10(file_pressure[i]);

  // fixed_mixing_ratios: [nb_pressure_levels][nb_species_total]
  // Only fill columns that map to known species
  const size_t nb_species_total = constants::species_data.size();

  fixed_mixing_ratios.resize(
    file_pressure.size(), std::vector<double>(nb_species_total, 0.0));

  for (size_t i = 0; i < file_pressure.size(); ++i)
    for (size_t col = 1; col < nb_columns; ++col)
      if (column_species[col] != _TOTAL)
        fixed_mixing_ratios[i][column_species[col]] = file_mixing_ratios[i][col];

  std::cout << "  - Pressure range: "
            << file_pressure.front() << " - " << file_pressure.back()
            << " bar (" << file_pressure.size() << " levels)\n";
}


bool FixedChemistry::calcChemicalComposition(
  const std::vector<double>& parameters,
  const std::vector<double>& temperature,
  const std::vector<double>& model_pressure,
  std::vector<std::vector<double>>& number_densities,
  std::vector<double>& mean_molecular_weight)
{
  for (size_t i = 0; i < number_densities.size(); ++i)
  {
    const double log_p = std::log10(model_pressure[i]);

    const double n_total =
      model_pressure[i] * 1.e6 / constants::boltzmann_k / temperature[i];

    number_densities[i][_TOTAL] = n_total;

    // Find bracketing indices in the file pressure grid (stored as log10)
    // Clamp to boundaries if outside range
    size_t k_lo = 0;
    size_t k_hi = 0;
    double t = 0;

    if (log_p >= pressure.front())
    {
      k_lo = 0;
      k_hi = 0;
    }
    else if (log_p <= pressure.back())
    {
      k_lo = pressure.size() - 1;
      k_hi = k_lo;
    }
    else
    {
      // Find the interval: pressure is decreasing (high to low)
      k_lo = 0;

      while (k_lo < pressure.size() - 1 && pressure[k_lo + 1] > log_p)
        ++k_lo;

      k_hi = k_lo + 1;
      t = (log_p - pressure[k_lo]) / (pressure[k_hi] - pressure[k_lo]);
    }

    for (auto& s : species)
    {
      double vmr;

      if (k_lo == k_hi)
      {
        vmr = fixed_mixing_ratios[k_lo][s];
      }
      else
      {
        const double vmr_lo = fixed_mixing_ratios[k_lo][s];
        const double vmr_hi = fixed_mixing_ratios[k_hi][s];

        if (vmr_lo > 0 && vmr_hi > 0)
          vmr = std::pow(10.0,
            std::log10(vmr_lo) + t * (std::log10(vmr_hi) - std::log10(vmr_lo)));
        else
          vmr = vmr_lo + t * (vmr_hi - vmr_lo);

        if (vmr < 0) vmr = 0;
      }

      number_densities[i][s] = n_total * vmr;
    }
  }

  meanMolecularWeight(number_densities, mean_molecular_weight);

  bool mixing_ratios_ok = checkMixingRatios(number_densities);

  return !mixing_ratios_ok;
}


}
