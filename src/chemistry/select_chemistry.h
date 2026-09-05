/*
* This file is part of the ngam code.
* Copyright (C) 2026 Daniel Kitzmann
*
* ngam is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

#ifndef _select_chemistry_h
#define _select_chemistry_h

#include <memory>
#include <string>
#include <vector>

#include "chemistry.h"
#include "fastchem_chemistry.h"
#include "isoprofile_chemistry.h"
#include "fixed_chemistry.h"
#include "mw_humidity_chemistry.h"
#include "../config/module_params.h"


namespace ngam {


// Chemistry modules, selected by a module spec {type, named parameters}:
//
//   equilibrium (eq)        FastChem equilibrium chemistry
//                           parameter_file   FastChem parameter file (required)
//                           metallicity      relative to solar, default 1
//                           c_to_o           C/O ratio, default 0.5
//   isoprofile (iso)        constant mixing ratios; every key is a species symbol, its value the
//                           volume mixing ratio, e.g. {H2: 0.85, He: 0.15, H2O: 1e-4}
//   fixed (fix)             tabulated composition profile
//                           file             path to the composition file (required)
//   manabe_wetherald (mw)   Manabe & Wetherald relative-humidity H2O profile (overrides H2O)
//                           surface_rh       surface relative humidity, default 0.77
namespace chemistry_modules{
  enum id {iso, eq, fixed, mw_humidity};
  const std::vector<std::string> description {"isoprofile", "equilibrium", "fixed", "manabe_wetherald"};
  const std::vector<std::string> description_short {"iso", "eq", "fix", "mw"};
}


inline std::unique_ptr<Chemistry> selectChemistryModule(const ModuleSpec& spec)
{
  const auto module_id = static_cast<chemistry_modules::id>(resolveModuleType(
    spec.type, chemistry_modules::description, chemistry_modules::description_short, "chemistry"));

  ParamReader reader(spec, "chemistry");
  std::unique_ptr<Chemistry> module;

  switch (module_id)
  {
    case chemistry_modules::eq :
    {
      const std::string parameter_file = reader.requireString("parameter_file");
      const double metallicity = reader.getDouble("metallicity", 1.0);
      const double c_to_o = reader.getDouble("c_to_o", 0.5);
      module = std::make_unique<FastChemChemistry>(parameter_file);
      module->parameters = {metallicity, c_to_o};
      break;
    }

    case chemistry_modules::iso :
    {
      const std::vector<std::string> species = reader.allKeys();
      if (species.empty())
        throw InvalidInput("chemistry",
          "isoprofile chemistry needs at least one species: {symbol: mixing ratio, ...}\n");
      std::vector<double> mixing_ratios;
      for (const auto& s : species) mixing_ratios.push_back(reader.requireDouble(s));
      module = std::make_unique<IsoprofileChemistry>(species);
      module->parameters = mixing_ratios;
      break;
    }

    case chemistry_modules::fixed :
      module = std::make_unique<FixedChemistry>(reader.requireString("file"));
      break;

    case chemistry_modules::mw_humidity :
      module = std::make_unique<ManabeWetherlandChemistry>(reader.getDouble("surface_rh", 0.77));
      break;
  }

  reader.finish();
  return module;
}


inline std::vector<std::unique_ptr<Chemistry>> selectChemistryModules(
  const std::vector<ModuleSpec>& specs)
{
  std::vector<std::unique_ptr<Chemistry>> modules;
  for (const auto& spec : specs)
    modules.push_back(selectChemistryModule(spec));
  return modules;
}


}
#endif
