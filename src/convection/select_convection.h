/*
* This file is part of the ngam code.
* Copyright (C) 2026 Daniel Kitzmann
*
* ngam is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

#ifndef _select_convection_h
#define _select_convection_h

#include <memory>
#include <string>
#include <vector>

#include "convection.h"
#include "dry_adiabatic.h"
#include "moist_adiabatic.h"
#include "mixing_length.h"
#include "../config/module_params.h"


namespace ngam {


// Convection schemes, selected by a module spec {type, named parameters}:
//
//   mlt          smooth mixing-length convection (doc/mlt_convection_design.md). The DEFAULT;
//                requires the ratio_ul temperature correction.
//                alpha          mixing length in scale heights, default 1 (the converged T(P) is
//                               insensitive to it)
//                min_pressure   no convection above this pressure [bar]; default set by the object
//   mlt_moist    as mlt, with the moist (condensation-limited) neutrality gradient
//   dry          hard dry-adiabatic adjustment (relaxation / flux_divergence / ptc schemes)
//                min_pressure, max_sweeps (default 10)
//   moist        hard moist-adiabatic adjustment; same parameters as dry
//   none         no convection
namespace convection_modules{
  enum id {mlt, mlt_moist, dry, moist, none};
  const std::vector<std::string> description {"mlt", "mlt_moist", "dry", "moist", "none"};
}


// blackadar_surface: apply the Blackadar wall law at the bottom of the domain -- only meaningful
// when the bottom IS a surface (terrestrial planets); a self-luminous domain bottom is not one.
inline std::unique_ptr<Convection> selectConvection(
  const ModuleSpec& spec,
  const double default_min_pressure,
  const bool blackadar_surface)
{
  const auto module_id = static_cast<convection_modules::id>(resolveModuleType(
    spec.type, convection_modules::description, {}, "convection"));

  ParamReader reader(spec, "convection");
  std::unique_ptr<Convection> module;

  switch (module_id)
  {
    case convection_modules::mlt :
    case convection_modules::mlt_moist :
    {
      const double alpha = reader.getDouble("alpha", 1.0);
      const double min_pressure = reader.getDouble("min_pressure", default_min_pressure);
      module = std::make_unique<MixingLengthConvection>(
        module_id == convection_modules::mlt_moist, alpha, min_pressure, blackadar_surface);
      break;
    }

    case convection_modules::dry :
    case convection_modules::moist :
    {
      const size_t max_sweeps = static_cast<size_t>(reader.getInt("max_sweeps", 10));
      const double min_pressure = reader.getDouble("min_pressure", default_min_pressure);
      if (module_id == convection_modules::dry)
        module = std::make_unique<DryAdiabaticAdjustment>(max_sweeps, min_pressure);
      else
        module = std::make_unique<MoistAdiabaticAdjustment>(max_sweeps, min_pressure);
      break;
    }

    case convection_modules::none :
      module = nullptr;
      break;
  }

  reader.finish();
  return module;
}


}
#endif
