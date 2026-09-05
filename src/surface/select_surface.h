/*
* This file is part of the ngam code.
* Copyright (C) 2026 Daniel Kitzmann
*
* ngam is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

#ifndef _select_surface_h
#define _select_surface_h

#include <memory>
#include <string>
#include <vector>

#include "generic_surface.h"
#include "simple_surface.h"
#include "var_albedo_surface.h"
#include "../spectral_grid/spectral_grid.h"
#include "../config/module_params.h"


namespace ngam {


// Surface models, selected by a module spec {type, named parameters}:
//
//   blackbody         non-reflecting blackbody surface (no parameters)
//   simple            grey albedo below a wavelength switch, black above it
//                     albedo             shortwave albedo (required)
//                     wavelength_switch  switch wavelength in micron (required)
//   variable_albedo   tabulated spectral albedo
//                     file               path to the albedo file (required)
namespace surface_modules{
  enum id {blackbody, simple, variable_albedo};
  const std::vector<std::string> description {"blackbody", "simple", "variable_albedo"};
}


inline std::unique_ptr<GenericSurface> selectSurface(
  const ModuleSpec& spec,
  SpectralGrid* spectral_grid)
{
  const auto module_id = static_cast<surface_modules::id>(resolveModuleType(
    spec.type, surface_modules::description, {}, "surface"));

  ParamReader reader(spec, "surface");
  std::unique_ptr<GenericSurface> module;

  switch (module_id)
  {
    case surface_modules::blackbody :
      module = std::make_unique<GenericSurface>(spectral_grid);
      break;

    case surface_modules::simple :
    {
      const double albedo = reader.requireDouble("albedo");
      const double wavelength_switch = reader.requireDouble("wavelength_switch");
      module = std::make_unique<SimpleSurface>(spectral_grid, albedo, wavelength_switch);
      break;
    }

    case surface_modules::variable_albedo :
      module = std::make_unique<VariableAlbedoSurface>(spectral_grid, reader.requireString("file"));
      break;
  }

  reader.finish();
  return module;
}


}
#endif
