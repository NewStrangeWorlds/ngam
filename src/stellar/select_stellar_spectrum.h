/*
* This file is part of the ngam code.
* Copyright (C) 2026 Daniel Kitzmann
*
* ngam is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

#ifndef _select_stellar_spectrum_h
#define _select_stellar_spectrum_h

#include <memory>
#include <string>
#include <vector>

#include "stellar_spectrum.h"
#include "../config/module_params.h"


namespace ngam {


// Host-star spectra, selected by a module spec {type, named parameters}. The spectrum's shape
// comes from the module; its normalisation (the instellation at the planet) and the incidence
// geometry are properties of the planet and are passed separately.
//
//   blackbody    temperature   stellar effective temperature in K (required)
//   tabulated    file          path to the spectrum file (required)
namespace stellar_modules{
  enum id {blackbody, tabulated};
  const std::vector<std::string> description {"blackbody", "tabulated"};
}


inline std::unique_ptr<StellarSpectrum> selectStellarSpectrum(
  const ModuleSpec& spec,
  const double instellation)
{
  const auto module_id = static_cast<stellar_modules::id>(resolveModuleType(
    spec.type, stellar_modules::description, {}, "stellar_spectrum"));

  ParamReader reader(spec, "stellar_spectrum");
  std::unique_ptr<StellarSpectrum> module;

  switch (module_id)
  {
    case stellar_modules::blackbody :
      module = std::make_unique<BlackbodyStar>(reader.requireDouble("temperature"), instellation);
      break;

    case stellar_modules::tabulated :
      module = std::make_unique<TabulatedStar>(reader.requireString("file"), instellation);
      break;
  }

  reader.finish();
  return module;
}


}
#endif
