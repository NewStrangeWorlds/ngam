#ifndef _select_surface_h
#define _select_surface_h

#include <vector>
#include <string>
#include <algorithm>
#include <memory>

#include "generic_surface.h"
#include "simple_surface.h"
#include "var_albedo_surface.h"
#include "../additional/exceptions.h"
#include "../spectral_grid/spectral_grid.h"


namespace ngam {


namespace surface_modules{
  enum id {blackbody, simple, variable_albedo};
  const std::vector<std::string> description {"blackbody", "simple", "variable_albedo"};
}


inline std::unique_ptr<GenericSurface> selectSurface(
  const std::string surface_type,
  const std::vector<std::string>& parameters,
  SpectralGrid* spectral_grid)
{
  auto it = std::find(
    surface_modules::description.begin(),
    surface_modules::description.end(),
    surface_type);

  if (it == surface_modules::description.end())
  {
    std::string error_message = "Surface type " + surface_type + " unknown!\n";
    throw InvalidInput(std::string ("surface.config"), error_message);
  }

  surface_modules::id module_id = static_cast<surface_modules::id>(
    std::distance(surface_modules::description.begin(), it));

  switch (module_id)
  {
    case surface_modules::blackbody :
      return std::make_unique<GenericSurface>(spectral_grid);

    case surface_modules::simple :
      if (parameters.size() != 2)
      {
        std::string error_message =
          "Simple surface requires exactly two parameters "
          "(shortwave albedo, wavelength switch in micron)!\n";
        throw InvalidInput(std::string ("surface.config"), error_message);
      }
      return std::make_unique<SimpleSurface>(
        spectral_grid,
        std::stod(parameters[0]),
        std::stod(parameters[1]));

    case surface_modules::variable_albedo :
      if (parameters.size() != 1)
      {
        std::string error_message =
          "Variable albedo surface requires exactly one parameter "
          "(path to albedo data file)!\n";
        throw InvalidInput(std::string ("surface.config"), error_message);
      }
      return std::make_unique<VariableAlbedoSurface>(
        spectral_grid,
        parameters[0]);
  }

  return nullptr;
}


}
#endif
