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


#include "global_config.h"

#include "../additional/exceptions.h"

#include <string>

namespace ngam {


GlobalConfig::GlobalConfig(
  const std::string forward_model_type_,
  const std::string cross_section_file_path_,
  const std::string spectral_discretisation_,
  const double resolution_)
  : forward_model_type(forward_model_type_),
    cross_section_file_path(cross_section_file_path_)
{
  if (spectral_discretisation_ != "const_wavenumber"
   && spectral_discretisation_ != "const_wavelength"
   && spectral_discretisation_ != "const_resolution")
  {
    std::string error_message = "Spectral discretisation parameter: "
      + spectral_discretisation_ + " unknown!\n";
    throw InvalidInput(std::string ("GlobalConfig::GlobalConfig"), error_message);
  }

  if (spectral_discretisation_ == "const_wavenumber")
  {
    spectral_discretisation = 0;
  }
  else if (spectral_discretisation_ == "const_wavelength")
  {
    spectral_discretisation = 1;
  }
  else if (spectral_discretisation_ == "const_resolution")
  {
    spectral_discretisation = 2;
  }

  spectral_resolution = resolution_;
}


}
