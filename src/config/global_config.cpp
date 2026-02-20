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

#include <exception>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include <omp.h>
#include <stdlib.h>

namespace ngam {


GlobalConfig::GlobalConfig(
  const std::string forward_model_type_,
  const std::string cross_section_file_path_,
  const std::string spectral_disecretisation_,
  const double resolution_)
  : forward_model_type(forward_model_type_),
    cross_section_file_path(cross_section_file_path_)
{
  if (spectral_disecretisation_ != "const_wavenumber" 
   && spectral_disecretisation_ != "const_wavelength" 
   && spectral_disecretisation_ != "const_resolution")
  {
    std::string error_message = "Spectral discretisation parameter: " 
      + spectral_disecretisation_ + " in retrieval.config unknown!\n";
    throw InvalidInput(std::string ("GlobalConfig::GlobalConfig"), error_message);
  }

  if (spectral_disecretisation_ == "const_wavenumber")
  {
    spectral_disecretisation = 0;
  }
  else if (spectral_disecretisation_ == "const_wavelength")
  {
    spectral_disecretisation = 1;
  }
  else if (spectral_disecretisation_ == "const_resolution")
  {
    spectral_disecretisation = 2;
  }

  spectral_resolution = resolution_;
}



bool GlobalConfig::loadConfigFile(std::string retrieval_folder)
{
  if (retrieval_folder.back() != '/')
    retrieval_folder.append("/");

  retrieval_folder_path = retrieval_folder;

  
  std::string file_path = retrieval_folder;
  file_path.append("retrieval.config");

  
  std::fstream file;
  file.open(file_path.c_str(), std::ios::in);

  if (file.fail()) 
  {
    std::cout << "Couldn't open retrieval options file " << file_path << "\n";
    
    return false;
  }

  
  std::cout << "\nParameters found in retrieval.config:\n";

  std::string line;
  std::string input;

  //Header General Config
  std::getline(file, line);
  std::getline(file, line);
  std::getline(file, line);
  std::cout << "General Program Parameters\n";

  std::getline(file, line);

  file.close();

  return true;
}


}
