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


#ifndef _fixed_chemistry_h
#define _fixed_chemistry_h

#include "chemistry.h"


#include <vector>
#include <string>


namespace ngam {


class FixedChemistry : public Chemistry{
  public:
    FixedChemistry(const std::string& file_path)
    {
      nb_parameters = 0;
      readFile(file_path);
    }
    virtual ~FixedChemistry() {}
    virtual bool calcChemicalComposition(
      const std::vector<double>& parameters,
      const std::vector<double>& temperature,
      const std::vector<double>& pressure,
      std::vector<std::vector<double>>& number_densities,
      std::vector<double>& mean_molecular_weight);
  protected:
    void readFile(const std::string& file_path);

    std::vector<std::vector<double>> fixed_mixing_ratios;
    std::vector<double> pressure;
};


}
#endif 
