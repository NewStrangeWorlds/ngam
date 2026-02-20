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


#ifndef _atmosphere_h
#define _atmosphere_h

#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>

#include "../chemistry/chemistry.h"


namespace ngam {


class Atmosphere {
  public:
    Atmosphere (
      const size_t nb_grid_points_,
      const std::vector<double>& atmos_boundaries);
    ~Atmosphere() {}

    const size_t nb_grid_points = 0;

    std::vector<double> pressure;
    std::vector<double> temperature;
    std::vector<double> altitude;
    std::vector<double> scale_height;
    std::vector<double> mass_density;
    std::vector< std::vector<double> > number_densities;

    void calcAtmosphereStructure(
      const double surface_gravity,
      const double bottom_radius,
      const bool use_variable_gravity,
      std::vector<Chemistry*>& chemistry,
      const std::vector<double>& chem_parameters);

  private:
    void createPressureGrid(const std::vector<double>& domain_boundaries);
    void calcMassDensity(
      const std::vector<double>& mean_molecular_weights);
    void calcAltitude(
      const double g, const std::vector<double>& mean_molecular_weights);
    void calcAltitudeVariableGravity(
      const double g, 
      const double bottom_radius,
      const std::vector<double>& mean_molecular_weights);
    void calcScaleHeight(
      const double g, const std::vector<double>& mean_molecular_weights);
};


}


#endif
