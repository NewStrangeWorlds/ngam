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


#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <iomanip>

#include "atmosphere.h"

#include "../chemistry/chem_species.h"
#include "../additional/physical_const.h"


namespace ngam{


Atmosphere::Atmosphere(
  const size_t nb_grid_points_,
  const std::vector<double>& atmos_boundaries) 
  : nb_grid_points(nb_grid_points_)
{
  createPressureGrid(atmos_boundaries);

  //initialise temperatures, altitudes, and number_densities to 0
  temperature.assign(nb_grid_points, 0.0);
  altitude.assign(nb_grid_points, 0.0);
  scale_height.assign(nb_grid_points, 0.0);
  mass_density.assign(nb_grid_points, 0.0);
  mean_molecular_weight.assign(nb_grid_points, 0.0);

  number_densities.assign(
    nb_grid_points,
    std::vector<double>(constants::species_data.size(), 0.0));

  convective.assign(nb_grid_points, 0);
  kzz.assign(nb_grid_points, 0.0);
}



void Atmosphere::calcMassDensity()
{
  for (size_t i=0; i<nb_grid_points; ++i)
    mass_density[i] = mean_molecular_weight[i] * pressure[i]*1e6
      / (constants::gas_constant * temperature[i]);
}



void Atmosphere::calcAtmosphereStructure(
  const double surface_gravity,
  const double bottom_radius,
  const bool use_variable_gravity)
{
  calcMassDensity();

  if (use_variable_gravity)
    calcAltitudeVariableGravity(surface_gravity, bottom_radius);
  else
    calcAltitude(surface_gravity);

  calcScaleHeight(surface_gravity);
}



//determine the vertical grid via hydrostatic equilibrium
void Atmosphere::calcAltitude(const double surface_gravity)
{
  altitude.assign(nb_grid_points, 0.0);

  for (size_t i=1; i<nb_grid_points; i++)
  {
    double delta_z = 
      (1.0/(mass_density[i]*surface_gravity) 
      + 1.0/(mass_density[i-1]*surface_gravity)) 
      * 0.5 * (pressure[i-1]*1e6 - pressure[i]*1e6);

    altitude[i] = altitude[i-1] + delta_z;
  }
}


//determine the vertical grid via hydrostatic equilibrium
void Atmosphere::calcAltitudeVariableGravity(
  const double surface_gravity_bottom,
  const double bottom_radius)
{ 
  altitude.assign(nb_grid_points, 0.0);
  
  for (size_t i=1; i<nb_grid_points; i++)
  { 
    const double local_radius = bottom_radius + altitude[i-1];
    const double surface_gravity = surface_gravity_bottom * std::pow(bottom_radius/local_radius, 2.0);
    
    const double delta_z = 
      (1.0/(mass_density[i]*surface_gravity) 
      + 1.0/(mass_density[i-1]*surface_gravity)) 
      * 0.5 * (pressure[i-1]*1e6 - pressure[i]*1e6);

    altitude[i] = altitude[i-1] + delta_z;
  }
}


void Atmosphere::calcScaleHeight(const double surface_gravity)
{
  scale_height.assign(nb_grid_points, 0.0);

  for (size_t i=0; i<nb_grid_points; ++i)
    scale_height[i] = constants::gas_constant * temperature[i] / (mean_molecular_weight[i] * surface_gravity);
}


//Creates a pressure grid between the two boundary points
//nb_grid_point pressures vales are equidistantly spread in the log(p) space
void Atmosphere::createPressureGrid(const std::vector<double>& atmos_boundaries)
{
  pressure.assign(nb_grid_points, 0.0);

  const double min_pressure = atmos_boundaries[1];
  const double max_pressure = atmos_boundaries[0];

  pressure.front() = std::log10(max_pressure);

  const double log_step = (std::log10(max_pressure) - std::log10(min_pressure)) / (nb_grid_points - 1.0);

  for (size_t i=1; i<nb_grid_points-1; ++i)
    pressure[i] = pressure[i-1] - log_step;

  for (size_t i=0; i<nb_grid_points-1; ++i)
    pressure[i] = pow(10.0, pressure[i]);

  pressure.back() = min_pressure;
}



}
