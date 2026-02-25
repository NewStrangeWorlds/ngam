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


#include "time_stepping_temperature.h"
#include "../additional/physical_const.h"
#include "../additional/thermodynamic_data.h"
#include "../atmosphere/atmosphere.h"
#include "../radiative_transfer/radiative_transfer.h"

#include <vector>
#include <cmath>


namespace ngam {



void TimeSteppingTemperature::calcCorrection(
  const double surface_gravity,
  Atmosphere& atmosphere,
  const RadiativeTransferOutput& radiation_field,
  const OpacityCalculation& /*opacity*/)
{
  const bool dynamic_mode = (dt_fixed <= 0);

  const size_t n = atmosphere.pressure.size();

  // For irradiated planets (target_flux <= 0): correct all levels via flux divergence.
  // For brown dwarfs (target_flux > 0): skip the bottom level (anchored separately).
  const size_t start_level =  (target_flux > 0) ? 1 : 0;

  for (size_t i = start_level; i < n; ++i)
  {
    const double T_i = atmosphere.temperature[i];
    const double p_i = atmosphere.pressure[i];

    const double c_p = ThermodynamicData::meanHeatCapacity(
      atmosphere.number_densities[i], T_i);

    double dt_i = dt_fixed;

    if (dynamic_mode)
    {
      // local radiative timescale: t_rad = c_p * p / (g * 4 * sigma * T^3)
      // pressure in bar -> dyne/cm^2: multiply by 1e6
      dt_i = alpha * c_p * p_i * 1e6
           / (surface_gravity * 4.0 * constants::stefan_boltzmann * T_i * T_i * T_i);
    }

    // temperature correction: dT = dt * g / c_p * dF_net/dp
    // flux_divergence is dF_net/dp, pressure in bar -> multiply by 1e-6
    atmosphere.temperature[i] += dt_i * surface_gravity / c_p
                                * radiation_field.flux_divergence[i] * 1e-6;

    // clamp to prevent negative or near-zero temperatures
    if (atmosphere.temperature[i] < 1.0)
      atmosphere.temperature[i] = 1.0;
  }

  // flux anchoring: only for brown dwarfs (target_flux > 0)
  // in equilibrium F_rad(TOA) = sigma * T_eff^4; adjust the bottom temperature
  // to drive the outgoing flux towards the target
  if (target_flux > 0)
  {
    const double F_actual = radiation_field.flux_total.back();
    const double T_i = atmosphere.temperature[0]; //std::pow(target_flux / constants::stefan_boltzmann, 0.25);
    const double p_i = atmosphere.pressure[0];

    const double c_p = ThermodynamicData::meanHeatCapacity(
      atmosphere.number_densities[0], T_i);
    double dt_i = alpha * c_p * p_i * 1e6
           / (surface_gravity * 4.0 * constants::stefan_boltzmann * T_i * T_i * T_i);

    atmosphere.temperature[0] -= dt_i * surface_gravity / c_p
                                * radiation_field.flux_total.back() * 1e-6;

    // atmosphere.temperature[0] += alpha * (target_flux - F_actual)
    //    / (4.0 * constants::stefan_boltzmann * T_eff * T_eff * T_eff);

    if (atmosphere.temperature[0] < 1.0)
      atmosphere.temperature[0] = 1.0;
  }
}



}
