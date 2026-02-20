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


#ifndef GENERIC_OBJECT_H
#define GENERIC_OBJECT_H

#include "../config/global_config.h"
#include "../spectral_grid/spectral_grid.h"
#include "../transport_coeff/transport_coeff.h"
#include "../transport_coeff/opacity_calc.h"
#include "../atmosphere/atmosphere.h"
#include "../chemistry/select_chemistry.h"
#include "../chemistry/chemistry.h"
#include "../temperature/select_temperature_profile.h"
#include "../temperature/temperature.h"
#include "../radiative_transfer/select_radiative_transfer.h"
#include "../radiative_transfer/radiative_transfer.h"


namespace ngam{


class GenericObject {
  public:
    GenericObject(
      GlobalConfig* config_,
      SpectralGrid* spectral_grid_) 
      : radiation_field(spectral_grid_->nbSpectralPoints(), config_->nb_grid_points),
        config(config_), 
        spectral_grid(spectral_grid_),
        atmosphere(config->nb_grid_points, config->atmos_boundary_pressures),
        opacity(
          config,
          spectral_grid,
          &atmosphere,
          config->opacity_species_symbol,
          config->opacity_species_folder,
          config->cloud_model.size() > 0)
    {
      temperature.assign(config->nb_grid_points, 100.0);
      pressure.assign(config->nb_grid_points, 1.0);

      chemistry.assign(config->chemistry_model.size(), nullptr);

      for (size_t i=0; i<config->chemistry_model.size(); ++i)
        chemistry[i] = selectChemistryModule(
          config->chemistry_model[i], 
          config->chemistry_parameters[i], 
          config);

      temperature_profile = selectTemperatureProfile(
        config->temperature_profile_type, 
        config->temperature_profile_parameters);

      radiative_transfer = selectRadiativeTransfer(
        config->radiative_transfer_type,
        config->radiative_transfer_parameters,
        config->nb_grid_points,
        config,
        spectral_grid);
    }
    virtual ~GenericObject() {}

    virtual bool computeAtmosphericStructure() = 0;

    std::vector<double> temperature;
    std::vector<double> pressure;

    RadiativeTransferOutput radiation_field;
  protected:
    GlobalConfig* config;
    SpectralGrid* spectral_grid;

    Atmosphere atmosphere;
    OpacityCalculation opacity;
    RadiativeTransfer* radiative_transfer = nullptr;

    std::vector<Chemistry*> chemistry;
    Temperature* temperature_profile = nullptr;

    void computeOpacity();
};


} // namespace ngam

#endif // GENERIC_OBJECT_H