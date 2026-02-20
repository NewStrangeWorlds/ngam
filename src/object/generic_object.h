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

#include <memory>
#include <vector>
#include <string>

#include "../spectral_grid/spectral_grid.h"
#include "../transport_coeff/transport_coeff.h"
#include "../transport_coeff/opacity_calc.h"
#include "../atmosphere/atmosphere.h"
#include "../chemistry/chemistry.h"
#include "../temperature/temperature.h"
#include "../radiative_transfer/radiative_transfer.h"


namespace ngam{


class GenericObject {
  public:
    GenericObject(
      SpectralGrid* spectral_grid_,
      size_t nb_grid_points,
      const std::vector<double>& atmos_boundary_pressures,
      const std::string& cross_section_file_path,
      const std::vector<std::string>& opacity_species_symbol,
      const std::vector<std::string>& opacity_species_folder,
      bool use_clouds,
      std::vector<std::unique_ptr<Chemistry>> chemistry_,
      std::unique_ptr<Temperature> temperature_profile_,
      std::unique_ptr<RadiativeTransfer> radiative_transfer_)
      : radiation_field(spectral_grid_->nbSpectralPoints(), nb_grid_points),
        spectral_grid(spectral_grid_),
        atmosphere(nb_grid_points, atmos_boundary_pressures),
        opacity(
          cross_section_file_path,
          spectral_grid_,
          &atmosphere,
          opacity_species_symbol,
          opacity_species_folder,
          use_clouds),
        chemistry(std::move(chemistry_)),
        temperature_profile(std::move(temperature_profile_)),
        radiative_transfer(std::move(radiative_transfer_))
    {}
    virtual ~GenericObject() {}

    virtual bool computeAtmosphericStructure() = 0;

    const Atmosphere& getAtmosphere() const { return atmosphere; }

    RadiativeTransferOutput radiation_field;
  protected:
    SpectralGrid* spectral_grid;

    Atmosphere atmosphere;
    OpacityCalculation opacity;

    std::vector<std::unique_ptr<Chemistry>> chemistry;
    std::unique_ptr<Temperature> temperature_profile;
    std::unique_ptr<RadiativeTransfer> radiative_transfer;
};


} // namespace ngam

#endif // GENERIC_OBJECT_H
