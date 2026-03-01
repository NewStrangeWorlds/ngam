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

#ifndef GENERIC_SURFACE_H
#define GENERIC_SURFACE_H

#include <memory>
#include <vector>
#include <cmath>

#include "../additional/physical_const.h"
#include "../spectral_grid/spectral_grid.h"
#include "../additional/quadrature.h"
#include "../radiative_transfer/radiative_transfer.h"


namespace ngam{


class GenericSurface {
  public:
    GenericSurface(
      SpectralGrid* spectral_grid_)
      : spectral_grid(spectral_grid_) {
        albedo.assign(spectral_grid->nbSpectralPoints(), 0.0);
      };
    virtual ~GenericSurface() {}
    virtual double heatCapacity() const {
      return heat_capacity;}
    const std::vector<double>& getAlbedo() const {
      return albedo;}
    virtual void calcTemperature(
      const RadiativeTransferOutput& radiation_field,
      const double time_step = 0){
        if (time_step > 0)
          temperature -= radiation_field.flux_total.back() * time_step / heat_capacity;
        else
          calcTemperatureFromAbsorbedFlux(radiation_field);
      }

    double temperature = 0;
  protected:
    SpectralGrid* spectral_grid;

    //specific heat capacity of a 50 cm deep slab ocean
    //in erg/K/cm2
    double heat_capacity = 50*4.18*1.e7;
    std::vector<double> albedo;

    virtual void calcTemperatureFromAbsorbedFlux(
      const RadiativeTransferOutput& radiation_field) {
        std::vector<double> absorbed_flux = radiation_field.flux_down.front();

        for (size_t i=0; i<absorbed_flux.size(); ++i)
          absorbed_flux[i] *= (1.0 - albedo[i]);

        const double absorbed_flux_total = aux::quadratureTrapezoidal(
          spectral_grid->wavenumber_list, absorbed_flux);

        if (absorbed_flux_total > 0)
          temperature = std::pow(absorbed_flux_total / constants::stefan_boltzmann, 0.25);
      }
};


} // namespace ngam

#endif // GENERIC_SURFACE_H
