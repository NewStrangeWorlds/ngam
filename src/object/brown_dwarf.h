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


#ifndef BROWN_DWARF_H
#define BROWN_DWARF_H


#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

#include "generic_object.h"
#include "../chemistry/chem_species.h"


namespace ngam{

class BrownDwarf : public GenericObject {
  public:
    BrownDwarf(
      SpectralGrid* spectral_grid,
      size_t nb_grid_points,
      const std::vector<double>& atmos_boundary_pressures,
      const std::string& cross_section_file_path,
      const std::vector<std::string>& opacity_species_symbol,
      const std::vector<std::string>& opacity_species_folder,
      bool use_clouds,
      std::vector<std::unique_ptr<Chemistry>> chemistry,
      std::unique_ptr<Temperature> temperature_profile,
      std::unique_ptr<RadiativeTransfer> radiative_transfer,
      double surface_gravity_,
      double bottom_radius_,
      bool use_variable_gravity_)
      : GenericObject(
          spectral_grid,
          nb_grid_points,
          atmos_boundary_pressures,
          cross_section_file_path,
          opacity_species_symbol,
          opacity_species_folder,
          use_clouds,
          std::move(chemistry),
          std::move(temperature_profile),
          std::move(radiative_transfer)),
        surface_gravity(surface_gravity_),
        bottom_radius(bottom_radius_),
        use_variable_gravity(use_variable_gravity_)
    {}
    virtual ~BrownDwarf() {}

    bool computeAtmosphericStructure() override
    {
      std::cout << "Computing brown dwarf atmospheric structure...\n";


      std::vector<double> chem_parameters{0.5, 0.5};
      std::vector<double> temp_parameters{1e-4, 3000.0};

      //temperature profile
      temperature_profile->calcProfile(
        temp_parameters,
        surface_gravity,
        atmosphere.pressure,
        atmosphere.temperature);

      //chemical composition
      std::vector<double> mean_molecular_weights(atmosphere.nb_grid_points, 0.0);

      atmosphere.number_densities.assign(
        atmosphere.nb_grid_points,
        std::vector<double>(constants::species_data.size(), 0.0));

      size_t nb_chem_param = 0;

      for (auto & chem : chemistry)
      {
        std::vector<double> params(
          chem_parameters.begin() + nb_chem_param,
          chem_parameters.begin() + nb_chem_param + chem->nbParameters());

        nb_chem_param += chem->nbParameters();

        chem->calcChemicalComposition(
          params, atmosphere.temperature, atmosphere.pressure,
          atmosphere.number_densities, mean_molecular_weights);
      }

      //atmospheric structure from composition
      atmosphere.calcAtmosphereStructure(
        surface_gravity,
        bottom_radius,
        use_variable_gravity,
        mean_molecular_weights);

      opacity.calculate();

      radiative_transfer->calculate(
        atmosphere,
        opacity,
        radiation_field);

      for (size_t i=0; i<radiation_field.flux_total.size(); ++i)
        std::cout << "Level " << i << ": Flux = " << radiation_field.flux_total[i] << " erg/cm2/s\n";

      std::string file_name = "spectrum.dat";

      std::fstream file(file_name.c_str(), std::ios::out);

      for (size_t i=0; i<spectral_grid->nb_spectral_points; ++i)
        file << std::setprecision(10) << std::scientific << spectral_grid->wavelength_list[i] << " " << radiation_field.spectrum[i] << "\n";

      file.close();

      return true;
    }

  private:
    double surface_gravity;
    double bottom_radius;
    bool use_variable_gravity;
};

} // namespace ngam

#endif // BROWN_DWARF_H
