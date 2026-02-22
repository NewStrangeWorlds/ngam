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


#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "generic_object.h"
#include "../additional/physical_const.h"
#include "../temperature/time_stepping_temperature.h"
#include "../convection/dry_adiabatic.h"


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
      double effective_temperature_,
      double metallicity_,
      double bottom_radius_,
      bool use_variable_gravity_,
      const std::vector<double>& temperature_parameters_,
      const std::vector<double>& chemistry_parameters_,
      size_t max_iterations_ = 100,
      double convergence_threshold_ = 1e-4,
      double iteration_gamma_ = 0.5,
      bool use_convective_adjustment_ = true)
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
        effective_temperature(effective_temperature_),
        metallicity(metallicity_),
        target_flux(constants::stefan_boltzmann * std::pow(effective_temperature_, 4)),
        bottom_radius(bottom_radius_),
        use_variable_gravity(use_variable_gravity_),
        temperature_parameters(temperature_parameters_),
        chemistry_parameters(chemistry_parameters_),
        max_iterations(max_iterations_),
        convergence_threshold(convergence_threshold_),
        iteration_gamma(iteration_gamma_)
    {
      if (use_convective_adjustment_)
        convection = std::make_unique<DryAdiabaticAdjustment>();
    }
    virtual ~BrownDwarf() {}

    void setTemperature(const std::vector<double>& temperature)
    {
      atmosphere.temperature = temperature;
      skip_init = true;
    }

    bool computeAtmosphericStructure() override
    {
      if (!skip_init)
      {
        // --- Initialization: analytic temperature profile ---
        auto init_params = temperature_parameters;
        if (init_params.size() < 3)
          init_params.resize(3);
        init_params[2] = target_flux;

        temperature_profile->calcProfile(
          init_params, surface_gravity, atmosphere, radiation_field);
      }
      else
      {
        std::cout << "\n  Using externally provided temperature profile for initialization.\n";
        skip_init = false;
      }

      // time-stepping correction profile (dynamic mode: dt <= 0)
      TimeSteppingTemperature temp_correction;
      std::vector<double> correction_params = {-1.0, iteration_gamma, target_flux};

      std::cout << "\n--- Starting iteration loop ---\n"
                << "  max iterations:        " << max_iterations << "\n"
                << "  convergence threshold: " << convergence_threshold << "\n"
                << "  target flux:           " << std::scientific << std::setprecision(4)
                << target_flux << " erg/cm2/s\n\n" << std::fixed;

      std::cout << "  " << std::setw(5) << "iter"
                << "  " << std::setw(12) << "max|dT/T|"
                << "  " << std::setw(9) << "max|dT|"
                << "  " << std::setw(5) << "@ lv"
                << "  " << std::setw(12) << "dF/F(TOA)"
                << "  " << std::setw(12) << "flux_divergence"
                << "  " << std::setw(8) << "T_bot"
                << "  " << std::setw(8) << "T_top"
                << "  " << std::setw(12) << "N_conv"
                << "\n";

      for (size_t iter = 0; iter < max_iterations; ++iter)
      {
        std::vector<double> old_temperature = atmosphere.temperature;

        // 1. Chemistry
        calcChemistry();

        // 2. Atmosphere structure (density, altitude, scale height)
        atmosphere.calcAtmosphereStructure(
          surface_gravity, bottom_radius, use_variable_gravity);

        // 3. Opacities
        opacity.calculate();

        // 4. Radiative transfer
        radiative_transfer->calculate(atmosphere, opacity, radiation_field);

        // 5. Flux divergence
        radiation_field.calcFluxDivergence(atmosphere.pressure);

        // 6. Temperature correction
        temp_correction.calcProfile(
          correction_params, surface_gravity, atmosphere, radiation_field);

        // 7. Convective adjustment
        if (convection)
          convection->adjust(atmosphere);

        // 8. Convergence check: max |dT/T| and max |dT|
        double max_change = 0;
        double max_abs_dT = 0;
        double max_dT = 0;
        size_t max_abs_dT_level = 0;

        for (size_t i = 0; i < atmosphere.temperature.size(); ++i)
        {
          double rel_change = std::abs(
            (atmosphere.temperature[i] - old_temperature[i]) / old_temperature[i]);
          max_change = std::max(max_change, rel_change);

          double abs_dT = std::abs(atmosphere.temperature[i] - old_temperature[i]);
          if (abs_dT > max_abs_dT)
          {
            max_abs_dT = abs_dT;
            max_dT = atmosphere.temperature[i] - old_temperature[i];
            max_abs_dT_level = i;
          }
        }

        const double flux_error = (radiation_field.flux_total.back() - target_flux) / target_flux;
        const int n_conv = std::count(
          atmosphere.convective.begin(), atmosphere.convective.end(), 1);

        std::cout << "  " << std::setw(5) << iter + 1
                  << "  " << std::setw(12) << std::scientific << std::setprecision(4) << max_change
                  << "  " << std::setw(9) << std::fixed << std::setprecision(3) << max_dT
                  << "  " << std::setw(5) << max_abs_dT_level
                  << "  " << std::setw(12) << std::scientific << std::setprecision(4) << flux_error
                  << "  " << std::setw(12) << std::scientific << std::setprecision(4) << radiation_field.flux_divergence[max_abs_dT_level]
                  << "  " << std::setw(8) << std::fixed << std::setprecision(0) << atmosphere.temperature[0]
                  << "  " << std::setw(8) << atmosphere.temperature.back()
                  << "  " << std::setw(6) << n_conv
                  << "\n";

        if (max_change < convergence_threshold)
        {
          std::cout << "\n  Converged after " << iter + 1 << " iterations.\n" << std::endl;
          return true;
        }
      }

      std::cout << "\n  Warning: did not converge after "
                << max_iterations << " iterations.\n" << std::endl;
      return false;
    }

  private:
    double surface_gravity;
    double effective_temperature;
    double metallicity;
    double target_flux;
    double bottom_radius;
    bool use_variable_gravity;

    std::vector<double> temperature_parameters;
    std::vector<double> chemistry_parameters;

    size_t max_iterations;
    double convergence_threshold;
    double iteration_gamma;
    bool skip_init = false;

    void calcChemistry()
    {
      size_t param_offset = 0;

      for (auto& chem : chemistry)
      {
        std::vector<double> params(
          chemistry_parameters.begin() + param_offset,
          chemistry_parameters.begin() + param_offset + chem->nbParameters());
        param_offset += chem->nbParameters();

        chem->calcChemicalComposition(
          params,
          atmosphere.temperature,
          atmosphere.pressure,
          atmosphere.number_densities,
          atmosphere.mean_molecular_weight);
      }
    }
};

} // namespace ngam

#endif // BROWN_DWARF_H
