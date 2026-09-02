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
#include "../additional/ng_accelerator.h"
#include "../temperature/select_temperature_profile.h"
#include "../temperature/time_stepping_temperature.h"
#include "../temperature/time_stepping_lre_temperature.h"
#include "../temperature/linearised_temperature_correction.h"
#include "../temperature/clima_rce_correction.h"
#include "../temperature/select_temperature_correction.h"
#include "../chemistry/select_chemistry.h"
#include "../convection/dry_adiabatic.h"
#include "../convection/moist_adiabatic.h"
#include "../convection/mixing_length.h"



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
      std::unique_ptr<RadiativeTransfer> radiative_transfer,
      double surface_gravity_,
      double effective_temperature_,
      double metallicity_,
      double c_to_o_ratio_,
      double bottom_radius_,
      bool use_variable_gravity_,
      size_t max_iterations_ = 100,
      double convergence_threshold_ = 1e-4,
      double iteration_gamma_ = 0.5,
      bool use_convective_adjustment_ = true,
      std::string convection_type_ = "mlt",
      size_t ng_interval_ = 10,
      double lre_fraction_ = 0.0,
      double min_convection_pressure_ = 1e-4,
      double max_change_per_iteration_ = 0.1,
      std::string temperature_correction_ = "ratio_ul",
      std::vector<std::string> temperature_correction_parameters_ = {},
      std::vector<std::string> convection_parameters_ = {})
      : GenericObject(
          spectral_grid,
          nb_grid_points,
          atmos_boundary_pressures,
          cross_section_file_path,
          opacity_species_symbol,
          opacity_species_folder,
          use_clouds,
          std::move(chemistry),
          std::move(radiative_transfer)),
        surface_gravity(surface_gravity_),
        effective_temperature(effective_temperature_),
        metallicity(metallicity_),
        c_to_o_ratio(c_to_o_ratio_),
        target_flux(constants::stefan_boltzmann * std::pow(effective_temperature_, 4)),
        bottom_radius(bottom_radius_),
        use_variable_gravity(use_variable_gravity_),
        max_iterations(max_iterations_),
        convergence_threshold(convergence_threshold_),
        iteration_gamma(iteration_gamma_),
        ng_interval(ng_interval_),
        lre_fraction(lre_fraction_),
        max_change_per_iteration(max_change_per_iteration_),

        temperature_correction(temperature_correction_),

        temperature_correction_parameters(temperature_correction_parameters_)
    {
      if (use_convective_adjustment_)
      {
        if (convection_type_ == "mlt" || convection_type_ == "mlt_moist")
        {
          // smooth mixing-length convection (doc/mlt_convection_design.md); requires ratio_ul.
          // No Blackadar wall law: a self-luminous domain bottom is not a surface.
          const double alpha =
            convection_parameters_.empty() ? 1.0 : std::stod(convection_parameters_[0]);
          convection = std::make_unique<MixingLengthConvection>(
            convection_type_ == "mlt_moist", alpha, min_convection_pressure_,
            /*blackadar_surface=*/false);
        }
        else if (convection_type_ == "moist")
          convection = std::make_unique<MoistAdiabaticAdjustment>(10, min_convection_pressure_);
        else
          convection = std::make_unique<DryAdiabaticAdjustment>(10, min_convection_pressure_);
      }
    }
    virtual ~BrownDwarf() {}

    // Profile-based initialization
    void initialize(
      const std::string& temperature_type,
      const std::vector<std::string>& temperature_config,
      const std::vector<double>& temperature_parameters,
      const std::vector<std::pair<std::string, std::vector<std::string>>>& init_chemistry_configs,
      const std::vector<double>& init_chemistry_parameters)
    {
      // 1. Create and apply temperature profile
      auto temp_profile = selectTemperatureProfile(temperature_type, temperature_config);

      auto init_params = temperature_parameters;
      if (init_params.size() < 3)
        init_params.resize(3);
      init_params[2] = target_flux;

      temp_profile->calcProfile(init_params, surface_gravity, atmosphere);

      // 2. Create init chemistry modules and compute chemical composition
      std::vector<std::unique_ptr<Chemistry>> init_chemistry;
      for (auto& [type, params] : init_chemistry_configs)
        init_chemistry.push_back(selectChemistryModule(type, params));

      size_t param_offset = 0;
      for (auto& chem : init_chemistry)
      {
        std::vector<double> params(
          init_chemistry_parameters.begin() + param_offset,
          init_chemistry_parameters.begin() + param_offset + chem->nbParameters());
        param_offset += chem->nbParameters();

        chem->calcChemicalComposition(
          params,
          atmosphere.temperature,
          atmosphere.pressure,
          atmosphere.number_densities,
          atmosphere.mean_molecular_weight);
      }

      // 3. Compute atmosphere structure from the initialized state
      atmosphere.calcAtmosphereStructure(
        surface_gravity, bottom_radius, use_variable_gravity);

      std::cout << "\n  Initialized from " << temperature_type
                << " temperature profile + chemistry.\n";
    }

    // Array-based initialization (restart from saved model)
    void initializeFromArrays(
      const std::vector<double>& temperature,
      const std::vector<std::vector<double>>& number_densities,
      const std::vector<double>& mean_molecular_weight)
    {
      atmosphere.temperature = temperature;
      atmosphere.number_densities = number_densities;
      atmosphere.mean_molecular_weight = mean_molecular_weight;

      atmosphere.calcAtmosphereStructure(
        surface_gravity, bottom_radius, use_variable_gravity);

      std::cout << "\n  Initialized from external arrays.\n";
    }

    bool computeAtmosphericStructure() override
    {
      // time-stepping correction (dynamic mode: dt <= 0)
      // optionally with opacity-weighted LRE correction
      // ---- temperature-correction scheme, selected by config string (see
      //      select_temperature_correction.h and doc/temperature_correction_schemes.tex).
      TemperatureCorrectionSetup tc_setup;
      tc_setup.target_flux              = target_flux;
      tc_setup.convection               = convection.get();
      tc_setup.max_change_per_iteration = max_change_per_iteration;
      tc_setup.iteration_gamma          = iteration_gamma;
      tc_setup.lre_fraction             = lre_fraction;
      tc_setup.flux_scale               = 0.0;
      // mask_band = 0: this object's radiative band runs at Delta tau >> 1, where the collocated
      // residual is Nyquist-degenerate and a one-level RCB placement error locks in a checkerboard.
      tc_setup.mask_band                = 0;

      std::unique_ptr<TemperatureCorrection> temp_correction =
        selectTemperatureCorrection(temperature_correction, temperature_correction_parameters, tc_setup);

      // installed unconditionally; correctors that do not need it inherit a no-op.
        // Collocated ratio residual + full Uns\"old-Lucy (the terrestrial default), applied to the
        // self-luminous case. Motivation: the per-level net-flux residual of the linearised corrector
        // has a near-null Nyquist eigenvalue (the net flux is an odd angular moment, so its Jacobian
        // diagonal cancels), which leaves a grid-scale checkerboard in the CONVERGED ROOT that no step
        // operator can remove -- measured here at ~10 K near the convective boundary (P ~ 2-20 bar),
        // i.e. ~6e-3 relative. The ratio residual's Nyquist eigenvalue is O(1) instead, and the
        // one-sided cumulative Lucy integral supplies level and gradient without re-exciting the mode.
        // The self-luminous case is the CLASSICAL Uns\"old-Lucy setting: F_star = sigma T_int^4 is a
        // known nonzero scalar (the corrector's bottom-boundary rows carry it) and there is no surface
        // energy balance to reconcile.
        // mask_band = 0: the self-luminous detached radiative band sits at Delta tau >> 1 per layer,
        // where a one-level RCB placement error locks in a large-amplitude checkerboard (see the
        // constructor note in clima_rce_correction.h). The terrestrial default dead band (2) left the
        // mask one level short of the detected top and cost 30x in the flux error bar.
        temp_correction->setForwardEvalFull(
          [this](const std::vector<double>& T, bool recompute_opacity, bool compute_jacobian,
                 std::vector<double>& flux_out, std::vector<double>& net_heating_out)
          {
            atmosphere.temperature = T;
            if (recompute_opacity)          // true residual: composition + structure + opacity
            {
              calcChemistry();
              atmosphere.calcAtmosphereStructure(surface_gravity, bottom_radius, use_variable_gravity);
              opacity.calculate();
            }
            radiation_field.compute_jacobian = compute_jacobian;
            radiative_transfer->calculate(atmosphere, opacity, radiation_field);
            flux_out = radiation_field.flux_total;
            net_heating_out = radiation_field.net_heating;
          });

      radiation_field.compute_jacobian = temp_correction->requiresRadiationJacobian();

      NgAccelerator ng(ng_interval);

      std::cout << "\n--- Starting iteration loop ---\n"
                << "  max iterations:        " << max_iterations << "\n"
                << "  convergence threshold: " << convergence_threshold << "\n"
                << "  Ng acceleration:       every " << ng_interval << " iterations"
                << (ng_interval == 0 ? " (disabled)" : "") << "\n"
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

        // 6. Temperature correction (flux divergence + optional LRE)
        temp_correction->calcCorrection(
          surface_gravity, atmosphere, radiation_field, opacity);

        // 6b. Limit maximum temperature change per iteration. Skipped when the corrector sets its own
        // step (NLEQ-ERR / trust region): an independent per-level clip here is a POST-HOC profile
        // modification the Newton never sees -- it breaks the adiabat slaving (a convective layer is
        // clipped independently of its anchor level) and moves the committed profile off the root, so
        // the residual floors (the measured terrestrial Shapiro-filter failure mode).
        if (max_change_per_iteration > 0 && !temp_correction->managesOwnStepSize())
        {
          for (size_t i = 0; i < atmosphere.temperature.size(); ++i)
          {
            const double dT = atmosphere.temperature[i] - old_temperature[i];
            const double limit = max_change_per_iteration * old_temperature[i];

            if (std::abs(dT) > limit)
              atmosphere.temperature[i] = old_temperature[i] + std::copysign(limit, dT);
          }
        }

        // 7. Convective adjustment (the linearisation and clima-RCE Newtons handle convection
        // internally, slaving convective layers to the adiabat, so the explicit adjustment is
        // skipped there -- it would overwrite both the committed profile and the corrector's
        // convective-mask bookkeeping)
        const bool newton_corrector = temp_correction->handlesConvectionInternally();
        if (convection && !newton_corrector)
          convection->adjust(atmosphere);

        // 8. Ng acceleration (relaxation correctors only; the Newton step self-converges)
        bool ng_applied = newton_corrector
          ? false : ng.accelerate(atmosphere.temperature, iter);

        // 9. Convergence check: max |dT/T| and max |dT|
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

        // flux conservation: max relative net-flux deviation from the internal flux over the
        // column -- the physical radiative-equilibrium metric (not the skin-dominated max|dT/T|).
        // convergence/display metric: the corrector's radiative flux residual (excludes
        // convective + optically-thin-skin layers) when provided, else the TOA flux error.
        const double lin_resid = temp_correction->lastConvergenceResidual();
        const double flux_error = (lin_resid >= 0.0)
          ? lin_resid : (radiation_field.flux_total.back() - target_flux) / target_flux;

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
                  << (ng_applied ? "  [Ng]" : "")
                  << "\n";

        // converge on the corrector's flux residual when it provides one, else on |dT/T|.
        const bool converged = (lin_resid >= 0.0)
          ? (lin_resid < convergence_threshold)
          : (max_change < convergence_threshold);

        if (converged)
        {
          // TOA energy-balance diagnostic (see gas_planet.h): the convergence residual excludes
          // the optically-thin skin, so the TOA net-flux offset -- the RT backend's thin-layer
          // accuracy floor -- is reported here instead of being gated on.
          const double toa_flux_error = radiation_field.flux_total.back() - target_flux;
          std::cout << "\n  Converged after " << iter + 1 << " iterations.\n"
                    << "  TOA net-flux error F_net(TOA) - F_int: "
                    << std::scientific << std::setprecision(4) << toa_flux_error
                    << " erg/cm2/s (" << toa_flux_error / target_flux
                    << " of F_int; optically-thin skin, excluded from the convergence gate)\n"
                    << std::endl;
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
    double c_to_o_ratio;
    double target_flux;
    double bottom_radius;
    bool use_variable_gravity;

    size_t max_iterations;
    double convergence_threshold;
    double iteration_gamma;
    size_t ng_interval;
    double lre_fraction;
    double max_change_per_iteration;
    std::string temperature_correction;
    std::vector<std::string> temperature_correction_parameters;

    void calcChemistry()
    {
      std::vector<double> params{metallicity, c_to_o_ratio};

      for (auto& chem : chemistry)
      {
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
