/*
* This file is part of the ngam code.
* Copyright (C) 2026 Daniel Kitzmann
*
* ngam is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

#ifndef GENERIC_OBJECT_H
#define GENERIC_OBJECT_H

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../config/module_params.h"
#include "../spectral_grid/spectral_grid.h"
#include "../transport_coeff/transport_coeff.h"
#include "../transport_coeff/opacity_calc.h"
#include "../atmosphere/atmosphere.h"
#include "../additional/thermodynamic_data.h"
#include "../chemistry/chemistry.h"
#include "../chemistry/select_chemistry.h"
#include "../radiative_transfer/radiative_transfer.h"
#include "../radiative_transfer/select_radiative_transfer.h"
#include "../convection/convection.h"
#include "../convection/select_convection.h"
#include "../temperature/temperature_correction.h"
#include "../temperature/select_temperature_correction.h"
#include "../temperature/select_temperature_profile.h"


namespace ngam{


// The part of a model's configuration that every object class shares: the pressure grid, the
// opacity sources, and the pluggable components (each a module spec, see module_params.h).
// Object-specific physical parameters (gravity, effective/internal temperature, instellation,
// stellar spectrum, surface, ...) are constructor arguments of the respective class.
struct ModelConfig {
  size_t nb_grid_points = 100;
  std::vector<double> boundary_pressures {1e2, 1e-6};   // [bottom, top] in bar

  std::string opacity_path;                                          // cross-section data root
  std::vector<std::pair<std::string, std::string>> opacity_species;  // (symbol, data folder)
  bool use_clouds = false;

  std::vector<ModuleSpec> chemistry;                                 // applied in order
  ModuleSpec radiative_transfer {"disort", {{"nb_streams", "4"}}};
  ModuleSpec convection {"mlt"};
  ModuleSpec solver {"ratio_ul"};
};


class GenericObject {
  public:
    // has_surface: the domain bottom is a solid surface (terrestrial planets) -- selects the
    // Blackadar wall law for MLT convection. default_min_convection_pressure: the object class's
    // default for the convection spec's min_pressure.
    GenericObject(
      SpectralGrid* spectral_grid_,
      const ModelConfig& config,
      const double surface_gravity_,
      const double bottom_radius_,
      const bool use_variable_gravity_,
      const bool has_surface,
      const double default_min_convection_pressure)
      : radiation_field(spectral_grid_, config.nb_grid_points),
        spectral_grid(spectral_grid_),
        atmosphere(config.nb_grid_points, config.boundary_pressures),
        opacity(
          config.opacity_path,
          spectral_grid_,
          &atmosphere,
          speciesSymbols(config.opacity_species),
          speciesFolders(config.opacity_species),
          config.use_clouds),
        chemistry(selectChemistryModules(config.chemistry)),
        radiative_transfer(selectRadiativeTransfer(
          config.radiative_transfer, config.nb_grid_points, spectral_grid_)),
        convection(selectConvection(
          config.convection, default_min_convection_pressure, has_surface)),
        solver(parseSolverSettings(config.solver)),
        surface_gravity(surface_gravity_),
        bottom_radius(bottom_radius_),
        use_variable_gravity(use_variable_gravity_)
    {
      if (config.chemistry.empty())
        throw InvalidInput("chemistry", "at least one chemistry module is required\n");

      checkSolverConvectionPairing(solver, convection.get());
    }
    virtual ~GenericObject() {}

    virtual bool computeAtmosphericStructure() = 0;

    // Initialise the atmosphere from an analytic temperature profile plus chemistry.
    //
    // profile: a module spec with named parameters
    //   adiabat   T_surface down-integrated along the ACTIVE convection scheme's neutrality
    //             gradient (dry, moist, mlt...), reduced 2%, floored at T_stratosphere -- clima's
    //             start. The 2% bias puts every link on the STABLE side of the scheme's own
    //             threshold, which is what lets the MLT corrector skip its easy-start homotopy
    //             (doc/mlt_convection_design.md Sec. 10.6).
    //               surface_temperature, stratosphere_temperature   [K]
    //   milne     grey Milne-Eddington solution
    //               kappa_ross [cm^2/g], effective_temperature (default: the object's T_eff/T_int)
    //   const     isothermal
    //               temperature
    //   guillot   Guillot (2010) irradiated-analytic profile (irradiated gas planets)
    //               kappa_ir [cm^2/g], t_irr, gamma (= kappa_vis/kappa_ir),
    //               t_int (default: the object's T_int), mode "isotropic" (default) with f
    //               (flux-redistribution factor, default 0.25) or mode "beam" with mu (default:
    //               the object's zenith angle)
    //
    // init_chemistry: the chemistry modules used to compute the initial composition; empty (the
    // default) = the model's own chemistry modules.
    void initialize(
      const ModuleSpec& profile,
      const std::vector<ModuleSpec>& init_chemistry_specs = {})
    {
      std::vector<std::unique_ptr<Chemistry>> own_modules;
      std::vector<Chemistry*> init_chemistry;

      if (init_chemistry_specs.empty())
        for (auto& c : chemistry) init_chemistry.push_back(c.get());
      else
      {
        own_modules = selectChemistryModules(init_chemistry_specs);
        for (auto& c : own_modules) init_chemistry.push_back(c.get());
      }

      auto runInitChemistry = [&]() {
        for (auto* chem : init_chemistry)
          chem->calcChemicalComposition(
            chem->parameters,
            atmosphere.temperature,
            atmosphere.pressure,
            atmosphere.number_densities,
            atmosphere.mean_molecular_weight);
      };

      ParamReader reader(profile, "initial_profile");

      if (profile.type == "adiabat")
      {
        const double t_surf  = reader.requireDouble("surface_temperature");
        const double t_strat = reader.requireDouble("stratosphere_temperature");
        reader.finish();

        // Using convectiveGradient keeps the init consistent with the configured scheme by
        // construction -- no duplicated formula can rot. The gradient needs the composition, so:
        // isothermal pass -> chemistry -> integrate -> chemistry again (the last pass matters for
        // T-dependent chemistry, e.g. the Manabe-Wetherald humidity profile).
        const size_t n = atmosphere.pressure.size();
        atmosphere.temperature.assign(n, t_surf);
        runInitChemistry();

        constexpr double stable_bias = 0.98;
        for (size_t i = 1; i < n; ++i)
        {
          const double nab = convection
            ? convection->convectiveGradient(
                atmosphere.number_densities[i-1], atmosphere.temperature[i-1],
                atmosphere.pressure[i-1])
            : ThermodynamicData::adiabaticGradient(
                atmosphere.number_densities[i-1], atmosphere.temperature[i-1]);
          atmosphere.temperature[i] = std::max(t_strat,
            atmosphere.temperature[i-1]
              * std::pow(atmosphere.pressure[i]/atmosphere.pressure[i-1], stable_bias*nab));
        }
        runInitChemistry();
      }
      else
      {
        std::vector<std::string> profile_config;
        const std::vector<double> parameters = profileParameters(profile.type, reader, profile_config);
        reader.finish();

        auto temp_profile = selectTemperatureProfile(profile.type, profile_config);
        temp_profile->calcProfile(parameters, surface_gravity, atmosphere);
        runInitChemistry();
      }

      calcAtmosphereStructure();
      onInitialized();

      std::cout << "\n  Initialized from " << profile.type << " temperature profile + chemistry.\n";
    }

    // Initialise from saved arrays (restart)
    void initializeFromArrays(
      const std::vector<double>& temperature,
      const std::vector<std::vector<double>>& number_densities,
      const std::vector<double>& mean_molecular_weight)
    {
      atmosphere.temperature = temperature;
      atmosphere.number_densities = number_densities;
      atmosphere.mean_molecular_weight = mean_molecular_weight;

      calcAtmosphereStructure();
      onInitialized();

      std::cout << "\n  Initialized from external arrays.\n";
    }

    const Atmosphere& getAtmosphere() const { return atmosphere; }

    RadiativeTransferOutput radiation_field;

  protected:
    SpectralGrid* spectral_grid;

    Atmosphere atmosphere;
    OpacityCalculation opacity;

    std::vector<std::unique_ptr<Chemistry>> chemistry;
    std::unique_ptr<RadiativeTransfer> radiative_transfer;
    std::unique_ptr<Convection> convection;
    SolverSettings solver;

    double surface_gravity;       // cm/s^2
    double bottom_radius;         // cm; 0 = plane-parallel with constant gravity
    bool use_variable_gravity;

    // hook for derived classes after any initialisation (e.g. set the surface temperature)
    virtual void onInitialized() {}

    // Object-supplied defaults for the analytic initial profiles; NaN = no default (the parameter
    // must then be given explicitly).
    virtual double defaultProfileTemperature() const { return NAN; }   // T_eff / T_int
    virtual double defaultZenithAngle() const { return NAN; }

    void calcChemistry()
    {
      for (auto& chem : chemistry)
        chem->calcChemicalComposition(
          chem->parameters,
          atmosphere.temperature,
          atmosphere.pressure,
          atmosphere.number_densities,
          atmosphere.mean_molecular_weight);
    }

    void calcAtmosphereStructure()
    {
      atmosphere.calcAtmosphereStructure(surface_gravity, bottom_radius, use_variable_gravity);
    }

    // Limit the relative temperature change per iteration (relaxation schemes). Skipped when the
    // corrector sets its own step (NLEQ-ERR / trust region): an independent per-level clip here is
    // a POST-HOC profile modification the Newton never sees -- it breaks the adiabat slaving (a
    // convective layer is clipped independently of its anchor level) and moves the committed
    // profile off the root, so the residual floors.
    void limitTemperatureChange(
      const std::vector<double>& old_temperature,
      const TemperatureCorrection& temp_correction)
    {
      if (solver.max_change <= 0 || temp_correction.managesOwnStepSize()) return;

      for (size_t i = 0; i < atmosphere.temperature.size(); ++i)
      {
        const double dT = atmosphere.temperature[i] - old_temperature[i];
        const double limit = solver.max_change * old_temperature[i];

        if (std::abs(dT) > limit)
          atmosphere.temperature[i] = old_temperature[i] + std::copysign(limit, dT);
      }
    }

    void shapiroFilter(std::vector<double>& data, const double alpha) {
      const size_t n = atmosphere.temperature.size();
      std::vector<double> data_smooth(n);
      data_smooth[0] = data[0];
      data_smooth[n-1] = data[n-1];

      for (size_t i = 1; i < n - 1; ++i)
        data_smooth[i] = (1.0 - 2.0*alpha) * data[i]
                + alpha * (data[i-1] + data[i+1]);

      data = std::move(data_smooth);
    }

  private:
    static std::vector<std::string> speciesSymbols(
      const std::vector<std::pair<std::string, std::string>>& species)
    {
      std::vector<std::string> out;
      for (const auto& s : species) out.push_back(s.first);
      return out;
    }

    static std::vector<std::string> speciesFolders(
      const std::vector<std::pair<std::string, std::string>>& species)
    {
      std::vector<std::string> out;
      for (const auto& s : species) out.push_back(s.second);
      return out;
    }

    // a parameter with an object-supplied default: explicit value > default > error
    static double valueOrDefault(ParamReader& reader, const std::string& key, const double dflt)
    {
      if (reader.has(key)) return reader.requireDouble(key);
      if (!std::isnan(dflt)) return dflt;
      return reader.requireDouble(key);   // throws the "required parameter missing" error
    }

    // map the named parameters of an analytic profile onto the profile class's parameter vector
    std::vector<double> profileParameters(
      const std::string& type,
      ParamReader& reader,
      std::vector<std::string>& profile_config)
    {
      if (type == "milne")
        return {
          reader.requireDouble("kappa_ross"),
          valueOrDefault(reader, "effective_temperature", defaultProfileTemperature())};

      if (type == "const")
        return {reader.requireDouble("temperature")};

      if (type == "guillot")
      {
        const std::string mode = reader.getString("mode", "isotropic");
        if (mode != "isotropic" && mode != "beam")
          throw InvalidInput("initial_profile",
            "guillot profile: mode must be 'isotropic' or 'beam', got '" + mode + "'\n");
        profile_config = {mode};
        const double last = (mode == "beam")
          ? valueOrDefault(reader, "mu", defaultZenithAngle())
          : reader.getDouble("f", 0.25);
        return {
          reader.requireDouble("kappa_ir"),
          reader.requireDouble("t_irr"),
          valueOrDefault(reader, "t_int", defaultProfileTemperature()),
          reader.requireDouble("gamma"),
          last};
      }

      throw InvalidInput("initial_profile",
        "initial profile type '" + type + "' unknown! Available: adiabat, milne, const, guillot\n");
    }
};


} // namespace ngam

#endif // GENERIC_OBJECT_H
