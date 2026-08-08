#ifndef GAS_PLANET_H
#define GAS_PLANET_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>

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
#include "../stellar/stellar_spectrum.h"
#include "../additional/quadrature.h"


namespace ngam {


// Gas (giant) planet forward model.
//
// A combination of the brown dwarf and terrestrial planet models: an
// illuminated, semi-infinite atmosphere *without* a surface.
//   - Like the terrestrial planet, the top of the atmosphere is illuminated
//     by an incident stellar beam (incident_flux + zenith_angle).
//   - Like the brown dwarf, there is no surface. The lower boundary is the
//     diffusion (semi-infinite) condition and an internal heat flux escapes
//     from below, set by the internal temperature via
//     target_flux = sigma * T_int^4. The deep atmosphere temperature is
//     anchored to this internal flux.
class GasPlanet : public GenericObject {
  public:
    GasPlanet(
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
      double internal_temperature_,
      double zenith_angle_,
      std::unique_ptr<StellarSpectrum> stellar_spectrum_,
      const std::vector<double>& chemistry_parameters_,
      double bottom_radius_ = 0.0,
      bool use_variable_gravity_ = false,
      size_t max_iterations_ = 100,
      double convergence_threshold_ = 1e-4,
      double iteration_gamma_ = 0.5,
      bool use_convective_adjustment_ = true,
      std::string convection_type_ = "dry",
      size_t ng_interval_ = 10,
      double lre_fraction_ = 0.0,
      double min_convection_pressure_ = 1e-4,
      double max_change_per_iteration_ = 0.1,
      std::string temperature_correction_ = "ratio_ul",
      std::vector<std::string> temperature_correction_parameters_ = {})
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
        internal_temperature(internal_temperature_),
        target_flux(constants::stefan_boltzmann * std::pow(internal_temperature_, 4)),
        zenith_angle(zenith_angle_),
        stellar_spectrum(std::move(stellar_spectrum_)),
        chemistry_parameters(chemistry_parameters_),
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
      // precompute stellar flux per wavenumber
      stellar_flux = stellar_spectrum->calcFlux(spectral_grid->wavenumber_list);

      if (use_convective_adjustment_)
      {
        if (convection_type_ == "moist")
          convection = std::make_unique<MoistAdiabaticAdjustment>(10, min_convection_pressure_);
        else
          convection = std::make_unique<DryAdiabaticAdjustment>(10, min_convection_pressure_);
      }
    }
    virtual ~GasPlanet() {}

    // Profile-based initialization
    void initialize(
      const std::string& temperature_type,
      const std::vector<std::string>& temperature_config,
      const std::vector<double>& temperature_parameters,
      const std::vector<std::pair<std::string, std::vector<std::string>>>& init_chemistry_configs,
      const std::vector<double>& init_chemistry_parameters)
    {
      auto temp_profile = selectTemperatureProfile(temperature_type, temperature_config);

      // the internal flux sets the deep temperature for analytic (e.g. Milne) profiles
      auto init_params = temperature_parameters;
      if (init_params.size() < 3)
        init_params.resize(3);
      init_params[2] = target_flux;

      temp_profile->calcProfile(init_params, surface_gravity, atmosphere);

      // create init chemistry modules and compute chemical composition
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
      // target_flux = sigma * T_int^4 (> 0): the internal heat flux escaping
      // through the semi-infinite lower boundary. With target_flux > 0 the
      // temperature correction anchors the deep atmosphere to this flux,
      // exactly as for the brown dwarf.
      // Flux scale for residual normalisation and the convergence metric. The natural target of the
      // correction is F_net = F_int at every level, but for an irradiated planet F_int is usually
      // TINY compared to the stellar flux the two streams actually carry (here F_int = sigma T_int^4
      // ~ 5.7e3 vs mu*S ~ 1.2e8 erg/cm2/s): judging |F_net - F_int| relative to F_int alone demands
      // a stream accuracy of tol * F_int / (mu*S) ~ 5e-10 -- unreachable for any sampled-opacity RT.
      // The honest scale is the flux the atmosphere actually transports, mu*S + F_int (reduces to
      // F_int for an isolated object, to mu*S for a strongly irradiated one). The CLIMA_RCE corrector
      // already normalises internally by max(F_int, F_down(TOA)); this scale feeds the linearised
      // (PTC) corrector and the driver's fallback display/convergence metric.
      const double flux_scale =
        aux::quadratureTrapezoidal(spectral_grid->wavenumber_list, stellar_flux) * zenith_angle
        + target_flux;

      // ---- temperature-correction scheme, selected by config string (see
      //      select_temperature_correction.h and doc/temperature_correction_schemes.tex).
      TemperatureCorrectionSetup tc_setup;
      tc_setup.target_flux              = target_flux;
      tc_setup.convection               = convection.get();
      tc_setup.max_change_per_iteration = max_change_per_iteration;
      tc_setup.iteration_gamma          = iteration_gamma;
      tc_setup.lre_fraction             = lre_fraction;
      tc_setup.flux_scale               = flux_scale;
      // mask_band = 0: this object's radiative band runs at Delta tau >> 1, where the collocated
      // residual is Nyquist-degenerate and a one-level RCB placement error locks in a checkerboard.
      tc_setup.mask_band                = 0;

      std::unique_ptr<TemperatureCorrection> temp_correction =
        selectTemperatureCorrection(temperature_correction, temperature_correction_parameters, tc_setup);

      // installed unconditionally; correctors that do not need it inherit a no-op.
        // Collocated ratio residual + full Uns\"old-Lucy (the terrestrial default), generalised to a
        // nonzero internal flux: at RCE the NET total flux (thermal + stellar) equals F_int at every
        // level, so the corrector's bottom/zone rows and the Lucy integral all target F_star =
        // sigma T_int^4 exactly as for the brown dwarf; the absorbed stellar flux enters through the
        // mean intensity in the ratio term. mask_band = 0 as for the brown dwarf: the self-luminous
        // deep sits at Delta tau >> 1 per layer, where a one-level RCB placement error locks in a
        // large-amplitude checkerboard (see the constructor note in clima_rce_correction.h).
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
            RadiativeBoundaryConditions bc;
            bc.incident_flux = stellar_flux;
            bc.zenith_angle = zenith_angle;
            bc.has_surface = false;         // semi-infinite bottom, irradiated top
            radiation_field.compute_jacobian = compute_jacobian;
            radiative_transfer->calculate(atmosphere, opacity, radiation_field, bc);
            flux_out = radiation_field.flux_total;
            net_heating_out = radiation_field.net_heating;
          });

      radiation_field.compute_jacobian = temp_correction->requiresRadiationJacobian();

      NgAccelerator ng(ng_interval);

      std::cout << "\n--- Starting iteration loop (gas planet) ---\n"
                << "  max iterations:        " << max_iterations << "\n"
                << "  convergence threshold: " << convergence_threshold << "\n"
                << "  Ng acceleration:       every " << ng_interval << " iterations"
                << (ng_interval == 0 ? " (disabled)" : "") << "\n"
                << "  internal temperature:  " << internal_temperature << " K\n"
                << "  internal (target) flux:" << std::scientific << std::setprecision(4)
                << target_flux << " erg/cm2/s\n"
                << "  zenith angle (cos):    " << std::fixed << zenith_angle << "\n\n";

      std::cout << "  " << std::setw(5) << "iter"
                << "  " << std::setw(12) << "max|dT/T|"
                << "  " << std::setw(9) << "max|dT|"
                << "  " << std::setw(5) << "@ lv"
                << "  " << std::setw(12) << "dF/F(s+i)"
                << "  " << std::setw(8) << "T_bot"
                << "  " << std::setw(8) << "T_top"
                << "  " << std::setw(6) << "N_conv"
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

        // 4. Boundary conditions: irradiated top, no surface (semi-infinite bottom)
        RadiativeBoundaryConditions bc;
        bc.incident_flux = stellar_flux;
        bc.zenith_angle = zenith_angle;
        bc.has_surface = false;

        // 5. Radiative transfer
        radiative_transfer->calculate(atmosphere, opacity, radiation_field, bc);

        // 6. Flux divergence
        radiation_field.calcFluxDivergence(atmosphere.pressure);

        // 7. Temperature correction (flux divergence + internal-flux anchoring)
        temp_correction->calcCorrection(
          surface_gravity, atmosphere, radiation_field, opacity);

        // 8. Limit maximum temperature change per iteration. Skipped when the corrector sets its
        // own step (NLEQ-ERR / trust region): a post-hoc per-level clip breaks the adiabat slaving
        // (a convective layer is clipped independently of its anchor) and moves the committed
        // profile off the root the Newton just found, flooring the residual.
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

        // 9. Convective adjustment (the linearisation and clima-RCE Newtons handle convection
        // internally, slaving convective layers to the adiabat, so the explicit adjustment is
        // skipped there -- it would overwrite both the committed profile and the corrector's
        // convective-mask bookkeeping)
        const bool newton_corrector = temp_correction->handlesConvectionInternally();
        if (convection && !newton_corrector)
          convection->adjust(atmosphere);

        // 10. Ng acceleration (only for the fixed-point relaxation correctors;
        // the Newton step converges on its own)
        bool ng_applied = newton_corrector
          ? false : ng.accelerate(atmosphere.temperature, iter);

        // 11. Convergence check
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

        // flux conservation: the largest relative deviation of the net flux from the
        // internal flux over the whole column. This is the physical radiative-equilibrium
        // convergence metric -- unlike max|dT/T| it is not dominated by the weakly-determined,
        // energetically-negligible optically-thin skin.
        // convergence/display metric: the corrector's radiative flux-conservation residual
        // (excludes convective + optically-thin-skin layers) when it provides one, else the
        // TOA flux error.
        const double lin_resid = temp_correction->lastConvergenceResidual();
        // fallback metric (relaxation correctors): TOA net-flux error relative to the TRANSPORTED
        // flux mu*S + F_int, not F_int alone (see the flux_scale note above).
        const double flux_error = (lin_resid >= 0.0)
          ? lin_resid : (radiation_field.flux_total.back() - target_flux) / flux_scale;

        const int n_conv = std::count(
          atmosphere.convective.begin(), atmosphere.convective.end(), 1);

        std::cout << "  " << std::setw(5) << iter + 1
                  << "  " << std::setw(12) << std::scientific << std::setprecision(4) << max_change
                  << "  " << std::setw(9) << std::fixed << std::setprecision(3) << max_dT
                  << "  " << std::setw(5) << max_abs_dT_level
                  << "  " << std::setw(12) << std::scientific << std::setprecision(4) << flux_error
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
    double internal_temperature;
    double target_flux;
    double zenith_angle;
    std::unique_ptr<StellarSpectrum> stellar_spectrum;
    std::vector<double> stellar_flux;

    std::vector<double> chemistry_parameters;

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
      for (auto& chem : chemistry)
      {
        chem->calcChemicalComposition(
          chemistry_parameters,
          atmosphere.temperature,
          atmosphere.pressure,
          atmosphere.number_densities,
          atmosphere.mean_molecular_weight);
      }
    }
};


} // namespace ngam

#endif // GAS_PLANET_H
