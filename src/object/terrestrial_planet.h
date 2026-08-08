#ifndef TERRESTRIAL_PLANET_H
#define TERRESTRIAL_PLANET_H

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
#include "../surface/generic_surface.h"


namespace ngam {


class TerrestrialPlanet : public GenericObject {
  public:
    TerrestrialPlanet(
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
      double zenith_angle_,
      std::unique_ptr<StellarSpectrum> stellar_spectrum_,
      std::unique_ptr<GenericSurface> surface_,
      const std::vector<double>& chemistry_parameters_,
      size_t max_iterations_ = 100,
      double convergence_threshold_ = 1e-4,
      double iteration_gamma_ = 0.5,
      bool use_convective_adjustment_ = true,
      std::string convection_type_ = "dry",
      size_t ng_interval_ = 10,
      double lre_fraction_ = 0.0,
      double min_convection_pressure_ = 1e-3,
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
        zenith_angle(zenith_angle_),
        stellar_spectrum(std::move(stellar_spectrum_)),
        surface(std::move(surface_)),
        chemistry_parameters(chemistry_parameters_),
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
    virtual ~TerrestrialPlanet() {}

    void initialize(
      const std::string& temperature_type,
      const std::vector<std::string>& temperature_config,
      const std::vector<double>& temperature_parameters,
      const std::vector<std::pair<std::string, std::vector<std::string>>>& init_chemistry_configs,
      const std::vector<double>& init_chemistry_parameters)
    {
      auto temp_profile = selectTemperatureProfile(temperature_type, temperature_config);

      auto init_params = temperature_parameters;

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

      atmosphere.calcAtmosphereStructure(surface_gravity, 0, false);

      // initialize surface temperature from bottom of atmosphere
      surface->temperature = atmosphere.temperature[0];

      std::cout << "\n  Initialized from " << temperature_type
                << " temperature profile + chemistry.\n";
    }

    void initializeFromArrays(
      const std::vector<double>& temperature,
      const std::vector<std::vector<double>>& number_densities,
      const std::vector<double>& mean_molecular_weight)
    {
      atmosphere.temperature = temperature;
      atmosphere.number_densities = number_densities;
      atmosphere.mean_molecular_weight = mean_molecular_weight;

      atmosphere.calcAtmosphereStructure(surface_gravity, 0, false);

      surface->temperature = atmosphere.temperature[0];

      std::cout << "\n  Initialized from external arrays.\n";
    }

    double getSurfaceTemperature() const { return surface->temperature; }

    // Diagnostic: run ONE forward solve at the given temperature profile (chemistry, structure,
    // opacity, radiative transfer) and return the per-level net flux. With compute_jac, also computes
    // the analytic temperature Jacobians; read them afterwards via the radiation_field. Lets a caller
    // finite-difference the FULL forward map (opacity recomputed) against the analytic Jacobian.
    std::vector<double> evalForward(const std::vector<double>& temperature, bool compute_jac,
                                    bool recompute_opacity = true)
    {
      atmosphere.temperature = temperature;
      if (recompute_opacity)
      {
        calcChemistry();
        atmosphere.calcAtmosphereStructure(surface_gravity, 0, false);
        opacity.calculate();
      }
      RadiativeBoundaryConditions bc;
      bc.incident_flux = stellar_flux;
      bc.zenith_angle = zenith_angle;
      bc.surface_albedo = surface->getAlbedo();
      bc.surface_temperature = temperature[0];
      bc.has_surface = true;
      radiation_field.compute_jacobian = compute_jac;
      radiative_transfer->calculate(atmosphere, opacity, radiation_field, bc);
      return radiation_field.flux_total;
    }

    bool computeAtmosphericStructure() override
    {
      // target_flux = 0: no internal heat. In radiative-convective equilibrium F_net = 0
      // everywhere (DISORT folds the stellar beam into F_down), and F_net(surface) = 0 is the
      // surface energy balance. The flux residual is normalised by the incident stellar flux.
      const double target_flux = 0.0;
      const double flux_scale = std::max(1.0,
        aux::quadratureTrapezoidal(spectral_grid->wavenumber_list, stellar_flux) * zenith_angle);

      // ---- temperature-correction scheme, selected by config string (see
      //      select_temperature_correction.h and doc/temperature_correction_schemes.tex).
      TemperatureCorrectionSetup tc_setup;
      tc_setup.target_flux             = target_flux;
      // LIN_NO_CONV (debug) forces a pure-radiative run -- a clean test bed for the solvers.
      tc_setup.convection              = std::getenv("LIN_NO_CONV") ? nullptr : convection.get();
      tc_setup.max_change_per_iteration= max_change_per_iteration;
      tc_setup.iteration_gamma         = iteration_gamma;
      tc_setup.lre_fraction            = lre_fraction;
      tc_setup.flux_scale              = flux_scale;
      // surface-anchored: the troposphere is slaved to lv 0 (= the surface), which keeps its flux row
      // so F_net[0]=0 (the surface balance) is enforced.
      tc_setup.surface_anchored        = true;
      tc_setup.mask_band               = 2;   // terrestrial runs at Delta tau <~ 1: placement-insensitive

      std::unique_ptr<TemperatureCorrection> temp_correction =
        selectTemperatureCorrection(temperature_correction, temperature_correction_parameters, tc_setup);

      // Both callbacks are installed unconditionally; each corrector uses the one it needs (the base
      // class declares the others as no-ops). FULL eval: temperatures in, flux AND net heating out,
      // with the option to FREEZE opacity+composition (for Jacobian columns / trial residuals) or
      // recompute them (for the true residual).
      temp_correction->setForwardEvalFull(
        [this](const std::vector<double>& T, bool recompute_opacity, bool compute_jacobian,
               std::vector<double>& flux_out, std::vector<double>& net_heating_out)
        {
          atmosphere.temperature = T;
          if (recompute_opacity)
          {
            calcChemistry();
            atmosphere.calcAtmosphereStructure(surface_gravity, 0, false);
            opacity.calculate();
          }
          RadiativeBoundaryConditions bc;
          bc.incident_flux = stellar_flux;
          bc.zenith_angle = zenith_angle;
          bc.surface_albedo = surface->getAlbedo();
          // BOA-tied surface: the folded sentinel (-1) gives the surface no independent DOF; it emits
          // at the bottom-level temperature.
          bc.surface_temperature = -1.0;
          bc.has_surface = true;
          radiation_field.compute_jacobian = compute_jacobian;
          radiative_transfer->calculate(atmosphere, opacity, radiation_field, bc);
          flux_out = radiation_field.flux_total;
          net_heating_out = radiation_field.net_heating;
        });

      // FLUX-only eval: the true residual at a trial temperature, for the NLEQ-ERR monotonicity test.
      temp_correction->setForwardFluxEval(
        [this](const std::vector<double>& T, std::vector<double>& flux_out)
        {
          atmosphere.temperature = T;
          calcChemistry();
          atmosphere.calcAtmosphereStructure(surface_gravity, 0, false);
          opacity.calculate();
          RadiativeBoundaryConditions bc;
          bc.incident_flux = stellar_flux;
          bc.zenith_angle = zenith_angle;
          bc.surface_albedo = surface->getAlbedo();
          bc.surface_temperature = T[0];
          bc.has_surface = true;
          radiation_field.compute_jacobian = false;
          radiative_transfer->calculate(atmosphere, opacity, radiation_field, bc);
          flux_out = radiation_field.flux_total;
        });


      radiation_field.compute_jacobian = temp_correction->requiresRadiationJacobian();

      NgAccelerator ng(ng_interval);

      std::cout << "\n--- Starting iteration loop (terrestrial planet) ---\n"
                << "  max iterations:        " << max_iterations << "\n"
                << "  convergence threshold: " << convergence_threshold << "\n"
                << "  Ng acceleration:       every " << ng_interval << " iterations"
                << (ng_interval == 0 ? " (disabled)" : "") << "\n"
                << "  zenith angle (cos):    " << zenith_angle << "\n\n" << std::fixed;

      std::cout << "  " << std::setw(5) << "iter"
                << "  " << std::setw(12) << "max|dT/T|"
                << "  " << std::setw(9) << "max|dT|"
                << "  " << std::setw(5) << "@ lv"
                << "  " << std::setw(12) << "F_net(TOA)"
                << "  " << std::setw(11) << "conv_resid"
                << "  " << std::setw(8) << "T_bot"
                << "  " << std::setw(8) << "T_top"
                << "  " << std::setw(8) << "T_surf"
                << "  " << std::setw(6) << "N_conv"
                << "\n";

      for (size_t iter = 0; iter < max_iterations; ++iter)
      {
        std::vector<double> old_temperature = atmosphere.temperature;
        
        // 1. Chemistry
        calcChemistry();

        // 2. Atmosphere structure (density, altitude, scale height)
        atmosphere.calcAtmosphereStructure(surface_gravity, 0, false);
        
        // 3. Opacities
        opacity.calculate();

        // 4. Build boundary conditions
        RadiativeBoundaryConditions bc;
        bc.incident_flux = stellar_flux;
        bc.zenith_angle = zenith_angle;
        bc.surface_albedo = surface->getAlbedo();
        bc.surface_temperature = surface->temperature;
        bc.has_surface = true;
        
        // 5. Radiative transfer (trial evaluations in the corrector leave compute_jacobian off,
        //    so re-arm it here for the base solve that the temperature Jacobian is taken from)
        radiation_field.compute_jacobian = temp_correction->requiresRadiationJacobian();
        radiative_transfer->calculate(atmosphere, opacity, radiation_field, bc);

        // 6. Flux divergence
        radiation_field.calcFluxDivergence(atmosphere.pressure);
        
        // 7. Temperature correction (target_flux = 0: corrects all levels, no anchoring)
        temp_correction->calcCorrection(
          surface_gravity, atmosphere, radiation_field, opacity);

        // 8. Surface temperature. The linearisation solves it inside the Newton (surface-anchored:
        // T_surf = the bottom level, set by F_net[0]=0), so just mirror it; otherwise the explicit
        // surface model + Shapiro smoothing of the relaxation update.
        if (temp_correction->solvesSurfaceTemperature())
          surface->temperature = atmosphere.temperature[0];
        else
        {
          surface->calcTemperature(radiation_field, 1000.0);
          shapiroFilter(atmosphere.temperature, 0.25);
        }

        // 9. Limit maximum temperature change per iteration. Skipped when the corrector sets its own
        // step (NLEQ-ERR): an independent per-level clip here breaks the adiabat slaving (it would clip
        // a convective layer differently from its surface anchor) -> a deep-layer bang-bang oscillation.
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
        
        // 10./11. Convective adjustment and Ng acceleration are RELAXATION-corrector machinery. A
        // Newton-type corrector handles convection internally (zone slaving) and self-converges, and
        // BOTH of these edit the committed profile AFTER the solve -- the post-hoc-mutation failure
        // mode: the solver never sees the change, the two fight, and the residual floors. Gate them on
        // the corrector's own declaration rather than on a legacy scheme flag.
        const bool newton_corrector = temp_correction->handlesConvectionInternally();

        if (convection && !newton_corrector)
          convection->adjust(atmosphere);

        bool ng_applied = newton_corrector ? false : ng.accelerate(atmosphere.temperature, iter);

        // 12. Convergence check
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

        const double flux_toa = radiation_field.flux_total.back()/radiation_field.flux_down_total.back();
        const int n_conv = std::count(
          atmosphere.convective.begin(), atmosphere.convective.end(), 1);

        // the actual convergence metric (max per-level flux imbalance / stellar, gated by |dT/T|): this is
        // what is tested below, so print it -- F_net(TOA) is only the top and floors long before the deep.
        const double lin_resid = temp_correction->lastConvergenceResidual();

        std::cout << "  " << std::setw(5) << iter + 1
                  << "  " << std::setw(12) << std::scientific << std::setprecision(4) << max_change
                  << "  " << std::setw(9) << std::fixed << std::setprecision(3) << max_dT
                  << "  " << std::setw(5) << max_abs_dT_level
                  << "  " << std::setw(12) << std::scientific << std::setprecision(4) << flux_toa
                  << "  " << std::setw(11) << std::scientific << std::setprecision(3) << lin_resid
                  << "  " << std::setw(8) << std::fixed << std::setprecision(0) << atmosphere.temperature[0]
                  << "  " << std::setw(8) << atmosphere.temperature.back()
                  << "  " << std::setw(8) << std::setprecision(1) << surface->temperature
                  << "  " << std::setw(6) << n_conv
                  << (ng_applied ? "  [Ng]" : "")
                  << "\n";

        // converge on the corrector's flux residual when it provides one (linearisation), else |dT/T|
        const bool converged = (lin_resid >= 0.0)
          ? (lin_resid < convergence_threshold) : (max_change < convergence_threshold);
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
    double zenith_angle;
    std::unique_ptr<StellarSpectrum> stellar_spectrum;
    std::vector<double> stellar_flux;
    std::unique_ptr<GenericSurface> surface;

    std::vector<double> chemistry_parameters;

    size_t max_iterations;
    double convergence_threshold;
    double iteration_gamma;
    size_t ng_interval;
    double lre_fraction;
    double max_change_per_iteration;
    std::string temperature_correction = "ratio_ul";
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

#endif // TERRESTRIAL_PLANET_H
