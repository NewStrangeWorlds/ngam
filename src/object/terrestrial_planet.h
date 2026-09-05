/*
* This file is part of the ngam code.
* Copyright (C) 2026 Daniel Kitzmann
*
* ngam is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

#ifndef TERRESTRIAL_PLANET_H
#define TERRESTRIAL_PLANET_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <memory>

#include "generic_object.h"
#include "../additional/physical_const.h"
#include "../additional/ng_accelerator.h"
#include "../additional/quadrature.h"
#include "../stellar/stellar_spectrum.h"
#include "../stellar/select_stellar_spectrum.h"
#include "../surface/generic_surface.h"
#include "../surface/select_surface.h"


namespace ngam {

// Terrestrial planet: an irradiated atmosphere on top of a solid surface. Plane-parallel with
// constant gravity; the surface energy balance F_net(surface) = 0 closes the problem (no
// internal heat flux).
class TerrestrialPlanet : public GenericObject {
  public:
    TerrestrialPlanet(
      SpectralGrid* spectral_grid,
      const ModelConfig& config,
      const double surface_gravity_,
      const double instellation,
      const double zenith_angle_,
      const ModuleSpec& stellar_spectrum_spec,
      const ModuleSpec& surface_spec)
      : GenericObject(
          spectral_grid, config, surface_gravity_, /*bottom_radius=*/0.0,
          /*use_variable_gravity=*/false,
          /*has_surface=*/true, /*default_min_convection_pressure=*/1e-3),
        zenith_angle(zenith_angle_),
        stellar_spectrum(selectStellarSpectrum(stellar_spectrum_spec, instellation)),
        surface(selectSurface(surface_spec, spectral_grid))
    {
      // precompute stellar flux per wavenumber
      stellar_flux = stellar_spectrum->calcFlux(spectral_grid->wavenumber_list);
    }
    virtual ~TerrestrialPlanet() {}

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
        calcAtmosphereStructure();
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

      // ---- temperature-correction scheme, selected by the solver spec (see
      //      select_temperature_correction.h and doc/temperature_correction_schemes.tex).
      TemperatureCorrectionSetup tc_setup;
      tc_setup.target_flux             = target_flux;
      // LIN_NO_CONV (debug) forces a pure-radiative run -- a clean test bed for the solvers.
      tc_setup.convection              = std::getenv("LIN_NO_CONV") ? nullptr : convection.get();
      tc_setup.flux_scale              = flux_scale;
      // surface-anchored: the troposphere is slaved to lv 0 (= the surface), which keeps its flux row
      // so F_net[0]=0 (the surface balance) is enforced.
      tc_setup.surface_anchored        = true;
      tc_setup.mask_band               = 2;   // terrestrial runs at Delta tau <~ 1: placement-insensitive

      std::unique_ptr<TemperatureCorrection> temp_correction =
        selectTemperatureCorrection(solver, tc_setup);

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
            calcAtmosphereStructure();
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
          calcAtmosphereStructure();
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

      NgAccelerator ng(solver.ng_interval);

      std::cout << std::defaultfloat << std::setprecision(6)
                << "\n--- Starting iteration loop (terrestrial planet) ---\n"
                << "  solver:                " << solver.scheme_name << "\n"
                << "  max iterations:        " << solver.max_iterations << "\n"
                << "  convergence threshold: " << solver.convergence_threshold << "\n"
                << "  Ng acceleration:       every " << solver.ng_interval << " iterations"
                << (solver.ng_interval == 0 ? " (disabled)" : "") << "\n"
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

      for (size_t iter = 0; iter < solver.max_iterations; ++iter)
      {
        std::vector<double> old_temperature = atmosphere.temperature;

        // 1. Chemistry
        calcChemistry();

        // 2. Atmosphere structure (density, altitude, scale height)
        calcAtmosphereStructure();

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

        // 9. Limit maximum temperature change per iteration (relaxation schemes only)
        limitTemperatureChange(old_temperature, *temp_correction);

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
          ? (lin_resid < solver.convergence_threshold) : (max_change < solver.convergence_threshold);
        if (converged)
        {
          std::cout << "\n  Converged after " << iter + 1 << " iterations.\n" << std::endl;
          return true;
        }
      }

      std::cout << "\n  Warning: did not converge after "
                << solver.max_iterations << " iterations.\n" << std::endl;
      return false;
    }

  protected:
    // the surface starts at the bottom-of-atmosphere temperature
    void onInitialized() override { surface->temperature = atmosphere.temperature[0]; }
    double defaultZenithAngle() const override { return zenith_angle; }

  private:
    double zenith_angle;
    std::unique_ptr<StellarSpectrum> stellar_spectrum;
    std::vector<double> stellar_flux;
    std::unique_ptr<GenericSurface> surface;
};


} // namespace ngam

#endif // TERRESTRIAL_PLANET_H
