/*
* This file is part of the ngam code.
* Copyright (C) 2026 Daniel Kitzmann
*
* ngam is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

#ifndef BROWN_DWARF_H
#define BROWN_DWARF_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>

#include "generic_object.h"
#include "../additional/physical_const.h"
#include "../additional/ng_accelerator.h"


namespace ngam{

// Brown dwarf: a self-luminous, semi-infinite atmosphere. No irradiation, no surface; the
// internal flux sigma T_eff^4 escapes through the diffusion lower boundary and anchors the
// deep temperature.
class BrownDwarf : public GenericObject {
  public:
    BrownDwarf(
      SpectralGrid* spectral_grid,
      const ModelConfig& config,
      const double effective_temperature_,
      const double surface_gravity_,
      const double radius_,
      const bool use_variable_gravity_)
      : GenericObject(
          spectral_grid, config, surface_gravity_, radius_, use_variable_gravity_,
          /*has_surface=*/false, /*default_min_convection_pressure=*/1e-4),
        effective_temperature(effective_temperature_),
        target_flux(constants::stefan_boltzmann * std::pow(effective_temperature_, 4))
    {}
    virtual ~BrownDwarf() {}

    bool computeAtmosphericStructure() override
    {
      // ---- temperature-correction scheme, selected by the solver spec (see
      //      select_temperature_correction.h and doc/temperature_correction_schemes.tex).
      TemperatureCorrectionSetup tc_setup;
      tc_setup.target_flux              = target_flux;
      tc_setup.convection               = convection.get();
      tc_setup.flux_scale               = 0.0;
      // mask_band = 0: this object's radiative band runs at Delta tau >> 1, where the collocated
      // residual is Nyquist-degenerate and a one-level RCB placement error locks in a checkerboard.
      tc_setup.mask_band                = 0;

      std::unique_ptr<TemperatureCorrection> temp_correction =
        selectTemperatureCorrection(solver, tc_setup);

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
              calcChemistry(/*update_kzz=*/false);
              calcAtmosphereStructure();
              opacity.calculate();
            }
            radiation_field.compute_jacobian = compute_jacobian;
            radiative_transfer->calculate(atmosphere, opacity, radiation_field);
            flux_out = radiation_field.flux_total;
            net_heating_out = radiation_field.net_heating;
          });

      radiation_field.compute_jacobian = temp_correction->requiresRadiationJacobian();

      NgAccelerator ng(solver.ng_interval);

      std::cout << std::defaultfloat << std::setprecision(6)
                << "\n--- Starting iteration loop ---\n"
                << "  solver:                " << solver.scheme_name << "\n"
                << "  max iterations:        " << solver.max_iterations << "\n"
                << "  convergence threshold: " << solver.convergence_threshold << "\n"
                << "  Ng acceleration:       every " << solver.ng_interval << " iterations"
                << (solver.ng_interval == 0 ? " (disabled)" : "") << "\n"
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

      for (size_t iter = 0; iter < solver.max_iterations; ++iter)
      {
        std::vector<double> old_temperature = atmosphere.temperature;

        // 1. Chemistry
        calcChemistry();

        // 2. Atmosphere structure (density, altitude, scale height)
        calcAtmosphereStructure();

        // 3. Opacities
        opacity.calculate();

        // 4. Radiative transfer
        radiative_transfer->calculate(atmosphere, opacity, radiation_field);

        // 5. Flux divergence
        radiation_field.calcFluxDivergence(atmosphere.pressure);

        // 6. Temperature correction
        temp_correction->calcCorrection(
          surface_gravity, atmosphere, radiation_field, opacity);

        // 6b. Limit maximum temperature change per iteration (relaxation schemes only)
        limitTemperatureChange(old_temperature, *temp_correction);

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
          ? (lin_resid < solver.convergence_threshold)
          : (max_change < solver.convergence_threshold);

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
                << solver.max_iterations << " iterations.\n" << std::endl;
      return false;
    }

  protected:
    double defaultProfileTemperature() const override { return effective_temperature; }

  private:
    double effective_temperature;
    double target_flux;
};

} // namespace ngam

#endif // BROWN_DWARF_H
