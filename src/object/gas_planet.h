/*
* This file is part of the ngam code.
* Copyright (C) 2026 Daniel Kitzmann
*
* ngam is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

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
#include "../additional/quadrature.h"
#include "../stellar/stellar_spectrum.h"
#include "../stellar/select_stellar_spectrum.h"


namespace ngam{

// Gas (giant) planet forward model.
//
// A combination of the brown dwarf and terrestrial planet models: an
// illuminated, semi-infinite atmosphere *without* a surface.
//   - Like the terrestrial planet, the top of the atmosphere is illuminated
//     by an incident stellar beam (instellation + zenith_angle).
//   - Like the brown dwarf, there is no surface. The lower boundary is the
//     diffusion (semi-infinite) condition and an internal heat flux escapes
//     from below, set by the internal temperature via
//     target_flux = sigma * T_int^4. The deep atmosphere temperature is
//     anchored to this internal flux.
class GasPlanet : public GenericObject {
  public:
    GasPlanet(
      SpectralGrid* spectral_grid,
      const ModelConfig& config,
      const double internal_temperature_,
      const double surface_gravity_,
      const double radius_,
      const bool use_variable_gravity_,
      const double instellation,
      const double zenith_angle_,
      const ModuleSpec& stellar_spectrum_spec)
      : GenericObject(
          spectral_grid, config, surface_gravity_, radius_, use_variable_gravity_,
          /*has_surface=*/false, /*default_min_convection_pressure=*/1e-4),
        internal_temperature(internal_temperature_),
        target_flux(constants::stefan_boltzmann * std::pow(internal_temperature_, 4)),
        zenith_angle(zenith_angle_),
        stellar_spectrum(selectStellarSpectrum(stellar_spectrum_spec, instellation))
    {
      // precompute stellar flux per wavenumber
      stellar_flux = stellar_spectrum->calcFlux(spectral_grid->wavenumber_list);
    }
    virtual ~GasPlanet() {}

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

      // ---- temperature-correction scheme, selected by the solver spec (see
      //      select_temperature_correction.h and doc/temperature_correction_schemes.tex).
      TemperatureCorrectionSetup tc_setup;
      tc_setup.target_flux              = target_flux;
      tc_setup.convection               = convection.get();
      tc_setup.flux_scale               = flux_scale;
      // mask_band = 0: this object's radiative band runs at Delta tau >> 1, where the collocated
      // residual is Nyquist-degenerate and a one-level RCB placement error locks in a checkerboard.
      tc_setup.mask_band                = 0;

      std::unique_ptr<TemperatureCorrection> temp_correction =
        selectTemperatureCorrection(solver, tc_setup);

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
              calcAtmosphereStructure();
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

      NgAccelerator ng(solver.ng_interval);

      std::cout << std::defaultfloat << std::setprecision(6)
                << "\n--- Starting iteration loop (gas planet) ---\n"
                << "  solver:                " << solver.scheme_name << "\n"
                << "  max iterations:        " << solver.max_iterations << "\n"
                << "  convergence threshold: " << solver.convergence_threshold << "\n"
                << "  Ng acceleration:       every " << solver.ng_interval << " iterations"
                << (solver.ng_interval == 0 ? " (disabled)" : "") << "\n"
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

      for (size_t iter = 0; iter < solver.max_iterations; ++iter)
      {
        std::vector<double> old_temperature = atmosphere.temperature;

        // 1. Chemistry
        calcChemistry();

        // 2. Atmosphere structure (density, altitude, scale height)
        calcAtmosphereStructure();

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

        // 8. Limit maximum temperature change per iteration (relaxation schemes only)
        limitTemperatureChange(old_temperature, *temp_correction);

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
          ? (lin_resid < solver.convergence_threshold)
          : (max_change < solver.convergence_threshold);

        if (converged)
        {
          // TOA energy-balance diagnostic. The corrector's convergence residual excludes the
          // optically-thin skin (zeta < skin_zeta, held on local RE), so a TOA net-flux offset --
          // the RT backend's thin-layer accuracy floor, ~1e-4 of the transported flux for
          // adding-doubling, invariant to stream count and grid resolution -- is never driven
          // below the threshold. Report it so it is not mistaken for converged energy balance.
          const double toa_flux_error = radiation_field.flux_total.back() - target_flux;
          std::cout << "\n  Converged after " << iter + 1 << " iterations.\n"
                    << "  TOA net-flux error F_net(TOA) - F_int: "
                    << std::scientific << std::setprecision(4) << toa_flux_error
                    << " erg/cm2/s (" << toa_flux_error / flux_scale
                    << " of mu*S + F_int; optically-thin skin, excluded from the convergence gate)\n"
                    << std::endl;
          return true;
        }
      }

      std::cout << "\n  Warning: did not converge after "
                << solver.max_iterations << " iterations.\n" << std::endl;
      return false;
    }

  protected:
    double defaultProfileTemperature() const override { return internal_temperature; }
    double defaultZenithAngle() const override { return zenith_angle; }

  private:
    double internal_temperature;
    double target_flux;
    double zenith_angle;
    std::unique_ptr<StellarSpectrum> stellar_spectrum;
    std::vector<double> stellar_flux;
};

} // namespace ngam

#endif // GAS_PLANET_H
