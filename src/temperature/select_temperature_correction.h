#ifndef _select_temperature_correction_h
#define _select_temperature_correction_h

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "temperature_correction.h"
#include "clima_rce_correction.h"
#include "linearised_temperature_correction.h"
#include "time_stepping_temperature.h"
#include "time_stepping_lre_temperature.h"
#include "../additional/exceptions.h"


namespace ngam {


// The available temperature-correction schemes, with a config keyword and a short form -- the same
// pattern as selectChemistryModule / selectRadiativeTransfer.
//
//   ratio_ul          collocated ratio local-RE residual + the full Uns"old-Lucy correction.
//                     RECOMMENDED for every object class: it is the only scheme whose residual has an
//                     O(1) Nyquist eigenvalue, so its converged root carries no grid-scale
//                     checkerboard, and the one-sided cumulative Lucy integral fixes the flux level
//                     AND gradient without re-exciting that mode.
//   flux_divergence   heat-capacity-weighted flux divergence (clima's residual). Conserves flux
//                     EXACTLY by construction at its root, but that root carries a sawtooth. Use it
//                     when per-level flux conservation matters more than a smooth profile.
//   ptc               the legacy pseudo-transient-continuation flux Newton.
//   time_stepping     explicit relaxation on the radiative heating.
//   time_stepping_lre as above, blended with a local-radiative-equilibrium update.
//
// See doc/temperature_correction_schemes.tex for the mathematics and the measured comparison.
namespace temperature_correction_modules {
  enum id {ratio_ul, flux_divergence, ptc, time_stepping, time_stepping_lre};
  const std::vector<std::string> description {
    "ratio_ul", "flux_divergence", "ptc", "time_stepping", "time_stepping_lre"};
  const std::vector<std::string> description_short {
    "rce", "flux", "lin", "ts", "ts_lre"};
}


// Everything the correctors may need from the calling object. Defaults are the terrestrial values;
// the self-luminous drivers override mask_band (see below) and supply flux_scale.
struct TemperatureCorrectionSetup {
  double target_flux = 0.0;
  const Convection* convection = nullptr;
  double max_change_per_iteration = 0.1;
  double iteration_gamma = 0.2;
  double lre_fraction = 0.0;
  double flux_scale = 0.0;
  bool surface_anchored = false;
  // RCB dead-band half-width. This is a Delta-tau REGIME decision, not a taste one: where the
  // radiative column runs at Delta tau <~ 1 (terrestrial) the collocated residual is insensitive to
  // the convective-boundary placement and a band of 2 is harmless, but where it runs at
  // Delta tau >> 1 (the self-luminous radiative band) the residual is itself Nyquist-degenerate and a
  // one-level placement error locks in a large-amplitude checkerboard -- so those objects pass 0.
  int mask_band = 2;
};


inline std::unique_ptr<TemperatureCorrection> selectTemperatureCorrection(
  const std::string& correction_type,
  const std::vector<std::string>& parameters,
  const TemperatureCorrectionSetup& setup)
{
  auto it = std::find(
    temperature_correction_modules::description.begin(),
    temperature_correction_modules::description.end(),
    correction_type);
  auto it_short = std::find(
    temperature_correction_modules::description_short.begin(),
    temperature_correction_modules::description_short.end(),
    correction_type);

  if (it == temperature_correction_modules::description.end()
   && it_short == temperature_correction_modules::description_short.end())
  {
    std::string error_message = "Temperature correction type " + correction_type + " unknown! Available: ";
    for (const auto& d : temperature_correction_modules::description) error_message += d + " ";
    error_message += "\n";
    throw InvalidInput(std::string("forward_model.config"), error_message);
  }

  temperature_correction_modules::id module_id = static_cast<temperature_correction_modules::id>(0);

  if (it != temperature_correction_modules::description.end())
    module_id = static_cast<temperature_correction_modules::id>(
      std::distance(temperature_correction_modules::description.begin(), it));
  else
    module_id = static_cast<temperature_correction_modules::id>(
      std::distance(temperature_correction_modules::description_short.begin(), it_short));

  (void) parameters;   // reserved for scheme-specific arguments

  if (module_id == temperature_correction_modules::ratio_ul)
    return std::make_unique<ClimaRCECorrection>(
      setup.target_flux, setup.convection, setup.mask_band, /*use_ratio=*/true);

  if (module_id == temperature_correction_modules::flux_divergence)
    return std::make_unique<ClimaRCECorrection>(
      setup.target_flux, setup.convection, setup.mask_band, /*use_ratio=*/false);

  if (module_id == temperature_correction_modules::ptc)
    return std::make_unique<LinearisedTemperatureCorrection>(
      setup.target_flux, 1.0, 1.0, setup.max_change_per_iteration, setup.convection,
      setup.flux_scale, setup.surface_anchored);

  if (module_id == temperature_correction_modules::time_stepping_lre)
    return std::make_unique<TimeSteppingLRETemperature>(
      -1.0, setup.iteration_gamma, setup.target_flux, setup.lre_fraction);

  return std::make_unique<TimeSteppingTemperature>(
    -1.0, setup.iteration_gamma, setup.target_flux);
}


}
#endif
