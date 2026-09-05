#ifndef _select_temperature_correction_h
#define _select_temperature_correction_h

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "temperature_correction.h"
#include "clima_rce_correction.h"
#include "../convection/convection.h"
#include "linearised_temperature_correction.h"
#include "time_stepping_temperature.h"
#include "time_stepping_lre_temperature.h"
#include "../config/module_params.h"
#include "../additional/exceptions.h"


namespace ngam {


// The available temperature-correction schemes ("solver" in the model config), with a config
// keyword and a short form -- the same pattern as the other module selectors.
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
//
// Named parameters, validated per scheme (a knob that a scheme does not use is REJECTED):
//   all schemes         max_iterations (100), convergence_threshold (1e-4)
//   time_stepping       gamma (0.5)  relaxation factor
//                       ng_interval (10)  Ng acceleration every n iterations, 0 = off
//                       max_change (0.1)  max relative temperature change per iteration, 0 = off
//   time_stepping_lre   as time_stepping, plus lre_fraction (0.5)  weight of the LRE update
//   ptc                 max_change (0.1)
//   ratio_ul, flux_divergence   no scheme parameters (Newton with its own step control)
namespace temperature_correction_modules {
  enum id {ratio_ul, flux_divergence, ptc, time_stepping, time_stepping_lre};
  const std::vector<std::string> description {
    "ratio_ul", "flux_divergence", "ptc", "time_stepping", "time_stepping_lre"};
  const std::vector<std::string> description_short {
    "rce", "flux", "lin", "ts", "ts_lre"};
}


// The parsed solver spec: the scheme plus the iteration-loop and scheme-specific settings the
// object drivers and the corrector constructors read.
struct SolverSettings {
  temperature_correction_modules::id scheme = temperature_correction_modules::ratio_ul;
  std::string scheme_name = "ratio_ul";

  size_t max_iterations = 100;
  double convergence_threshold = 1e-4;

  // relaxation-scheme parameters (time_stepping, time_stepping_lre); ptc uses max_change too
  double gamma = 0.5;
  double lre_fraction = 0.5;
  size_t ng_interval = 10;
  double max_change = 0.1;
};


inline SolverSettings parseSolverSettings(const ModuleSpec& spec)
{
  using namespace temperature_correction_modules;

  SolverSettings s;
  s.scheme = static_cast<id>(resolveModuleType(
    spec.type, description, description_short, "solver"));
  s.scheme_name = description[s.scheme];

  ParamReader reader(spec, "solver");
  s.max_iterations = static_cast<size_t>(reader.getInt("max_iterations", 100));
  s.convergence_threshold = reader.getDouble("convergence_threshold", 1e-4);

  const bool relaxation = (s.scheme == time_stepping || s.scheme == time_stepping_lre);

  if (relaxation)
  {
    s.gamma = reader.getDouble("gamma", 0.5);
    s.ng_interval = static_cast<size_t>(reader.getInt("ng_interval", 10));
  }
  else
    s.ng_interval = 0;

  if (s.scheme == time_stepping_lre)
    s.lre_fraction = reader.getDouble("lre_fraction", 0.5);

  if (relaxation || s.scheme == ptc)
    s.max_change = reader.getDouble("max_change", 0.1);
  else
    s.max_change = 0.0;

  reader.finish();
  return s;
}


// Everything the correctors need from the calling object (as opposed to the user's solver spec).
// Defaults are the terrestrial values; the self-luminous drivers override mask_band (see below)
// and supply flux_scale.
struct TemperatureCorrectionSetup {
  double target_flux = 0.0;
  const Convection* convection = nullptr;
  double flux_scale = 0.0;
  bool surface_anchored = false;
  // RCB dead-band half-width. This is a Delta-tau REGIME decision, not a taste one: where the
  // radiative column runs at Delta tau <~ 1 (terrestrial) the collocated residual is insensitive to
  // the convective-boundary placement and a band of 2 is harmless, but where it runs at
  // Delta tau >> 1 (the self-luminous radiative band) the residual is itself Nyquist-degenerate and a
  // one-level placement error locks in a large-amplitude checkerboard -- so those objects pass 0.
  int mask_band = 2;
};


// Flux-law convection (mlt_dry/mlt_moist, the default) is consumed inside the ratio_ul corrector;
// every other scheme relies on the hard adjustment (applied by the driver or by internal
// slaving), which a flux-law Convection deliberately does NOT perform. flux_divergence was
// tried and measured NOT to converge with all levels free (its zone rows/slaving are
// load-bearing -- see the note in clima_rce_correction.cpp); the relaxation schemes would
// silently run without convection. Fail loudly instead -- at model construction.
// EXCEPTION (experimental, CLIMA_TIKH>0): flux_divergence+mlt is unlocked when the Tikhonov
// objective regularisation is active -- the measured stall was the diagonally-deficient
// pure-flux Newton, which is exactly the near-null-mode disease the regularisation removes
// (see the corresponding gate in clima_rce_correction.cpp).
inline void checkSolverConvectionPairing(
  const SolverSettings& solver,
  const Convection* convection)
{
  using namespace temperature_correction_modules;

  const bool tikh_active = [] {
    const char* e = std::getenv("CLIMA_TIKH");
    return e != nullptr && std::atof(e) > 0.0; }();
  if (convection != nullptr && convection->providesFlux()
      && solver.scheme != ratio_ul
      && !(solver.scheme == flux_divergence && tikh_active))
    throw InvalidInput(std::string("solver"),
      "convection mlt_dry/mlt_moist requires solver ratio_ul; "
      "use convection dry/moist with the other schemes "
      "(or flux_divergence with CLIMA_TIKH>0 for the experimental regularised pairing)\n");
}


inline std::unique_ptr<TemperatureCorrection> selectTemperatureCorrection(
  const SolverSettings& solver,
  const TemperatureCorrectionSetup& setup)
{
  using namespace temperature_correction_modules;

  checkSolverConvectionPairing(solver, setup.convection);

  switch (solver.scheme)
  {
    case ratio_ul :
      return std::make_unique<ClimaRCECorrection>(
        setup.target_flux, setup.convection, setup.mask_band, /*use_ratio=*/true);

    case flux_divergence :
      return std::make_unique<ClimaRCECorrection>(
        setup.target_flux, setup.convection, setup.mask_band, /*use_ratio=*/false);

    case ptc :
      return std::make_unique<LinearisedTemperatureCorrection>(
        setup.target_flux, 1.0, 1.0, solver.max_change, setup.convection,
        setup.flux_scale, setup.surface_anchored);

    case time_stepping_lre :
      return std::make_unique<TimeSteppingLRETemperature>(
        -1.0, solver.gamma, setup.target_flux, solver.lre_fraction);

    case time_stepping :
      return std::make_unique<TimeSteppingTemperature>(
        -1.0, solver.gamma, setup.target_flux);
  }

  return nullptr;
}


}
#endif
