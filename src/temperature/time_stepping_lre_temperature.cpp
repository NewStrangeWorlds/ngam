#include "time_stepping_lre_temperature.h"
#include "../additional/aux_functions.h"
#include "../additional/quadrature.h"
#include "../transport_coeff/opacity_calc.h"
#include "../radiative_transfer/radiative_transfer.h"

#include <vector>
#include <cmath>


namespace ngam {


void TimeSteppingLRETemperature::calcCorrection(
  const double surface_gravity,
  Atmosphere& atmosphere,
  const RadiativeTransferOutput& radiation_field,
  const OpacityCalculation& opacity)
{
  // save pre-correction temperature for computing B_ν
  // (consistent with the radiation field from this iteration)
  const std::vector<double> T_before = atmosphere.temperature;

  // first apply the flux-divergence time stepping correction
  TimeSteppingTemperature::calcCorrection(
    surface_gravity, atmosphere, radiation_field, opacity);

  // then apply the opacity-weighted LRE correction:
  //   T_LRE = T * (∫κ_ν J_ν dν / ∫κ_ν B_ν dν)^{1/4}
  const auto& wn = radiation_field.spectral_grid->wavenumber_list;
  const size_t nb_spectral = wn.size();

  std::vector<double> kJ_integrand(nb_spectral);
  std::vector<double> kB_integrand(nb_spectral);

  const size_t start_level = (target_flux > 0) ? 1 : 0;

  for (size_t i = start_level; i < atmosphere.temperature.size(); ++i)
  {
    const double T_i = T_before[i];

    for (size_t nu = 0; nu < nb_spectral; ++nu)
    {
      const double kappa = opacity.absorption_coeff[nu][i];
      kJ_integrand[nu] = kappa * radiation_field.mean_intensity[i][nu];
      kB_integrand[nu] = kappa * aux::planckFunctionWavenumber(T_i, wn[nu]);
    }

    const double kJ = aux::quadratureTrapezoidal(wn, kJ_integrand);
    const double kB = aux::quadratureTrapezoidal(wn, kB_integrand);

    if (kB > 0)
    {
      const double T_lre = T_i * std::pow(kJ / kB, 0.25);
      atmosphere.temperature[i] += lre_fraction * (T_lre - T_i);

      if (atmosphere.temperature[i] < 1.0)
        atmosphere.temperature[i] = 1.0;
    }
  }
}


}
