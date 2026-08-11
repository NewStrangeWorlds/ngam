#include "mixing_length.h"
#include "../additional/thermodynamic_data.h"

#include <cmath>
#include <vector>


namespace ngam {


double MixingLengthConvection::convectiveGradient(
  const std::vector<double>& number_densities,
  double temperature,
  double pressure) const
{
  return moist
    ? ThermodynamicData::moistAdiabaticGradient(number_densities, temperature, pressure)
    : ThermodynamicData::adiabaticGradient(number_densities, temperature);
}


void MixingLengthConvection::adjust(Atmosphere& atmosphere) const
{
  // Diagnostic only: flag the levels bounding super-adiabatic links. The temperatures are never
  // modified -- with providesFlux() the corrector handles convection through the flux law, and the
  // driver's post-step adjust() call (used by the relaxation schemes) must be a no-op on T.
  const size_t n = atmosphere.pressure.size();
  std::fill(atmosphere.convective.begin(), atmosphere.convective.end(), 0);

  for (size_t i = 0; i + 1 < n; ++i)
  {
    const double d_ln_p =
      std::log(atmosphere.pressure[i]) - std::log(atmosphere.pressure[i+1]);
    if (d_ln_p <= 0) continue;

    const double nabla =
      (std::log(atmosphere.temperature[i]) - std::log(atmosphere.temperature[i+1])) / d_ln_p;
    const double nabla_ad = 0.5 * (
      convectiveGradient(
        atmosphere.number_densities[i], atmosphere.temperature[i], atmosphere.pressure[i]) +
      convectiveGradient(
        atmosphere.number_densities[i+1], atmosphere.temperature[i+1], atmosphere.pressure[i+1]));

    if (nabla > nabla_ad)
    {
      atmosphere.convective[i]   = 1;
      atmosphere.convective[i+1] = 1;
    }
  }
}


}
