#include "moist_adiabatic.h"
#include "../additional/thermodynamic_data.h"

#include <cmath>
#include <vector>


namespace ngam {


double MoistAdiabaticAdjustment::convectiveGradient(
  const std::vector<double>& number_densities,
  double temperature,
  double pressure) const
{
  return ThermodynamicData::moistAdiabaticGradient(number_densities, temperature, pressure);
}


void MoistAdiabaticAdjustment::adjust(Atmosphere& atmosphere) const
{
  const size_t n = atmosphere.pressure.size();
  if (n < 2) return;

  std::fill(atmosphere.convective.begin(), atmosphere.convective.end(), 0);

  for (size_t sweep = 0; sweep < max_sweeps; ++sweep)
  {
    bool adjusted = false;

    // sweep from bottom (i=0, high P) to top (i=n-2)
    for (size_t i = 0; i < n - 1; ++i)
    {
      if (atmosphere.pressure[i+1] < min_pressure) break;

      const double ln_P_i   = std::log(atmosphere.pressure[i]);
      const double ln_P_ip1 = std::log(atmosphere.pressure[i+1]);
      const double d_ln_P   = ln_P_i - ln_P_ip1;  // > 0

      if (d_ln_P <= 0) continue;

      const double ln_T_i   = std::log(atmosphere.temperature[i]);
      const double ln_T_ip1 = std::log(atmosphere.temperature[i+1]);

      const double nabla_actual = (ln_T_i - ln_T_ip1) / d_ln_P;

      // moist adiabatic gradient at the midpoint (average of both levels)
      const double nabla_ad_i   = convectiveGradient(
        atmosphere.number_densities[i], atmosphere.temperature[i], atmosphere.pressure[i]);
      const double nabla_ad_ip1 = convectiveGradient(
        atmosphere.number_densities[i+1], atmosphere.temperature[i+1], atmosphere.pressure[i+1]);
      const double nabla_ad = 0.5 * (nabla_ad_i + nabla_ad_ip1);

      if (nabla_actual <= nabla_ad) continue;

      // superadiabatic: adjust temperatures to the adiabat
      adjusted = true;
      atmosphere.convective[i]   = 1;
      atmosphere.convective[i+1] = 1;

      // pressure ratio factor: T[i+1]_new = T[i]_new * r
      const double r = std::pow(
        atmosphere.pressure[i+1] / atmosphere.pressure[i], nabla_ad);

      if (i == 0)
      {
        // bottom level is the anchor (set by flux balance);
        // only adjust the level above
        atmosphere.temperature[i+1] = atmosphere.temperature[i] * r;
      }
      else
      {
        // enthalpy-conserving adjustment for interior pairs
        const double c_p_i = ThermodynamicData::meanHeatCapacity(
          atmosphere.number_densities[i], atmosphere.temperature[i]);
        const double c_p_ip1 = ThermodynamicData::meanHeatCapacity(
          atmosphere.number_densities[i+1], atmosphere.temperature[i+1]);

        const double H_total = c_p_i * atmosphere.temperature[i]
                              + c_p_ip1 * atmosphere.temperature[i+1];

        atmosphere.temperature[i]   = H_total / (c_p_i + c_p_ip1 * r);
        atmosphere.temperature[i+1] = atmosphere.temperature[i] * r;
      }
    }

    if (!adjusted) break;
  }
}


}
