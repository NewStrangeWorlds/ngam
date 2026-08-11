
#include "guillot_temperature.h"

#include <algorithm>
#include <vector>
#include <cmath>


namespace ngam {


// Guillot (2010), Eqs. 27 (beam) and 29 (isotropic average): T^4(tau_ir) from the internal and
// irradiation temperatures, with the grey IR optical depth tau_ir = kappa_ir * P / g.
void GuillotTemperature::calcProfile(
  const std::vector<double>& parameters,
  const double surface_gravity,
  Atmosphere& atmosphere)
{
  const size_t n = atmosphere.pressure.size();
  atmosphere.temperature.assign(n, 0.0);

  const double kappa_ir        = parameters[0];
  const double temperature_irr = parameters[1];
  const double temperature_int = parameters[2];
  const double gamma           = parameters[3];
  const double profile_param   = parameters[4];   // mu (beam) or f (isotropic)

  std::vector<double> tau_ir(n, 0.0);
  for (size_t i = 0; i < n; ++i)
    tau_ir[i] = kappa_ir * atmosphere.pressure[i] * 1e6 / surface_gravity;   // bar -> dyn cm^-2

  if (profile == 0)
    profileBeamSource(
      tau_ir, temperature_irr, temperature_int, profile_param, gamma,
      atmosphere.temperature);
  else
    profileIsotropicSource(
      tau_ir, temperature_irr, temperature_int, profile_param, gamma,
      atmosphere.temperature);

  // floor against unphysically cold analytic values (matches BeAR)
  for (auto& t : atmosphere.temperature)
    if (t < 50.0) t = 50.0;
}


void GuillotTemperature::profileBeamSource(
  const std::vector<double>& optical_depth,
  const double temperature_irr,
  const double temperature_int,
  const double mu,
  const double gamma,
  std::vector<double>& temperature)
{
  for (size_t i = 0; i < temperature.size(); ++i)
  {
    temperature[i] = 3.0/4.0 * std::pow(temperature_int, 4) * (2.0/3.0 + optical_depth[i])
                   + 3.0/4.0 * std::pow(temperature_irr, 4) * mu
                   * (2.0/3.0 + mu/gamma + (gamma/(3.0 * mu) - mu/gamma)
                      * std::exp(-gamma*optical_depth[i]/mu));

    temperature[i] = std::pow(temperature[i], 0.25);
  }
}


void GuillotTemperature::profileIsotropicSource(
  const std::vector<double>& optical_depth,
  const double temperature_irr,
  const double temperature_int,
  const double flux_distribution,
  const double gamma,
  std::vector<double>& temperature)
{
  for (size_t i = 0; i < temperature.size(); ++i)
  {
    temperature[i] = 3.0/4.0 * std::pow(temperature_int, 4) * (2.0/3.0 + optical_depth[i])
                   + 3.0/4.0 * std::pow(temperature_irr, 4) * flux_distribution
                   * (2.0/3.0 + 1.0/gamma/std::sqrt(3.0) + (gamma/std::sqrt(3.0) - 1.0/gamma/std::sqrt(3.0))
                     * std::exp(-gamma*optical_depth[i]*std::sqrt(3.0)));

    temperature[i] = std::pow(temperature[i], 0.25);
  }
}


}
