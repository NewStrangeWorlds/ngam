#include "helios_temperature.h"
#include "../atmosphere/atmosphere.h"
#include "../radiative_transfer/radiative_transfer.h"
#include "../transport_coeff/opacity_calc.h"
#include "../additional/aux_functions.h"
#include "../additional/quadrature.h"

#include <vector>
#include <cmath>
#include <algorithm>


namespace ngam {


void HeliosTemperature::calcCorrection(
  const double /*surface_gravity*/,
  Atmosphere& atmosphere,
  const RadiativeTransferOutput& radiation_field,
  const OpacityCalculation& opacity)
{
  const size_t n = atmosphere.temperature.size();
  const std::vector<double>& p = atmosphere.pressure;
  const std::vector<double>& F = radiation_field.flux_total;
  std::vector<double>& T = atmosphere.temperature;

  if (step.size() != n)
  {
    step.assign(n, step_init);
    T_store = T;
    iteration = 0;
  }

  const bool anchored = (target_flux > 0);

  // dimensionless flux imbalance per level: the net-flux difference across the level (HELIOS'
  // layer flux divergence), positive = heating. Level 0 of an internally heated object is the BOA
  // ghost layer, driven by the deep-flux anchoring residual.
  std::vector<double> r(n);

  for (size_t i = 0; i < n; ++i)
  {
    if (i == 0 && anchored)
    {
      r[i] = (target_flux - F[0]) / flux_scale;
      continue;
    }

    // net flux difference across the level, positive = heating (F grows with pressure)
    double dF = 0;

    if (stencil == "centered")
    {
      double dp = 0;
      if (i == 0)          dp = std::abs(p[1] - p[0]);
      else if (i == n - 1) dp = std::abs(p[n-1] - p[n-2]);
      else                 dp = 0.5 * std::abs(p[i+1] - p[i-1]);

      dF = radiation_field.flux_divergence[i] * dp;
    }
    else if (stencil == "forward")
      dF = (i < n - 1) ? F[i] - F[i+1] : F[n-2] - F[n-1];
    else  // backward
      dF = (i > 0) ? F[i-1] - F[i] : F[0] - F[1];

    r[i] = dF / flux_scale;
  }

  // HELIOS' layer residual in collocated form: the local radiative heating of the level's slab,
  // 4 pi Delta z_i int kappa_nu (J_nu - B_nu) dnu, positive = heating (replaces the flux stencil)
  if (residual_type == "heating")
  {
    const std::vector<double>& wn = radiation_field.spectral_grid->wavenumber_list;
    const std::vector<double>& z = atmosphere.altitude;
    const size_t nb_spectral = wn.size();

    #pragma omp parallel for
    for (size_t i = (anchored ? 1 : 0); i < n; ++i)
    {
      std::vector<double> integrand(nb_spectral);

      for (size_t nu = 0; nu < nb_spectral; ++nu)
        integrand[nu] = opacity.absorption_coeff[nu][i]
          * (radiation_field.mean_intensity[i][nu] - aux::planckFunctionWavenumber(T[i], wn[nu]));

      double dz = 0;
      if (i == 0)          dz = std::abs(z[1] - z[0]);
      else if (i == n - 1) dz = std::abs(z[n-1] - z[n-2]);
      else                 dz = 0.5 * std::abs(z[i+1] - z[i-1]);

      r[i] = 4.0 * M_PI * dz * aux::quadratureTrapezoidal(wn, integrand) / flux_scale;
    }
  }

  // HELIOS bookkeeping: remember the profile at the start of each adaptation interval, and at its
  // end judge every level by the drift it made against the drift a monotonic march would have made
  const bool store_now = (iteration % adapt_interval == 0);
  const bool adapt_now = (iteration % adapt_interval == adapt_interval - 1);

  if (store_now) T_store = T;

  for (size_t i = 0; i < n; ++i)
  {
    double dT = 0;

    if (r[i] != 0)
      dT = step[i] * std::copysign(std::pow(std::abs(r[i]), step_exponent), r[i]);

    if (std::abs(dT) > max_step)
      dT = std::copysign(max_step, dT);

    if (adapt_now)
    {
      const double drift = std::abs(T[i] - T_store[i]);

      if (drift < 0.5 * adapt_interval * std::abs(dT))
        step[i] /= step_shrink;    // back-and-forth: oscillating, shrink
      else
        step[i] *= step_grow;      // monotonic progress: grow
    }

    T[i] += dT;

    if (T[i] < 1.0) T[i] = 1.0;
  }

  // HELIOS' local radiative-equilibrium criterion over the radiative levels
  residual = 0;

  for (size_t i = 0; i < n; ++i)
  {
    if (atmosphere.convective[i] == 1) continue;

    residual = std::max(residual, std::abs(F[i] - target_flux) / flux_scale);
  }

  ++iteration;
}


}
