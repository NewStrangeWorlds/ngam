#include "mw_humidity_chemistry.h"
#include "chem_species.h"
#include "../additional/thermodynamic_data.h"
#include "../additional/physical_const.h"

#include <algorithm>
#include <cmath>
#include <iostream>


namespace ngam {


bool ManabeWetherlandChemistry::calcChemicalComposition(
  const std::vector<double>& parameters,
  const std::vector<double>& temperature,
  const std::vector<double>& pressure,
  std::vector<std::vector<double>>& number_densities,
  std::vector<double>& mean_molecular_weight)
{
  const double rh0 = parameters.empty() ? rh_surface : parameters[0];
  const double p_s = pressure.front();  // surface pressure (bottom of atmosphere, index 0)

  // Build the H2O profile from the surface (index 0) upward, enforcing a COLD TRAP: the H2O volume
  // mixing ratio is not allowed to INCREASE with height. Above the cold point (the coldest/driest
  // level, ~ the tropopause) it is held at that minimum, so a warm (e.g. ozone-heated) stratosphere
  // cannot re-inflate the water via saturation. This is the discrete analogue of holding the
  // composition fixed above the tropopause as in the clima/atmos terrestrial models. Without it the
  // fixed-RH water-vapour feedback fills the stratosphere and drives an unphysical runaway greenhouse.
  double vmr_cold = 1.0;   // running minimum of the saturation VMR below (the cold-trap value)
  for (size_t i = 0; i < number_densities.size(); ++i)
  {
    const double p     = pressure[i];
    const double T     = temperature[i];
    const double n_tot = number_densities[i][_TOTAL];

    // Manabe & Wetherald (1967) Eq. (2): RH linear in p/p_s, reaching zero at p/p_s = 0.02 (a dry
    // stratosphere -- the parameterisation's own cold trap; it is NOT floored to a finite value).
    const double p_ratio = p / p_s;
    double rh = rh0 * (p_ratio - 0.02) / (1.0 - 0.02);
    if (rh < 0.0) rh = 0.0;

    // H2O VMR at saturation (Murphy & Koop 2005), scaled by local RH, capped at a physical maximum
    const double e_s = ThermodynamicData::saturationVaporPressure(T);
    double vmr_sat = rh * e_s / p;
    vmr_sat = std::max(0.0, std::min(vmr_sat, 0.99));

    // cold trap: water can only decrease upward (held at the coldest-point value above the trap)
    if (vmr_sat < vmr_cold) vmr_cold = vmr_sat;

    number_densities[i][_H2O] = n_tot * vmr_cold;
  }

  meanMolecularWeight(number_densities, mean_molecular_weight);
  return false;
}


}
