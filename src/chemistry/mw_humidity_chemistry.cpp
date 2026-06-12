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

  for (size_t i = 0; i < number_densities.size(); ++i)
  {
    const double p     = pressure[i];
    const double T     = temperature[i];
    const double n_tot = number_densities[i][_TOTAL];

    // Manabe & Wetherald (1967) Eq. (2): linear in p/p_s, zero above p/p_s = 0.02
    const double p_ratio = p / p_s;
    double rh = rh0 * (p_ratio - 0.02) / (1.0 - 0.02);

    if (rh < 0.08) rh = 0.08;

    // H2O VMR at saturation (Murphy & Koop 2005), scaled by local RH
    const double e_s    = ThermodynamicData::saturationVaporPressure(T);
    double vmr_h2o = rh * e_s / p;
    vmr_h2o = std::max(0.0, std::min(vmr_h2o, 0.99));

    number_densities[i][_H2O] = n_tot * vmr_h2o;
  }

  meanMolecularWeight(number_densities, mean_molecular_weight);
  return false;
}


}
