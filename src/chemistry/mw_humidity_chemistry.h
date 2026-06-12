#ifndef _mw_humidity_chemistry_h
#define _mw_humidity_chemistry_h

#include "chemistry.h"

#include <vector>
#include <string>


namespace ngam {


// Manabe & Wetherald (1967, J. Atmos. Sci. 24:241) prescribed relative humidity profile.
// RH(p) = rh_surface * (p/p_s - 0.02) / (1 - 0.02)  for p/p_s > 0.02, else 0.
// H2O VMR at each level = RH * e_s(T) / P  using Murphy & Koop (2005) saturation pressure.
// Intended to run after a dry-composition chemistry module (e.g. "fixed") so that only H2O
// is overridden; all other species are left unchanged.
class ManabeWetherlandChemistry : public Chemistry {
  public:
    ManabeWetherlandChemistry(double rh_surface_ = 0.77)
      : rh_surface(rh_surface_)
    {
      nb_parameters = 1;
    }
    virtual ~ManabeWetherlandChemistry() {}

    virtual bool calcChemicalComposition(
      const std::vector<double>& parameters,
      const std::vector<double>& temperature,
      const std::vector<double>& pressure,
      std::vector<std::vector<double>>& number_densities,
      std::vector<double>& mean_molecular_weight);

  private:
    double rh_surface;
};


}
#endif
