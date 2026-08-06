#ifndef _dry_adiabatic_h
#define _dry_adiabatic_h

#include <iostream>

#include "convection.h"
#include "../atmosphere/atmosphere.h"


namespace ngam {


class DryAdiabaticAdjustment : public Convection {
  public:
    DryAdiabaticAdjustment(size_t max_sweeps_ = 10, double min_pressure_ = 1e-4)
      : max_sweeps(max_sweeps_), min_pressure(min_pressure_)
    {
      std::cout << "\n- Convection: dry adiabatic adjustment"
                << " (min pressure: " << min_pressure << " bar)\n\n";
    }
    virtual ~DryAdiabaticAdjustment() {}
    void adjust(Atmosphere& atmosphere) const override;
    double convectiveGradient(
      const std::vector<double>& number_densities,
      double temperature,
      double /*pressure*/) const override;
  private:
    size_t max_sweeps;
    double min_pressure;
};


}

#endif
