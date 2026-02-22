#ifndef _dry_adiabatic_h
#define _dry_adiabatic_h

#include <iostream>

#include "convection.h"
#include "../atmosphere/atmosphere.h"


namespace ngam {


class DryAdiabaticAdjustment : public Convection {
  public:
    DryAdiabaticAdjustment(size_t max_sweeps_ = 10)
      : max_sweeps(max_sweeps_)
    {
      std::cout << "\n- Convection: dry adiabatic adjustment\n\n";
    }
    virtual ~DryAdiabaticAdjustment() {}
    void adjust(Atmosphere& atmosphere) override;
  private:
    size_t max_sweeps;
};


}

#endif
