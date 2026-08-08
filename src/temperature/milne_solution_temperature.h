#ifndef _milne_solution_temperature_h
#define _milne_solution_temperature_h

#include <iostream>
#include <vector>

#include "temperature.h"
#include "../atmosphere/atmosphere.h"


namespace ngam {


class MilneTemperature : public TemperatureProfile{
  public:
    MilneTemperature() {
        nb_parameters = 2;
        std::cout << "\n- Temperature profile: Milne's solution\n\n";
      }
    virtual ~MilneTemperature() {}
    virtual void calcProfile(
      const std::vector<double>& parameters,
      const double surface_gravity,
      Atmosphere& atmosphere);
  private:
    //fit coefficients for the Hopf function
    const std::vector<double> fit_p {0.6162, -0.3799, 2.395, -2.041, 2.578};
    const std::vector<double> fit_q {-0.9799, 3.917, -3.17, 3.69};

    double hopfFunction(const double optical_depth);
};


}
#endif
