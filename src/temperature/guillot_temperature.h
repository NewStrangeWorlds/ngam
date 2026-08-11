#ifndef _guillot_temperature_h
#define _guillot_temperature_h

#include <iostream>
#include <vector>
#include <string>

#include "temperature.h"
#include "../atmosphere/atmosphere.h"


namespace ngam {


// Guillot (2010) irradiated-analytic temperature profile (ported from BeAR's
// GuillotTemperature). The recommended initial profile for irradiated gas planets: it starts
// both the quasi-isothermal deep and the upper irradiation-dominated region near the answer,
// where Milne (self-luminous) is qualitatively wrong and isothermal-at-T_eq starts the deep
// ~hundreds of K low.
//
// parameters: [kappa_ir (cm^2/g), T_irr (K), T_int (K), gamma (= kappa_vis/kappa_ir), mu-or-f]
//   profile_type "beam"      (temperature_config[0]): collimated irradiation, last parameter mu
//   profile_type "isotropic" (default):               averaged irradiation, last parameter f
//                                                     (flux distribution: 1/4 global, 1/2 dayside)
class GuillotTemperature : public TemperatureProfile{
  public:
    GuillotTemperature(const std::string profile_type)
    {
      nb_parameters = 5;

      if (profile_type == "beam")
        profile = 0;
      else if (profile_type == "isotropic" || profile_type.empty())
        profile = 1;
      else
      {
        std::cout << "Guillot profile type " << profile_type
                  << " unknown, using isotropic!\n";
        profile = 1;
      }

      std::cout << "\n- Temperature profile: Guillot, "
                << (profile == 0 ? "beam" : "isotropic") << " source\n\n";
    }
    virtual ~GuillotTemperature() {}
    virtual void calcProfile(
      const std::vector<double>& parameters,
      const double surface_gravity,
      Atmosphere& atmosphere);
  private:
    unsigned int profile = 1;

    void profileBeamSource(
      const std::vector<double>& optical_depth,
      const double temperature_irr,
      const double temperature_int,
      const double mu,
      const double gamma,
      std::vector<double>& temperature);
    void profileIsotropicSource(
      const std::vector<double>& optical_depth,
      const double temperature_irr,
      const double temperature_int,
      const double flux_distribution,
      const double gamma,
      std::vector<double>& temperature);
};


}
#endif
