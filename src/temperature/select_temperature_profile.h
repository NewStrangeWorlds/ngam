#ifndef _select_temperature_h
#define _select_temperature_h

#include <vector>
#include <string>
#include <algorithm>
#include <memory>

#include "temperature.h"

#include "../additional/exceptions.h"

#include "milne_solution_temperature.h"
#include "constant_temperature.h"
#include "guillot_temperature.h"


namespace ngam {

//definition of the different chemistry modules with an
//identifier, a keyword to be located in the config file and a short version of the keyword
namespace temp_profile_modules{
  enum id {milne, constant, guillot};
  const std::vector<std::string> description {"milne", "const", "guillot"};
}



inline std::unique_ptr<TemperatureProfile> selectTemperatureProfile(
  const std::string profile_type,
  const std::vector<std::string>& parameters)
{
  //find the corresponding radiative transfer module to the supplied type string
  auto it = std::find(
    temp_profile_modules::description.begin(),
    temp_profile_modules::description.end(),
    profile_type);


  //no module is found
  if (it == temp_profile_modules::description.end())
  {
    std::string error_message = "Temperature profile type " + profile_type + " unknown!\n";
    throw InvalidInput(std::string ("forward_model.config"), error_message);
  }


  //get the id of the chosen module
  temp_profile_modules::id module_id = static_cast<temp_profile_modules::id>(
    std::distance(temp_profile_modules::description.begin(), it));


  switch (module_id)
  {
    case temp_profile_modules::milne :
      return std::make_unique<MilneTemperature>();

    case temp_profile_modules::constant :
      return std::make_unique<ConstantTemperature>();

    // Guillot (2010) irradiated-analytic profile; temperature_config[0] selects "beam" or
    // "isotropic" (default). The recommended init for irradiated gas planets.
    case temp_profile_modules::guillot :
      return std::make_unique<GuillotTemperature>(
        parameters.empty() ? std::string("isotropic") : parameters[0]);
  }

  return nullptr;
}


}
#endif

