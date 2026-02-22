#ifndef _thermodynamic_data_h
#define _thermodynamic_data_h

#include <vector>
#include <array>

#include "../chemistry/chem_species.h"


namespace ngam {


struct NASACoefficients {
    double temp_low;
    double temp_mid;
    double temp_high;
    std::array<double, 7> low;   // coefficients for [temp_low, temp_mid]
    std::array<double, 7> high;  // coefficients for [temp_mid, temp_high]
};


class ThermodynamicData {
  public:
    // c_p/R for a single species (dimensionless)
    static double cpOverR(chemical_species_id species, double temperature);

    // c_p for a single species [erg/(mol·K)]
    static double heatCapacityMolar(chemical_species_id species, double temperature);

    // Mean c_p of the gas mixture [erg/(g·K)]
    // number_densities: vector indexed by chemical_species_id for a single level
    static double meanHeatCapacity(
      const std::vector<double>& number_densities,
      double temperature);

    // Adiabatic temperature gradient: d ln T / d ln P (dimensionless)
    static double adiabaticGradient(
      const std::vector<double>& number_densities,
      double temperature);

  private:
    static double evaluateCpOverR(const std::array<double, 7>& coeffs, double T);
    static bool isMonatomic(chemical_species_id species);
};


}

#endif
