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

    // Saturation vapor pressure of H2O [bar]
    // Murphy & Koop (2005, QJRMS 131:1539): liquid branch T > 273.15 K, ice branch T <= 273.15 K
    static double saturationVaporPressure(double temperature);

    // d(ln e_s)/dT [K^-1] — analytic derivative of Murphy & Koop formula
    static double dLnSatVaporPressure_dT(double temperature);

    // Latent heat of H2O [erg/g] from Clausius-Clapeyron applied to Murphy & Koop:
    // L = dLnSatVaporPressure_dT * (R_gas / M_H2O) * T^2
    static double latentHeat(double temperature);

    // Moist adiabatic gradient d ln T / d ln P, using NDIV=10 sublevel integration.
    // pressure in bar. Falls back to adiabaticGradient() when e_s >= pressure.
    static double moistAdiabaticGradient(
      const std::vector<double>& number_densities,
      double temperature,
      double pressure);

  private:
    static double evaluateCpOverR(const std::array<double, 7>& coeffs, double T);
    static bool isMonatomic(chemical_species_id species);
};


}

#endif
