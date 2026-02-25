#ifndef STELLAR_SPECTRUM_H
#define STELLAR_SPECTRUM_H

#include <vector>
#include <cmath>

#include "../additional/aux_functions.h"
#include "../additional/physical_const.h"
#include "../additional/quadrature.h"


namespace ngam {


class StellarSpectrum {
  public:
    virtual ~StellarSpectrum() {}
    virtual std::vector<double> calcFlux(
      const std::vector<double>& wavenumber_list) const = 0;
};


class BlackbodyStar : public StellarSpectrum {
  public:
    BlackbodyStar(double stellar_temperature, double instellation_flux)
      : stellar_temperature(stellar_temperature),
        instellation_flux(instellation_flux)
    {}

    // Returns spectral flux [erg/cm^2/s/cm^-1] at each wavenumber.
    // The spectrum is a Planck function scaled so that its integral
    // over wavenumber equals instellation_flux.
    std::vector<double> calcFlux(
      const std::vector<double>& wavenumber_list) const override
    {
      const size_t n = wavenumber_list.size();
      std::vector<double> flux(n);

      for (size_t i = 0; i < n; ++i)
        flux[i] = constants::pi
                * aux::planckFunctionWavenumber(stellar_temperature, wavenumber_list[i]);

      // integrate to get the bolometric flux of the raw Planck spectrum
      double bolometric = aux::quadratureTrapezoidal(wavenumber_list, flux);

      if (bolometric > 0)
      {
        double scale = instellation_flux / bolometric;
        for (size_t i = 0; i < n; ++i)
          flux[i] *= scale;
      }

      return flux;
    }

  private:
    double stellar_temperature;
    double instellation_flux;
};


} // namespace ngam

#endif // STELLAR_SPECTRUM_H
