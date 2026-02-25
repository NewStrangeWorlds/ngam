#include "species_definition.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include "../additional/physical_const.h"


namespace ngam{


bool GasH2OCIA::calcContinuumAbsorption(
  const double temperature, 
  const std::vector<double>& number_densities, 
  std::vector<double>& absorption_coeff)
{
  std::vector<double> radiation_term = this->radiationTerm(temperature);
  
  if (number_densities[_H2O] == 0) return false;

  const double vmr_h2o = number_densities[_H2O] / number_densities[_TOTAL];
  const double vmr_rest = 1.0 - vmr_h2o;

  //pressure in bar
  const double pressure = number_densities[_TOTAL] * constants::boltzmann_k * temperature * 1e-6; 
  const double density_ratio = (pressure/p_ref) * (t_ref/temperature);
  
  #pragma omp parallel for
  for (size_t i = 0; i < spectral_grid->nbSpectralPoints(); ++i)
  {
    // Apply temperature dependence to reference water vapor self continuum coefficients
    // and scale to given density.
    const double self_continuum = self_continuum_reference[i] * 
      std::pow(t_ref/temperature, self_temp_exp_reference[i]) * vmr_h2o * density_ratio;

    // Compute water vapor foreign continuum absorption coefficient.
    const double foreign_continuum = foreign_continuum_reference[i] * vmr_rest * density_ratio;

    absorption_coeff[i] += (self_continuum + foreign_continuum) 
      * radiation_term[i] * number_densities[_H2O];
  }

  return true;
}


void GasH2OCIA::loadContinuumData(const std::string& filepath)
{
  std::ifstream file(filepath);

  if (!file) {
    throw std::runtime_error(
      "GasH2OCIA::loadContinuumData: could not open file: " + filepath);
  }

  std::vector<double> cia_wavenumbers_;
  std::vector<double> self_continuum_;
  std::vector<double> self_temp_exp_;
  std::vector<double> foreign_continuum_;

  // skip the header line
  std::string header;
  std::getline(file, header);

  double wavenumber, self_cont, self_temp, for_cont, for_cont_alt;

  while (file >> wavenumber >> self_cont >> self_temp >> for_cont >> for_cont_alt)
  {
    cia_wavenumbers_.push_back(wavenumber);
    self_continuum_.push_back(self_cont);
    self_temp_exp_.push_back(self_temp);
    foreign_continuum_.push_back(for_cont);
  }

  self_continuum_reference = spectral_grid->interpolateToWavenumberGrid(
    cia_wavenumbers_, self_continuum_, true);

  self_temp_exp_reference = spectral_grid->interpolateToWavenumberGrid(
    cia_wavenumbers_, self_temp_exp_, false);

  foreign_continuum_reference = spectral_grid->interpolateToWavenumberGrid(
    cia_wavenumbers_, foreign_continuum_, true);

  data_loaded_ = true;
}


// FUNCTION RADFN CALCULATES THE RADIATION TERM FOR THE LINE SHAPE
// Converted from the MT_CKD FORTRAN version 4.2
std::vector<double> GasH2OCIA::radiationTerm(
  const double temperature)
{ 
  //second radiation constant in cm·K
  constexpr double rad_const2 = 1.4387752;
  //temperature in cm^{-1} 
  const double xkt = temperature / rad_const2; // xkt = h*c/(k*T) in cm^{-1}

  std::vector<double> radiation_term(spectral_grid->nbSpectralPoints(), 0.0);
  
  // Note: IN THE SMALL XVIOKT REGION 0.5 IS REQUIRED
  #pragma omp parallel for
  for (size_t i = 0; i < spectral_grid->nbSpectralPoints(); ++i) 
  {
    const double nu = spectral_grid->wavenumber_list[i];
    double xviokt = nu / xkt;

    if (xviokt <= 0.01f) {
      radiation_term[i] = 0.5f * xviokt * nu;
    } 
    else if (xviokt <= 10.0f) {
      const double expvkt = std::exp(-xviokt);
      radiation_term[i] = nu * (1.0f - expvkt) / (1.0f + expvkt);
    } 
    else {
      // Default: rad = xvi
      radiation_term[i] = nu;
    }
  }

  return radiation_term;
}


}