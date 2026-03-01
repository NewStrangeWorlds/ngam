/*
* This file is part of the BeAR code (https://github.com/newstrangeworlds/BeAR).
* Copyright (C) 2024 Daniel Kitzmann
*
* BeAR is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* BeAR is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You find a copy of the GNU General Public License in the main
* BeAR directory under <LICENSE>. If not, see
* <http://www.gnu.org/licenses/>.
*/


#ifndef SPECIES_DEFINITION_H
#define SPECIES_DEFINITION_H

#include "opacity_species.h"

#include <vector>
#include <iostream>
#include <array>

#include "../chemistry/chem_species.h"
#include "../additional/physical_const.h"
#include "../spectral_grid/spectral_grid.h"


namespace ngam{


class SpectralGrid;


class GasGeneric : public OpacitySpecies {
  public:
    GasGeneric(
      const std::string& cross_section_path, 
      SpectralGrid* spectral_grid_ptr, 
      const unsigned int index, 
      const std::string name, 
      const std::string folder) 
        : OpacitySpecies(index, name, folder) 
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          init();
        }
    GasGeneric(
      const std::string& cross_section_path, 
      SpectralGrid* spectral_grid_ptr, 
      const unsigned int index, 
      const std::string name, 
      const std::string folder, 
      const size_t reference_species) 
        : OpacitySpecies(index, name, folder) 
        {
          cross_section_file_path = cross_section_path; spectral_grid = spectral_grid_ptr; 
          pressure_reference_species = reference_species; 
          init();
        }
    GasGeneric(
      const std::string& cross_section_path, 
      SpectralGrid* spectral_grid_ptr, 
      const unsigned int index, 
      const std::string name, 
      const std::string folder, 
      const std::vector<size_t> cia_collision_species) 
        : OpacitySpecies(index, name, folder)
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          cia_collision_partner = cia_collision_species; 
          init();
        }
    virtual ~GasGeneric() {}
};


class GasHm : public OpacitySpecies {
  public:
    GasHm(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr) 
        : OpacitySpecies(_Hm, "H-", "Continuum")
        { 
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          continuum_available = true;
          init();
        }
    virtual ~GasHm() {}
  protected:
    virtual bool calcContinuumAbsorption(
      const double temperature,
      const std::vector<double>& number_densities,
      std::vector<double>& absorption_coeff);
    private:
      std::vector<double> boundFreeAbsorption(const double temperature);
      std::vector<double> freeFreeAbsorption(const double temperature);
};


class GasHRayleigh : public OpacitySpecies {
  public:
    GasHRayleigh(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr, const std::string folder) 
        : OpacitySpecies(_H, "H Rayleigh", "Rayleigh")
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          rayleigh_available = true;
          init();
        }
    GasHRayleigh(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr) 
        : OpacitySpecies(_H, "H Rayleigh", "Rayleigh")
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          rayleigh_available = true;
          init();
        }
    virtual ~GasHRayleigh() {}
  protected:
    virtual bool calcRayleighCrossSections(std::vector<double>& cross_sections);
};


class GasH2Rayleigh : public OpacitySpecies {
  public:
    GasH2Rayleigh(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr, const std::string folder) 
        : OpacitySpecies(_H2, "H2 Rayleigh", "Rayleigh")
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          rayleigh_available = true;
          init();
        }
    GasH2Rayleigh(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr) 
        : OpacitySpecies(_H2, "H2 Rayleigh", "Rayleigh")
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          rayleigh_available = true;
          init();
        }
    virtual ~GasH2Rayleigh() {}
  protected:
    virtual bool calcRayleighCrossSections(std::vector<double>& cross_sections);
};


class GasHeRayleigh : public OpacitySpecies {
  public:
    GasHeRayleigh(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr, const std::string folder)
        : OpacitySpecies(_He, "He Rayleigh", "Rayleigh")
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          rayleigh_available = true;
          init();
        }
    GasHeRayleigh(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr) 
        : OpacitySpecies(_He, "He Rayleigh", "Rayleigh")
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          rayleigh_available = true;
          init();
        }
    virtual ~GasHeRayleigh() {}
  protected:
    virtual bool calcRayleighCrossSections(std::vector<double>& cross_sections);
};


class GasCORayleigh : public OpacitySpecies {
  public:
    GasCORayleigh(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr, const std::string folder)
        : OpacitySpecies(_CO, "CO Rayleigh", "Rayleigh")
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          rayleigh_available = true;
          init();
        }
    GasCORayleigh(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr) 
        : OpacitySpecies(_CO, "CO Rayleigh", "Rayleigh")
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          rayleigh_available = true;
          init();
        }
    virtual ~GasCORayleigh() {}
  protected:
    virtual bool calcRayleighCrossSections(std::vector<double>& cross_sections);
};


class GasCO2Rayleigh : public OpacitySpecies {
  public:
    GasCO2Rayleigh(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr, const std::string folder)
        : OpacitySpecies(_CO2, "CO2 Rayleigh", "Rayleigh")
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          rayleigh_available = true;
          init();
        }
    GasCO2Rayleigh(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr) 
        : OpacitySpecies(_CO2, "CO2 Rayleigh", "Rayleigh")
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          rayleigh_available = true;
          init();
        }
    virtual ~GasCO2Rayleigh() {}
  protected:     
    virtual bool calcRayleighCrossSections(std::vector<double>& cross_sections);
};


class GasCH4Rayleigh : public OpacitySpecies {
  public:
    GasCH4Rayleigh(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr, const std::string folder)
        : OpacitySpecies(_CH4, "CH4 Rayleigh", "Rayleigh")
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          rayleigh_available = true;
          init();
        }
    GasCH4Rayleigh(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr) 
        : OpacitySpecies(_CH4, "CH4 Rayleigh", "Rayleigh")
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          rayleigh_available = true;
          init();
        }
    virtual ~GasCH4Rayleigh() {}
  protected:
    virtual bool calcRayleighCrossSections(std::vector<double>& cross_sections);
};



class GasH2ORayleigh : public OpacitySpecies {
  public:
    GasH2ORayleigh(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr, const std::string folder)
        : OpacitySpecies(_H2O, "H2O Rayleigh", "Rayleigh")
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          rayleigh_available = true;
          init();
        }
    GasH2ORayleigh(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr) 
        : OpacitySpecies(_H2O, "H2O Rayleigh", "Rayleigh")
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          rayleigh_available = true;
          init();
        }
    virtual ~GasH2ORayleigh() {}
  protected:
    virtual bool calcRayleighCrossSections(std::vector<double>& cross_sections);
};


class GasN2Rayleigh : public OpacitySpecies {
  public:
    GasN2Rayleigh(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr, const std::string folder)
        : OpacitySpecies(_N2, "N2 Rayleigh", "Rayleigh")
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          rayleigh_available = true;
          init();
        }
    GasN2Rayleigh(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr) 
        : OpacitySpecies(_N2, "N2 Rayleigh", "Rayleigh")
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          rayleigh_available = true;
          init();
        }
    virtual ~GasN2Rayleigh() {}
  protected:
    virtual bool calcRayleighCrossSections(std::vector<double>& cross_sections);
};


class GasO2Rayleigh : public OpacitySpecies {
  public:
    GasO2Rayleigh(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr, const std::string folder)
        : OpacitySpecies(_O2, "O2 Rayleigh", "Rayleigh")
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          rayleigh_available = true;
          init();
        }
    GasO2Rayleigh(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr) 
        : OpacitySpecies(_O2, "O2 Rayleigh", "Rayleigh")
        {
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          rayleigh_available = true;
          init();
        }
    virtual ~GasO2Rayleigh() {}
  protected:
    virtual bool calcRayleighCrossSections(std::vector<double>& cross_sections);
};


class GasCO2CIA : public OpacitySpecies {
  public:
    GasCO2CIA(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr) 
        : OpacitySpecies(_CO2, "CO2-CO2 CIA", "Continuum")
        { 
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          continuum_available = true;

          const std::string& dimer_data_path = "src/transport_coeff/data/CO2_dimer_data.bin";
          loadDimerData(dimer_data_path);
        
          init();
        }
    virtual ~GasCO2CIA() {}
  protected:
    virtual bool calcContinuumAbsorption(
      const double temperature,
      const std::vector<double>& number_densities,
      std::vector<double>& absorption_coeff);
    private:
      static constexpr int nS_ = 1713;
      static constexpr int nT_ = 9;

      bool data_loaded_ = false;
      std::array<double, nS_> wn_arr_{};
      std::array<double, nT_> temp_arr_{};
      std::array<double, nS_ * nT_> dim_arr_{};  // column-major (FORTRAN order)
      void loadDimerData(const std::string& filepath);

      // Main entry point: compute CIA coefficient (cm^{-1}).
      //   T  : temperature (K)
      //   nu : wavenumber (cm^{-1})
      //   Ps : partial pressure of CO2 (bar)
      double compute(double T, double nu, double Ps) const;

      // Analytical CIA for 0-500 cm^{-1}.  Returns cm^{-1} amagat^{-2}.
      static double getspc(double temp, double wn);

      // Data-driven CIA dimer spectrum for 1000-2000 cm^{-1}.
      // Returns cm^{-1} amagat^{-2}.  Requires dimer data to be loaded.
      double baranov(double temp, double wn) const;
        std::vector<double> boundFreeAbsorption(const double temperature);
        std::vector<double> freeFreeAbsorption(const double temperature);

      // Modified Bessel function K1(x) * x.
      // Precision better than 2.2e-7 (Abramowitz & Stegun, p.379).
      static double xk1(double x);

      // Bilinear interpolation on an irregular 2D grid (column-major storage).
      static double bilinear(
        const double* x_arr, const double* y_arr,
        int nX, int nY, const double* f2d_arr,
        double x, double y);
};


class GasH2OCIA : public OpacitySpecies {
  public:
    GasH2OCIA(const std::string& cross_section_path, SpectralGrid* spectral_grid_ptr) 
        : OpacitySpecies(_H2O, "H2O CIA", "Continuum")
        { 
          cross_section_file_path = cross_section_path; 
          spectral_grid = spectral_grid_ptr; 
          continuum_available = true;

          const std::string& dimer_data_path = "src/transport_coeff/data/mt_ckd_h2o_continuum.dat";
          loadContinuumData(dimer_data_path);
        
          init();
        }
    virtual ~GasH2OCIA() {}
  protected:
    virtual bool calcContinuumAbsorption(
      const double temperature,
      const std::vector<double>& number_densities,
      std::vector<double>& absorption_coeff);
    private:
      static constexpr int t_ref = 296; //reference temperature for the CIA data (K)
      static constexpr int p_ref = 1013 * 0.001; //reference pressure for the CIA data (bar)

      bool data_loaded_ = false;

      std::vector<double> self_continuum_reference;
      std::vector<double> foreign_continuum_reference;
      std::vector<double> self_temp_exp_reference;

      void loadContinuumData(const std::string& filepath);
      std::vector<double> radiationTerm(const double temperature);
};



}

#endif
