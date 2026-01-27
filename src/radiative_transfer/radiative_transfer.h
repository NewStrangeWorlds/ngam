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


#ifndef _radiative_transfer_h
#define _radiative_transfer_h

#include <vector>

#include "../atmosphere/atmosphere.h"
#include "../transport_coeff/opacity_calc.h"
#include "../spectral_grid/spectral_grid.h"


namespace bear {


struct RadiativeTransferOutput {
  RadiativeTransferOutput(size_t nb_spectral_points, size_t nb_grid_points) 
  {
    spectrum.assign(nb_spectral_points, 0.0);

    flux_total.assign(nb_grid_points, 0.0);
    flux_up_total.assign(nb_grid_points, 0.0);
    flux_down_total.assign(nb_grid_points, 0.0);
    mean_intensity_total.assign(nb_grid_points, 0.0);

    flux.assign(nb_spectral_points, std::vector<double>(nb_grid_points, 0.0));
    flux_up.assign(nb_spectral_points, std::vector<double>(nb_grid_points, 0.0));
    flux_down.assign(nb_spectral_points, std::vector<double>(nb_grid_points, 0.0));
    mean_intensity.assign(nb_spectral_points, std::vector<double>(nb_grid_points, 0.0));
  }
  
  std::vector<double> spectrum;

  std::vector<double> flux_total;
  std::vector<double> flux_up_total;
  std::vector<double> flux_down_total;
  std::vector<double> mean_intensity_total;
  
  std::vector< std::vector<double> > flux;
  std::vector< std::vector<double> > flux_up;
  std::vector< std::vector<double> > flux_down;
  std::vector< std::vector<double> > mean_intensity;
};



class RadiativeTransfer{
  public:
    RadiativeTransfer(SpectralGrid* spectral_grid_) {
      spectral_grid = spectral_grid_;};
    virtual ~RadiativeTransfer() {}
    virtual void calcSpectrum(
      const Atmosphere& atmosphere,
      const OpacityCalculation& opacity,
      RadiativeTransferOutput& output) = 0;
    
    //change units of high-res spectrum from cm to micron^-1
    void changeSpectrumUnits(std::vector<double>& spectrum) {
      for (size_t i=0; i<spectrum.size(); ++i)
        spectrum[i] = spectrum[i]/spectral_grid->wavelength_list[i]/spectral_grid->wavelength_list[i]*10000.0;};
  
  protected:
    SpectralGrid* spectral_grid = nullptr;
};


}
#endif

