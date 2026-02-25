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


#ifndef _discrete_ordinate_h
#define _discrete_ordinate_h


#include <vector>
#include <iostream>
#include <cmath>
#include <functional>

#include "radiative_transfer.h"
#include "../atmosphere/atmosphere.h"
#include "../spectral_grid/spectral_grid.h"


#include "../../_deps/disortpp-src/src/DisortFluxConfig.hpp"
#include "../../_deps/disortpp-src/src/FluxResult.hpp"
#include "../../_deps/disortpp-src/src/FluxSolver.hpp"


namespace ngam {


class DiscreteOrdinates : public RadiativeTransfer{
  public:
    DiscreteOrdinates(
      SpectralGrid* spectral_grid_ptr,
      const size_t nb_streams,
      const size_t nb_grid_points);
    virtual ~DiscreteOrdinates() {}
    
    virtual void calculate(
      const Atmosphere& atmosphere,
      const OpacityCalculation& opacity,
      RadiativeTransferOutput& output,
      const RadiativeBoundaryConditions& bc = RadiativeBoundaryConditions{}) override;
  private:
    size_t nb_streams = 0;
    size_t nb_grid_points = 0;
    size_t nb_layers = 0;

    int nb_threads = 0;

    //std::vector<disort_state> ds;
    //std::vector<disort_output> out;
    std::vector<disortpp::DisortFluxConfig> configs;
    std::vector<std::function<disortpp::FluxResult(const disortpp::DisortFluxConfig&)>> solvers;

    void calculate(
      const OpacityCalculation& opacity,
      const std::vector<double>& vertical_grid,
      const double surface_albedo,
      const bool has_surface,
      const double incident_radiation,
      const double zenith_angle,
      const size_t nu_index,
      const double max_temperature,
      RadiativeTransferOutput& output);
    void calcTotalTransportCoeff(
      const OpacityCalculation& opacity,
      const std::vector<double>& vertical_grid, 
      const size_t nu_index,
      std::vector<double>& optical_depth,
      std::vector<double>& single_scattering_albedo,
      std::vector<double>& asymmetry_parameter);
    void setTemperatureStructure(
      const std::vector<double>& temperature_structure,
      const double& surface_temperature);
    void setDISORTParam(
      const int thread_id,
      const double wavenumber_input,
      const std::vector<double>& optical_depth,
      const std::vector<double>& single_scattering_albedo,
      const std::vector<double>& asymmetry_parameter,
      const double incident_radiation,
      const double zenith_angle,
      const double surface_albedo,
      const bool has_surface);
    void initDISORT();
};


}
#endif


