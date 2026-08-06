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


#ifndef _adding_doubling_h
#define _adding_doubling_h


#include <vector>
#include <iostream>
#include <cmath>

#include "radiative_transfer.h"
#include "../atmosphere/atmosphere.h"
#include "../spectral_grid/spectral_grid.h"

// Provided by the AddingDoublingRT dependency (target "adding_doubling", whose
// PUBLIC include directory is propagated through ngam_lib).
#include "adding_doubling.hpp"
#include "workspace.hpp"


namespace ngam {


class AddingDoubling : public RadiativeTransfer{
  public:
    AddingDoubling(
      SpectralGrid* spectral_grid_ptr,
      const size_t num_quadrature,
      const size_t nb_grid_points);
    virtual ~AddingDoubling() {}

    virtual void calculate(
      const Atmosphere& atmosphere,
      const OpacityCalculation& opacity,
      RadiativeTransferOutput& output,
      const RadiativeBoundaryConditions& bc = RadiativeBoundaryConditions{}) override;
  private:
    size_t num_quadrature = 0;
    size_t nb_grid_points = 0;
    size_t nb_layers = 0;

    int nb_threads = 0;

    double surface_temperature = 0.0;
    // Value passed to the solver's surface_temperature: a positive value gives a DISTINCT surface
    // DOF; a negative value (the "folded" sentinel) makes the surface emit at the bottom-level
    // temperature and folds its emission into the bottom-level Jacobian column. We use the folded
    // form whenever the surface temperature is tied to the BOA temperature (the usual case, incl.
    // the RCE), so the single BOA knob does not silently drive a dropped surface column.
    double surface_temperature_config = -1.0;

    // one solver configuration and one (caller-owned, single-thread) workspace per
    // OpenMP thread; the adding-doubling solve() is a free function rather than a
    // templated solver object, so no per-stream dispatch is needed.
    std::vector<adrt::ADConfig> configs;
    std::vector<adrt::SolverWorkspace> workspaces;

    // scratch for the in-loop temperature-Jacobian accumulation (only used when
    // output.compute_jacobian is set): per-point trapezoidal weights and per-thread
    // accumulators reduced after the parallel spectral loop.
    std::vector<double> quad_weights;
    std::vector< std::vector<double> >                  heating_value_thread;       // [thread][i]
    std::vector< std::vector< std::vector<double> > >   jac_net_heating_thread;     // [thread][i][j]
    std::vector< std::vector< std::vector<double> > >   jac_net_flux_thread;        // [thread][i][j]
    std::vector< std::vector< std::vector<double> > >   jac_meanint_kappa_thread;   // [thread][i][j]

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
      const std::vector<double>& temperature_structure);
    void setSolverParam(
      const int thread_id,
      const double wavenumber_input,
      const std::vector<double>& optical_depth,
      const std::vector<double>& single_scattering_albedo,
      const std::vector<double>& asymmetry_parameter,
      const double incident_radiation,
      const double zenith_angle,
      const double surface_albedo,
      const bool has_surface);
    void initSolver();
};


}
#endif
