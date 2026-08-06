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


#include <iostream>
#include <string>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "spectral_grid.h"

#include "../additional/exceptions.h"
#include "../additional/aux_functions.h"


namespace ngam{


SpectralGrid::SpectralGrid(
  const std::string& cross_section_file_path_,
  const std::string& wavenumber_file_path_,
  unsigned int spectral_discretisation_,
  double spectral_resolution_)
  : cross_section_file_path(cross_section_file_path_),
    wavenumber_file_path(wavenumber_file_path_),
    spectral_discretisation(spectral_discretisation_),
    spectral_resolution(spectral_resolution_)
{
  loadWavenumberList();

  wavelength_list_full = wavenumberToWavelength(wavenumber_list_full);
}


SpectralGrid::SpectralGrid(
  const std::string& cross_section_file_path_,
  const std::string& wavenumber_file_path_,
  unsigned int spectral_discretisation_,
  double spectral_resolution_,
  double wavelength_min,
  double wavelength_max)
  : cross_section_file_path(cross_section_file_path_),
    wavenumber_file_path(wavenumber_file_path_),
    spectral_discretisation(spectral_discretisation_),
    spectral_resolution(spectral_resolution_)
{
  loadWavenumberList();

  wavelength_list_full = wavenumberToWavelength(wavenumber_list_full);

  std::vector<std::vector<double>> wavenumber_edges =
    {{wavelengthToWavenumber(wavelength_max),
      wavelengthToWavenumber(wavelength_min)}};

  std::vector<std::vector<size_t>> edge_indices;
  findBinEdges(wavenumber_edges, edge_indices);

  createHighResGrid(edge_indices);
}


SpectralGrid::SpectralGrid(
  const std::string& cross_section_file_path_,
  const std::string& wavenumber_file_path_,
  unsigned int spectral_discretisation_,
  double spectral_resolution_,
  double wavelength_min,
  double wavelength_max,
  double cov_temperature_min_,
  double cov_temperature_max_,
  unsigned int cov_nb_temperatures_,
  size_t target_nb_points_,
  double cov_stellar_temperature_,
  size_t target_nb_points_stellar_)
  : cross_section_file_path(cross_section_file_path_),
    wavenumber_file_path(wavenumber_file_path_),
    spectral_discretisation(spectral_discretisation_),
    spectral_resolution(spectral_resolution_),
    cov_temperature_min(cov_temperature_min_),
    cov_temperature_max(cov_temperature_max_),
    cov_nb_temperatures(cov_nb_temperatures_),
    target_nb_points(target_nb_points_),
    cov_stellar_temperature(cov_stellar_temperature_),
    target_nb_points_stellar(target_nb_points_stellar_)
{
  loadWavenumberList();

  wavelength_list_full = wavenumberToWavelength(wavenumber_list_full);

  std::vector<std::vector<double>> wavenumber_edges =
    {{wavelengthToWavenumber(wavelength_max),
      wavelengthToWavenumber(wavelength_min)}};

  std::vector<std::vector<size_t>> edge_indices;
  findBinEdges(wavenumber_edges, edge_indices);

  createHighResGrid(edge_indices);
}


void SpectralGrid::createHeliosWavenumberList()
{
  const double wavenumber_step = 0.01;
  const double max_wavenumber = 90000.0;

  wavenumber_list_full.assign(max_wavenumber/wavenumber_step+1, 0.);

  for (size_t i=0; i<wavenumber_list_full.size(); ++i)
    wavenumber_list_full[i] = i * wavenumber_step;
}


void SpectralGrid::loadWavenumberList()
{
  std::string file_name = wavenumber_file_path;


  std::fstream file;

  file.open(file_name.c_str(), std::ios::in);


  if (file.fail())
  {
    std::cout << "Did not find wavenumber list file at: " << file_name << "\n";
    std::cout << "Assuming now that the standard HELIOS-k grid is used!\n\n";

    createHeliosWavenumberList();

    return;
  }


  size_t nb_wavenumbers;
  file >> nb_wavenumbers;

  wavenumber_list_full.resize(nb_wavenumbers);


  for (std::vector<double>::iterator it = wavenumber_list_full.begin(); it != wavenumber_list_full.end(); ++it)
    file >> *it;


  file.close();
}



void SpectralGrid::createHighResGridConstWavenumber(
  const std::vector<std::vector<size_t>>& edge_indices,
  std::vector<int>& included_points)
{ 
  for (size_t i=0; i<edge_indices.size(); ++i)
  {
    included_points[edge_indices[i][0]] = 1;

    size_t last_index = edge_indices[i][0];

    for (size_t j=edge_indices[i][0]; j<edge_indices[i][1]; ++j)
    { 
      const double next_wavenumber = wavenumber_list_full[last_index] + spectral_resolution;
      
      if (next_wavenumber == wavenumber_list_full[j] || wavenumber_list_full[j+1] > next_wavenumber)
      { 
        included_points[j] = 1;
        last_index = j;
      }
      
      included_points[edge_indices[i][1]] = 1;
    }
  }
}


void SpectralGrid::createHighResGridConstWavelength(
  const std::vector<std::vector<size_t>>& edge_indices,
  std::vector<int>& included_points)
{
  for (size_t i=0; i<edge_indices.size(); ++i)
  {
    included_points[edge_indices[i][0]] = 1;

    size_t last_index = edge_indices[i][0];

    for (size_t j=edge_indices[i][0]; j<edge_indices[i][1]; ++j)
    {
      const double next_wavelength = wavelength_list_full[last_index] - spectral_resolution;

      if (next_wavelength == wavelength_list_full[j] || wavelength_list_full[j+1] < next_wavelength)
      { 
        included_points[j] = 1;
        last_index = j;
      }
      
      included_points[edge_indices[i][1]] = 1;
    }
  }
}


void SpectralGrid::createHighResGridConstResolution(
  const std::vector<std::vector<size_t>>& edge_indices,
  std::vector<int>& included_points)
{
  for (size_t i=0; i<edge_indices.size(); ++i)
  {
    included_points[edge_indices[i][0]] = 1;

    size_t last_index = edge_indices[i][0];

    for (size_t j=edge_indices[i][0]; j<edge_indices[i][1]; ++j)
    { 
      const double next_wavelength = wavelength_list_full[last_index] * (1 - 1./spectral_resolution);

      if (next_wavelength == wavelength_list_full[j] || wavelength_list_full[j+1] < next_wavelength)
      { 
        included_points[j] = 1;
        last_index = j;
      }
      
      included_points[edge_indices[i][1]] = 1;
    }
  }
}


//Composite-Planck "covering" distribution (Helling & Jorgensen 1998, A&A 337, 477).
//Point density is proportional to the covering curve, i.e. the pointwise maximum over a
//set of normalised Planck energy densities spanning the atmospheric temperature range,
//so that more points are placed where the radiation field carries energy. The thermal
//(atmospheric) covering and the stellar irradiation covering are selected independently
//with their own point budgets and unioned. Implemented as a selector over the native
//grid, exactly like the constant-step modes, so opacity sampling (direct indexing) is
//unaffected.

//Set of covering temperatures spanning [cov_temperature_min, cov_temperature_max]
//(Helling-style ~500 K steps by default).
std::vector<double> SpectralGrid::coveringTemperatures() const
{
  unsigned int nb_temperatures = cov_nb_temperatures;

  if (nb_temperatures == 0)
  {
    const double step = 500.0;
    nb_temperatures = static_cast<unsigned int>(
      std::round((cov_temperature_max - cov_temperature_min)/step)) + 1;
  }

  if (nb_temperatures < 2) nb_temperatures = 2;

  std::vector<double> temperatures;

  for (unsigned int k=0; k<nb_temperatures; ++k)
    temperatures.push_back(
      cov_temperature_min
      + (cov_temperature_max - cov_temperature_min) * k / (nb_temperatures - 1.0));

  return temperatures;
}


//covering(nu) = max_k  B_nu(T_k) / T_k^4  over the given temperatures (Helling & Jorgensen
//eq. 2, up to the universal constant pi/sigma that cancels in the maximum and the threshold)
std::vector<double> SpectralGrid::computeCoveringCurve(
  const std::vector<std::vector<size_t>>& edge_indices,
  const std::vector<double>& temperatures) const
{
  std::vector<double> covering(wavenumber_list_full.size(), 0.0);

  for (size_t i=0; i<edge_indices.size(); ++i)
  {
    for (size_t j=edge_indices[i][0]; j<=edge_indices[i][1]; ++j)
    {
      double max_value = 0.0;

      for (size_t k=0; k<temperatures.size(); ++k)
      {
        const double t = temperatures[k];
        const double e =
          aux::planckFunctionWavenumber(t, wavenumber_list_full[j]) / (t*t*t*t);

        if (e > max_value) max_value = e;
      }

      covering[j] = max_value;
    }
  }

  return covering;
}


//Mark native points such that the covering measure accumulated since the last marked
//point reaches 'threshold' (the discretised form of Helling's recursion
//nu_{i+1} = nu_i + const/E_nu). Bin edges are always kept. Returns the number of points.
size_t SpectralGrid::markCovering(
  const std::vector<std::vector<size_t>>& edge_indices,
  const std::vector<double>& covering,
  const double threshold,
  std::vector<int>& included_points)
{
  std::fill(included_points.begin(), included_points.end(), 0);

  size_t count = 0;

  for (size_t i=0; i<edge_indices.size(); ++i)
  {
    const size_t lo = edge_indices[i][0];
    const size_t hi = edge_indices[i][1];

    if (included_points[lo] == 0) { included_points[lo] = 1; ++count; }

    double acc = 0.0;

    for (size_t j=lo+1; j<hi; ++j)
    {
      acc += 0.5 * (covering[j] + covering[j-1])
        * (wavenumber_list_full[j] - wavenumber_list_full[j-1]);

      if (acc >= threshold)
      {
        included_points[j] = 1;
        ++count;
        acc = 0.0;
      }
    }

    if (included_points[hi] == 0) { included_points[hi] = 1; ++count; }
  }

  return count;
}


//Select ~target_points native points distributed according to the given covering curve,
//by bisecting the marking threshold (the selected count decreases monotonically with it).
void SpectralGrid::selectByCovering(
  const std::vector<std::vector<size_t>>& edge_indices,
  const std::vector<double>& covering,
  const size_t target_points,
  std::vector<int>& included_points)
{
  double total_measure = 0.0;
  size_t nb_native = 0;

  for (size_t i=0; i<edge_indices.size(); ++i)
  {
    nb_native += edge_indices[i][1] - edge_indices[i][0] + 1;

    for (size_t j=edge_indices[i][0]+1; j<=edge_indices[i][1]; ++j)
      total_measure += 0.5 * (covering[j] + covering[j-1])
        * (wavenumber_list_full[j] - wavenumber_list_full[j-1]);
  }

  size_t target = target_points;
  if (target < 2) target = 2;
  if (target > nb_native) target = nb_native;

  std::vector<int> scratch(wavenumber_list_full.size(), 0);

  double threshold_hi = total_measure;             //high threshold -> few points
  double threshold_lo = total_measure / nb_native; //low threshold  -> ~all points
  if (threshold_lo <= 0.0) threshold_lo = threshold_hi * 1e-12;

  double threshold = total_measure / target;       //initial guess

  for (unsigned int iter=0; iter<60; ++iter)
  {
    const size_t count = markCovering(edge_indices, covering, threshold, scratch);

    if (count == target) break;

    if (count > target)
      threshold_lo = threshold;   //too many points -> raise the threshold
    else
      threshold_hi = threshold;   //too few points  -> lower the threshold

    threshold = std::sqrt(threshold_lo * threshold_hi);  //geometric bisection
  }

  markCovering(edge_indices, covering, threshold, included_points);
}


void SpectralGrid::createHighResGridPlanckCovering(
  const std::vector<std::vector<size_t>>& edge_indices,
  std::vector<int>& included_points)
{
  //thermal (atmospheric) covering -> target_nb_points
  const std::vector<double> cov_thermal =
    computeCoveringCurve(edge_indices, coveringTemperatures());
  selectByCovering(edge_indices, cov_thermal, target_nb_points, included_points);

  //stellar irradiation covering -> target_nb_points_stellar (only when irradiated);
  //selected independently and unioned with the thermal points
  if (cov_stellar_temperature > 0.0 && target_nb_points_stellar > 0)
  {
    const std::vector<double> cov_stellar =
      computeCoveringCurve(edge_indices, {cov_stellar_temperature});

    std::vector<int> stellar_points(wavenumber_list_full.size(), 0);
    selectByCovering(edge_indices, cov_stellar, target_nb_points_stellar, stellar_points);

    for (size_t i=0; i<included_points.size(); ++i)
      if (stellar_points[i] == 1) included_points[i] = 1;
  }
}


void SpectralGrid::createHighResGrid(
  const std::vector<std::vector<size_t>>& edge_indices)
{
  std::vector<int> included_points(wavenumber_list_full.size(), 0);

  if (spectral_discretisation == 0)
    createHighResGridConstWavenumber(edge_indices, included_points);

  if (spectral_discretisation == 1)
    createHighResGridConstWavelength(edge_indices, included_points);

  if (spectral_discretisation == 2)
    createHighResGridConstResolution(edge_indices, included_points);

  if (spectral_discretisation == 3)
    createHighResGridPlanckCovering(edge_indices, included_points);


  index_list.resize(0);
  index_list.reserve(wavelength_list_full.size());

  for (size_t i=0; i<included_points.size(); ++i)
  {
    if (included_points[i] == 1)
      index_list.push_back(i);
  }


  index_list.shrink_to_fit();

  wavenumber_list.assign(index_list.size(), 0);
  wavelength_list.assign(index_list.size(), 0);

  for (size_t i=0; i<index_list.size(); ++i)
  {
    wavenumber_list[i] = wavenumber_list_full[index_list[i]];
    wavelength_list[i] = wavelength_list_full[index_list[i]];
  }

  nb_spectral_points = wavelength_list.size();
}



size_t SpectralGrid::findClosestIndexAsc(
  const double x,
  std::vector<double>& data,
  std::vector<double>::iterator start)
{
  auto iter_geq = std::lower_bound(
    start, 
    data.end(),
    x);

  if (iter_geq == start)
    return start - data.begin();

  double a = *(iter_geq - 1);
  double b = *(iter_geq);

  if (std::fabs(x - a) < fabs(x - b)) 
    return iter_geq - data.begin() - 1;

  return iter_geq - data.begin();
}


size_t SpectralGrid::findClosestIndexDesc(
  const double x,
  std::vector<double>& data,
  std::vector<double>::iterator start)
{
  auto iter_geq = std::lower_bound(
    start, 
    data.end(),
    x,
    std::greater<double>());

  if (iter_geq == start)
    return start - data.begin();

  double a = *(iter_geq - 1);
  double b = *(iter_geq);

  if (std::fabs(x - a) < fabs(x - b)) 
    return iter_geq - data.begin() - 1;

  return iter_geq - data.begin();
}


size_t SpectralGrid::findClosestIndex(
  const double x,
  std::vector<double>& data,
  std::vector<double>::iterator start)
{
  if (data.size() < 2)
    return 0;
  
  if (data[0] < data[1])
    return findClosestIndexAsc(x, data, start);

  if (data[0] > data[1])
    return findClosestIndexDesc(x, data, start);

  return 0;
}



void SpectralGrid::findBinEdges(
  const std::vector< std::vector<double> >& wavenumber_edges,
  std::vector<std::vector<size_t>>& edge_indices)
{
  edge_indices.assign(wavenumber_edges.size(), std::vector<size_t>{0, 0});

  for (size_t i=0; i<wavenumber_edges.size(); ++i)
  {
    auto it_left = std::lower_bound(
      wavenumber_list_full.begin(), 
      wavenumber_list_full.end(), 
      wavenumber_edges[i][0]);

    auto it_right = std::lower_bound(
      wavenumber_list_full.begin(), 
      wavenumber_list_full.end(), 
      wavenumber_edges[i][1]);

    edge_indices[i][0] = it_left - wavenumber_list_full.begin();

    if (wavenumber_list_full[edge_indices[i][0]] > wavenumber_edges[i][0])
      edge_indices[i][0] -= 1;

    edge_indices[i][1] = it_right - wavenumber_list_full.begin();
  }

}


/*std::vector<double> SpectralGrid::wavenumberList(const std::vector<size_t>& indices)
{
  std::vector<double> output(indices.size(), 0.0);

  for (size_t i=0; i<indices.size(); ++i)
    output[i] = wavenumber_list[indices[i]];

  return output;
}



std::vector<double> SpectralGrid::wavelengthList(const std::vector<size_t>& indices)
{
  std::vector<double> output(indices.size(), 0.0);

  for (size_t i=0; i<indices.size(); ++i)
    output[i] = wavelength_list[indices[i]];

  return output;
}*/



SpectralGrid::~SpectralGrid()
{ 

}


}






