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


#ifndef _spectral_grid_h
#define _spectral_grid_h

#include <vector>
#include <string>


namespace ngam {


class SpectralGrid{
  public:
    SpectralGrid(
      const std::string& cross_section_file_path_,
      const std::string& wavenumber_file_path_,
      unsigned int spectral_discretisation_,
      double spectral_resolution_);
    SpectralGrid(
      const std::string& cross_section_file_path_,
      const std::string& wavenumber_file_path_,
      unsigned int spectral_discretisation_,
      double spectral_resolution_,
      double wavelength_min,
      double wavelength_max);
    SpectralGrid(
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
      size_t target_nb_points_stellar_);
    ~SpectralGrid();

    std::vector<double> wavenumber_list;      //wavenumber list used to calculate the high-res spectra
    std::vector<double> wavelength_list;      //wavelength list used to calculate the high-res spectra

    std::vector<double> wavelengthToWavenumber(
      const std::vector<double>& wavelengths);
    std::vector<double> wavenumberToWavelength(
      const std::vector<double>& wavenumbers);

    double wavelengthToWavenumber(const double wavelength)
     {return 1.0/wavelength * 1e4;}
    double wavenumberToWavelength(const double wavenumber)
     {return 1.0/wavenumber * 1e4;}

    size_t nb_spectral_points_full;           //number of points in the global wavenumber list
    size_t nb_spectral_points;                //number of points in the spectral grid

    size_t nbSpectralPointsFull() {
      return nb_spectral_points_full;}
    size_t nbSpectralPoints() {
      return nb_spectral_points;}

    std::vector<size_t> spectralIndexList() {
      return index_list;}

    //band-closure correction (clima_rce_correction): fixed-width bands over the native grid.
    //The per-band native opacity integral minus its sampled representation closes the local
    //energy balance for the line cores the sampled list misses (they carry the Planck-mean
    //emissivity aloft; see doc). 30 cm^-1 keeps the kappa-B covariance error below ~1 K
    //equivalent at all pressures (measured against the direct native integral).
    static constexpr double correction_band_width = 30.0;   //cm^-1
    size_t nbCorrectionBands() const {
      return wavenumber_list_full.empty() ? 0
        : static_cast<size_t>(wavenumber_list_full.back()/correction_band_width) + 1;}
    const std::vector<double>& nativeWavenumbers() const {
      return wavenumber_list_full;}

    void findBinEdges(
      const std::vector< std::vector<double> >& wavenumber_edges,
      std::vector<std::vector<size_t>>& edge_indices);

    size_t findClosestIndex(
      const double search_value,
      std::vector<double>& data,
      std::vector<double>::iterator it_start);

    std::vector<double> interpolateToWavenumberGrid(
      const std::vector<double>& data_x,
      const std::vector<double>& data_y,
      const bool log_interpolation,
      const bool extrapolate=false);
    std::vector<double> interpolateToWavelengthGrid(
      const std::vector<double>& data_x,
      const std::vector<double>& data_y,
      const bool log_interpolation,
      const bool extrapolate=false);
    std::vector<double> interpolateToWavelengthGrid(
      const std::vector<double>& data_x,
      const std::vector<double>& data_y,
      const std::vector<double>& new_x,
      const bool log_interpolation,
      const bool extrapolate=false);
    std::string crossSectionFilePath() const {
      return cross_section_file_path;}

  private:
    std::string cross_section_file_path;
    std::string wavenumber_file_path;
    unsigned int spectral_discretisation;
    double spectral_resolution;

    //parameters for the composite-Planck covering distribution (spectral_discretisation == 3)
    double cov_temperature_min = 0.0;       //lower bound of the atmospheric covering temperature range
    double cov_temperature_max = 0.0;       //upper bound of the atmospheric covering temperature range
    unsigned int cov_nb_temperatures = 0;   //number of covering temperatures (0 -> internal ~500 K-step default)
    size_t target_nb_points = 0;            //desired number of points for the thermal (atmospheric) covering
    double cov_stellar_temperature = 0.0;   //host-star temperature for the stellar covering term (<= 0 -> omitted)
    size_t target_nb_points_stellar = 0;    //desired number of points for the stellar irradiation covering

    std::vector<double> wavenumber_list_full; //the full, global wavenumber list, the opacities have been calculated at
    std::vector<double> wavelength_list_full; //the full, global wavelength list, the opacities have been calculated at

    std::vector<size_t> index_list;
    std::vector<std::vector<double>> observation_wavelength_edges;

    void loadWavenumberList();
    void createHeliosWavenumberList();

    void createHighResGrid(
      const std::vector<std::vector<size_t>>& edge_indices);

    void createHighResGridConstWavenumber(
      const std::vector<std::vector<size_t>>& edge_indices,
      std::vector<int>& included_points);
    void createHighResGridConstWavelength(
      const std::vector<std::vector<size_t>>& edge_indices,
      std::vector<int>& included_points);
    void createHighResGridConstResolution(
      const std::vector<std::vector<size_t>>& edge_indices,
      std::vector<int>& included_points);
    void createHighResGridPlanckCovering(
      const std::vector<std::vector<size_t>>& edge_indices,
      std::vector<int>& included_points);

    std::vector<double> coveringTemperatures() const;
    std::vector<double> computeCoveringCurve(
      const std::vector<std::vector<size_t>>& edge_indices,
      const std::vector<double>& temperatures) const;
    void selectByCovering(
      const std::vector<std::vector<size_t>>& edge_indices,
      const std::vector<double>& covering,
      const size_t target_points,
      std::vector<int>& included_points);
    size_t markCovering(
      const std::vector<std::vector<size_t>>& edge_indices,
      const std::vector<double>& covering,
      const double threshold,
      std::vector<int>& included_points);

    size_t findClosestIndexDesc(
      const double search_value,
      std::vector<double>& data,
      std::vector<double>::iterator it_start);

    size_t findClosestIndexAsc(
      const double search_value,
      std::vector<double>& data,
      std::vector<double>::iterator it_start);
};



}


#endif
