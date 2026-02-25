
#include <iostream>
#include <memory>

#include "config/global_config.h"
#include "spectral_grid/spectral_grid.h"

#include "chemistry/select_chemistry.h"
#include "radiative_transfer/select_radiative_transfer.h"

#include "object/brown_dwarf.h"


int main(int argc, char *argv[])
{
  ngam::GlobalConfig config("browndwarf", "/media/data/opacity_data/helios-k/", "const_resolution", 1000.0);

  ngam::SpectralGrid spectral_grid(
    config.cross_section_file_path,
    config.wavenumber_file_path,
    config.spectral_discretisation,
    config.spectral_resolution,
    0.3, 100);

  // create chemistry modules
  std::vector<std::unique_ptr<ngam::Chemistry>> chemistry(config.chemistry_model.size());

  for (size_t i=0; i<config.chemistry_model.size(); ++i)
    chemistry[i] = ngam::selectChemistryModule(
      config.chemistry_model[i],
      config.chemistry_parameters[i]);

  // create radiative transfer solver
  auto radiative_transfer = ngam::selectRadiativeTransfer(
    config.radiative_transfer_type,
    config.radiative_transfer_parameters,
    config.nb_grid_points,
    &spectral_grid);

  // fundamental model parameters
  double effective_temperature = 3000.0;
  double metallicity = 0.5;
  double c_to_o_ratio = 0.5;

  // sub-model parameters
  std::vector<double> temperature_parameters{1e-4, effective_temperature};  // kappa_ross, T_eff (for Milne init)

  // assemble the model
  ngam::BrownDwarf brown_dwarf(
    &spectral_grid,
    config.nb_grid_points,
    config.atmos_boundary_pressures,
    config.cross_section_file_path,
    config.opacity_species_symbol,
    config.opacity_species_folder,
    !config.cloud_model.empty(),
    std::move(chemistry),
    std::move(radiative_transfer),
    config.surface_gravity,
    effective_temperature,
    metallicity,
    c_to_o_ratio,
    config.bottom_radius,
    config.use_variable_gravity);

  // initialize from analytic profile + chemistry
  brown_dwarf.initialize(
    config.temperature_profile_type,
    config.temperature_profile_parameters,
    temperature_parameters,
    {{"eq", {"fastchem_parameters.dat"}}},
    {metallicity, c_to_o_ratio});

  brown_dwarf.computeAtmosphericStructure();

  return 0;
}
