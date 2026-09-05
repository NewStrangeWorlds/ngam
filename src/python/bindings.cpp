/*
* This file is part of the ngam code.
* Copyright (C) 2026 Daniel Kitzmann
*
* ngam is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

// The compiled core of the `pyngam` package. The pure-Python layer (pyngam/__init__.py) wraps
// these classes to record the configuration for provenance and to add file-based helpers; the
// constructor and initialize() signatures here ARE the model configuration schema.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../config/module_params.h"
#include "../spectral_grid/spectral_grid.h"
#include "../atmosphere/atmosphere.h"
#include "../chemistry/chem_species.h"
#include "../radiative_transfer/radiative_transfer.h"
#include "../object/generic_object.h"
#include "../object/brown_dwarf.h"
#include "../object/terrestrial_planet.h"
#include "../object/gas_planet.h"

namespace py = pybind11;
using namespace ngam;


// ---- module specs from Python objects -------------------------------------------------------
//
// A module spec can be written as
//   "mlt_dry"                                a bare type name
//   ("mlt_dry", {"alpha": 1.0})              a (type, parameters) tuple
//   {"type": "mlt_dry", "alpha": 1.0}        a dict with a "type" key (the config-file form)
// Parameter values are passed to C++ as strings and converted/validated by the module selector.

static std::string paramToString(const py::handle& value)
{
  if (py::isinstance<py::bool_>(value))
    return value.cast<bool>() ? "true" : "false";
  if (py::isinstance<py::str>(value))
    return value.cast<std::string>();
  if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value))
  {
    std::string joined;
    for (const auto& item : value)
      joined += (joined.empty() ? "" : ",") + paramToString(item);
    return joined;
  }
  return py::str(value).cast<std::string>();   // int, float, numpy scalars, ...
}


static ModuleParams paramsFromDict(const py::dict& dict, const std::string& context)
{
  ModuleParams params;
  for (const auto& item : dict)
  {
    if (!py::isinstance<py::str>(item.first))
      throw py::type_error(context + ": parameter names must be strings");
    params[item.first.cast<std::string>()] = paramToString(item.second);
  }
  return params;
}


static ModuleSpec toModuleSpec(const py::handle& obj, const std::string& context)
{
  if (py::isinstance<py::str>(obj))
    return ModuleSpec(obj.cast<std::string>());

  if (py::isinstance<py::dict>(obj))
  {
    py::dict dict = py::reinterpret_borrow<py::dict>(obj);
    if (!dict.contains("type"))
      throw py::type_error(context + ": a module given as a dict needs a 'type' entry");
    ModuleSpec spec(paramToString(dict["type"]));
    py::dict rest;
    for (const auto& item : dict)
      if (!(py::isinstance<py::str>(item.first) && item.first.cast<std::string>() == "type"))
        rest[item.first] = item.second;
    spec.params = paramsFromDict(rest, context);
    return spec;
  }

  if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj))
  {
    py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
    if (seq.size() == 0 || seq.size() > 2 || !py::isinstance<py::str>(seq[0]))
      throw py::type_error(context + ": expected (type, {parameters})");
    ModuleSpec spec(seq[0].cast<std::string>());
    if (seq.size() == 2)
    {
      if (!py::isinstance<py::dict>(seq[1]))
        throw py::type_error(context + ": module parameters must be a dict");
      spec.params = paramsFromDict(py::reinterpret_borrow<py::dict>(seq[1]), context);
    }
    return spec;
  }

  throw py::type_error(
    context + ": expected a type name, (type, {parameters}) or {'type': ..., parameters}");
}


// a LIST is a sequence of specs; anything else (str, tuple, dict) is a single spec
static std::vector<ModuleSpec> toModuleSpecs(const py::handle& obj, const std::string& context)
{
  std::vector<ModuleSpec> specs;
  if (obj.is_none()) return specs;

  if (py::isinstance<py::list>(obj))
    for (const auto& item : obj) specs.push_back(toModuleSpec(item, context));
  else
    specs.push_back(toModuleSpec(obj, context));

  return specs;
}


static ModuleSpec specOrDefault(
  const py::object& obj, const ModuleSpec& default_spec, const std::string& context)
{
  return obj.is_none() ? default_spec : toModuleSpec(obj, context);
}


// ---- shared model configuration ---------------------------------------------------------------

static ModelConfig makeModelConfig(
  SpectralGrid& grid,
  const size_t nb_grid_points,
  const std::vector<double>& boundary_pressures,
  const std::string& opacity_path,
  const std::vector<std::pair<std::string, std::string>>& opacity_species,
  const bool use_clouds,
  const py::object& chemistry,
  const py::object& radiative_transfer,
  const py::object& convection,
  const py::object& solver)
{
  ModelConfig config;
  config.nb_grid_points = nb_grid_points;
  config.boundary_pressures = boundary_pressures;
  config.opacity_path = opacity_path.empty() ? grid.crossSectionFilePath() : opacity_path;
  config.opacity_species = opacity_species;
  config.use_clouds = use_clouds;
  config.chemistry = toModuleSpecs(chemistry, "chemistry");
  config.radiative_transfer = specOrDefault(
    radiative_transfer, ModelConfig().radiative_transfer, "radiative_transfer");
  config.convection = specOrDefault(convection, ModelConfig().convection, "convection");
  config.solver = specOrDefault(solver, ModelConfig().solver, "solver");
  return config;
}

// the shared keyword arguments, in the order they appear in every constructor
#define NGAM_MODEL_CONFIG_ARGS \
  py::arg("nb_grid_points") = 100, \
  py::arg("boundary_pressures") = std::vector<double>{1e2, 1e-6}, \
  py::arg("opacity_path") = "", \
  py::arg("opacity_species") = std::vector<std::pair<std::string, std::string>>{}, \
  py::arg("use_clouds") = false, \
  py::arg("chemistry"), \
  py::arg("radiative_transfer") = py::none(), \
  py::arg("convection") = py::none(), \
  py::arg("solver") = py::none()

static const char* model_config_doc =
  "Shared configuration (keyword-only):\n"
  "  nb_grid_points      number of pressure levels (default 100)\n"
  "  boundary_pressures  [bottom, top] in bar\n"
  "  opacity_path        cross-section data root (default: the grid's)\n"
  "  opacity_species     list of (species symbol, data folder) pairs\n"
  "  use_clouds          include cloud opacity (default False)\n"
  "  chemistry           list of chemistry module specs, applied in order:\n"
  "                        ('equilibrium', {parameter_file, metallicity=1, c_to_o=0.5})\n"
  "                        ('isoprofile', {symbol: mixing ratio, ...})\n"
  "                        ('fixed', {file})\n"
  "                        ('manabe_wetherald', {surface_rh=0.77})\n"
  "  radiative_transfer  ('disort', {nb_streams=4}) [default] or\n"
  "                      ('adding_doubling', {nb_streams=2})\n"
  "  convection          'mlt_dry' [default] / 'mlt_moist' ({alpha=1, min_pressure}),\n"
  "                      'dry' / 'moist' ({min_pressure, max_sweeps=10}), 'none'\n"
  "  solver              'ratio_ul' [default] / 'flux_divergence' / 'ptc' /\n"
  "                      'time_stepping' / 'time_stepping_lre' / 'helios', with\n"
  "                      {max_iterations=100, convergence_threshold=1e-4} for all,\n"
  "                      {gamma, ng_interval, max_change, lre_fraction} for the\n"
  "                      relaxation schemes (max_change also for ptc), and\n"
  "                      {ng_interval=0, step_init=10, step_grow=1.1, step_shrink=1.5,\n"
  "                      adapt_interval=20, step_exponent=0.1, max_step=500,\n"
  "                      stencil='backward'|'centered'|'forward',\n"
  "                      residual='flux'|'heating'} for helios\n"
  "A module spec is a type name, a (type, {parameters}) tuple, or a dict with a 'type' key.\n";


static std::unique_ptr<BrownDwarf> make_brown_dwarf(
  SpectralGrid& grid,
  double effective_temperature,
  double surface_gravity,
  double radius,
  bool variable_gravity,
  size_t nb_grid_points,
  const std::vector<double>& boundary_pressures,
  const std::string& opacity_path,
  const std::vector<std::pair<std::string, std::string>>& opacity_species,
  bool use_clouds,
  const py::object& chemistry,
  const py::object& radiative_transfer,
  const py::object& convection,
  const py::object& solver)
{
  const ModelConfig config = makeModelConfig(
    grid, nb_grid_points, boundary_pressures, opacity_path, opacity_species, use_clouds,
    chemistry, radiative_transfer, convection, solver);

  return std::make_unique<BrownDwarf>(
    &grid, config, effective_temperature, surface_gravity, radius, variable_gravity);
}


static std::unique_ptr<GasPlanet> make_gas_planet(
  SpectralGrid& grid,
  double internal_temperature,
  double surface_gravity,
  double instellation,
  double zenith_angle,
  const py::object& stellar_spectrum,
  double radius,
  bool variable_gravity,
  size_t nb_grid_points,
  const std::vector<double>& boundary_pressures,
  const std::string& opacity_path,
  const std::vector<std::pair<std::string, std::string>>& opacity_species,
  bool use_clouds,
  const py::object& chemistry,
  const py::object& radiative_transfer,
  const py::object& convection,
  const py::object& solver)
{
  const ModelConfig config = makeModelConfig(
    grid, nb_grid_points, boundary_pressures, opacity_path, opacity_species, use_clouds,
    chemistry, radiative_transfer, convection, solver);

  return std::make_unique<GasPlanet>(
    &grid, config, internal_temperature, surface_gravity, radius, variable_gravity,
    instellation, zenith_angle, toModuleSpec(stellar_spectrum, "stellar_spectrum"));
}


static std::unique_ptr<TerrestrialPlanet> make_terrestrial_planet(
  SpectralGrid& grid,
  double surface_gravity,
  double instellation,
  double zenith_angle,
  const py::object& stellar_spectrum,
  const py::object& surface,
  size_t nb_grid_points,
  const std::vector<double>& boundary_pressures,
  const std::string& opacity_path,
  const std::vector<std::pair<std::string, std::string>>& opacity_species,
  bool use_clouds,
  const py::object& chemistry,
  const py::object& radiative_transfer,
  const py::object& convection,
  const py::object& solver)
{
  const ModelConfig config = makeModelConfig(
    grid, nb_grid_points, boundary_pressures, opacity_path, opacity_species, use_clouds,
    chemistry, radiative_transfer, convection, solver);

  return std::make_unique<TerrestrialPlanet>(
    &grid, config, surface_gravity, instellation, zenith_angle,
    toModuleSpec(stellar_spectrum, "stellar_spectrum"), toModuleSpec(surface, "surface"));
}


static std::vector<std::string> get_species_symbols()
{
  std::vector<std::string> symbols;
  symbols.reserve(constants::species_data.size());
  for (auto& s : constants::species_data)
    symbols.push_back(s.symbol);
  return symbols;
}


// the methods every object class shares
template <class Object>
static void defineObjectMethods(py::class_<Object>& cls)
{
  cls
    .def("initialize",
      [](Object& self, const py::object& profile, const py::object& chemistry) {
        self.initialize(
          toModuleSpec(profile, "initial_profile"), toModuleSpecs(chemistry, "chemistry"));
      },
      py::arg("profile"), py::arg("chemistry") = py::none(),
      "Initialize from an analytic temperature profile plus chemistry.\n"
      "profile: ('adiabat', {surface_temperature, stratosphere_temperature}),\n"
      "         ('milne', {kappa_ross, effective_temperature=<object's>}),\n"
      "         ('const', {temperature}),\n"
      "         ('guillot', {kappa_ir, t_irr, gamma, t_int=<object's>, mode='isotropic', f=0.25\n"
      "                      | mode='beam', mu=<zenith angle>})\n"
      "chemistry: module specs for the initial composition (default: the model's own).")
    .def("initialize_from_arrays", &Object::initializeFromArrays,
      py::arg("temperature"),
      py::arg("number_densities"),
      py::arg("mean_molecular_weight"),
      "Initialize from saved temperature, number densities, and mean molecular weight arrays")
    .def("compute", &Object::computeAtmosphericStructure,
      "Iterate to radiative-convective equilibrium; returns True on convergence")
    .def_readonly("radiation_field", &Object::radiation_field)
    .def_property_readonly("atmosphere", &Object::getAtmosphere);
}


PYBIND11_MODULE(_pyngam, m) {
    m.doc() = "ngam compiled core -- use the pyngam package";

    m.def("species_symbols", &get_species_symbols,
        "Return the list of all chemical species symbols (indexed by species ID)");

    // ---- SpectralGrid ----
    // Raw constructors; the pyngam.SpectralGrid factories (constant_resolution, covering, ...)
    // are the intended interface.
    py::class_<SpectralGrid>(m, "SpectralGrid", py::dynamic_attr())
        .def(py::init<
            const std::string&,
            const std::string&,
            unsigned int,
            double,
            double,
            double>(),
            py::arg("opacity_path"),
            py::arg("wavenumber_file"),
            py::arg("discretisation"),
            py::arg("step"),
            py::arg("wavelength_min"),
            py::arg("wavelength_max"),
            "discretisation: 0 constant wavenumber step, 1 constant wavelength step,\n"
            "2 constant resolving power; `step` is the step or the resolution R")
        .def(py::init<
            const std::string&,
            const std::string&,
            unsigned int,
            double,
            double,
            double,
            double,
            double,
            unsigned int,
            size_t,
            double,
            size_t>(),
            py::arg("opacity_path"),
            py::arg("wavenumber_file"),
            py::arg("discretisation"),
            py::arg("step"),
            py::arg("wavelength_min"),
            py::arg("wavelength_max"),
            py::arg("temperature_min"),
            py::arg("temperature_max"),
            py::arg("nb_temperatures"),
            py::arg("nb_points"),
            py::arg("stellar_temperature"),
            py::arg("nb_points_stellar"),
            "discretisation 3: composite-Planck covering distribution")
        .def_property_readonly("opacity_path",
            [](SpectralGrid& g) { return g.crossSectionFilePath(); })
        .def_readonly("wavenumber_list", &SpectralGrid::wavenumber_list)
        .def_readonly("wavelength_list", &SpectralGrid::wavelength_list)
        .def_readonly("nb_spectral_points", &SpectralGrid::nb_spectral_points)
        .def_readonly("nb_spectral_points_full", &SpectralGrid::nb_spectral_points_full)
        .def("wavelength_to_wavenumber",
            py::overload_cast<const double>(&SpectralGrid::wavelengthToWavenumber))
        .def("wavenumber_to_wavelength",
            py::overload_cast<const double>(&SpectralGrid::wavenumberToWavelength));

    // ---- Atmosphere (read-only view) ----
    py::class_<Atmosphere>(m, "Atmosphere")
        .def_readonly("nb_grid_points", &Atmosphere::nb_grid_points)
        .def_readonly("pressure", &Atmosphere::pressure)
        .def_readonly("temperature", &Atmosphere::temperature)
        .def_readonly("altitude", &Atmosphere::altitude)
        .def_readonly("scale_height", &Atmosphere::scale_height)
        .def_readonly("mass_density", &Atmosphere::mass_density)
        .def_readonly("mean_molecular_weight", &Atmosphere::mean_molecular_weight)
        .def_readonly("number_densities", &Atmosphere::number_densities)
        .def_readonly("convective", &Atmosphere::convective);

    // ---- RadiativeTransferOutput ----
    py::class_<RadiativeTransferOutput>(m, "RadiativeTransferOutput")
        .def_readonly("spectrum", &RadiativeTransferOutput::spectrum)
        .def_readonly("flux_total", &RadiativeTransferOutput::flux_total)
        .def_readonly("flux_up_total", &RadiativeTransferOutput::flux_up_total)
        .def_readonly("flux_down_total", &RadiativeTransferOutput::flux_down_total)
        .def_readonly("mean_intensity_total", &RadiativeTransferOutput::mean_intensity_total)
        .def_readonly("flux", &RadiativeTransferOutput::flux)
        .def_readonly("flux_up", &RadiativeTransferOutput::flux_up)
        .def_readonly("flux_down", &RadiativeTransferOutput::flux_down)
        .def_readonly("mean_intensity", &RadiativeTransferOutput::mean_intensity)
        .def_readonly("flux_divergence", &RadiativeTransferOutput::flux_divergence)
        .def_readonly("net_heating", &RadiativeTransferOutput::net_heating)
        .def_readonly("net_flux_jacobian", &RadiativeTransferOutput::net_flux_jacobian)
        .def_readonly("net_heating_jacobian", &RadiativeTransferOutput::net_heating_jacobian);

    // ---- BrownDwarf ----
    py::class_<BrownDwarf> brown_dwarf(m, "BrownDwarf", py::dynamic_attr());
    brown_dwarf.def(py::init(&make_brown_dwarf),
        py::arg("grid"),
        py::kw_only(),
        py::arg("effective_temperature"),
        py::arg("surface_gravity"),
        py::arg("radius") = 0.0,
        py::arg("variable_gravity") = false,
        NGAM_MODEL_CONFIG_ARGS,
        py::keep_alive<1, 2>(),
        "Self-luminous brown dwarf.\n"
        "  effective_temperature  K; sets the internal flux sigma T_eff^4\n"
        "  surface_gravity        cm/s^2\n"
        "  radius                 cm, bottom-of-domain radius (used with variable_gravity)\n"
        "  variable_gravity       gravity varies with altitude (default False)\n");
    defineObjectMethods(brown_dwarf);

    // ---- GasPlanet ----
    py::class_<GasPlanet> gas_planet(m, "GasPlanet", py::dynamic_attr());
    gas_planet.def(py::init(&make_gas_planet),
        py::arg("grid"),
        py::kw_only(),
        py::arg("internal_temperature"),
        py::arg("surface_gravity"),
        py::arg("instellation"),
        py::arg("zenith_angle"),
        py::arg("stellar_spectrum"),
        py::arg("radius") = 0.0,
        py::arg("variable_gravity") = false,
        NGAM_MODEL_CONFIG_ARGS,
        py::keep_alive<1, 2>(),
        "Irradiated gas planet without a surface.\n"
        "  internal_temperature   K; sets the internal flux sigma T_int^4\n"
        "  surface_gravity        cm/s^2\n"
        "  instellation           incident stellar flux at the planet, erg/cm^2/s\n"
        "  zenith_angle           cosine of the stellar zenith angle\n"
        "  stellar_spectrum       ('tabulated', {file}) or ('blackbody', {temperature})\n"
        "  radius, variable_gravity   as for BrownDwarf\n");
    defineObjectMethods(gas_planet);

    // ---- TerrestrialPlanet ----
    py::class_<TerrestrialPlanet> terrestrial_planet(m, "TerrestrialPlanet", py::dynamic_attr());
    terrestrial_planet.def(py::init(&make_terrestrial_planet),
        py::arg("grid"),
        py::kw_only(),
        py::arg("surface_gravity"),
        py::arg("instellation"),
        py::arg("zenith_angle"),
        py::arg("stellar_spectrum"),
        py::arg("surface"),
        NGAM_MODEL_CONFIG_ARGS,
        py::keep_alive<1, 2>(),
        "Irradiated planet with a solid surface.\n"
        "  surface_gravity        cm/s^2\n"
        "  instellation           incident stellar flux at the planet, erg/cm^2/s\n"
        "  zenith_angle           cosine of the stellar zenith angle\n"
        "  stellar_spectrum       ('tabulated', {file}) or ('blackbody', {temperature})\n"
        "  surface                'blackbody', ('simple', {albedo, wavelength_switch}) or\n"
        "                         ('variable_albedo', {file})\n");
    defineObjectMethods(terrestrial_planet);
    terrestrial_planet
        .def("eval_forward", &TerrestrialPlanet::evalForward,
            py::arg("temperature"), py::arg("compute_jac"), py::arg("recompute_opacity") = true,
            "Diagnostic: one forward solve at the given T; returns net flux per level. With "
            "compute_jac, also fills radiation_field.net_flux_jacobian / net_heating_jacobian.")
        .def_property_readonly("surface_temperature", &TerrestrialPlanet::getSurfaceTemperature);

    m.attr("model_config_doc") = model_config_doc;
}
