#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../spectral_grid/spectral_grid.h"
#include "../atmosphere/atmosphere.h"
#include "../chemistry/chem_species.h"
#include "../chemistry/chemistry.h"
#include "../chemistry/select_chemistry.h"
#include "../temperature/temperature.h"
#include "../temperature/select_temperature_profile.h"
#include "../radiative_transfer/radiative_transfer.h"
#include "../radiative_transfer/select_radiative_transfer.h"
#include "../object/brown_dwarf.h"

namespace py = pybind11;
using namespace ngam;


// Factory: construct BrownDwarf from Python-friendly config arguments.
// Component creation (chemistry, temperature, RT) happens inside C++ to
// avoid the unique_ptr ownership transfer issues with pybind11.
static std::unique_ptr<BrownDwarf> make_brown_dwarf(
    SpectralGrid& spectral_grid,
    size_t nb_grid_points,
    const std::vector<double>& atmos_boundary_pressures,
    const std::string& cross_section_file_path,
    const std::vector<std::string>& opacity_species_symbol,
    const std::vector<std::string>& opacity_species_folder,
    bool use_clouds,
    const std::vector<std::pair<std::string, std::vector<std::string>>>& chemistry_configs,
    const std::string& temperature_type,
    const std::vector<std::string>& temperature_params,
    const std::string& rt_type,
    const std::vector<std::string>& rt_params,
    double surface_gravity,
    double effective_temperature,
    double metallicity,
    double bottom_radius,
    bool use_variable_gravity,
    const std::vector<double>& temperature_parameters,
    const std::vector<double>& chemistry_parameters,
    size_t max_iterations,
    double convergence_threshold,
    double iteration_gamma,
    bool use_convective_adjustment)
{
    std::vector<std::unique_ptr<Chemistry>> chemistry;
    for (auto& [type, params] : chemistry_configs)
        chemistry.push_back(selectChemistryModule(type, params));

    auto temperature = selectTemperatureProfile(temperature_type, temperature_params);
    auto rt = selectRadiativeTransfer(
        rt_type, rt_params, nb_grid_points, &spectral_grid);

    return std::make_unique<BrownDwarf>(
        &spectral_grid,
        nb_grid_points,
        atmos_boundary_pressures,
        cross_section_file_path,
        opacity_species_symbol,
        opacity_species_folder,
        use_clouds,
        std::move(chemistry),
        std::move(temperature),
        std::move(rt),
        surface_gravity,
        effective_temperature,
        metallicity,
        bottom_radius,
        use_variable_gravity,
        temperature_parameters,
        chemistry_parameters,
        max_iterations,
        convergence_threshold,
        iteration_gamma,
        use_convective_adjustment);
}


static std::vector<std::string> get_species_symbols()
{
    std::vector<std::string> symbols;
    symbols.reserve(constants::species_data.size());
    for (auto& s : constants::species_data)
        symbols.push_back(s.symbol);
    return symbols;
}


PYBIND11_MODULE(pyngam, m) {
    m.doc() = "ngam — atmospheric modeling framework";

    m.def("species_symbols", &get_species_symbols,
        "Return the list of all chemical species symbols (indexed by species ID)");

    // ---- SpectralGrid ----
    py::class_<SpectralGrid>(m, "SpectralGrid")
        .def(py::init<
            const std::string&,
            const std::string&,
            unsigned int,
            double>(),
            py::arg("cross_section_file_path"),
            py::arg("wavenumber_file_path"),
            py::arg("spectral_discretisation"),
            py::arg("spectral_resolution"))
        .def(py::init<
            const std::string&,
            const std::string&,
            unsigned int,
            double,
            double,
            double>(),
            py::arg("cross_section_file_path"),
            py::arg("wavenumber_file_path"),
            py::arg("spectral_discretisation"),
            py::arg("spectral_resolution"),
            py::arg("wavelength_min"),
            py::arg("wavelength_max"))
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
        .def_readonly("flux_divergence", &RadiativeTransferOutput::flux_divergence);

    // ---- BrownDwarf ----
    py::class_<BrownDwarf>(m, "BrownDwarf")
        .def(py::init(&make_brown_dwarf),
            py::arg("spectral_grid"),
            py::arg("nb_grid_points"),
            py::arg("atmos_boundary_pressures"),
            py::arg("cross_section_file_path"),
            py::arg("opacity_species_symbol"),
            py::arg("opacity_species_folder"),
            py::arg("use_clouds"),
            py::arg("chemistry"),
            py::arg("temperature_type"),
            py::arg("temperature_params"),
            py::arg("rt_type"),
            py::arg("rt_params"),
            py::arg("surface_gravity"),
            py::arg("effective_temperature"),
            py::arg("metallicity"),
            py::arg("bottom_radius"),
            py::arg("use_variable_gravity"),
            py::arg("temperature_parameters"),
            py::arg("chemistry_parameters"),
            py::arg("max_iterations") = 100,
            py::arg("convergence_threshold") = 1e-4,
            py::arg("iteration_gamma") = 0.5,
            py::arg("use_convective_adjustment") = true,
            py::keep_alive<1, 2>())
        .def("compute", &BrownDwarf::computeAtmosphericStructure)
        .def("set_temperature", &BrownDwarf::setTemperature,
            py::arg("temperature"),
            "Set initial temperature profile from an external array (skips analytic init)")
        .def_readonly("radiation_field", &BrownDwarf::radiation_field)
        .def_property_readonly("atmosphere", &BrownDwarf::getAtmosphere);
}
