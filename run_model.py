import sys
import numpy as np

sys.path.insert(0, "build")
import pyngam
from ngam_io import save_model, load_model_data


# --- Model configuration ---

opacity_data_path = "/media/data/opacity_data/helios-k/"

opacity_species = [
    ("CIA-H2-H2", "CIA/H2-H2"),
    ("CIA-H2-He",  "CIA/H2-He"),
    ("K",          "Alkali_Allard/K"),
    ("Na",         "Alkali_Allard/Na"),
    ("H2O",        "Molecules/1H2-16O__POKAZATEL_e2b"),
    ("CH4",        "Molecules/12C-1H4__YT34to10_e2b"),
    ("NH3",        "Molecules/14N-1H3__CoYuTe_e2b"),
    ("H2S",        "Molecules/1H2-32S__AYT2_e2b"),
    ("CO",         "Molecules/12C-16O__Li2015_e2b"),
    ("CO2",        "Molecules/12C-16O2__CDSD_4000_e2b"),
]

species_symbols = [s[0] for s in opacity_species]
species_folders = [s[1] for s in opacity_species]

model_config = dict(
    effective_temperature=1000.0,
    surface_gravity=10**4.5,
    metallicity=1.0,
    c_to_o_ratio=0.5,
    bottom_radius=7.1492e9,
)


# --- Build the model ---

grid = pyngam.SpectralGrid(
    opacity_data_path, "",
    2, 1000.0,
    0.3, 100.0)

model = pyngam.BrownDwarf(
    spectral_grid=grid,
    nb_grid_points=100,
    atmos_boundary_pressures=[1e2, 1e-6],
    cross_section_file_path=opacity_data_path,
    opacity_species_symbol=species_symbols,
    opacity_species_folder=species_folders,
    use_clouds=False,
    chemistry=[("eq", ["fastchem_parameters.dat"])],
    rt_type="disort",
    rt_params=["4"],
    use_variable_gravity=False,
    max_iterations=100,
    convergence_threshold=1e-5,
    iteration_gamma=0.5,
    lre_fraction=0.5,
    ng_interval=0,
    use_convective_adjustment=True,
    convection_type="dry",
    max_change_per_iteration=0.1,
    use_linearisation=True,
    **model_config)


# --- Initialize ---

# Option 1: Initialize from analytic profile + chemistry
model.initialize(
    temperature_type="milne",
    temperature_config=[],
    temperature_parameters=[1e-2, 1000.0],  # kappa_ross, T_eff (for Milne init)
    init_chemistry=[("eq", ["fastchem_parameters.dat"])],
    init_chemistry_parameters=[model_config["metallicity"], model_config["c_to_o_ratio"]])

# Option 2: Restart from a saved file
# ds = load_model_data("output.nc")
# model.initialize_from_arrays(
#     temperature=ds["temperature"].values.tolist(),
#     number_densities=ds["number_densities"].values.tolist(),
#     mean_molecular_weight=ds["mean_molecular_weight"].values.tolist())
# ds.close()


# --- Run ---

model.compute()


# --- Output ---

rf = model.radiation_field
atm = model.atmosphere

wavelengths = np.array(grid.wavelength_list)
spectrum = np.array(rf.spectrum)
flux_total = np.array(rf.flux_total)
pressure = np.array(atm.pressure)
temperature = np.array(atm.temperature)

print(f"\nSpectral grid: {grid.nb_spectral_points} points")
print(f"Wavelength range: {wavelengths[-1]:.3f} - {wavelengths[0]:.3f} um")
print(f"Atmosphere: {atm.nb_grid_points} levels, "
      f"P = {pressure[0]:.1e} - {pressure[-1]:.1e} bar, "
      f"T = {temperature[0]:.0f} - {temperature[-1]:.0f} K")

print(f"\nFlux at top: {flux_total[0]:.4e} erg/cm2/s")
print(f"Flux at bottom: {flux_total[-1]:.4e} erg/cm2/s")


# --- Save ---

save_model("output.nc", model, grid, config=model_config)
