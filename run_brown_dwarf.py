import numpy as np

import pyngam
from pyngam import save_model


# --- Model configuration ---

opacity_path = "/media/data/opacity_data/helios-k/"

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


# --- Build the model ---

grid = pyngam.SpectralGrid.constant_resolution(
    opacity_path, resolution=1000.0, wavelength_min=0.3, wavelength_max=100.0)

model = pyngam.BrownDwarf(
    grid,
    effective_temperature=1000.0,      # K
    surface_gravity=10**4.5,           # cm/s^2
    radius=7.1492e9,                   # cm (~ 1 R_Jup)
    nb_grid_points=100,
    boundary_pressures=[1e2, 1e-6],    # bar: bottom -> top
    opacity_species=opacity_species,
    chemistry=[("equilibrium", dict(parameter_file="fastchem_parameters.dat",
                                    metallicity=1.0, c_to_o=0.5))],
    radiative_transfer=("disort", dict(nb_streams=4)),
    convection="mlt_dry",
    solver=("ratio_ul", dict(max_iterations=200, convergence_threshold=1e-5)))


# --- Initialize ---

# Option 1: analytic profile (T_eff defaults to the model's effective temperature) + chemistry
model.initialize(("milne", dict(kappa_ross=1e-2)))

# Option 2: restart from a saved file
# model.initialize_from_file("output_brown_dwarf.nc")


# --- Run ---

model.compute()


# --- Output ---

rf = model.radiation_field
atm = model.atmosphere

wavelengths = np.array(grid.wavelength_list)
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


# --- Save (the full configuration is stored in the file) ---

save_model("output_brown_dwarf.nc", model)
