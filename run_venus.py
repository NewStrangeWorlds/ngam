import numpy as np

import pyngam
from pyngam import save_model


# --- Model configuration ---

opacity_path = "/media/data/opacity_data/helios-k/"

opacity_species = [
    ("N2",         "Rayleigh"),
    ("CO2",        "Rayleigh"),
    ("CO2",        "Molecules/12C-16O2__CDSD_4000_e2b"),
    ("CO2-CIA",    "none"),
]


# --- Build the model ---

grid = pyngam.SpectralGrid.constant_resolution(
    opacity_path, resolution=3000.0, wavelength_min=0.15, wavelength_max=100.0)

model = pyngam.TerrestrialPlanet(
    grid,
    surface_gravity=887.0,             # cm/s^2
    instellation=2601e3 * 0.5,         # erg/cm^2/s (Venus solar constant, 0.5 for a fast rotator)
    zenith_angle=0.5,                  # cos(60 deg) = global-average approximation
    stellar_spectrum=("tabulated", dict(file="data/stellar_spectra/spectrum_sun.dat")),
    surface=("simple", dict(albedo=0.1, wavelength_switch=2.0)),
    nb_grid_points=100,
    boundary_pressures=[92, 1e-5],     # bar: surface -> top
    opacity_species=opacity_species,
    chemistry=[("isoprofile", dict(CO2=0.965, N2=0.035))],
    radiative_transfer=("adding_doubling", dict(nb_streams=2)),
    convection=("dry", dict(min_pressure=1e-2)),
    solver=("flux_divergence", dict(max_iterations=100, convergence_threshold=1e-5)))


# --- Initialize ---

model.initialize(("adiabat", dict(surface_temperature=650.0, stratosphere_temperature=160.0)))


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
print(f"Surface temperature: {model.surface_temperature:.1f} K")

print(f"\nNet flux at TOA: {flux_total[-1]:.4e} erg/cm2/s")
print(f"Net flux at bottom: {flux_total[0]:.4e} erg/cm2/s")


# --- Save (the full configuration is stored in the file) ---

save_model("output_venus.nc", model)
