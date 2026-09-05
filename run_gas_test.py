import numpy as np

import pyngam
from pyngam import save_model


# --- Model configuration ---
#
# Gas (giant) planet: an illuminated, semi-infinite atmosphere without a
# surface. The top is irradiated by the host star (like the terrestrial
# planet) while an internal heat flux escapes from below, set by the
# internal temperature (like the brown dwarf).

opacity_path = "/media/data/opacity_data/helios-k/"

opacity_species = [
    ("CIA-H2-H2",  "CIA/H2-H2"),
    ("CIA-H2-He",  "CIA/H2-He"),
    ("H2",         "Rayleigh"),
    ("He",         "Rayleigh"),
    ("H2O",        "Molecules/H2O_HITRAN_cut25"),
    ("CH4",        "Molecules/12C-1H4__YT34to10_e2b"),
    ("CO2",        "Molecules/12C-16O2__CDSD_4000_e2b"),
    ("TiO",        "Molecules/48Ti-16O__Toto"),
    ("VO",         "Molecules/51V-16O__HyVO"),
    ("HCN",        "Molecules/1H-12C-14N__Harris_e2b"),
]


# --- Build the model ---

grid = pyngam.SpectralGrid.constant_resolution(
    opacity_path, resolution=10000.0, wavelength_min=0.3, wavelength_max=100.0)

model = pyngam.GasPlanet(
    grid,
    internal_temperature=100.0,        # K -> internal heat flux = sigma * T_int^4
    surface_gravity=2500.0,            # cm/s^2 (~ warm Jupiter)
    radius=7.0e9,                      # cm (~ 1 R_Jup), for the deep atmosphere
    instellation=2.48e8,               # erg/cm^2/s (incident stellar flux)
    zenith_angle=0.5,                  # cos(60 deg), global-average approximation
    stellar_spectrum=("tabulated", dict(file="data/stellar_spectra/spectrum_sun.dat")),
    nb_grid_points=100,
    boundary_pressures=[1e3, 1e-6],    # bar: deep (semi-infinite) -> TOA
    opacity_species=opacity_species,
    chemistry=[("equilibrium", dict(parameter_file="fastchem_parameters.dat",
                                    metallicity=1.0, c_to_o=0.55))],
    radiative_transfer=("adding_doubling", dict(nb_streams=2)),
    #convection=("dry", dict(min_pressure=1e-4)),
    convection=("none"),
    solver=("helios", dict(ng_interval=4, max_iterations=200, convergence_threshold=1e-4)))


# --- Initialize ---

# isothermal start; the initial composition uses prescribed mixing ratios instead of the
# model's equilibrium chemistry
model.initialize(
    ("const", dict(temperature=2000.0)),
    chemistry=[("isoprofile", dict(H2=0.85, He=0.15, H2O=1e-4, CH4=5e-4, CO2=1e-6))])

# Guillot (2010) irradiated-analytic start (T_int defaults to the model's internal temperature):
# model.initialize(("guillot", dict(kappa_ir=1e-2, t_irr=1500.0, gamma=0.4, f=0.25)))

# Restart from a saved file:
# model.initialize_from_file("output_gas.nc")


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

print(f"\nNet flux at TOA:    {flux_total[-1]:.4e} erg/cm2/s")
print(f"Net flux at bottom: {flux_total[0]:.4e} erg/cm2/s")


# --- Save (the full configuration is stored in the file) ---

save_model("output_gas.nc", model)
