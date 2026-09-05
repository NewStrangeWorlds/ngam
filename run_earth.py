import numpy as np

import pyngam
from pyngam import save_model


# --- Model configuration ---

opacity_path = "/media/data/opacity_data/helios-k/"

opacity_species = [
    ("N2",         "Rayleigh"),
    ("O2",         "Rayleigh"),
    ("O3",         "Molecules/16O3__HITRAN2020_e2b"),
    ("O3",         "Molecules/O3_visible_HITRAN"),
    ("H2O",        "Molecules/H2O_HITRAN_cut25"),
    ("CH4",        "Molecules/12C-1H4__YT34to10_e2b"),
    #("N2O",        "Molecules/14N2-16O__HITEMP2019_e2b"),
    ("CO2",        "Molecules/12C-16O2__CDSD_4000_e2b"),
    ("CO2-CIA",    "none"),
    ("H2O-CIA",    "none"),
]


# --- Build the model ---

# Composite-Planck covering grid: points where the Planck functions between the coldest and the
# deepest layer (and the star's) carry energy. Set temperature_max ~ the deepest layer T.
grid = pyngam.SpectralGrid.covering(
    opacity_path, wavelength_min=0.15, wavelength_max=100.0,
    temperature_min=100.0, temperature_max=350.0, nb_temperatures=30,
    nb_points=10000, stellar_temperature=5772.0, nb_points_stellar=10000)

# grid = pyngam.SpectralGrid.constant_resolution(
#     opacity_path, resolution=5000.0, wavelength_min=0.15, wavelength_max=100.0)

model = pyngam.TerrestrialPlanet(
    grid,
    surface_gravity=980.0,             # cm/s^2
    instellation=1361e3 * 0.5,         # erg/cm^2/s (solar constant, 0.5 for a fast rotator)
    zenith_angle=0.5,                  # cos(60 deg) = global-average approximation
    stellar_spectrum=("tabulated", dict(file="data/stellar_spectra/spectrum_sun.dat")),
    surface=("simple", dict(albedo=0.3, wavelength_switch=2.0)),
    # surface=("variable_albedo", dict(file="data/Earth/earth_spectral_surface_reflection.dat")),
    nb_grid_points=100,
    boundary_pressures=[1, 1e-5],      # bar: surface -> top
    opacity_species=opacity_species,
    chemistry=[
        ("fixed", dict(file="data/Earth/earth_standard_composition.dat")),
        # ("manabe_wetherald", dict(surface_rh=0.77)),   # overrides H2O with an RH profile
    ],
    radiative_transfer=("adding_doubling", dict(nb_streams=2)),
    convection=("mlt_moist", dict(min_pressure=1e-2)),
    solver=("ratio_ul", dict(max_iterations=100, convergence_threshold=1e-5)))


# --- Initialize ---

# clima-style start: an adiabat from a guessed surface temperature up to a stratosphere floor,
# integrated along the ACTIVE convection scheme's neutrality gradient (dry, moist, mlt...) reduced
# by 2%, so every layer starts on the stable side of the scheme's own threshold. This is what lets
# the MLT corrector skip its easy-start homotopy (doc/mlt_convection_design.md Sec. 10.6).
model.initialize(("adiabat", dict(surface_temperature=288.0, stratosphere_temperature=160.0)))


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

save_model("output_earth.nc", model)
