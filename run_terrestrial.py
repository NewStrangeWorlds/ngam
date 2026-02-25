import sys
import numpy as np

sys.path.insert(0, "build")
import pyngam
from ngam_io import save_model, load_model_data


# --- Model configuration ---

opacity_data_path = "/media/data/opacity_data/helios-k/"

opacity_species = [
    ("H2O",        "Rayleigh"),
    ("CO2",        "Rayleigh"),
    ("H2O",        "Molecules/1H2-16O__POKAZATEL_e2b"),
    ("CO2",        "Molecules/12C-16O2__CDSD_4000_e2b"),
    ("CO2-CIA",    "none"),
    ("H2O-CIA",    "none")
]


species_symbols = [s[0] for s in opacity_species]
species_folders = [s[1] for s in opacity_species]


# --- Build the model ---

grid = pyngam.SpectralGrid(
    opacity_data_path, "",
    2, 1000.0,
    0.3, 100.0)

# Chemistry: isoprofile with prescribed mixing ratios.
# The chemistry_parameters vector contains the volume mixing ratios
# for each species in the order of species_data (see pyngam.species_symbols()).
# Here we set up a simple H2/He dominated atmosphere with trace H2O and CO2.
all_species = pyngam.species_symbols()
mix_ratios = [0.0] * len(all_species)

# set mixing ratios by species name
composition = {
    "CO2":  1e-6,
    "O2":  0.2,
    "H2O": 1e-4,
    "N2": 1-0.2-1e-6,
}

mix_ratios = [0.7, 0.2, 1e-4, 1-0.2-0.7-1e-4]  # H2, O2, H2O, N2

# for name, vmr in composition.items():
#     idx = all_species.index(name)
#     mix_ratios[idx] = vmr

model = pyngam.TerrestrialPlanet(
    spectral_grid=grid,
    nb_grid_points=100,
    atmos_boundary_pressures=[1, 1e-6],
    cross_section_file_path=opacity_data_path,
    opacity_species_symbol=species_symbols,
    opacity_species_folder=species_folders,
    use_clouds=False,
    chemistry=[("iso", ["CO2", "O2", "H2O", "N2"])],
    rt_type="disort",
    rt_params=["4"],
    surface_gravity=980.0,       # cm/s^2 (~ Earth)
    surface_albedo=0.3,
    zenith_angle=0.5,            # cos(60 deg) = global average approximation
    stellar_temperature=5780.0,  # K (Sun)
    instellation=1361e3*0.5,         # erg/cm^2/s (solar constant: 1361 W/m^2 * 1e3)
    chemistry_parameters=mix_ratios,
    max_iterations=2000,
    convergence_threshold=1e-5,
    iteration_gamma=0.1,
    use_convective_adjustment=True,
    ng_interval=0,
    max_change_per_iteration=0.1)


# --- Initialize ---

# Option 2: Restart from a saved file
ds = load_model_data("output_terrestrial_nc.nc")
model.initialize_from_arrays(
    temperature=ds["temperature"].values.tolist(),
    number_densities=ds["number_densities"].values.tolist(),
    mean_molecular_weight=ds["mean_molecular_weight"].values.tolist())
ds.close()

# model.initialize(
#     temperature_type="const",
#     temperature_config=[],
#     temperature_parameters=[300.0],
#     init_chemistry=[("iso", ["CO2", "O2", "H2O", "N2"])],
#     init_chemistry_parameters=mix_ratios)


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
print(f"Surface temperature: {model.surface_temperature:.1f} K")

print(f"\nNet flux at TOA: {flux_total[-1]:.4e} erg/cm2/s")
print(f"Net flux at bottom: {flux_total[0]:.4e} erg/cm2/s")


# --- Save ---

save_model("output_terrestrial.nc", model, grid, config={
    "surface_gravity": 980.0,
    "surface_albedo": 0.3,
    "stellar_temperature": 5780.0,
    "instellation": 1361e3,
    "surface_temperature": model.surface_temperature,
})
