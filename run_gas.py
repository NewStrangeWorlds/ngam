import sys
import numpy as np

sys.path.insert(0, "build")
import pyngam
from ngam_io import save_model, load_model_data


# --- Model configuration ---
#
# Gas (giant) planet: an illuminated, semi-infinite atmosphere without a
# surface. The top is irradiated by the host star (like the terrestrial
# planet) while an internal heat flux escapes from below, set by the
# internal temperature (like the brown dwarf).

opacity_data_path = "/media/data/opacity_data/helios-k/"

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

species_symbols = [s[0] for s in opacity_species]
species_folders = [s[1] for s in opacity_species]


# --- Build the model ---

grid = pyngam.SpectralGrid(
    opacity_data_path, "",
    2, 10000.0,
    0.3, 100.0)

# Chemistry: isoprofile with prescribed volume mixing ratios. The isoprofile
# module takes the species symbols as its configuration; the corresponding
# mixing ratios are passed (in the same order) via chemistry_parameters.
# (For equilibrium chemistry one would instead use ("equilibrium", [...]) with
# chemistry_parameters = [metallicity, C/O].)
chem_species = ["H2", "He", "H2O", "CH4", "CO2"]

# H2/He dominated atmosphere with trace H2O, CH4 and CO2.
mix_ratios = [0.85, 0.15, 1e-4, 5e-4, 1e-6]

model = pyngam.GasPlanet(
    spectral_grid=grid,
    nb_grid_points=100,
    atmos_boundary_pressures=[1e3, 1e-6],   # bar: deep (semi-infinite) -> TOA
    cross_section_file_path=opacity_data_path,
    opacity_species_symbol=species_symbols,
    opacity_species_folder=species_folders,
    use_clouds=False,
    #chemistry=[("isoprofile", chem_species)],
    chemistry=[("eq", ["fastchem_parameters.dat"])],
    rt_type="adding_doubling",
    rt_params=["2"],
    surface_gravity=2500.0,        # cm/s^2 (~ warm Jupiter)
    internal_temperature=100.0,    # K -> internal heat flux = sigma * T_int^4
    zenith_angle=0.5,              # cos(60 deg), global-average approximation
    stellar_type="tabulated",
    stellar_params=["data/stellar_spectra/spectrum_sun.dat"],
    instellation=2.48e8, # 1361e3,         # erg/cm^2/s (incident stellar flux) 
    chemistry_parameters=[1.0, 0.55],  # equilibrium chemistry: [metallicity, C/O]
    bottom_radius=7.0e9,           # cm (~ 1 R_Jup), for the deep atmosphere
    use_variable_gravity=False,
    max_iterations=100,
    temperature_correction="ptc",
    convergence_threshold=1e-5,
    iteration_gamma=0.2,
    lre_fraction=0.5,
    use_convective_adjustment=True,
    convection_type="dry",
    ng_interval=0,
    min_convection_pressure=1e-4,
    max_change_per_iteration=0.1)


# --- Initialize ---

# model.initialize(
#     temperature_type="milne",
#     temperature_config=[],
#     temperature_parameters=[1e-2, 500.0],   # kappa_ross, T_eff (Milne init)
#     init_chemistry=[("isoprofile", chem_species)],
#     init_chemistry_parameters=mix_ratios)

model.initialize(
    temperature_type="const",
    temperature_config=[],
    temperature_parameters=[2000],   # kappa_ross, T_eff (Milne init)
    init_chemistry=[("isoprofile", chem_species)],
    init_chemistry_parameters=mix_ratios)

# Option 2: restart from a saved file
# ds = load_model_data("output_gas.nc")
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

print(f"\nNet flux at TOA:    {flux_total[-1]:.4e} erg/cm2/s")
print(f"Net flux at bottom: {flux_total[0]:.4e} erg/cm2/s")


# --- Save ---

save_model("output_gas.nc", model, grid, config={
    "surface_gravity": 2500.0,
    "internal_temperature": 100.0,
    "instellation": 1361e3,
})
