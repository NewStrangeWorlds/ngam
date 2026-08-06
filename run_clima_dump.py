import sys
import numpy as np

sys.path.insert(0, "build")
import pyngam
from ngam_io import save_model, load_model_data


# --- Model configuration ---

opacity_data_path = "/media/data/opacity_data/helios-k/"

opacity_species = [
    ("N2",         "Rayleigh"),
    ("O2",         "Rayleigh"),
    ("H2O",        "Molecules/H2O_HITRAN_cut25"),
    ("CH4",        "Molecules/12C-1H4__YT34to10_e2b"),
    #("N2O",        "Molecules/14N2-16O__HITEMP2019_e2b"),
    ("CO2",        "Molecules/12C-16O2__CDSD_4000_e2b"),
    #("CO2-CIA",    "none"),
    #("H2O-CIA",    "none")
]

species_symbols = [s[0] for s in opacity_species]
species_folders = [s[1] for s in opacity_species]


# --- Build the model ---

grid = pyngam.SpectralGrid(
    opacity_data_path, "",
    3, 1000.0,
    0.15, 100.0,
    cov_temperature_min=120.0, cov_temperature_max=340.0, cov_nb_temperatures=15,
    target_nb_points=20000,
    cov_stellar_temperature=5780.0, target_nb_points_stellar=20000)

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
    atmos_boundary_pressures=[1, 1e-4],
    cross_section_file_path=opacity_data_path,
    opacity_species_symbol=species_symbols,
    opacity_species_folder=species_folders,
    use_clouds=False,
    chemistry=[
        ("fixed", ["data/Earth/earth_standard_composition.dat"]),
        #("manabe_wetherald", []),   # overrides H2O with RH profile (RH0=0.77)
    ],
    rt_type="adding_doubling",
    rt_params=["2"],
    surface_gravity=980.0,       # cm/s^2 (~ Earth)
    zenith_angle=0.5,            # cos(60 deg) = global average approximation
    stellar_type="tabulated",
    stellar_params=["data/stellar_spectra/spectrum_sun.dat"],   # stellar temperature in K (Sun)
    instellation=1361e3*0.5,     # erg/cm^2/s (solar constant: 1361 W/m^2 * 1e3, 0.5 for fast rotator)
    surface_type="variable_albedo",       # variable: albedo from file
    surface_params=["data/Earth/earth_spectral_surface_reflection.dat"],  #albedo file
    chemistry_parameters=mix_ratios,
    max_iterations=1,
    convergence_threshold=1e-5,
    iteration_gamma=0.2,
    lre_fraction=0.5,
    use_convective_adjustment=True,
    convection_type="dry",
    ng_interval=0,
    max_change_per_iteration=0.1,
    use_linearisation=True)

# model = pyngam.TerrestrialPlanet(
#     spectral_grid=grid,
#     nb_grid_points=100,
#     atmos_boundary_pressures=[1, 1e-4],
#     cross_section_file_path=opacity_data_path,
#     opacity_species_symbol=species_symbols,
#     opacity_species_folder=species_folders,
#     use_clouds=False,
#     chemistry=[("fixed", ["data/Earth/earth_standard_composition.dat"])],
#     rt_type="disort",
#     rt_params=["4"],
#     surface_gravity=980.0,       # cm/s^2 (~ Earth)
#     zenith_angle=0.5,            # cos(60 deg) = global average approximation
#     stellar_type="tabulated",
#     stellar_params=["data/stellar_spectra/spectrum_sun.dat"],   # stellar temperature in K (Sun)
#     instellation=1361e3*0.5,     # erg/cm^2/s (solar constant: 1361 W/m^2 * 1e3, 0.5 for fast rotator)
#     surface_type="simple",       # variable: albedo from file
#     surface_params=["0.3", "2.0"],  #albedo file
#     chemistry_parameters=mix_ratios,
#     max_iterations=2000,
#     convergence_threshold=1e-5,
#     iteration_gamma=0.2,
#     lre_fraction=0.5,
#     use_convective_adjustment=True,
#     min_convection_pressure=1e-3,
#     convection_type="moist",
#     ng_interval=0,
#     max_change_per_iteration=0.1)

# --- Initialize ---

# Option 2: Restart from a saved file
# ds = load_model_data("output_terrestrial.nc")
# model.initialize_from_arrays(
#     temperature=ds["temperature"].values.tolist(),
#     number_densities=ds["number_densities"].values.tolist(),
#     mean_molecular_weight=ds["mean_molecular_weight"].values.tolist())
# ds.close()

# clima-style start: dry adiabat from a guessed surface temperature up to a tropopause floor, then
# isothermal -> the Newton starts on the convective profile (not a radiative Milne profile).

model.initialize(
    temperature_type="milne",
    temperature_config=[],
    temperature_parameters=[1e-2, 300.0],   # kappa_ross, T_eff (Milne init)
    init_chemistry=[("fixed", ["data/Earth/earth_standard_composition.dat"])],
    init_chemistry_parameters=mix_ratios)

# clima-style start: dry adiabat from a guessed surface temperature up to a tropopause floor.
_P = np.array(model.atmosphere.pressure)
_Tsurf, _Ttrop, _nabla = 288.0, 160.0, 0.286
_Tinit = np.maximum(_Ttrop, _Tsurf * (_P / _P[0])**_nabla)
model.initialize_from_arrays(
    temperature=_Tinit.tolist(),
    number_densities=[list(r) for r in np.array(model.atmosphere.number_densities)],
    mean_molecular_weight=list(np.array(model.atmosphere.mean_molecular_weight)))

_P = np.array(model.atmosphere.pressure)
_Tsurf, _Ttrop, _nabla = 288.0, 160.0, 0.286
_Tinit = np.maximum(_Ttrop, _Tsurf * (_P / _P[0])**_nabla)
model.initialize_from_arrays(
    temperature=_Tinit.tolist(),
    number_densities=[list(r) for r in np.array(model.atmosphere.number_densities)],
    mean_molecular_weight=list(np.array(model.atmosphere.mean_molecular_weight)))




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

save_model("output_clima_rad.nc", model, grid, config={
    "surface_gravity": 980.0,
    "surface_albedo": 0.3,
    "stellar_temperature": 5780.0,
    "instellation": 1361e3,
    "surface_temperature": model.surface_temperature,
})
