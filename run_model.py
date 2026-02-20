import sys
import numpy as np

sys.path.insert(0, "build")
import pyngam


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
    temperature_type="milne",
    temperature_params=[],
    rt_type="disort",
    rt_params=["4"],
    surface_gravity=100.0,
    bottom_radius=7.1492e9,
    use_variable_gravity=False)


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

# Save spectrum
np.savetxt("spectrum.dat",
    np.column_stack([wavelengths, spectrum]),
    header="wavelength [um]    flux [erg/cm2/s/cm-1]")

print(f"\nSpectrum saved to spectrum.dat ({len(spectrum)} points)")
