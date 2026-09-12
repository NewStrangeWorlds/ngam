import numpy as np

import os
import sys

import pyngam
from pyngam import save_model, mixing_ratios, KineticsCoupling


# --- Model configuration ---
#
# The irradiated gas planet of run_gas.py, coupled to the neoVULCAN chemical kinetics code with
# photochemistry on (see run_brown_dwarf_neovulcan.py for the coupling itself). neoVULCAN's own
# configuration, configs/neovulcan_gas_planet.cfg, carries the stellar UV flux and geometry that
# match this model's instellation and zenith angle.

opacity_path = "/media/data/opacity_data/helios-k/"

# neoVULCAN's own configuration for this run; the code itself is the checkout fetched by the
# build (pinned in CMakeLists.txt; -DFETCHCONTENT_SOURCE_DIR_NEOVULCAN=... for a local one)
neovulcan_config = "configs/neovulcan_gas_planet.cfg"

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
    ("CO",         "Molecules/12C-16O__Li2015_e2b"),
    ("NH3",        "Molecules/14N-1H3__CoYuTe_e2b"),
    ("H2S",        "Molecules/1H2-32S__AYT2_e2b"),
    ("SO2",        "Molecules/32S-16O2__ExoAmes_e2b"),
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
                                    metallicity=1.0, c_to_o=0.55)),
               ("quench", dict(metallicity=1.0)),   # Zahnle & Marley 2014: first-pass guess
               ("external", {})],                   # filled by neoVULCAN through the driver
    # Kzz: the radiative envelope of an irradiated planet is mixed by the circulation, not by
    # convection (Parmentier et al. 2013 tracer-transport fit for HD 209458b)
    kzz=("power_law", dict(value=5e8, pressure=1.0, slope=-0.5)),
    radiative_transfer=("adding_doubling", dict(nb_streams=2)),
    convection=("mlt_dry", dict(min_pressure=1e-4)),
    solver=("ratio_ul", dict(max_iterations=100, convergence_threshold=1e-5)))
    # The quench module is left out: neoVULCAN handles the disequilibrium, and with the module in
    # the chain the temperature correction crawled at ~1 K per iteration up to its cap in every
    # pass, while without it no pass needed more than 23 iterations (see the note in the README).


# --- Initialize ---

# Guillot (2010) irradiated-analytic start: t_irr = (instellation / sigma)^(1/4) = 1446 K for this
# planet. Starting near the answer matters here: from an isothermal 2000 K profile the first pass
# took 99 iterations (2.5 h at this resolution), 80 of them a damped-Newton crawl far from the root.
model.initialize(("guillot", dict(kappa_ir=1e-2, t_irr=1446.0, gamma=0.4, f=0.25)))

# Alternatives: isothermal start (slow, see above), or a restart from a saved run
# model.initialize(("const", dict(temperature=2000.0)),
#                  chemistry=[("isoprofile", dict(H2=0.85, He=0.15, H2O=1e-4, CH4=5e-4, CO2=1e-6))])
# model.initialize_from_file("output_gas.nc")


# --- Build the kinetics code ---
#
# neoVULCAN reads its thermodynamic data relative to its base directory, ngam's FastChem
# parameter file relative to this directory, so the kinetics code is initialised (and later
# driven) from its own directory. load_vulcan takes care of the former, workdir= of the latter.

chem, neovulcan_path = load_vulcan(neovulcan_config)   # regenerates neoVULCAN's chemistry module


# --- Run the coupled iteration ---
#
# rce_tolerance: RCE convergence threshold of the first pass and the final one (tightened by a
# decade per pass). dead_band / composition_tolerance in dex of mixing ratio: with photochemistry
# neoVULCAN stops at a 5-10% residual change (its slope criterion is met long before the
# photochemical species aloft settle), so both must stay above ~0.1 dex here.

coupling = KineticsCoupling(
    model, chem,
    rce_tolerance=(1e-3, 1e-5),
    dead_band=0.1,
    composition_tolerance=0.1,
    seed_continuation=True,    # only acts with a quench module in the chain: equilibrium first,
                               # then the quench module on a converged profile (irradiated planets)
    workdir=neovulcan_path)

converged = coupling.run()


# --- Output ---

atm = model.atmosphere
pressure = np.array(atm.pressure)
temperature = np.array(atm.temperature)

print(f"\nCoupling {'converged' if converged else 'NOT converged'} "
      f"after {len(coupling.history)} passes")
for h in coupling.history:
    print(f"  pass {h['pass_']}: composition change {h['change_dex']:.3f} dex"
          f"{' (committed, weight %.2f)' % h['weight'] if h['committed'] else ' (below dead band)'},"
          f" kinetics {h['kinetics'].get('count')} steps,"
          f" RCE tolerance {h['rce_tolerance']:.0e} {'ok' if h['rce_converged'] else 'not converged'}")

species = ["CO", "CH4", "H2O", "NH3", "N2", "HCN", "CO2", "H2S", "SO2", "C2H2", "OH", "H"]
symbols, table = mixing_ratios(model, species)
table = np.array(table)

print(f"\nAtmosphere: {atm.nb_grid_points} levels, "
      f"P = {pressure[0]:.1e} - {pressure[-1]:.1e} bar, "
      f"T = {temperature[0]:.0f} - {temperature[-1]:.0f} K")
print("\nMixing ratios of the quenched and photochemical species:")
print(f"{'p [bar]':>10} " + " ".join(f"{s:>9}" for s in symbols))
for p_bar in (10.0, 1.0, 1e-2, 1e-4, 1e-6):
    i = np.argmin(np.abs(np.log10(pressure) - np.log10(p_bar)))
    print(f"{pressure[i]:10.2e} " + " ".join(f"{table[i, k]:9.2e}" for k in range(len(symbols))))


# --- Save (the full ngam configuration is stored in the file) ---

save_model("output_gas_neovulcan.nc", model)
