import numpy as np

import pyngam
from pyngam import save_model, mixing_ratios, KineticsCoupling, load_vulcan


# --- Model configuration ---
#
# The brown dwarf of run_brown_dwarf.py, coupled to the neoVULCAN chemical kinetics code
# (https://github.com/NewStrangeWorlds/neoVULCAN). ngam solves the radiative-convective
# equilibrium with the composition frozen, neoVULCAN the chemical steady state for that
# atmosphere on its own grid, and pyngam.KineticsCoupling alternates the two until both agree.
#
# The chemistry list ends in ("external", {}), the module that receives neoVULCAN's composition;
# the quench approximation in between seeds the kinetics on the first pass.

opacity_path = "/media/data/opacity_data/helios-k/"

# neoVULCAN's own configuration for this run; the code itself is the checkout fetched by the
# build (pinned in CMakeLists.txt; -DFETCHCONTENT_SOURCE_DIR_NEOVULCAN=... for a local one)
neovulcan_config = "configs/neovulcan_brown_dwarf.cfg"

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
                                    metallicity=1.0, c_to_o=0.5)),
               ("quench", dict(metallicity=1.0)),   # Zahnle & Marley 2014: first-pass guess
               ("external", {})],                   # filled by neoVULCAN through the driver
    radiative_transfer=("disort", dict(nb_streams=4)),
    convection="mlt_dry",
    solver=("ratio_ul", dict(max_iterations=200, convergence_threshold=1e-5)))
    # kzz defaults to the self-consistent mixing-length profile, which the driver passes on

model.initialize(("milne", dict(kappa_ross=1e-2)))


# --- Build the kinetics code ---
#
# neoVULCAN reads its thermodynamic data relative to its base directory, ngam's FastChem
# parameter file relative to this directory, so the kinetics code is initialised (and later
# driven) from its own directory. load_vulcan takes care of the former, workdir= of the latter.

chem, neovulcan_path = load_vulcan(neovulcan_config)   # regenerates neoVULCAN's chemistry module


# --- Run the coupled iteration ---
#
# rce_tolerance: RCE convergence threshold of the first pass and the final one (tightened by a
# decade per pass). dead_band / composition_tolerance in dex of mixing ratio; neoVULCAN's own
# convergence noise is ~0.03 dex, so neither should go below that.

coupling = KineticsCoupling(
    model, chem,
    rce_tolerance=(1e-3, 1e-5),
    dead_band=0.05,
    composition_tolerance=0.05,
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

species = ["CO", "CH4", "H2O", "NH3", "N2", "HCN", "CO2"]
symbols, table = mixing_ratios(model, species)
table = np.array(table)

print(f"\nAtmosphere: {atm.nb_grid_points} levels, "
      f"P = {pressure[0]:.1e} - {pressure[-1]:.1e} bar, "
      f"T = {temperature[0]:.0f} - {temperature[-1]:.0f} K")
print("\nMixing ratios of the quenched species:")
print(f"{'p [bar]':>10} " + " ".join(f"{s:>9}" for s in symbols))
for p_bar in (10.0, 1.0, 0.1, 0.01):
    i = np.argmin(np.abs(np.log10(pressure) - np.log10(p_bar)))
    print(f"{pressure[i]:10.2e} " + " ".join(f"{table[i, k]:9.2e}" for k in range(len(symbols))))


# --- Save (the full ngam configuration is stored in the file) ---

save_model("output_brown_dwarf_neovulcan.nc", model)
