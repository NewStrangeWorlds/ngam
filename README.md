# Next Generation Atmosphere Model

Under development :)

ngam computes radiative-convective equilibrium atmospheres of brown dwarfs, gas planets and
terrestrial planets. The C++ core is used through the `pyngam` Python package.

## Building

```sh
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

This places the compiled core `_pyngam*.so` inside the `pyngam/` package directory, so
`import pyngam` works from the repository root (or with the repository root on `PYTHONPATH`).

## Configuring a model

A model is configured with keyword arguments. Object-specific physical parameters (gravity,
effective or internal temperature, instellation, ...) are plain keyword arguments; every
pluggable component is a *module spec*: a type name, a `(type, {parameters})` tuple, or a dict
with a `"type"` key. Each component accepts only its own parameters and rejects everything
else, so e.g. a relaxation-solver knob can never be passed silently to the Newton solver.

```python
import pyngam

grid = pyngam.SpectralGrid.constant_resolution(
    "/data/opacities/", resolution=1000.0, wavelength_min=0.3, wavelength_max=100.0)

model = pyngam.BrownDwarf(
    grid,
    effective_temperature=1000.0,      # K
    surface_gravity=10**4.5,           # cm/s^2
    nb_grid_points=100,
    boundary_pressures=[1e2, 1e-6],    # bar, bottom -> top
    opacity_species=[("CIA-H2-H2", "CIA/H2-H2"), ("H2O", "Molecules/1H2-16O__POKAZATEL_e2b")],
    chemistry=[("equilibrium", dict(parameter_file="fastchem_parameters.dat",
                                    metallicity=1.0, c_to_o=0.5))],
    radiative_transfer=("disort", dict(nb_streams=4)),
    convection="mlt",
    solver=("ratio_ul", dict(max_iterations=200, convergence_threshold=1e-5)))

model.initialize(("milne", dict(kappa_ross=1e-2)))     # or model.initialize_from_file("out.nc")
model.compute()
pyngam.save_model("output_brown_dwarf.nc", model)      # stores the full configuration
```

See `run_brown_dwarf.py`, `run_gas.py`, `run_earth.py` and `run_venus.py` for complete
examples, and `help(pyngam.BrownDwarf)` / `pyngam.model_config_doc` for the argument lists.

### Components

| keyword              | types and parameters |
|----------------------|----------------------|
| `chemistry` (list)   | `equilibrium` {parameter_file, metallicity=1, c_to_o=0.5}; `isoprofile` {symbol: mixing ratio, ...}; `fixed` {file}; `manabe_wetherald` {surface_rh=0.77} |
| `radiative_transfer` | `disort` {nb_streams=4}; `adding_doubling` {nb_streams=2} |
| `convection`         | `mlt`, `mlt_moist` {alpha=1, min_pressure}; `dry`, `moist` {min_pressure, max_sweeps=10}; `none` |
| `solver`             | `ratio_ul` (default), `flux_divergence`, `ptc`, `time_stepping`, `time_stepping_lre`; all take {max_iterations=100, convergence_threshold=1e-4}; the relaxation schemes add {gamma, ng_interval, max_change, lre_fraction}; `ptc` adds {max_change} |
| `stellar_spectrum`   | `tabulated` {file}; `blackbody` {temperature} |
| `surface`            | `blackbody`; `simple` {albedo, wavelength_switch}; `variable_albedo` {file} |
| `initialize(profile)`| `adiabat` {surface_temperature, stratosphere_temperature}; `milne` {kappa_ross, effective_temperature=object's}; `const` {temperature}; `guillot` {kappa_ir, t_irr, gamma, t_int=object's, mode="isotropic"/"beam", f=0.25 / mu=zenith angle} |

`convection` mlt/mlt_moist (the default) requires the `ratio_ul` solver; use `dry`/`moist`
with the other schemes.

### Spectral grids

`SpectralGrid.constant_resolution`, `.constant_wavenumber_step`, `.constant_wavelength_step`
and `.covering` (the composite-Planck covering distribution for irradiated atmospheres) all
take `opacity_path` first and the wavelength range in micron.

## Config files

The same configuration can be written as a TOML (or YAML / JSON) file whose sections mirror
the keyword arguments, see `configs/earth.toml`:

```sh
python -m pyngam configs/earth.toml
```

or from Python: `model, grid = pyngam.build_model(pyngam.load_config("configs/earth.toml"))`.

## Output and provenance

`pyngam.save_model` writes a NetCDF4 file with the atmosphere, the radiation field and the
spectrum, plus the complete grid and model configuration as the JSON attribute `config`
(scalars are also stored as flat attributes). `pyngam.load_output_config(file)` reads it back
and `pyngam.build_model(...)` rebuilds the model from it; `model.initialize_from_file(file)`
restarts from the saved profile and composition.
