"""File-based configuration.

A config file (TOML, YAML or JSON) is just the serialised form of the keyword arguments::

    [grid]
    type = "constant_resolution"        # a SpectralGrid factory name + its arguments
    opacity_path = "/data/opacities/"
    resolution = 1000.0
    wavelength_min = 0.3
    wavelength_max = 100.0

    [model]
    type = "BrownDwarf"                 # BrownDwarf | GasPlanet | TerrestrialPlanet
    effective_temperature = 1000.0      # ... followed by the class's keyword arguments
    surface_gravity = 31622.8
    opacity_species = [["H2O", "Molecules/1H2-16O__POKAZATEL_e2b"], ["CH4", "..."]]
    chemistry = [{type = "equilibrium", parameter_file = "fastchem_parameters.dat"}]
    radiative_transfer = {type = "disort", nb_streams = 4}
    convection = "mlt_dry"
    solver = {type = "ratio_ul", max_iterations = 200, convergence_threshold = 1e-5}

    [initial_profile]
    type = "milne"
    kappa_ross = 1e-2

    [output]
    file = "output_brown_dwarf.nc"

Module specs are written as inline tables with a ``type`` entry (or a bare type name).
``initial_profile`` may also be ``{type = "file", file = "previous_output.nc"}`` to restart from
a saved model; an optional ``initial_chemistry`` list overrides the chemistry used for the
initial composition. The ``config`` attribute of an output file has the same structure (with the
model class under ``model.type``), so an output file can be rebuilt with ``build_model``.
"""

import copy
import json
from pathlib import Path


def load_config(path):
    """Read a TOML / YAML / JSON config file into a dict."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".toml":
        try:
            import tomllib as toml
        except ImportError:                      # Python < 3.11
            import tomli as toml
        with open(path, "rb") as f:
            return toml.load(f)

    if suffix in (".yaml", ".yml"):
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)

    if suffix == ".json":
        with open(path) as f:
            return json.load(f)

    raise ValueError(f"unknown config file format {suffix!r} (use .toml, .yaml or .json)")


def build_model(config):
    """Build (and initialise) a model from a config dict; returns ``(model, grid)``.

    Accepts a config-file dict or the ``config`` attribute of an output file.
    """
    from . import SpectralGrid, MODEL_CLASSES

    cfg = copy.deepcopy(config)
    if "grid" not in cfg or "model" not in cfg:
        raise ValueError("config needs 'grid' and 'model' sections")

    grid = SpectralGrid.from_config(cfg["grid"])

    model_cfg = dict(cfg["model"])
    kind = model_cfg.pop("type", None)
    if kind not in MODEL_CLASSES:
        raise ValueError(f"model.type must be one of {sorted(set(MODEL_CLASSES))}, got {kind!r}")

    # the initial profile may sit at the top level or inside the model section (output files)
    initial_profile = cfg.get("initial_profile", model_cfg.pop("initial_profile", None))
    initial_chemistry = cfg.get("initial_chemistry", model_cfg.pop("initial_chemistry", None))

    # (symbol, folder) pairs may arrive as two-element lists from a config file
    if "opacity_species" in model_cfg:
        model_cfg["opacity_species"] = [tuple(s) for s in model_cfg["opacity_species"]]

    model = MODEL_CLASSES[kind](grid, **model_cfg)

    if initial_profile is not None:
        if initial_profile.get("type") == "file":
            model.initialize_from_file(initial_profile["file"])
        else:
            model.initialize(initial_profile, initial_chemistry)

    return model, grid


def run(config):
    """Build, initialise and run a model from a config dict or file; saves the output if the
    config has an ``[output]`` section with a ``file`` entry. Returns ``(model, grid)``."""
    from .io import save_model

    if not isinstance(config, dict):
        config = load_config(config)

    model, grid = build_model(config)
    model.compute()

    output = config.get("output", {})
    if output.get("file"):
        save_model(output["file"], model, grid)

    return model, grid
