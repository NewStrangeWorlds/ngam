"""pyngam -- the Python interface of the ngam atmosphere model.

The compiled core (``pyngam._pyngam``) provides the spectral grid and the three object classes
(BrownDwarf, GasPlanet, TerrestrialPlanet). This package wraps them so that

* every grid and model remembers the configuration it was built from (``grid.config``,
  ``model.config``), which :func:`pyngam.save_model` writes into the output file, and
* the same configuration can come from a file (:func:`pyngam.load_config`,
  :func:`pyngam.build_model`, ``python -m pyngam config.toml``).

Models are configured with keyword arguments. Pluggable components (chemistry, radiative
transfer, convection, solver, stellar spectrum, surface, initial profile) are *module specs*:
a type name, a ``(type, {parameters})`` tuple, or a dict with a ``"type"`` key. Parameters that
a module does not accept are rejected, so a knob belonging to one scheme can never be passed
silently to another. See ``pyngam.model_config_doc`` and the class docstrings for the lists.
"""

from . import _pyngam
from ._pyngam import species_symbols, Atmosphere, RadiativeTransferOutput, model_config_doc
from .io import save_model, load_model_data, load_init_arrays, load_temperature, load_output_config
from .config import load_config, build_model, run

__all__ = [
    "SpectralGrid", "BrownDwarf", "GasPlanet", "TerrestrialPlanet",
    "species_symbols", "Atmosphere", "RadiativeTransferOutput", "model_config_doc",
    "save_model", "load_model_data", "load_init_arrays", "load_temperature", "load_output_config",
    "load_config", "build_model", "run", "spec_to_dict",
]


# --- module specs ------------------------------------------------------------------------------

def spec_to_dict(spec):
    """Normalise a module spec (``"type"``, ``(type, {params})`` or ``{"type": ..}``) to a dict."""
    if isinstance(spec, str):
        return {"type": spec}
    if isinstance(spec, dict):
        if "type" not in spec:
            raise ValueError("a module spec given as a dict needs a 'type' entry")
        return {k: _jsonable(v) for k, v in spec.items()}
    if isinstance(spec, (tuple, list)) and 1 <= len(spec) <= 2 and isinstance(spec[0], str):
        out = {"type": spec[0]}
        if len(spec) == 2:
            out.update({k: _jsonable(v) for k, v in dict(spec[1]).items()})
        return out
    raise TypeError(f"not a module spec: {spec!r}")


def _specs_to_dicts(specs):
    """A list is a sequence of specs; anything else is a single spec."""
    if isinstance(specs, list):
        return [spec_to_dict(s) for s in specs]
    return [spec_to_dict(specs)]


_SPEC_KEYS = ("radiative_transfer", "convection", "solver", "stellar_spectrum", "surface", "kzz")
_SPEC_LIST_KEYS = ("chemistry",)


def _jsonable(value):
    """Plain Python containers/scalars only (tuples -> lists, numpy scalars -> Python)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and callable(value.item):   # numpy scalar
        return value.item()
    return value


def _model_kwargs_to_config(kwargs):
    config = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if key in _SPEC_KEYS:
            config[key] = spec_to_dict(value)
        elif key in _SPEC_LIST_KEYS:
            config[key] = _specs_to_dicts(value)
        else:
            config[key] = _jsonable(value)
    return config


# --- spectral grid -----------------------------------------------------------------------------

class SpectralGrid(_pyngam.SpectralGrid):
    """The spectral grid: a subset of the native wavenumber list of the opacity data.

    Build one with a factory (all wavelengths in micron):

    * ``SpectralGrid.constant_resolution(opacity_path, resolution, wavelength_min, wavelength_max)``
    * ``SpectralGrid.constant_wavenumber_step(opacity_path, step, wavelength_min, wavelength_max)``
    * ``SpectralGrid.constant_wavelength_step(opacity_path, step, wavelength_min, wavelength_max)``
    * ``SpectralGrid.covering(opacity_path, wavelength_min, wavelength_max, temperature_min,
      temperature_max, nb_points, nb_temperatures=0, stellar_temperature=0, nb_points_stellar=0)``
      -- the composite-Planck covering distribution: points are placed where the Planck
      functions between temperature_min and temperature_max (and the star's, if a stellar
      temperature is given) carry energy. Set temperature_max to about the deepest layer's
      temperature.

    ``wavenumber_file`` (optional) overrides the native wavenumber list of the opacity data.
    """

    @classmethod
    def constant_resolution(cls, opacity_path, resolution, wavelength_min, wavelength_max,
                            wavenumber_file=""):
        return cls._sampled("constant_resolution", 2, opacity_path, resolution,
                            wavelength_min, wavelength_max, wavenumber_file)

    @classmethod
    def constant_wavenumber_step(cls, opacity_path, step, wavelength_min, wavelength_max,
                                 wavenumber_file=""):
        return cls._sampled("constant_wavenumber_step", 0, opacity_path, step,
                            wavelength_min, wavelength_max, wavenumber_file)

    @classmethod
    def constant_wavelength_step(cls, opacity_path, step, wavelength_min, wavelength_max,
                                 wavenumber_file=""):
        return cls._sampled("constant_wavelength_step", 1, opacity_path, step,
                            wavelength_min, wavelength_max, wavenumber_file)

    @classmethod
    def _sampled(cls, kind, discretisation, opacity_path, step, wavelength_min, wavelength_max,
                 wavenumber_file):
        grid = cls(str(opacity_path), str(wavenumber_file), int(discretisation), float(step),
                   float(wavelength_min), float(wavelength_max))
        parameter = "resolution" if kind == "constant_resolution" else "step"
        grid.config = {
            "type": kind, "opacity_path": str(opacity_path), parameter: float(step),
            "wavelength_min": float(wavelength_min), "wavelength_max": float(wavelength_max)}
        if wavenumber_file:
            grid.config["wavenumber_file"] = str(wavenumber_file)
        return grid

    @classmethod
    def covering(cls, opacity_path, wavelength_min, wavelength_max, temperature_min,
                 temperature_max, nb_points, nb_temperatures=0, stellar_temperature=0.0,
                 nb_points_stellar=0, wavenumber_file=""):
        grid = cls(str(opacity_path), str(wavenumber_file), 3, 0.0,
                   float(wavelength_min), float(wavelength_max),
                   float(temperature_min), float(temperature_max), int(nb_temperatures),
                   int(nb_points), float(stellar_temperature), int(nb_points_stellar))
        grid.config = {
            "type": "covering", "opacity_path": str(opacity_path),
            "wavelength_min": float(wavelength_min), "wavelength_max": float(wavelength_max),
            "temperature_min": float(temperature_min), "temperature_max": float(temperature_max),
            "nb_temperatures": int(nb_temperatures), "nb_points": int(nb_points),
            "stellar_temperature": float(stellar_temperature),
            "nb_points_stellar": int(nb_points_stellar)}
        if wavenumber_file:
            grid.config["wavenumber_file"] = str(wavenumber_file)
        return grid

    @classmethod
    def from_config(cls, config):
        """Build a grid from a config dict: ``{"type": <factory name>, <its keyword arguments>}``."""
        cfg = dict(config)
        kind = cfg.pop("type")
        factories = {
            "constant_resolution": cls.constant_resolution,
            "constant_wavenumber_step": cls.constant_wavenumber_step,
            "constant_wavelength_step": cls.constant_wavelength_step,
            "covering": cls.covering}
        if kind not in factories:
            raise ValueError(f"unknown spectral grid type {kind!r}; "
                             f"choose from {', '.join(factories)}")
        return factories[kind](**cfg)


# --- model classes -----------------------------------------------------------------------------

# defaults of the shared components in the compiled core (see ModelConfig in generic_object.h)
_COMPONENT_DEFAULTS = {
    "radiative_transfer": {"type": "disort", "nb_streams": 4},
    "convection": {"type": "mlt_dry"},
    "solver": {"type": "ratio_ul"},
}

def _wrap_model(kind, base):
    """Subclass a compiled model class so that it records its configuration."""

    class Model(base):
        __doc__ = (base.__doc__ or "") + "\n" + model_config_doc

        def __init__(self, grid, **kwargs):
            base.__init__(self, grid, **kwargs)
            self.grid = grid
            self.config = {"type": kind}
            self.config.update(_model_kwargs_to_config(kwargs))
            self.config.setdefault("opacity_path", grid.opacity_path)
            # record the compiled defaults of the omitted components for provenance
            for key, default in _COMPONENT_DEFAULTS.items():
                self.config.setdefault(key, dict(default))

        def initialize(self, profile, chemistry=None):
            """Initialise from an analytic profile spec (see the compiled docstring)."""
            base.initialize(self, profile, chemistry)
            self.config["initial_profile"] = spec_to_dict(profile)
            if chemistry is not None:
                self.config["initial_chemistry"] = _specs_to_dicts(chemistry)
            return self

        def initialize_from_file(self, filename):
            """Restart from a model output file written by :func:`pyngam.save_model`."""
            arrays = load_init_arrays(filename)
            base.initialize_from_arrays(self, **arrays)
            self.config["initial_profile"] = {"type": "file", "file": str(filename)}
            return self

        initialize.__doc__ = base.initialize.__doc__

    Model.__name__ = kind
    Model.__qualname__ = kind
    Model.__module__ = __name__
    return Model


BrownDwarf = _wrap_model("BrownDwarf", _pyngam.BrownDwarf)
GasPlanet = _wrap_model("GasPlanet", _pyngam.GasPlanet)
TerrestrialPlanet = _wrap_model("TerrestrialPlanet", _pyngam.TerrestrialPlanet)

MODEL_CLASSES = {
    "BrownDwarf": BrownDwarf, "brown_dwarf": BrownDwarf,
    "GasPlanet": GasPlanet, "gas_planet": GasPlanet,
    "TerrestrialPlanet": TerrestrialPlanet, "terrestrial_planet": TerrestrialPlanet,
}
