"""Model output: NetCDF4 files carrying the atmosphere, radiation field, spectrum and the full
configuration the model was built from."""

import json

import numpy as np
import xarray as xr


def _json_default(value):
    if hasattr(value, "item") and callable(value.item):   # numpy scalar
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _flat_attrs(config, prefix=""):
    """Scalar entries of a nested config as flat NetCDF attributes (nested keys joined by '.')."""
    attrs = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            attrs.update(_flat_attrs(value, name + "."))
        elif isinstance(value, bool):
            attrs[name] = int(value)
        elif isinstance(value, (int, float, str)):
            attrs[name] = value
    return attrs


def save_model(filename, model, grid=None, attrs=None):
    """Save model output to a NetCDF4 file.

    Parameters
    ----------
    filename : str
        Output file path (e.g. "output.nc").
    model : pyngam.BrownDwarf / GasPlanet / TerrestrialPlanet
        The model object after compute() has been called.
    grid : pyngam.SpectralGrid, optional
        The spectral grid; defaults to the grid the model was built with.
    attrs : dict, optional
        Additional global attributes.

    The configuration of the grid and the model (``grid.config`` / ``model.config``) is stored
    verbatim as the JSON attribute ``config``; ``load_output_config`` reads it back, and
    ``build_model`` can rebuild the model from it. Scalar entries are additionally stored as
    flat attributes for convenience.
    """
    from . import _pyngam

    if grid is None:
        grid = getattr(model, "grid", None)
    if grid is None:
        raise ValueError("save_model: pass the spectral grid (the model does not carry one)")

    atm = model.atmosphere
    rf = model.radiation_field
    species_names = _pyngam.species_symbols()
    nb_species = len(species_names)

    ds = xr.Dataset(
        # 1D atmospheric structure (grid_point dimension)
        data_vars={
            "pressure": (["grid_point"], np.array(atm.pressure),
                {"units": "bar", "long_name": "Pressure"}),
            "temperature": (["grid_point"], np.array(atm.temperature),
                {"units": "K", "long_name": "Temperature"}),
            "altitude": (["grid_point"], np.array(atm.altitude),
                {"units": "cm", "long_name": "Altitude"}),
            "scale_height": (["grid_point"], np.array(atm.scale_height),
                {"units": "cm", "long_name": "Pressure scale height"}),
            "mass_density": (["grid_point"], np.array(atm.mass_density),
                {"units": "g/cm3", "long_name": "Mass density"}),
            "mean_molecular_weight": (["grid_point"],
                np.array(atm.mean_molecular_weight),
                {"units": "amu", "long_name": "Mean molecular weight"}),
            "convective": (["grid_point"], np.array(atm.convective),
                {"long_name": "Convective zone flag (1 = convective)"}),

            # 2D number densities (grid_point x species)
            "number_densities": (["grid_point", "species"],
                np.array(atm.number_densities)[:, :nb_species],
                {"units": "cm-3", "long_name": "Number densities"}),

            # 1D integrated radiation (grid_point)
            "flux_total": (["grid_point"], np.array(rf.flux_total),
                {"units": "erg/cm2/s", "long_name": "Net flux (integrated)"}),
            "flux_up_total": (["grid_point"], np.array(rf.flux_up_total),
                {"units": "erg/cm2/s", "long_name": "Upward flux (integrated)"}),
            "flux_down_total": (["grid_point"], np.array(rf.flux_down_total),
                {"units": "erg/cm2/s", "long_name": "Downward flux (integrated)"}),
            "mean_intensity_total": (["grid_point"],
                np.array(rf.mean_intensity_total),
                {"units": "erg/cm2/s", "long_name": "Mean intensity (integrated)"}),
            "flux_divergence": (["grid_point"], np.array(rf.flux_divergence),
                {"units": "erg/cm2/s/bar",
                 "long_name": "Flux divergence dF_net/dp"}),

            # 1D spectrum (wavenumber)
            "spectrum": (["wavenumber_index"], np.array(rf.spectrum),
                {"units": "erg/cm2/s/cm-1",
                 "long_name": "Top-of-atmosphere upward flux"}),
            "wavenumber": (["wavenumber_index"],
                np.array(grid.wavenumber_list),
                {"units": "cm-1", "long_name": "Wavenumber"}),
            "wavelength": (["wavenumber_index"],
                np.array(grid.wavelength_list),
                {"units": "um", "long_name": "Wavelength"}),
        },
        coords={
            "species_name": ("species", species_names[:nb_species]),
        },
        attrs={
            "title": "ngam model output",
        },
    )

    # provenance: the full configuration, plus its scalars as flat attributes
    config = {
        "grid": getattr(grid, "config", None),
        "model": getattr(model, "config", None),
    }
    ds.attrs["config"] = json.dumps(config, default=_json_default)
    if config["model"] is not None:
        ds.attrs.update(_flat_attrs(config["model"]))
    if hasattr(model, "surface_temperature"):
        ds.attrs["surface_temperature"] = float(model.surface_temperature)
    if attrs:
        for key, val in attrs.items():
            ds.attrs[key] = str(val) if isinstance(val, (list, tuple)) else val

    ds.to_netcdf(filename, engine="h5netcdf")
    print(f"Model saved to {filename}")


def load_output_config(filename):
    """The configuration stored in a model output file: ``{"grid": ..., "model": ...}``."""
    ds = xr.open_dataset(filename, engine="h5netcdf")
    config = json.loads(ds.attrs["config"])
    ds.close()
    return config


def load_temperature(filename):
    """Temperature profile (K) from a saved model file, shape (nb_grid_points,)."""
    ds = xr.open_dataset(filename, engine="h5netcdf")
    temperature = ds["temperature"].values.copy()
    ds.close()
    return temperature


def load_model_data(filename):
    """All model data from a saved file as an xarray Dataset."""
    return xr.open_dataset(filename, engine="h5netcdf")


def load_init_arrays(filename):
    """Temperature, number densities and mean molecular weight from a saved model file, as plain
    lists suitable for ``initialize_from_arrays()``."""
    ds = xr.open_dataset(filename, engine="h5netcdf")
    result = {
        "temperature": ds["temperature"].values.tolist(),
        "number_densities": ds["number_densities"].values.tolist(),
        "mean_molecular_weight": ds["mean_molecular_weight"].values.tolist(),
    }
    ds.close()
    return result
