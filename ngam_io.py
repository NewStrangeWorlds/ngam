import numpy as np
import xarray as xr


def save_model(filename, model, grid, config=None):
    """Save model output to a NetCDF4 file.

    Parameters
    ----------
    filename : str
        Output file path (e.g. "output.nc").
    model : pyngam.BrownDwarf
        The model object after compute() has been called.
    grid : pyngam.SpectralGrid
        The spectral grid used by the model.
    config : dict, optional
        Additional model configuration parameters to store as attributes.
    """
    import sys
    sys.path.insert(0, "build")
    import pyngam

    atm = model.atmosphere
    rf = model.radiation_field
    species_names = pyngam.species_symbols()

    nb_grid = atm.nb_grid_points
    nb_spec = grid.nb_spectral_points
    nb_species = len(species_names)

    # --- Build xarray Dataset ---
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

    # Store model configuration as global attributes
    if config is not None:
        for key, val in config.items():
            if isinstance(val, (list, tuple)):
                ds.attrs[key] = str(val)
            else:
                ds.attrs[key] = val

    ds.to_netcdf(filename, engine="h5netcdf")
    print(f"Model saved to {filename}")


def load_temperature(filename):
    """Load temperature profile from a saved model file.

    Parameters
    ----------
    filename : str
        Path to a NetCDF4 file previously written by save_model().

    Returns
    -------
    temperature : numpy.ndarray
        Temperature profile (K), shape (nb_grid_points,).
    """
    ds = xr.open_dataset(filename, engine="h5netcdf")
    temperature = ds["temperature"].values.copy()
    ds.close()
    return temperature


def load_model_data(filename):
    """Load all model data from a saved file as an xarray Dataset.

    Parameters
    ----------
    filename : str
        Path to a NetCDF4 file previously written by save_model().

    Returns
    -------
    ds : xarray.Dataset
        The full dataset with all atmospheric, radiation, and spectral data.
    """
    return xr.open_dataset(filename, engine="h5netcdf")


def load_init_arrays(filename):
    """Load temperature, number densities, and mean molecular weight from a
    saved model file, as plain Python lists suitable for
    ``BrownDwarf.initialize_from_arrays()``.

    Parameters
    ----------
    filename : str
        Path to a NetCDF4 file previously written by save_model().

    Returns
    -------
    dict with keys ``temperature``, ``number_densities``,
    ``mean_molecular_weight``, each as a Python list (or list of lists).
    """
    ds = xr.open_dataset(filename, engine="h5netcdf")
    result = {
        "temperature": ds["temperature"].values.tolist(),
        "number_densities": ds["number_densities"].values.tolist(),
        "mean_molecular_weight": ds["mean_molecular_weight"].values.tolist(),
    }
    ds.close()
    return result
