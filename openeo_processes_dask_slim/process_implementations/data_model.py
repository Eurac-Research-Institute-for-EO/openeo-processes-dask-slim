from __future__ import annotations

from typing import Union

import dask_geopandas
import geopandas as gpd
import xarray as xr

RasterCube = xr.Dataset
VectorCube = Union[gpd.GeoDataFrame, dask_geopandas.GeoDataFrame, xr.Dataset]

X_GUESSES = ["x", "lon", "longitude"]
Y_GUESSES = ["y", "lat", "latitude"]


def is_raster_cube(obj) -> bool:
    if not isinstance(obj, xr.Dataset):
        return False
    lowercase_dims = [str(d).casefold() for d in obj.dims]
    has_x = any(d in lowercase_dims for d in X_GUESSES)
    has_y = any(d in lowercase_dims for d in Y_GUESSES)
    return has_x and has_y


def is_vector_cube(obj) -> bool:
    if isinstance(obj, (gpd.GeoDataFrame, dask_geopandas.GeoDataFrame)):
        return True
    if isinstance(obj, xr.Dataset):
        return not is_raster_cube(obj)
    return False


def to_raster_dataset(obj) -> xr.Dataset:
    if isinstance(obj, xr.Dataset):
        return obj
    if isinstance(obj, xr.DataArray):
        if "bands" in obj.dims:
            return obj.to_dataset(dim="bands")
        name = obj.name or "bands"
        return obj.to_dataset(name=name)
    raise TypeError(f"Cannot convert {type(obj)} to raster Dataset")


def band_names(cube: xr.Dataset) -> list:
    return list(cube.data_vars)


def select_bands(cube: xr.Dataset, bands: list) -> xr.Dataset:
    missing = [b for b in bands if b not in cube.data_vars]
    if missing:
        raise ValueError(f"Bands not found: {missing}")
    return cube[bands]


def rename_bands(cube: xr.Dataset, mapping: dict) -> xr.Dataset:
    return cube.rename(mapping)


def add_band(cube: xr.Dataset, name: str, array) -> xr.Dataset:
    return cube.assign({name: array})


def _stack_bands(cube: xr.Dataset, dim: str = "bands") -> xr.DataArray:
    result = cube.to_dataarray(dim=dim)
    result.attrs.update(cube.attrs)
    time_dim = "t" if "t" in cube.dims else None
    if time_dim:
        target_order = [time_dim, dim] + [d for d in cube.dims if d != time_dim]
    else:
        target_order = [dim] + list(cube.dims)
    return result.transpose(*target_order)


def _unstack_bands(array: xr.DataArray, dim: str = "bands") -> xr.Dataset:
    if dim not in array.dims:
        raise ValueError(f"Dimension '{dim}' not found in array")
    # Fix coordinate if it has mismatched length (can happen after concat+reduce)
    if dim in array.coords and len(array.coords[dim]) != array.sizes[dim]:
        old_coord = array.coords[dim].values
        valid_labels = old_coord[:array.sizes[dim]]
        array = array.assign_coords({dim: valid_labels})
    result = array.to_dataset(dim=dim)
    stale = [
        c
        for c in result.coords
        if c not in result.dims
        and c not in result.data_vars
        and dim in result.coords[c].dims
    ]
    if stale:
        result = result.drop_vars(stale)
    return result


# Public aliases kept for backward compatibility during migration
stack_bands = _stack_bands
unstack_bands = _unstack_bands


def all_data_vars_dask_backed(cube: xr.Dataset) -> bool:
    import dask

    for var in cube.data_vars.values():
        if not isinstance(var.data, dask.array.Array):
            return False
    return True
