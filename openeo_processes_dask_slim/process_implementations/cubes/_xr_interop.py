from typing import Optional

import odc.geo.xr  # Required for the .geo accessor on xarrays.
import xarray as xr

from openeo_processes_dask_slim.process_implementations.data_model import (
    X_GUESSES,
    Y_GUESSES,
    is_raster_cube,
)

TEMPORAL_GUESSES = [
    "DATE",
    "time",
    "t",
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "hour",
    "second",
]
BANDS_GUESSES = ["b", "bands", "band"]


def _guess_dims_for_type(dims, guesses):
    dims_list = list(dims)
    lowercase_dims = [str(d).casefold() for d in dims_list]
    found_dims = []
    for guess in guesses:
        if guess in lowercase_dims:
            i = lowercase_dims.index(guess)
            found_dims.append(dims_list[i])
    return found_dims


def _get_existing_dims_and_pop_missing(dims, expected_dims):
    existing_dims = []
    remaining = list(expected_dims)
    for dim in remaining[:]:
        if dim in dims:
            existing_dims.append(dim)
        else:
            remaining.remove(dim)
    return tuple(existing_dims)


@xr.register_dataarray_accessor("openeo")
class OpenEOExtensionDa:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._spatial_dims = _guess_dims_for_type(
            self._obj.dims, X_GUESSES
        ) + _guess_dims_for_type(self._obj.dims, Y_GUESSES)
        self._temporal_dims = _guess_dims_for_type(self._obj.dims, TEMPORAL_GUESSES)
        self._bands_dims = _guess_dims_for_type(self._obj.dims, BANDS_GUESSES)
        self._other_dims = [
            dim
            for dim in self._obj.dims
            if dim not in self._spatial_dims + self._temporal_dims + self._bands_dims
        ]

    @property
    def _lowercase_dims(self):
        return [str(dim).casefold() for dim in self._obj.dims]

    @property
    def spatial_dims(self) -> tuple[str]:
        """Find and return all spatial dimensions of the datacube as a tuple."""
        return _get_existing_dims_and_pop_missing(self._obj.dims, self._spatial_dims)

    @property
    def temporal_dims(self) -> tuple[str]:
        """Find and return all temporal dimensions of the datacube as a list."""
        return _get_existing_dims_and_pop_missing(self._obj.dims, self._temporal_dims)

    @property
    def band_dims(self) -> tuple[str]:
        """Find and return all bands dimensions of the datacube as a list."""
        return _get_existing_dims_and_pop_missing(self._obj.dims, self._bands_dims)

    @property
    def other_dims(self) -> tuple[str]:
        """Find and return any dimensions with type other as s list."""
        return _get_existing_dims_and_pop_missing(self._obj.dims, self._other_dims)

    @property
    def x_dim(self) -> Optional[str]:
        return next(
            iter(
                [
                    dim
                    for dim in self.spatial_dims
                    if str(dim).casefold() in X_GUESSES and dim in self._obj.dims
                ]
            ),
            None,
        )

    @property
    def y_dim(self) -> Optional[str]:
        return next(
            iter(
                [
                    dim
                    for dim in self.spatial_dims
                    if str(dim).casefold() in Y_GUESSES and dim in self._obj.dims
                ]
            ),
            None,
        )

    @property
    def z_dim(self):
        raise NotImplementedError()

    def add_dim_type(self, name: str, type: str) -> None:
        """Add dimension name to the list of guesses when calling add_dimension."""

        if name not in self._obj.dims:
            raise ValueError("Trying to add a dimension that doesn't exist")

        if type == "spatial":
            self._spatial_dims.append(name)
        elif type == "temporal":
            self._temporal_dims.append(name)
        elif type == "bands":
            self._bands_dims.append(name)
        elif type == "other":
            self._other_dims.append(name)
        else:
            raise ValueError(f"Type {type} is not understood")


@xr.register_dataset_accessor("openeo")
class OpenEOExtensionDs:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._spatial_dims = _guess_dims_for_type(
            self._obj.dims, X_GUESSES
        ) + _guess_dims_for_type(self._obj.dims, Y_GUESSES)
        self._temporal_dims = _guess_dims_for_type(self._obj.dims, TEMPORAL_GUESSES)
        self._other_dims = [
            dim
            for dim in self._obj.dims
            if dim not in self._spatial_dims + self._temporal_dims
        ]

    @property
    def spatial_dims(self) -> tuple[str]:
        return _get_existing_dims_and_pop_missing(self._obj.dims, self._spatial_dims)

    @property
    def temporal_dims(self) -> tuple[str]:
        return _get_existing_dims_and_pop_missing(self._obj.dims, self._temporal_dims)

    @property
    def band_dims(self) -> tuple[str]:
        return ("bands",)

    @property
    def band_names(self) -> list:
        return list(self._obj.data_vars)

    @property
    def other_dims(self) -> tuple[str]:
        return _get_existing_dims_and_pop_missing(self._obj.dims, self._other_dims)

    @property
    def x_dim(self) -> Optional[str]:
        return next(
            iter(
                [
                    dim
                    for dim in self.spatial_dims
                    if str(dim).casefold() in X_GUESSES and dim in self._obj.dims
                ]
            ),
            None,
        )

    @property
    def y_dim(self) -> Optional[str]:
        return next(
            iter(
                [
                    dim
                    for dim in self.spatial_dims
                    if str(dim).casefold() in Y_GUESSES and dim in self._obj.dims
                ]
            ),
            None,
        )

    @property
    def z_dim(self):
        raise NotImplementedError()

    def add_dim_type(self, name: str, type: str) -> None:
        if name not in self._obj.dims:
            raise ValueError("Trying to add a dimension that doesn't exist")

        if type == "spatial":
            self._spatial_dims.append(name)
        elif type == "temporal":
            self._temporal_dims.append(name)
        elif type == "bands":
            pass
        elif type == "other":
            self._other_dims.append(name)
        else:
            raise ValueError(f"Type {type} is not understood")

        dim_types = self._obj.attrs.get("openeo:dim_types", {})
        dim_types[name] = type
        self._obj.attrs["openeo:dim_types"] = dim_types
