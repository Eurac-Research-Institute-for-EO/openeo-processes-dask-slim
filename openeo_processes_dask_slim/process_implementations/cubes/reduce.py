from typing import Callable, Optional

import numpy as np
import xarray as xr

from openeo_processes_dask_slim.process_implementations.data_model import (
    RasterCube,
    _stack_bands,
)
from openeo_processes_dask_slim.process_implementations.exceptions import (
    DimensionNotAvailable,
)

try:
    from numpy._core._exceptions import UFuncTypeError
except ImportError:
    from numpy.core._exceptions import UFuncTypeError

__all__ = ["reduce_dimension", "reduce_spatial"]


def reduce_dimension(
    data: RasterCube,
    reducer: Callable,
    dimension: str,
    context: Optional[dict] = None,
) -> RasterCube:
    if dimension == "bands" and isinstance(data, xr.Dataset):
        stacked = _stack_bands(data)
        result = reduce_dimension(stacked, reducer, dimension, context)
        return result

    if dimension not in data.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({dimension}) not found in data.dims: {data.dims}"
        )

    dim_labels = data[dimension].values

    positional_parameters = {"data": 0}
    reduced_data = data.reduce(
        reducer,
        dim=dimension,
        keep_attrs=True,
        positional_parameters=positional_parameters,
        context=context,
        dim_labels=dim_labels,
    )

    # Preset
    if "reduced_dimensions_min_values" not in data.attrs:
        reduced_data.attrs["reduced_dimensions_min_values"] = {}
    try:
        reduced_data.attrs["reduced_dimensions_min_values"][dimension] = data.coords[
            dimension
        ].values.min()
    except (UFuncTypeError, AttributeError):
        reduced_data.attrs["reduced_dimensions_min_values"][dimension] = 0

    return reduced_data


def reduce_spatial(
    data: RasterCube, reducer: Callable, context: Optional[dict] = None
) -> RasterCube:
    positional_parameters = {"data": 0}
    named_parameters = {"context": context}

    spatial_dims = data.openeo.spatial_dims if data.openeo.spatial_dims else None
    return data.reduce(
        reducer,
        dim=spatial_dims,
        keep_attrs=True,
        positional_parameters=positional_parameters,
        named_parameters=named_parameters,
    )
