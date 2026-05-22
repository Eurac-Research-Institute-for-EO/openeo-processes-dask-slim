from typing import Callable, Optional, Union

import numpy as np
import odc.geo.xr
import scipy.ndimage
import xarray as xr

from openeo_processes_dask_slim.process_implementations.data_model import (
    RasterCube,
    _stack_bands,
    _unstack_bands,
)
from openeo_processes_dask_slim.process_implementations.exceptions import (
    DimensionNotAvailable,
    KernelDimensionsUneven,
)

__all__ = ["apply", "apply_dimension", "apply_kernel"]


def apply(
    data: RasterCube, process: Callable, context: Optional[dict] = None
) -> RasterCube:
    positional_parameters = {"x": 0}
    named_parameters = {"context": context}
    result = xr.apply_ufunc(
        process,
        data,
        dask="allowed",
        kwargs={
            "positional_parameters": positional_parameters,
            "named_parameters": named_parameters,
        },
        keep_attrs=True,
    )
    return result


def apply_dimension(
    data: RasterCube,
    process: Callable,
    dimension: str,
    target_dimension: Optional[str] = None,
    context: Optional[dict] = None,
) -> RasterCube:
    if context is None:
        context = {}

    if isinstance(data, xr.Dataset):
        if dimension == "bands":
            stacked = _stack_bands(data)
            result = apply_dimension(
                stacked, process, dimension, target_dimension, context
            )
            if isinstance(result, xr.Dataset):
                return result
            return _unstack_bands(result)

        result_vars = {}
        for var_name in data.data_vars:
            result_vars[var_name] = apply_dimension(
                data[var_name], process, dimension, target_dimension, context
            )
        data_vars = {}
        for name, da in result_vars.items():
            data_vars[name] = (list(da.dims), da.data)
        result_ds = xr.Dataset(data_vars, attrs=data.attrs)
        try:
            result_ds = odc.geo.xr.assign_crs(result_ds, crs=data.odc.crs)
        except Exception:
            pass
        return result_ds

    if dimension not in data.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({dimension}) not found in data.dims: {data.dims}"
        )

    keepdims = False
    is_new_dim_added = target_dimension is not None
    if is_new_dim_added:
        keepdims = True

    if target_dimension is None:
        target_dimension = dimension

    positional_parameters = {"data": 0}
    named_parameters = {"context": context}

    # This transpose (and back later) is needed because apply_ufunc automatically moves
    # input_core_dimensions to the last axes
    reordered_data = data.transpose(..., dimension)

    # Save original dimension coordinate before apply_ufunc (may drop labels for core dims)
    orig_dim_coord = data.coords[dimension].values if dimension in data.coords else None

    result = xr.apply_ufunc(
        process,
        reordered_data,
        input_core_dims=[[dimension]],
        output_core_dims=[[dimension]],
        dask="allowed",
        kwargs={
            "positional_parameters": positional_parameters,
            "named_parameters": named_parameters,
            "axis": list(reordered_data.dims).index(dimension),
            "keepdims": keepdims,
            "source_transposed_axis": list(data.dims).index(dimension),
            "context": context,
        },
        exclude_dims={dimension},
        keep_attrs=True,
    )

    reordered_result = result.transpose(*data.dims, ...)

    # Restore dimension coordinate labels only if sizes match (dimension may have changed size)
    if orig_dim_coord is not None and dimension in reordered_result.dims:
        if len(orig_dim_coord) == len(reordered_result[dimension]):
            reordered_result = reordered_result.assign_coords(
                {dimension: orig_dim_coord}
            )

    if dimension in reordered_result.dims:
        result_len = len(reordered_result[dimension])
    else:
        result_len = 1

    # Case 1: target_dimension is not defined/ is source dimension
    if dimension == target_dimension:
        # dimension labels preserved
        # if the number of source dimension's values is equal to the number of computed values
        if len(reordered_data[dimension]) == result_len:
            reordered_result[dimension] = reordered_data[dimension].values
        else:
            reordered_result[dimension] = np.arange(result_len)
    elif target_dimension in reordered_result.dims:
        # source dimension is not target dimension
        # target dimension exists with a single label only
        if len(reordered_result[target_dimension]) == 1:
            reordered_result = reordered_result.drop_vars(target_dimension).squeeze(
                target_dimension
            )
            reordered_result = reordered_result.rename({dimension: target_dimension})
            reordered_result[dimension] = np.arange(result_len)
        else:
            raise Exception(
                f"Cannot rename dimension {dimension} to {target_dimension} as {target_dimension} already exists in dataset and contains more than one label: {reordered_result[target_dimension]}. See process definition. "
            )
    else:
        # source dimension is not the target dimension and the latter does not exist
        reordered_result = reordered_result.rename({dimension: target_dimension})
        reordered_result[target_dimension] = np.arange(result_len)

    if data.odc.crs is not None:
        try:
            reordered_result = odc.geo.xr.assign_crs(reordered_result, crs=data.odc.crs)
        except ValueError:
            pass

    return reordered_result


def apply_kernel(
    data: RasterCube,
    kernel: np.ndarray,
    factor: Optional[float] = 1,
    border: Union[float, str, None] = 0,
    replace_invalid: Optional[float] = 0,
) -> RasterCube:
    kernel = np.asarray(kernel)
    if any(dim % 2 == 0 for dim in kernel.shape):
        raise KernelDimensionsUneven(
            "Each dimension of the kernel must have an uneven number of elements."
        )

    if isinstance(data, xr.Dataset):
        results = {}
        for var_name in data.data_vars:
            results[var_name] = apply_kernel(
                data[var_name], kernel, factor, border, replace_invalid
            )
        return xr.Dataset(results, coords=data.coords, attrs=data.attrs)

    def convolve(data, kernel, mode="constant", cval=0, fill_value=0):
        dims = data.openeo.spatial_dims
        convolved = lambda data: scipy.ndimage.convolve(
            data, kernel, mode=mode, cval=cval
        )

        data_masked = data.fillna(fill_value)

        return xr.apply_ufunc(
            convolved,
            data_masked,
            vectorize=True,
            dask="parallelized",
            input_core_dims=[dims],
            output_core_dims=[dims],
            output_dtypes=[data.dtype],
            dask_gufunc_kwargs={"allow_rechunk": True},
        ).transpose(*data.dims)

    openeo_scipy_modes = {
        "replicate": "nearest",
        "reflect": "reflect",
        "reflect_pixel": "mirror",
        "wrap": "wrap",
    }
    if isinstance(border, int) or isinstance(border, float):
        mode = "constant"
        cval = border
    else:
        mode = openeo_scipy_modes[border]
        cval = 0

    result = convolve(data, kernel, mode, cval, replace_invalid) * factor
    result.attrs = data.attrs
    return result
