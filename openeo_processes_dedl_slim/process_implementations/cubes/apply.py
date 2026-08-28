from collections.abc import Callable

import numpy as np
import odc.geo.xr
import scipy.ndimage
import xarray as xr

from openeo_processes_dedl_slim.process_implementations.cubes.dataset_bridge import (
    dataset_to_virtual_bands,
    detect_band_permutation,
    restore_dataset_metadata,
    virtual_bands_to_dataset,
)
from openeo_processes_dedl_slim.process_implementations.cubes.utils import (
    ensure_raster_cube,
)
from openeo_processes_dedl_slim.process_implementations.data_model import RasterCube
from openeo_processes_dedl_slim.process_implementations.exceptions import (
    DimensionNotAvailable,
    KernelDimensionsUneven,
)

__all__ = ["apply", "apply_dimension", "apply_kernel"]


def _is_apply_ufunc_dimension_mismatch(exc: ValueError) -> bool:
    return (
        "applied function returned data with an unexpected number of dimensions"
        in str(exc)
    )


def _first_data_array(data: xr.Dataset | xr.DataArray) -> xr.DataArray:
    if isinstance(data, xr.Dataset):
        return next(iter(data.data_vars.values()))
    return data


def _sample_core_vector(data: xr.Dataset | xr.DataArray, dimension: str):
    sample = _first_data_array(data)
    return np.zeros(sample.sizes[dimension], dtype=sample.dtype)


def _ensure_apply_dimension_array(result):
    result = np.asarray(result)
    if result.ndim == 0:
        result = result.reshape((1,))
    if result.ndim != 1:
        raise ValueError(
            "apply_dimension callback must return a one-dimensional array when "
            "called for a single dimension slice."
        )
    return result


def _apply_dimension_process(process, values, context, axis):
    return process(
        values,
        positional_parameters={"data": 0},
        named_parameters={"context": context},
        axis=axis,
        keepdims=True,
        source_transposed_axis=axis,
        context=context,
    )


def _apply_dimension_vectorized(
    data: xr.Dataset | xr.DataArray,
    process: Callable,
    dimension: str,
    context: dict,
) -> xr.Dataset | xr.DataArray:
    sample_result = _ensure_apply_dimension_array(
        _apply_dimension_process(
            process=process,
            values=_sample_core_vector(data, dimension),
            context=context,
            axis=0,
        )
    )

    def vectorized_process(values, **kwargs):
        return _ensure_apply_dimension_array(process(values, **kwargs))

    return xr.apply_ufunc(
        vectorized_process,
        data,
        input_core_dims=[[dimension]],
        output_core_dims=[[dimension]],
        dask="parallelized",
        vectorize=True,
        kwargs={
            "positional_parameters": {"data": 0},
            "named_parameters": {"context": context},
            "axis": 0,
            "keepdims": True,
            "source_transposed_axis": 0,
            "context": context,
        },
        exclude_dims={dimension},
        output_dtypes=[sample_result.dtype],
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {dimension: sample_result.shape[0]},
        },
        keep_attrs=True,
    )


def apply(
    data: RasterCube, process: Callable, context: dict | None = None
) -> RasterCube:
    ensure_raster_cube(data, "apply")
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
    target_dimension: str | None = None,
    context: dict | None = None,
) -> RasterCube:
    ensure_raster_cube(data, "apply_dimension")
    if context is None:
        context = {}

    if dimension == "bands":
        band_array, meta = dataset_to_virtual_bands(data, dim="bands")
        original_band_labels = band_array[dimension].values
        positional_parameters = {"data": 0}
        named_parameters = {"context": context}
        if target_dimension is None:
            target_dimension = dimension
        keepdims = target_dimension is not None
        reordered_band_array = band_array.transpose(..., dimension)
        result = xr.apply_ufunc(
            process,
            reordered_band_array,
            input_core_dims=[[dimension]],
            output_core_dims=[[dimension]],
            dask="allowed",
            kwargs={
                "positional_parameters": positional_parameters,
                "named_parameters": named_parameters,
                "axis": reordered_band_array.get_axis_num(dimension),
                "keepdims": keepdims,
                "source_transposed_axis": band_array.get_axis_num(dimension),
                "context": context,
            },
            exclude_dims={dimension},
            keep_attrs=True,
        )
        if isinstance(result, xr.DataArray) and dimension in result.dims:
            out_len = len(result[dimension])
            if out_len == len(original_band_labels):
                permuted = detect_band_permutation(
                    result, reordered_band_array, dimension
                )
                if permuted is not None:
                    labels = permuted
                    meta["order"] = permuted
                else:
                    labels = list(original_band_labels)
                try:
                    result = result.assign_coords({dimension: labels})
                except ValueError:
                    pass
            elif out_len < len(original_band_labels):
                try:
                    result = result.assign_coords(
                        {dimension: original_band_labels[:out_len]}
                    )
                    meta["order"] = list(original_band_labels[:out_len])
                except ValueError:
                    pass
            result = virtual_bands_to_dataset(result, meta, dim=dimension)
        elif not isinstance(result, xr.Dataset):
            result = result.to_dataset(name="result")
            result = restore_dataset_metadata(result, meta)
        else:
            result = restore_dataset_metadata(result, meta)
        return result

    if dimension not in data.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({dimension}) not found in data.dims: {data.dims}"
        )

    # `apply_dimension` keeps the dimension (unlike `reduce_dimension` which
    # drops it). The process is applied along `dimension` and the computed
    # values replace the source/target dimension's values, so the reducer-style
    # callbacks must keep the axis (`keepdims=True`) for `apply_ufunc`'s
    # `output_core_dims=[[dimension]]` contract to hold.
    keepdims = True

    if target_dimension is None:
        target_dimension = dimension

    positional_parameters = {"data": 0}
    named_parameters = {"context": context}

    # This transpose (and back later) is needed because apply_ufunc automatically moves
    # input_core_dimensions to the last axes
    reordered_data = data.transpose(..., dimension)

    # Dataset lacks get_axis_num; compute axis from first variable
    if isinstance(data, xr.Dataset):
        sample_var = _first_data_array(data)
        axis = sample_var.get_axis_num(dimension)
        source_axis = sample_var.get_axis_num(dimension)
        reordered_sample = _first_data_array(reordered_data)
        reordered_axis = reordered_sample.get_axis_num(dimension)
    else:
        axis = data.get_axis_num(dimension)
        source_axis = axis
        reordered_axis = reordered_data.get_axis_num(dimension)

    try:
        result = xr.apply_ufunc(
            process,
            reordered_data,
            input_core_dims=[[dimension]],
            output_core_dims=[[dimension]],
            dask="allowed",
            kwargs={
                "positional_parameters": positional_parameters,
                "named_parameters": named_parameters,
                "axis": reordered_axis,
                "keepdims": keepdims,
                "source_transposed_axis": source_axis,
                "context": context,
            },
            exclude_dims={dimension},
            keep_attrs=True,
        )
    except ValueError as exc:
        if not _is_apply_ufunc_dimension_mismatch(exc):
            raise
        result = _apply_dimension_vectorized(
            data=reordered_data,
            process=process,
            dimension=dimension,
            context=context,
        )

    reordered_result = result.transpose(*data.dims, ...)

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
            raise ValueError(
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
    factor: float | None = 1,
    border: float | str | None = 0,
    replace_invalid: float | None = 0,
) -> RasterCube:
    ensure_raster_cube(data, "apply_kernel")
    kernel = np.asarray(kernel)
    if any(dim % 2 == 0 for dim in kernel.shape):
        raise KernelDimensionsUneven(
            "Each dimension of the kernel must have an uneven number of elements."
        )

    def convolve(data, kernel, mode="constant", cval=0, fill_value=0):
        dims = data.openeo.spatial_dims
        convolved = lambda data: scipy.ndimage.convolve(
            data, kernel, mode=mode, cval=cval
        )

        data_masked = data.fillna(fill_value)

        result_vars = {}
        for var_name in data.data_vars:
            var_data = data_masked[var_name]
            result = xr.apply_ufunc(
                convolved,
                var_data,
                vectorize=True,
                dask="parallelized",
                input_core_dims=[dims],
                output_core_dims=[dims],
                output_dtypes=[var_data.dtype],
                dask_gufunc_kwargs={"allow_rechunk": True},
            ).transpose(*data[var_name].dims)
            result_vars[var_name] = result
        return xr.Dataset(result_vars, coords=data.coords, attrs=data.attrs).transpose(
            *data.dims
        )

    openeo_scipy_modes = {
        "replicate": "nearest",
        "reflect": "reflect",
        "reflect_pixel": "mirror",
        "wrap": "wrap",
    }
    if isinstance(border, (int, float)):
        mode = "constant"
        cval = border
    else:
        mode = openeo_scipy_modes[border]
        cval = 0

    result = convolve(data, kernel, mode, cval, replace_invalid) * factor
    result.attrs = data.attrs
    return result
