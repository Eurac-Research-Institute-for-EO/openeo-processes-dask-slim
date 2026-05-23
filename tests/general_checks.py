# Checks here are inspired by makepath/xarray-spatial/tests/general_checks.py
from typing import List

import dask.array as da
import numpy as np
import pyproj
import xarray as xr

from openeo_processes_dask_slim.process_implementations.data_model import RasterCube


def _get_crs(cube):
    crs = cube.odc.crs
    if crs is None and "crs" in cube.attrs:
        return pyproj.CRS(cube.attrs["crs"])
    return crs


def general_output_checks(
    input_cube: RasterCube,
    output_cube: RasterCube,
    expected_results=None,
    verify_crs: bool = False,
    verify_attrs: bool = False,
    expected_dims: list = None,
    rtol=1e-06,
):
    if isinstance(output_cube, xr.Dataset):
        if isinstance(input_cube, xr.Dataset) and set(output_cube.data_vars) == set(
            input_cube.data_vars
        ):
            pass
        else:
            assert len(output_cube.data_vars) > 0
    else:
        assert isinstance(output_cube.data, type(input_cube.data))

    if hasattr(input_cube, "openeo"):
        assert input_cube.openeo is not None
    if hasattr(output_cube, "openeo"):
        assert output_cube.openeo is not None

    if verify_crs:
        assert _get_crs(input_cube) == _get_crs(output_cube)

    if verify_attrs:
        assert input_cube.attrs == output_cube.attrs

    if expected_results is not None:
        if isinstance(output_cube, xr.Dataset):
            if not isinstance(expected_results, xr.Dataset):
                if "bands" in expected_results.dims:
                    expected_results = expected_results.to_dataset(dim="bands")
                else:
                    expected_results = expected_results.to_dataset(name="result")
            xr.testing.assert_allclose(output_cube, expected_results)
        elif isinstance(expected_results, xr.Dataset):
            xr.testing.assert_allclose(output_cube, expected_results)
        else:
            if isinstance(output_cube.data, np.ndarray):
                output_data = output_cube.data
            elif isinstance(output_cube.data, da.Array):
                output_data = output_cube.data.compute()
            else:
                raise TypeError(f"Unsupported array type: {type(output_cube.data)}")
            np.testing.assert_allclose(
                output_data, expected_results, equal_nan=True, rtol=rtol
            )

    if expected_dims is not None:
        actual_dims = output_cube.dims
        assert len(expected_dims) == len(actual_dims)
        assert set(actual_dims) == set(expected_dims)


def assert_numpy_equals_dask_numpy(numpy_cube, dask_cube, func):
    numpy_result = func(numpy_cube)
    dask_result = func(dask_cube)
    general_output_checks(dask_cube, dask_result)
    np.testing.assert_allclose(
        numpy_result.data, dask_result.data.compute(), equal_nan=True
    )
