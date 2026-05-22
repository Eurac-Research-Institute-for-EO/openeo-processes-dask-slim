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


def _get_data(cube):
    """Get underlying data from a cube, handling Dataset per-variable."""
    if isinstance(cube, xr.Dataset):
        first_var = list(cube.data_vars.values())[0]
        return first_var.data
    return cube.data


def general_output_checks(
    input_cube: RasterCube,
    output_cube: RasterCube,
    expected_results=None,
    verify_crs: bool = False,
    verify_attrs: bool = False,
    expected_dims: list = None,
    rtol=1e-06,
):
    input_data = _get_data(input_cube)
    output_data = _get_data(output_cube)

    assert isinstance(output_data, type(input_data))

    assert input_cube.openeo is not None
    assert output_cube.openeo is not None

    if verify_crs:
        assert _get_crs(input_cube) == _get_crs(output_cube)

    if verify_attrs:
        assert input_cube.attrs == output_cube.attrs

    if expected_results is not None:
        if isinstance(expected_results, xr.Dataset):
            for var_name in expected_results.data_vars:
                expected_var = expected_results[var_name]
                if isinstance(expected_var.data, np.ndarray):
                    computed_var = expected_var.data
                elif isinstance(expected_var.data, da.Array):
                    computed_var = expected_var.data.compute()
                else:
                    computed_var = expected_var.data
                actual_var = output_cube[var_name]
                if isinstance(actual_var.data, da.Array):
                    actual_var = actual_var.data.compute()
                else:
                    actual_var = actual_var.data
                np.testing.assert_allclose(
                    actual_var, computed_var, equal_nan=True, rtol=rtol
                )
        else:
            if isinstance(output_data, np.ndarray):
                computed = output_data
            elif isinstance(output_data, da.Array):
                computed = output_data.compute()
            else:
                raise TypeError(f"Unsupported array type: {type(output_data)}")

            np.testing.assert_allclose(
                computed, expected_results, equal_nan=True, rtol=rtol
            )

    if expected_dims is not None:
        actual_dims = output_cube.dims
        assert len(expected_dims) == len(actual_dims)
        assert set(actual_dims) == set(expected_dims)


def assert_numpy_equals_dask_numpy(numpy_cube, dask_cube, func):
    numpy_result = func(numpy_cube)
    dask_result = func(dask_cube)
    general_output_checks(dask_cube, dask_result)
    numpy_data = _get_data(numpy_result)
    dask_data = _get_data(dask_result)
    np.testing.assert_allclose(
        numpy_data, dask_data.compute(), equal_nan=True
    )
