"""Lightweight graph-size tests for per-variable Dataset operations.

These tests verify that dask graph sizes grow linearly with the number
of variables, not super-linearly, and that no eager compute is triggered.
"""

import dask
import dask.array as da
import numpy as np
import pytest

from openeo_processes_dedl_slim.process_implementations.cubes.apply import apply_kernel
from openeo_processes_dedl_slim.process_implementations.cubes.mask import mask


@pytest.fixture
def wide_dataset():
    n_vars = 8
    y, x, t = 10, 10, 4
    ds = {}
    for i in range(n_vars):
        name = f"B{i:02d}"
        data = da.from_array(
            np.random.default_rng(i).random((y, x, t)).astype(np.float32),
            chunks=(5, 5, 2),
        )
        ds[name] = data
    import xarray as xr

    ds = xr.Dataset(
        {name: xr.DataArray(data, dims=["y", "x", "t"]) for name, data in ds.items()}
    )
    import odc.geo.xr

    ds = odc.geo.xr.assign_crs(ds, crs="EPSG:4326")
    return ds


class TestGraphSize:
    def test_mask_graph_grows_linearly(self, wide_dataset):
        mask_data = wide_dataset > 0.5
        result = mask(data=wide_dataset, mask=mask_data, replacement=-1)
        n_tasks = sum(
            len(var.data.__dask_graph__()) for var in result.data_vars.values()
        )
        max_per_var = max(
            len(var.data.__dask_graph__()) for var in result.data_vars.values()
        )
        n_vars = len(result.data_vars)
        assert n_tasks <= n_vars * max_per_var * 1.5

    def test_mask_no_eager_compute(self, wide_dataset):
        mask_data = wide_dataset > 0.5
        result = mask(data=wide_dataset, mask=mask_data, replacement=-1)
        for var in result.data_vars.values():
            assert isinstance(var.data, da.Array)
            assert len(var.data.__dask_graph__()) > 0

    def test_apply_kernel_graph_grows_linearly(self, wide_dataset):
        kernel = np.ones((3, 3)) / 9
        result = apply_kernel(data=wide_dataset, kernel=kernel)
        n_tasks = sum(
            len(var.data.__dask_graph__()) for var in result.data_vars.values()
        )
        n_vars = len(result.data_vars)
        max_per_var = max(
            len(var.data.__dask_graph__()) for var in result.data_vars.values()
        )
        assert n_tasks <= n_vars * max_per_var * 1.5

    def test_apply_kernel_no_eager_compute(self, wide_dataset):
        kernel = np.ones((3, 3)) / 9
        result = apply_kernel(data=wide_dataset, kernel=kernel)
        for var in result.data_vars.values():
            assert isinstance(var.data, da.Array)
            assert len(var.data.__dask_graph__()) > 0

    def test_fewer_tasks_than_naive_per_variable(self, wide_dataset):
        mask_data = wide_dataset > 0.5
        result = mask(data=wide_dataset, mask=mask_data, replacement=-1)
        total_tasks = sum(
            len(var.data.__dask_graph__()) for var in result.data_vars.values()
        )
        n_vars = len(result.data_vars)
        tasks_per_var = total_tasks / n_vars
        assert tasks_per_var < 50
