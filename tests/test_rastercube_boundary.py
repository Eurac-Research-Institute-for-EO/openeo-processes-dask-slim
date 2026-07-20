import numpy as np
import pytest
import xarray as xr

from openeo_processes_dedl_slim.process_implementations.cubes._filter import (
    filter_bands,
    filter_bbox,
    filter_labels,
    filter_temporal,
)
from openeo_processes_dedl_slim.process_implementations.cubes.apply import (
    apply,
    apply_dimension,
    apply_kernel,
)
from openeo_processes_dedl_slim.process_implementations.cubes.general import (
    add_dimension,
    dimension_labels,
    drop_dimension,
    rename_dimension,
    rename_labels,
    trim_cube,
)
from openeo_processes_dedl_slim.process_implementations.cubes.indices import ndvi
from openeo_processes_dedl_slim.process_implementations.cubes.mask import mask
from openeo_processes_dedl_slim.process_implementations.cubes.merge import merge_cubes
from openeo_processes_dedl_slim.process_implementations.cubes.reduce import (
    reduce_dimension,
    reduce_spatial,
)


@pytest.fixture
def dataarray_cube():
    return xr.DataArray(
        np.ones((3, 3, 2)),
        dims=["y", "x", "t"],
        coords={
            "y": [0.0, 1.0, 2.0],
            "x": [0.0, 1.0, 2.0],
            "t": [0, 1],
        },
    )


class TestRejectsDataArray:
    def test_filter_temporal(self, dataarray_cube):
        with pytest.raises(TypeError, match="RasterCube must be an xr.Dataset"):
            filter_temporal(dataarray_cube, extent=[None, None])

    def test_filter_labels(self, dataarray_cube):
        with pytest.raises(TypeError, match="RasterCube must be an xr.Dataset"):
            filter_labels(dataarray_cube, condition=lambda x: True, dimension="x")

    def test_filter_bands(self, dataarray_cube):
        with pytest.raises(TypeError, match="RasterCube must be an xr.Dataset"):
            filter_bands(dataarray_cube, bands=["B02"])

    def test_filter_bbox(self, dataarray_cube):
        with pytest.raises(TypeError, match="RasterCube must be an xr.Dataset"):
            filter_bbox(dataarray_cube, extent=None)

    def test_apply(self, dataarray_cube):
        with pytest.raises(TypeError, match="RasterCube must be an xr.Dataset"):
            apply(dataarray_cube, process=lambda x: x)

    def test_apply_dimension(self, dataarray_cube):
        with pytest.raises(TypeError, match="RasterCube must be an xr.Dataset"):
            apply_dimension(dataarray_cube, process=lambda x: x, dimension="x")

    def test_apply_kernel(self, dataarray_cube):
        with pytest.raises(TypeError, match="RasterCube must be an xr.Dataset"):
            apply_kernel(dataarray_cube, kernel=np.ones((3, 3)))

    def test_mask(self, dataarray_cube):
        with pytest.raises(TypeError, match="RasterCube must be an xr.Dataset"):
            mask(data=dataarray_cube, mask=dataarray_cube > 0.5)

    def test_ndvi(self, dataarray_cube):
        ds = xr.Dataset(
            {
                "nir": xr.DataArray(np.ones((3, 3, 2)), dims=["y", "x", "t"]),
                "red": xr.DataArray(np.zeros((3, 3, 2)), dims=["y", "x", "t"]),
            }
        )
        with pytest.raises(TypeError, match="RasterCube must be an xr.Dataset"):
            ndvi(dataarray_cube)

    def test_reduce_dimension(self, dataarray_cube):
        with pytest.raises(TypeError, match="RasterCube must be an xr.Dataset"):
            reduce_dimension(dataarray_cube, reducer=np.mean, dimension="x")

    def test_reduce_spatial(self, dataarray_cube):
        with pytest.raises(TypeError, match="RasterCube must be an xr.Dataset"):
            reduce_spatial(dataarray_cube, reducer=np.mean)

    def test_merge_cubes(self, dataarray_cube):
        with pytest.raises(Exception):
            merge_cubes(dataarray_cube, dataarray_cube)

    def test_trim_cube(self, dataarray_cube):
        with pytest.raises(TypeError, match="RasterCube must be an xr.Dataset"):
            trim_cube(dataarray_cube)

    def test_dimension_labels(self, dataarray_cube):
        with pytest.raises(TypeError, match="RasterCube must be an xr.Dataset"):
            dimension_labels(dataarray_cube, dimension="x")

    def test_drop_dimension(self, dataarray_cube):
        with pytest.raises(TypeError, match="RasterCube must be an xr.Dataset"):
            drop_dimension(dataarray_cube, "x")

    def test_add_dimension(self, dataarray_cube):
        with pytest.raises(TypeError, match="RasterCube must be an xr.Dataset"):
            add_dimension(dataarray_cube, name="new_dim", label="test")

    def test_rename_dimension(self, dataarray_cube):
        with pytest.raises(TypeError, match="RasterCube must be an xr.Dataset"):
            rename_dimension(dataarray_cube, source="x", target="new_x")

    def test_rename_labels(self, dataarray_cube):
        with pytest.raises(TypeError, match="RasterCube must be an xr.Dataset"):
            rename_labels(dataarray_cube, dimension="x", target=["a", "b"])
