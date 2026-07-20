import numpy as np
import pytest
import xarray as xr

from openeo_processes_dedl_slim.process_implementations.cubes._filter import (
    filter_bands,
)
from openeo_processes_dedl_slim.process_implementations.cubes.general import (
    dimension_labels,
    rename_labels,
)
from openeo_processes_dedl_slim.process_implementations.cubes.mask import mask
from openeo_processes_dedl_slim.process_implementations.cubes.merge import merge_cubes
from tests.mockdata import create_multiresolution_rastercube


@pytest.fixture
def multires_cube(bounding_box, temporal_interval):
    return create_multiresolution_rastercube(
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
    )


@pytest.fixture
def multires_cube_dask(bounding_box, temporal_interval):
    pytest.importorskip("dask")
    return create_multiresolution_rastercube(
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        backend="dask",
    )


class TestMultiresolutionFixture:
    def test_variables_have_different_nan_patterns(self, multires_cube):
        """B02 (fine) has data everywhere; B08 (coarse) has NaN at every other pixel."""
        b02_nan = multires_cube["B02"].isnull().sum().item()
        b08_nan = multires_cube["B08"].isnull().sum().item()
        assert b08_nan > b02_nan

    def test_variables_share_spatial_dims(self, multires_cube):
        assert multires_cube["B02"].dims == multires_cube["B08"].dims

    def test_has_different_dtypes(self, multires_cube):
        assert multires_cube["B02"].dtype != multires_cube["B08"].dtype

    def test_has_per_variable_attrs(self, multires_cube):
        assert multires_cube["B02"].attrs.get("common_name") == "blue"
        assert multires_cube["B08"].attrs.get("common_name") == "nir"

    def test_has_dataset_attrs(self, multires_cube):
        assert multires_cube.attrs.get("title") == "multi-resolution test cube"

    def test_has_crs(self, multires_cube):
        assert multires_cube.odc.crs is not None

    def test_shared_temporal_dim(self, multires_cube):
        assert "t" in multires_cube.dims
        assert len(multires_cube.t) == 4


class TestNonHarmonizing:
    def test_filter_bands_selects_variables(self, multires_cube):
        result = filter_bands(data=multires_cube, bands=["B02"])
        assert set(result.data_vars) == {"B02"}

    def test_filter_bands_preserves_all_vars(self, multires_cube):
        result = filter_bands(data=multires_cube, bands=["B02", "B03", "B08"])
        assert set(result.data_vars) == {"B02", "B03", "B08"}

    def test_dimension_labels_returns_band_names(self, multires_cube):
        labels = dimension_labels(data=multires_cube, dimension="bands")
        assert list(labels) == ["B02", "B03", "B08"]

    def test_dimension_labels_returns_temporal(self, multires_cube):
        labels = dimension_labels(data=multires_cube, dimension="t")
        assert len(labels) == 4

    def test_rename_labels_renames_bands(self, multires_cube):
        result = rename_labels(
            data=multires_cube,
            dimension="bands",
            target=["blue", "green", "nir"],
            source=["B02", "B03", "B08"],
        )
        assert set(result.data_vars) == {"blue", "green", "nir"}
        assert "B02" not in result.data_vars

    def test_rename_labels_sequential_rename(self, multires_cube):
        labels = list(multires_cube.data_vars)
        target = [f"var_{i}" for i in range(len(labels))]
        result = rename_labels(data=multires_cube, dimension="bands", target=target)
        assert set(result.data_vars) == set(target)

    def test_mask_with_per_variable_mask(self, multires_cube):
        mask_data = multires_cube > 50
        result = mask(data=multires_cube, mask=mask_data, replacement=-1)
        assert set(result.data_vars) == {"B02", "B03", "B08"}

    def test_mask_with_dataarray_mask(self, multires_cube):
        mask_da = multires_cube["B02"] > 50
        result = mask(data=multires_cube, mask=mask_da, replacement=-1)
        assert set(result.data_vars) == {"B02", "B03", "B08"}

    def test_merge_cubes_non_overlapping(self, multires_cube):
        cube_a = multires_cube[["B02"]]
        cube_b = multires_cube[["B08"]].copy()
        result = merge_cubes(cube1=cube_a, cube2=cube_b)
        assert set(result.data_vars) == {"B02", "B08"}

    def test_mask_preserves_dask_laziness(self, multires_cube_dask):
        mask_data = multires_cube_dask > 50
        result = mask(data=multires_cube_dask, mask=mask_data, replacement=-1)
        for var in result.data_vars:
            import dask

            assert isinstance(result[var].data, dask.array.Array)

    def test_filter_bands_preserves_crs(self, multires_cube):
        result = filter_bands(data=multires_cube, bands=["B02"])
        assert result.odc.crs == multires_cube.odc.crs


class TestHarmonizingVirtualBand:
    """Tests for processes that stack variables into a virtual band dimension.

    After Dataset-level alignment (union of coordinates), to_array succeeds
    but coarser variables may contribute NaN. These tests document current behavior.
    """

    def test_apply_dimension_bands_produces_dataset(
        self, multires_cube, process_registry
    ):
        from functools import partial

        from openeo_pg_parser_networkx.pg_schema import ParameterReference

        process = partial(
            process_registry["mean"].implementation,
            ignore_nodata=True,
            data=ParameterReference(from_parameter="data"),
        )
        from openeo_processes_dedl_slim.process_implementations.cubes.apply import (
            apply_dimension,
        )

        result = apply_dimension(
            data=multires_cube,
            process=process,
            dimension="bands",
        )
        assert isinstance(result, xr.Dataset)

    def test_reduce_dimension_bands_produces_dataset(
        self, multires_cube, process_registry
    ):
        from functools import partial

        from openeo_pg_parser_networkx.pg_schema import ParameterReference

        process = partial(
            process_registry["mean"].implementation,
            ignore_nodata=True,
            data=ParameterReference(from_parameter="data"),
        )
        from openeo_processes_dedl_slim.process_implementations.cubes.reduce import (
            reduce_dimension,
        )

        result = reduce_dimension(
            data=multires_cube,
            reducer=process,
            dimension="bands",
        )
        assert isinstance(result, xr.Dataset)

    def test_fit_curve_along_temporal(self, multires_cube):
        from openeo_processes_dedl_slim.process_implementations.ml.curve_fitting import (
            fit_curve,
        )

        result = fit_curve(
            data=multires_cube,
            parameters=[1, 1],
            function=lambda x, parameters: x,
            dimension="t",
        )
        assert isinstance(result, xr.Dataset)

    def test_run_udf_produces_dataset(self, multires_cube):
        from openeo_processes_dedl_slim.process_implementations.udf.udf import run_udf

        simple_udf = """
from openeo.udf import XarrayDataCube, UdfData

def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    return cube
"""
        result = run_udf(
            data=multires_cube,
            udf=simple_udf,
            runtime="Python",
        )
        assert isinstance(result, xr.Dataset)


class TestMultiResolutionDask:
    def test_dask_fixture_is_lazy(self, multires_cube_dask):
        import dask

        for var in multires_cube_dask.data_vars:
            assert isinstance(multires_cube_dask[var].data, dask.array.Array)

    def test_filter_bands_on_dask(self, multires_cube_dask):
        result = filter_bands(data=multires_cube_dask, bands=["B08"])
        assert set(result.data_vars) == {"B08"}
