from functools import partial

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask_slim.process_implementations.cubes.apply import (
    apply,
    apply_dimension,
    apply_kernel,
)
from openeo_processes_dask_slim.process_implementations.data_model import (
    _stack_bands,
    _unstack_bands,
)
from tests.general_checks import assert_numpy_equals_dask_numpy, general_output_checks
from tests.mockdata import create_fake_rastercube


def _first_var_data(cube):
    first_var = list(cube.data_vars.values())[0]
    return first_var.data


def _first_var_shape(cube):
    first_var = list(cube.data_vars.values())[0]
    return first_var.shape


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply(temporal_interval, bounding_box, random_raster_data, process_registry):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process = partial(
        process_registry["add"].implementation,
        y=1,
        x=ParameterReference(from_parameter="x"),
    )

    output_cube = apply(data=input_cube, process=_process)

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
        expected_results=(input_cube + 1),
    )

    xr.testing.assert_equal(output_cube, input_cube + 1)


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_add(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process = partial(
        process_registry["add"].implementation,
        y=1,
        x=ParameterReference(from_parameter="data"),
    )

    # Target dimension is null and therefore defaults to the source dimension
    output_cube_same_pixels = apply_dimension(
        data=input_cube, process=_process, dimension="x"
    )

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube_same_pixels,
        verify_attrs=True,
        verify_crs=True,
        expected_results=(input_cube + 1),
    )


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_ordering_processes(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    var_name = list(input_cube.data_vars)[0]

    _process_order = partial(
        process_registry["order"].implementation,
        data=ParameterReference(from_parameter="data"),
        nodata=True,
    )

    output_cube_order = apply_dimension(
        data=input_cube[var_name],
        process=_process_order,
        dimension="x",
        target_dimension="target",
    )

    assert isinstance(output_cube_order.data, np.ndarray)


@pytest.mark.parametrize("size", [(6, 5, 30, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_quantile_processes(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )
    probability = 4

    _process_quantile = partial(
        process_registry["quantiles"].implementation,
        data=ParameterReference(from_parameter="data"),
        probabilities=probability,
    )

    output_cube_quantile = apply_dimension(
        data=input_cube,
        process=_process_quantile,
        dimension="t",
    )
    assert isinstance(output_cube_quantile, xr.Dataset)


@pytest.mark.parametrize("size", [(6, 5, 10, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_interpolate_processes(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )
    stacked = _stack_bands(input_cube)
    stacked[3, 2, 4, 0] = np.nan
    input_cube = _unstack_bands(stacked)
    var_name = list(input_cube.data_vars)[0]

    _process_interpolate = partial(
        process_registry["array_interpolate_linear"].implementation,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube = apply_dimension(
        data=input_cube,
        process=_process_interpolate,
        dimension="t",
    )
    assert isinstance(output_cube, xr.Dataset)


@pytest.mark.parametrize("size", [(6, 5, 10, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_modify_processes(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process_modify = partial(
        process_registry["array_modify"].implementation,
        data=ParameterReference(from_parameter="data"),
        values=[2, 3],
        index=3,
    )

    output_cube = apply_dimension(
        data=input_cube,
        process=_process_modify,
        dimension="bands",
    )
    assert output_cube is not None


@pytest.mark.parametrize("size", [(6, 5, 10, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_filter_processes(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _condition = partial(
        process_registry["gt"].implementation,
        x=ParameterReference(from_parameter="x"),
        y=10,
    )

    _process_filter = partial(
        process_registry["array_filter"].implementation,
        data=ParameterReference(from_parameter="data"),
        condition=_condition,
    )

    output_cube = apply_dimension(
        data=input_cube,
        process=_process_filter,
        dimension="bands",
    )
    assert output_cube is not None


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_kernel(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    # Following kernel should leave cube unchanged
    kernel = np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    output_cube = apply_kernel(data=input_cube, kernel=kernel)

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
        expected_results=input_cube,
    )

    xr.testing.assert_equal(output_cube, input_cube)


@pytest.mark.parametrize("size", [(6, 5, 30, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_cumsum_process(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process_cumsum = partial(
        process_registry["cumsum"].implementation,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube_cumsum = apply_dimension(
        data=input_cube,
        process=_process_cumsum,
        dimension="t",
    ).compute()

    assert isinstance(output_cube_cumsum, xr.Dataset)

    stacked = _stack_bands(input_cube)
    stacked[15, ...] = np.nan
    input_cube = _unstack_bands(stacked)

    _process_cumsum_with_nan = partial(
        process_registry["cumsum"].implementation,
        data=ParameterReference(from_parameter="data"),
        ignore_nodata=False,
    )

    output_cube_cumsum_with_nan = apply_dimension(
        data=input_cube,
        process=_process_cumsum_with_nan,
        dimension="t",
    ).compute()

    assert isinstance(output_cube_cumsum_with_nan, xr.Dataset)


@pytest.mark.parametrize("size", [(6, 5, 30, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_cumproduct_process(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process_cumsum = partial(
        process_registry["cumproduct"].implementation,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube_cumprod = apply_dimension(
        data=input_cube,
        process=_process_cumsum,
        dimension="t",
    ).compute()

    assert isinstance(output_cube_cumprod, xr.Dataset)

    stacked = _stack_bands(input_cube)
    stacked[15, ...] = np.nan
    input_cube = _unstack_bands(stacked)

    _process_cumprod_with_nan = partial(
        process_registry["cumproduct"].implementation,
        data=ParameterReference(from_parameter="data"),
        ignore_nodata=False,
    )

    output_cube_cumprod_with_nan = apply_dimension(
        data=input_cube,
        process=_process_cumprod_with_nan,
        dimension="t",
    ).compute()

    assert isinstance(output_cube_cumprod_with_nan, xr.Dataset)


@pytest.mark.parametrize("size", [(6, 5, 30, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_cummax_process(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process_cummax = partial(
        process_registry["cummax"].implementation,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube_cummax = apply_dimension(
        data=input_cube,
        process=_process_cummax,
        dimension="t",
    ).compute()

    assert isinstance(output_cube_cummax, xr.Dataset)

    stacked = _stack_bands(input_cube)
    stacked[15, ...] = np.nan
    input_cube = _unstack_bands(stacked)

    _process_cummax_with_nan = partial(
        process_registry["cummax"].implementation,
        data=ParameterReference(from_parameter="data"),
        ignore_nodata=False,
    )

    output_cube_cummax_with_nan = apply_dimension(
        data=input_cube,
        process=_process_cummax_with_nan,
        dimension="t",
    ).compute()

    assert isinstance(output_cube_cummax_with_nan, xr.Dataset)


@pytest.mark.parametrize("size", [(6, 5, 30, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_cummin_process(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process_cummin = partial(
        process_registry["cummin"].implementation,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube_cummin = apply_dimension(
        data=input_cube,
        process=_process_cummin,
        dimension="t",
    ).compute()

    assert isinstance(output_cube_cummin, xr.Dataset)

    stacked = _stack_bands(input_cube)
    stacked[15, ...] = np.nan
    input_cube = _unstack_bands(stacked)

    _process_cummin_with_nan = partial(
        process_registry["cummin"].implementation,
        data=ParameterReference(from_parameter="data"),
        ignore_nodata=False,
    )

    output_cube_cummin_with_nan = apply_dimension(
        data=input_cube,
        process=_process_cummin_with_nan,
        dimension="t",
    ).compute()

    assert isinstance(output_cube_cummin_with_nan, xr.Dataset)
