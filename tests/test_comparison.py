from datetime import datetime
from functools import partial

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask_slim.process_implementations.cubes.utils import isnull
from openeo_processes_dask_slim.process_implementations import merge_cubes
from openeo_processes_dask_slim.process_implementations.comparison import *
from openeo_processes_dask_slim.process_implementations.logic import *
from openeo_processes_dask_slim.process_implementations.cubes.apply import apply
from openeo_processes_dask_slim.process_implementations.cubes.reduce import (
    reduce_dimension,
)
from tests.general_checks import assert_numpy_equals_dask_numpy, general_output_checks
from tests.mockdata import create_fake_rastercube


def _get_data(cube):
    if isinstance(cube, xr.Dataset):
        first_var = list(cube.data_vars.values())[0]
        return first_var.data
    return cube.data


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, True),
        (np.nan, False),
        (np.array([1, np.nan]), np.array([True, False])),
        ({"test": "ok"}, True),
        ([1, 2, np.nan], np.array([True, True, False])),
    ],
)
@pytest.mark.parametrize("is_dask", [True, False])
def test_is_valid(value, expected, is_dask):
    value = np.asarray(value)

    if is_dask:
        value = da.from_array(value)

    output = is_valid(value)
    np.testing.assert_array_equal(output, expected)

    if is_dask:
        assert hasattr(output, "dask")


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, False),
        (np.nan, False),
    ],
)
def test_is_nodata(value, expected):
    value = np.asarray(value)
    output = is_nodata(value)
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, False),
        (np.inf, True),
        (-np.inf, True),
        (np.nan, False),
    ],
)
def test_is_infinite(value, expected):
    output = is_infinite(value)
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("x,expected", [(True, False), (False, True)])
def test_not(x, expected):
    output = _not(x)
    assert output == expected


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_and(x, y, expected):
    output = _and(x, y)
    assert output == expected


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (True, True, True),
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
def test_or(x, y, expected):
    output = _or(x, y)
    assert output == expected


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
def test_xor(x, y, expected):
    output = xor(x, y)
    assert output == expected


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (True, True, True),
        (True, False, False),
        (False, True, np.nan),
        (False, False, np.nan),
    ],
)
def test_if(x, y, expected):
    output = _if(x, y)
    if isinstance(expected, float) and np.isnan(expected):
        assert isnull(output)
    else:
        assert output == expected


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (True, False, True),
        (False, True, True),
        (True, True, False),
        (False, False, False),
    ],
)
def test_neq_op(x, y, expected):
    output = neq(x, y)
    assert output == expected


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (True, False, True),
        (False, True, True),
        (True, True, True),
        (False, False, False),
    ],
)
def test_or(x, y, expected):
    output = _or(x, y)
    assert output == expected


@pytest.mark.parametrize(
    "x",
    [True, False, 0, 1, 1.0, np.array([1, 2, 3]), np.array([[1, 2], [3, 4]])],
)
@pytest.mark.parametrize(
    "y",
    [True, False, 0, 1, 1.0, np.array([1, 2, 3]), np.array([[1, 2], [3, 4]])],
)
def test_eq_numpy(x, y):
    try:
        output = eq(x, y)
    except ValueError:
        return

    try:
        expected = np.equal(x, y)
        assert output == expected
    except ValueError:
        pass


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (3, 3, True),
        (3, 0, False),
        (0, 3, False),
    ],
)
def test_eq(x, y, expected):
    output = eq(x, y)
    assert output == expected


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (3, 3, False),
        (3, 0, True),
        (0, 3, False),
    ],
)
def test_gt(x, y, expected):
    output = gt(x, y)
    assert output == expected


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (3, 3, True),
        (3, 0, True),
        (0, 3, False),
    ],
)
def test_gte(x, y, expected):
    output = gte(x, y)
    assert output == expected


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (3, 3, False),
        (3, 0, False),
        (0, 3, True),
    ],
)
def test_lt(x, y, expected):
    output = lt(x, y)
    assert output == expected


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (3, 3, True),
        (3, 0, False),
        (0, 3, True),
    ],
)
def test_lte(x, y, expected):
    output = lte(x, y)
    assert output == expected


@pytest.mark.parametrize(
    "x,min,max,exclude_max,expected",
    [
        (3, 2, 4, False, True),
        (4, 2, 4, False, True),
        (4, 2, 4, True, False),
        (2, 2, 4, False, True),
        (1, 2, 4, False, False),
    ],
)
def test_between(x, min, max, exclude_max, expected):
    assert between(x, min, max, exclude_max) == expected
    assert between(np.array([x]), min, max, exclude_max) == expected
    assert between(da.from_array(np.array([x])), min, max, exclude_max) == expected


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_is(temporal_interval, bounding_box, random_raster_data, process_registry):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process = partial(
        process_registry["is_valid"].implementation,
        x=ParameterReference(from_parameter="x"),
    )
    output_cube = apply(data=input_cube, process=_process)
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
    )
    assert isinstance(_get_data(output_cube), dask.array.Array)
    xr.testing.assert_equal(output_cube, xr.ones_like(input_cube))

    _process = partial(
        process_registry["is_infinite"].implementation,
        x=ParameterReference(from_parameter="x"),
    )
    output_cube = apply(data=input_cube, process=_process)
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
    )
    assert isinstance(_get_data(output_cube), dask.array.Array)
    xr.testing.assert_equal(output_cube, xr.zeros_like(input_cube))


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_compare(temporal_interval, bounding_box, random_raster_data, process_registry):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process = partial(
        process_registry["eq"].implementation,
        y=200,
        x=ParameterReference(from_parameter="x"),
    )
    output_cube = apply(data=input_cube, process=_process)
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
    )
    assert isinstance(_get_data(output_cube), dask.array.Array)
    xr.testing.assert_equal(output_cube, xr.zeros_like(input_cube))

    _process = partial(
        process_registry["neq"].implementation,
        y=200,
        x=ParameterReference(from_parameter="x"),
    )
    output_cube = apply(data=input_cube, process=_process)
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
    )
    assert isinstance(_get_data(output_cube), dask.array.Array)
    xr.testing.assert_equal(output_cube, xr.ones_like(input_cube))

    _process = partial(
        process_registry["gt"].implementation,
        x=ParameterReference(from_parameter="x"),
        y=200,
    )
    output_cube_gt = apply(data=input_cube, process=_process)
    _process = partial(
        process_registry["gte"].implementation,
        x=ParameterReference(from_parameter="x"),
        y=200,
    )
    output_cube_gte = apply(data=input_cube, process=_process)
    assert isinstance(_get_data(output_cube_gt), dask.array.Array)
    xr.testing.assert_equal(output_cube_gt, output_cube_gte)

    _process = partial(
        process_registry["lt"].implementation,
        x=ParameterReference(from_parameter="x"),
        y=200,
    )
    output_cube_lt = apply(data=input_cube, process=_process)
    _process = partial(
        process_registry["lte"].implementation,
        x=ParameterReference(from_parameter="x"),
        y=200,
    )
    output_cube_lte = apply(data=input_cube, process=_process)
    assert isinstance(_get_data(output_cube_lt), dask.array.Array)
    xr.testing.assert_equal(output_cube_lt, output_cube_lte)

    _process = partial(
        process_registry["between"].implementation,
        x=ParameterReference(from_parameter="x"),
        min=200,
        max=300,
    )
    output_cube = apply(data=input_cube, process=_process)
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
    )
    xr.testing.assert_equal(output_cube, xr.zeros_like(input_cube))
    _process = partial(
        process_registry["between"].implementation,
        x=ParameterReference(from_parameter="x"),
        min=200,
        max=300,
        exclude_max=True,
    )
    output_cube_b2 = apply(data=input_cube, process=_process)
    assert isinstance(_get_data(output_cube), dask.array.Array)
    xr.testing.assert_equal(output_cube, output_cube_b2)


@pytest.mark.parametrize("size", [(6, 5, 4, 3)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_merge_cubes_eq(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    origin_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B01", "B02", "B03"],
        backend="dask",
    )

    cube_1 = origin_cube
    cube_2 = origin_cube

    merged_cube_eq = merge_cubes(
        cube_1,
        cube_2,
        partial(
            process_registry["eq"].implementation,
            x=ParameterReference(from_parameter="x"),
            y=ParameterReference(from_parameter="y"),
        ),
    )

    assert isinstance(_get_data(merged_cube_eq), dask.array.Array)

    merged_cube_neq = merge_cubes(
        cube_1,
        cube_2,
        partial(
            process_registry["neq"].implementation,
            x=ParameterReference(from_parameter="x"),
            y=ParameterReference(from_parameter="y"),
        ),
    )

    assert isinstance(_get_data(merged_cube_neq), dask.array.Array)
    xr.testing.assert_equal(merged_cube_eq, merged_cube_neq + 1)

    merged_cube_lt = merge_cubes(
        cube_1,
        cube_2,
        partial(
            process_registry["lt"].implementation,
            x=ParameterReference(from_parameter="x"),
            y=ParameterReference(from_parameter="y"),
        ),
    )

    assert isinstance(_get_data(merged_cube_lt), dask.array.Array)
    xr.testing.assert_equal(merged_cube_lt, merged_cube_neq)

from openeo_processes_dask_slim.process_implementations.utils import get_scalar_type


@pytest.mark.parametrize(
    "value, expected",
    [
        (1, np.int64),
        ("test", np.str_),
        (None, np.object_),
        (np.array([1, 2]), np.int64),
        (da.from_array(np.array([1, 2])), np.int64),
    ],
)
def test_get_scalar_type(value, expected):
    assert get_scalar_type(value) is expected
