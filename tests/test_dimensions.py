import numpy as np
import pytest

from openeo_processes_dask_slim.process_implementations.cubes.general import *
from openeo_processes_dask_slim.process_implementations.exceptions import (
    DimensionLabelCountMismatch,
    DimensionNotAvailable,
)
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(30, 30, 20, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_add_dimension(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    output_cube = add_dimension(data=input_cube, name="other", label="test")

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        expected_dims=["x", "y", "t", "other"],
    )
    assert set(output_cube.data_vars) == {"B02", "B03", "B04", "B08"}
    assert output_cube.openeo.temporal_dims[0] == "t"
    assert output_cube.openeo.spatial_dims == ("x", "y")
    assert output_cube.openeo.other_dims[0] == "other"

    output_cube_2 = add_dimension(
        data=input_cube, name="weird", label="test", type="temporal"
    )
    assert output_cube_2.openeo.temporal_dims[1] == "weird"


@pytest.mark.parametrize("size", [(30, 30, 20, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_add_dimension_dataset(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
        as_dataset=True,
    )

    output_cube = add_dimension(data=input_cube, name="other", label="test")

    assert "other" in output_cube.dims
    assert set(output_cube.data_vars) == {"B02", "B03", "B04", "B08"}

    output_cube_2 = add_dimension(
        data=input_cube, name="weird", label="test", type="temporal"
    )
    assert "weird" in output_cube_2.dims


@pytest.mark.parametrize("size", [(30, 30, 1, 2)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_drop_dimension(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B04"],
        backend="dask",
    )
    with pytest.raises(DimensionNotAvailable):
        drop_dimension(input_cube, "notthere")

    with pytest.raises(DimensionLabelCountMismatch):
        drop_dimension(input_cube, "x")

    output_cube = drop_dimension(input_cube, "t")
    assert "t" not in output_cube.dims
    assert set(output_cube.dims) == {"x", "y"}


@pytest.mark.parametrize("size", [(30, 30, 1, 2)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_rename_dimension(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B04"],
        backend="dask",
    )
    output_cube = rename_dimension(input_cube, source="t", target="time")

    assert "t" not in output_cube.dims
    assert "time" in output_cube.dims
    assert "time" in output_cube.openeo.temporal_dims
    assert "time" not in output_cube.openeo.spatial_dims

    with pytest.raises(DimensionNotAvailable):
        rename_dimension(input_cube, source="notthere", target="there")

    with pytest.raises(Exception):
        rename_dimension(input_cube, source="y", target="x")


@pytest.mark.parametrize("size", [(30, 30, 1, 5)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_rename_labels(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B05", "B08"],
        backend="dask",
    )
    x_target = list(range(100, 130))
    output_cube = rename_labels(input_cube, dimension="x", target=x_target)

    assert 100 in output_cube["x"]

    with pytest.raises(DimensionNotAvailable):
        rename_labels(input_cube, dimension="nonexistent", target=[0])

    with pytest.raises(Exception):
        rename_labels(
            input_cube,
            dimension="x",
            target=[float(input_cube["x"].values[0])] + list(range(100, 129)),
        )

    with pytest.raises(Exception):
        rename_labels(
            input_cube,
            dimension="x",
            target=list(range(100, 150)),
        )


@pytest.mark.parametrize("size", [(30, 30, 2, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_rename_labels_time(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    t_labels = dimension_labels(input_cube, dimension="t")
    output_cube = rename_labels(
        input_cube, dimension="t", source=t_labels, target=["first_date", "second_date"]
    )
    assert "first_date" in output_cube["t"].values

    output_cube_2 = rename_labels(
        input_cube, dimension="t", source=[t_labels[-1]], target=["second_date"]
    )
    assert "second_date" in output_cube_2["t"].values


@pytest.mark.parametrize("size", [(30, 30, 20, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_trim_cube(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )
    input_cube["B04"] = (("x", "y", "t"), np.zeros((30, 30, 20)) * np.nan)
    output_cube = trim_cube(input_cube)
    assert set(output_cube.data_vars) == {"B02", "B03", "B04", "B08"}

    all_nan = input_cube * np.nan
    with pytest.raises(ValueError):
        output_cube = trim_cube(all_nan)
