import copy
import sys
from functools import partial
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import (
    BoundingBox,
    ParameterReference,
    TemporalInterval,
)

from openeo_processes_dedl_slim.process_implementations.cubes._filter import *
from openeo_processes_dedl_slim.process_implementations.cubes.reduce import (
    reduce_dimension,
)
from openeo_processes_dedl_slim.process_implementations.exceptions import (
    DimensionNotAvailable,
    TemporalExtentEmpty,
)
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(30, 30, 30, 1)])
@pytest.mark.parametrize("dtype", [np.uint8])
def test_filter_temporal(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02"],
        backend="dask",
    )

    temporal_interval_part = TemporalInterval.model_validate(
        ["2018-05-15T00:00:00", "2018-06-01T00:00:00"]
    )
    output_cube = filter_temporal(data=input_cube, extent=temporal_interval_part)

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=True,
    )

    xr.testing.assert_equal(
        output_cube,
        input_cube.loc[{"t": slice("2018-05-15T00:00:00", "2018-05-31T23:59:59")}],
    )

    with pytest.raises(DimensionNotAvailable):
        filter_temporal(
            data=input_cube, extent=temporal_interval_part, dimension="immissing"
        )

    with pytest.raises(TemporalExtentEmpty):
        filter_temporal(
            data=input_cube,
            extent=["2018-05-31T23:59:59", "2018-05-15T00:00:00"],
        )

    temporal_interval_open = TemporalInterval.model_validate(
        [None, "2018-05-03T00:00:00"]
    )
    output_cube = filter_temporal(data=input_cube, extent=temporal_interval_open)

    xr.testing.assert_equal(
        output_cube,
        input_cube.loc[{"t": slice("2018-05-01T00:00:00", "2018-05-02T23:59:59")}],
    )

    new_coords = list(copy.deepcopy(input_cube.coords["t"].data))
    new_coords[1] = pd.NaT
    invalid_input_cube = input_cube.assign_coords({"t": np.array(new_coords)})
    filter_temporal(invalid_input_cube, temporal_interval)


@pytest.mark.parametrize("size", [(30, 30, 30, 3)])
@pytest.mark.parametrize("dtype", [np.uint8])
def test_filter_labels(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04"],
        backend="dask",
    )
    _process = partial(
        process_registry["eq"].implementation,
        y="B04",
        x=ParameterReference(from_parameter="x"),
    )

    output_cube = filter_labels(data=input_cube, condition=_process, dimension="bands")
    assert len(list(output_cube.data_vars)) == 1


def test_filter_labels_virtual_bands(process_registry):
    ds = xr.Dataset(
        {
            "B02": xr.DataArray(np.ones(2), dims=["x"]),
            "B03": xr.DataArray(np.ones(2) * 2, dims=["x"]),
            "B04": xr.DataArray(np.ones(2) * 3, dims=["x"]),
        }
    )
    _process = partial(
        process_registry["eq"].implementation,
        y="B04",
        x=ParameterReference(from_parameter="x"),
    )
    result = filter_labels(data=ds, condition=_process, dimension="bands")
    assert list(result.data_vars) == ["B04"]


@pytest.mark.parametrize("size", [(1, 1, 1, 2)])
@pytest.mark.parametrize("dtype", [np.uint8])
def test_filter_bands(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "SCL"],
        backend="dask",
    )

    output_cube = filter_bands(data=input_cube, bands=["SCL"])

    assert list(output_cube.data_vars) == ["SCL"]


@pytest.mark.parametrize("size", [(30, 30, 1, 1)])
@pytest.mark.parametrize("dtype", [np.uint8])
def test_filter_bbox(
    temporal_interval,
    bounding_box,
    random_raster_data,
    bounding_box_small,
    process_registry,
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02"],
        backend="dask",
        as_dataset=True,
    )

    output_cube = filter_bbox(data=input_cube, extent=bounding_box_small)

    assert len(output_cube.y) < len(input_cube.y)
    assert len(output_cube.x) < len(input_cube.x)

    _process = partial(
        process_registry["mean"].implementation,
        ignore_nodata=True,
        data=ParameterReference(from_parameter="data"),
    )

    input_cube_no_y = reduce_dimension(data=input_cube, reducer=_process, dimension="y")
    output_cube_cube_no_y = filter_bbox(data=input_cube_no_y, extent=bounding_box_small)

    input_cube_no_x = reduce_dimension(data=input_cube, reducer=_process, dimension="x")
    output_cube_cube_no_x = filter_bbox(data=input_cube_no_x, extent=bounding_box_small)

    output_cube_cube_no_x_y = reduce_dimension(
        data=input_cube_no_x, reducer=_process, dimension="y"
    )

    assert len(output_cube.y) == len(output_cube_cube_no_x.y)
    assert len(output_cube.x) == len(output_cube_cube_no_y.x)

    with pytest.raises(DimensionNotAvailable):
        filter_bbox(data=output_cube_cube_no_x_y, extent=bounding_box_small)


def test_filter_bbox_healpix_uses_lat_lon_center_coords():
    input_cube = xr.Dataset(
        {"ch1": (["t", "healpix_index"], da.ones((2, 5), chunks=(1, 5)))},
        coords={
            "t": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            "healpix_index": np.array([10, 11, 12, 13, 14]),
            "lat": ("healpix_index", np.array([49.0, 49.75, 50.0, 50.25, 51.0])),
            "lon": ("healpix_index", np.array([7.0, 8.25, 8.5, 8.75, 10.0])),
        },
        attrs={"crs": "healpix:1024", "healpix_nside": 1024, "healpix_order": "ring"},
    )
    bbox = BoundingBox(west=8.0, south=49.5, east=9.0, north=50.5, crs="EPSG:4326")

    output_cube = filter_bbox(data=input_cube, extent=bbox)

    assert output_cube.sizes == {"t": 2, "healpix_index": 3}
    assert output_cube["healpix_index"].values.tolist() == [11, 12, 13]
    assert output_cube.attrs["crs"] == "healpix:1024"
    assert hasattr(output_cube["ch1"].data, "dask")


def test_filter_bbox_healpix_handles_antimeridian_with_lon_coords():
    input_cube = xr.Dataset(
        {"ch1": (["t", "healpix_index"], da.ones((1, 3), chunks=(1, 3)))},
        coords={
            "t": np.array(["2024-01-01"], dtype="datetime64[ns]"),
            "healpix_index": np.array([10, 11, 12]),
            "lat": ("healpix_index", np.array([0.0, 0.0, 0.0])),
            "lon": ("healpix_index", np.array([175.0, -175.0, 0.0])),
        },
        attrs={"crs": "healpix:1024"},
    )
    bbox = BoundingBox(west=170.0, south=-1.0, east=-170.0, north=1.0)

    output_cube = filter_bbox(data=input_cube, extent=bbox)

    assert output_cube.sizes == {"t": 1, "healpix_index": 2}
    assert output_cube["healpix_index"].values.tolist() == [10, 11]
    assert hasattr(output_cube["ch1"].data, "dask")


def test_filter_bbox_healpix_can_use_pixel_ids_without_lat_lon(monkeypatch):
    calls = []

    def fake_pix2ang(nside, pixels, nest):
        calls.append((nside, nest))
        lons = np.array([0.0, 20.0, 8.5, 40.0, 60.0, 8.0, 9.5, 8.25])
        lats = np.array([0.0, 10.0, 50.0, 20.0, 30.0, 49.75, 50.25, 49.9])
        theta = np.deg2rad(90.0 - lats)
        phi = np.deg2rad(lons)
        return theta[np.asarray(pixels)], phi[np.asarray(pixels)]

    fake_healpy = SimpleNamespace(
        pix2ang=fake_pix2ang,
        nside2npix=lambda nside: 12,
    )
    monkeypatch.setitem(sys.modules, "healpy", fake_healpy)

    input_cube = xr.Dataset(
        {"ch1": (["t", "healpix_index"], da.ones((1, 8), chunks=(1, 4)))},
        coords={
            "t": np.array(["2024-01-01"], dtype="datetime64[ns]"),
            "healpix_index": np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        },
        attrs={"crs": "healpix:1", "healpix_nside": 1, "healpix_order": "nested"},
    )
    bbox = BoundingBox(west=8.0, south=49.5, east=9.0, north=50.5)

    output_cube = filter_bbox(data=input_cube, extent=bbox)

    assert calls == [(1, True)]
    assert output_cube.sizes == {"t": 1, "healpix_index": 3}
    assert output_cube["healpix_index"].values.tolist() == [2, 5, 7]
    assert hasattr(output_cube["ch1"].data, "dask")


@pytest.mark.parametrize("size", [(30, 30, 30, 4)])
@pytest.mark.parametrize("dtype", [np.uint8])
def test_filter_bands_dataset(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
        as_dataset=True,
    )

    output_cube = filter_bands(data=input_cube, bands=["B02", "B04"])
    assert set(output_cube.data_vars) == {"B02", "B04"}

    with pytest.raises(Exception):  # noqa: B017
        filter_bands(data=input_cube, bands=["nonexistent"])


@pytest.mark.parametrize("size", [(30, 30, 30, 4)])
@pytest.mark.parametrize("dtype", [np.uint8])
def test_filter_temporal_dataset(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
        as_dataset=True,
    )

    temporal_interval_part = TemporalInterval.model_validate(
        ["2018-05-15T00:00:00", "2018-06-01T00:00:00"]
    )
    output_cube = filter_temporal(data=input_cube, extent=temporal_interval_part)

    assert set(output_cube.data_vars) == {"B02", "B03", "B04", "B08"}
    assert len(output_cube.t) < len(input_cube.t)
