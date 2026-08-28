import numpy as np
import pytest
import xarray as xr

from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_openeo_accessor(temporal_interval, bounding_box, random_raster_data):
    raster_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
    )

    assert raster_cube.openeo is not None
    assert raster_cube.openeo.x_dim == "x"
    assert raster_cube.openeo.y_dim == "y"
    assert raster_cube.openeo.temporal_dims[0] == "t"
    assert raster_cube.openeo.band_dims == ()

    with pytest.raises(NotImplementedError):
        _ = raster_cube.openeo.z_dim

    raster_cube = raster_cube.rename({"t": "month"})
    assert raster_cube.openeo.temporal_dims[0] == "month"

    raster_cube = raster_cube.rename({"month": "NotATimeDim"})
    assert raster_cube.openeo.temporal_dims == ()

    assert raster_cube.openeo.band_dims == ()
    assert list(raster_cube.data_vars) == ["B02", "B03", "B04", "B08"]


def test_openeo_accessor_detects_healpix_index_as_spatial_dimension():
    raster_cube = xr.Dataset(
        {"B02": (["t", "healpix_index"], np.ones((1, 3)))},
        coords={
            "t": np.array(["2024-01-01"], dtype="datetime64[ns]"),
            "healpix_index": np.array([10, 11, 12]),
            "lat": ("healpix_index", np.array([49.0, 50.0, 51.0])),
            "lon": ("healpix_index", np.array([8.0, 9.0, 10.0])),
        },
        attrs={"crs": "healpix:1024", "healpix_nside": 1024, "healpix_order": "ring"},
    )

    assert raster_cube.openeo.spatial_dims == ("healpix_index",)
    assert raster_cube.openeo.x_dim is None
    assert raster_cube.openeo.y_dim is None
    assert raster_cube.openeo.other_dims == ()
