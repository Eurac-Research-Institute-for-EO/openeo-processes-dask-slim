import numpy as np
import pytest
import xarray as xr

from openeo_processes_dedl_slim.process_implementations.cubes.indices import ndvi
from openeo_processes_dedl_slim.process_implementations.exceptions import (
    BandExists,
    NirBandAmbiguous,
    RedBandAmbiguous,
)
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(20, 20, 10, 2)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_ndvi_common_name_resolution(
    temporal_interval, bounding_box, random_raster_data
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["red", "nir"],
        backend="dask",
        as_dataset=True,
    )

    input_cube["red"].attrs["common_name"] = "red"
    input_cube["nir"].attrs["common_name"] = "nir"

    output = ndvi(input_cube)
    assert "ndvi" in output.data_vars
    expected = (input_cube["nir"] - input_cube["red"]) / (
        input_cube["nir"] + input_cube["red"]
    )
    xr.testing.assert_allclose(output["ndvi"], expected)

    with pytest.raises(NirBandAmbiguous):
        ndvi(input_cube, nir="nonexistent")

    with pytest.raises(RedBandAmbiguous):
        ndvi(input_cube, red="nonexistent")

    target_band = "yay"
    output_with_target = ndvi(input_cube, target_band=target_band)
    assert target_band in output_with_target.data_vars
    xr.testing.assert_allclose(output_with_target[target_band], expected)

    with pytest.raises(BandExists):
        ndvi(input_cube, target_band="red")


@pytest.mark.parametrize("size", [(20, 20, 10, 2)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_ndvi_dataset(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["red", "nir"],
        backend="dask",
        as_dataset=True,
    )

    output = ndvi(input_cube)
    assert "ndvi" in output.data_vars
    expected = (input_cube["nir"] - input_cube["red"]) / (
        input_cube["nir"] + input_cube["red"]
    )
    xr.testing.assert_allclose(output["ndvi"], expected)

    output_with_target = ndvi(input_cube, target_band="ndvi_custom")
    assert "ndvi_custom" in output_with_target.data_vars

    with pytest.raises(NirBandAmbiguous):
        ndvi(input_cube, nir="nonexistent")

    with pytest.raises(RedBandAmbiguous):
        ndvi(input_cube, red="nonexistent")
