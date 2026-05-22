import numpy as np
import pytest

from openeo_processes_dask_slim.process_implementations.cubes.indices import ndvi
from openeo_processes_dask_slim.process_implementations.exceptions import (
    BandExists,
    NirBandAmbiguous,
    RedBandAmbiguous,
)
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(20, 20, 10, 2)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_ndvi(temporal_interval, bounding_box, random_raster_data, process_registry):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["red", "nir"],
        backend="dask",
    )

    output = ndvi(input_cube)

    assert "ndvi" in list(output.data_vars)

    expected_results = (
        input_cube["nir"] - input_cube["red"]
    ) / (input_cube["nir"] + input_cube["red"])

    first_output_var = list(output.data_vars.values())[0]
    np.testing.assert_allclose(
        first_output_var.data, expected_results.data, equal_nan=True
    )

    # Test with mismatched band names
    cube_with_wrong_names = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["blue", "yellow"],
        backend="dask",
    )
    with pytest.raises(NirBandAmbiguous):
        ndvi(cube_with_wrong_names)

    # ndvi on ndvi result should fail (no red/nir bands)
    with pytest.raises(NirBandAmbiguous):
        ndvi(output)

    # Test target_band parameter
    target_band = "yay"
    output_with_target = ndvi(input_cube, target_band=target_band)
    assert target_band in list(output_with_target.data_vars)
    # Original bands plus new one
    assert len(output_with_target.data_vars) == len(input_cube.data_vars) + 1

    with pytest.raises(BandExists):
        ndvi(input_cube, target_band="t")
