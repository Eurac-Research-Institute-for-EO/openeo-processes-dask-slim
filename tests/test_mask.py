import numpy as np
import pytest
import xarray as xr

from openeo_processes_dask_slim.process_implementations.cubes.mask import mask
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_mask_dataset(temporal_interval, bounding_box, random_raster_data):
    data_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
        as_dataset=True,
    )

    mask_data = data_cube > 50
    output_cube = mask(data=data_cube, mask=mask_data, replacement=np.nan)

    assert set(output_cube.data_vars) == {"B02", "B03", "B04", "B08"}
    assert output_cube["B02"].isnull().any()
