import logging
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import BoundingBox, TemporalInterval

logger = logging.getLogger(__name__)


def create_fake_rastercube(
    data,
    spatial_extent: BoundingBox,
    temporal_extent: TemporalInterval,
    bands: list,
    backend="numpy",
    chunks=("auto", "auto", "auto", -1),
):
    # Calculate the desired resolution based on how many samples we desire on the longest axis.
    len_x = max(spatial_extent.west, spatial_extent.east) - min(
        spatial_extent.west, spatial_extent.east
    )
    len_y = max(spatial_extent.south, spatial_extent.north) - min(
        spatial_extent.south, spatial_extent.north
    )

    x_coords = np.arange(
        min(spatial_extent.west, spatial_extent.east),
        max(spatial_extent.west, spatial_extent.east),
        step=len_x / data.shape[0],
    )
    y_coords = np.arange(
        min(spatial_extent.south, spatial_extent.north),
        max(spatial_extent.south, spatial_extent.north),
        step=len_y / data.shape[1],
    )

    # This line raises a deprecation warning, which according to this thread
    # will never actually be deprecated:
    # https://github.com/numpy/numpy/issues/23904
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        t_coords = pd.date_range(
            start=np.datetime64(temporal_extent.root[0].root),
            end=np.datetime64(temporal_extent.root[1].root),
            periods=data.shape[2],
        ).values

    # Original data shape: (x, y, t, bands)
    # Dataset model: bands are data vars with shape (t, y, x)
    var_chunks = chunks[:3] if len(chunks) >= 3 else chunks
    data_vars = {}
    for i, band in enumerate(bands):
        band_data = np.transpose(data[:, :, :, i], (2, 1, 0))  # (x, y, t) -> (t, y, x)
        if "dask" in backend:
            import dask.array as da
            band_data = da.from_array(band_data, chunks=var_chunks)
        data_vars[band] = (("t", "y", "x"), band_data)

    coords = {"x": x_coords, "y": y_coords, "t": t_coords}

    raster_cube = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={"crs": spatial_extent.crs},
    )
    import odc.geo.xr

    raster_cube = odc.geo.xr.assign_crs(raster_cube, crs=spatial_extent.crs)

    return raster_cube
