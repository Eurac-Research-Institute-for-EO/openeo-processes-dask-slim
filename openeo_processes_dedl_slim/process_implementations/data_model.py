from __future__ import annotations

from typing import Union

import dask_geopandas
import geopandas as gpd
import xarray as xr

RasterCube = xr.Dataset
VectorCube = Union[gpd.GeoDataFrame, dask_geopandas.GeoDataFrame, xr.Dataset]
