import xarray as xr

from openeo_processes_dask_slim.process_implementations.cubes.general import (
    add_dimension,
)
from openeo_processes_dask_slim.process_implementations.cubes.merge import merge_cubes
from openeo_processes_dask_slim.process_implementations.data_model import RasterCube

__all__ = ["ddmc"]


def _band_sel(data, band_name):
    if isinstance(data, xr.Dataset):
        return data[band_name]
    return data.sel(bands=band_name)


def ddmc(
    data: RasterCube,
    nir08="nir08",
    nir09="nir09",
    cirrus="cirrus",
    swir16="swir16",
    swir22="swir22",
    gain=2.5,
    target_band=None,
):
    if isinstance(data, xr.Dataset):
        dimension = "bands"
    else:
        dimension = data.openeo.band_dims[0]
    if target_band is None:
        target_band = dimension

    # Mid-Level Clouds
    def MIDCL(data):
        B08 = _band_sel(data, nir08)
        B09 = _band_sel(data, nir09)
        return (B08 - B09) * gain

    # Deep moist convection
    def DC(data):
        B10 = _band_sel(data, cirrus)
        B12 = _band_sel(data, swir22)
        return (B10 - B12) * gain

    # low-level cloudiness
    def LOWCL(data):
        B10 = _band_sel(data, cirrus)
        B11 = _band_sel(data, swir16)
        return (B11 - B10) * gain

    midcl = MIDCL(data)
    midcl = add_dimension(midcl, name=target_band, label="midcl", type=dimension)

    dc = DC(data)
    dc = add_dimension(dc, target_band, label="dc", type=dimension)

    lowcl = LOWCL(data)
    lowcl = add_dimension(lowcl, target_band, label="lowcl", type=dimension)

    ddmc1 = merge_cubes(midcl, lowcl)
    ddmc1.openeo.add_dim_type(name=target_band, type=dimension)
    ddmc = merge_cubes(dc, ddmc1, overlap_resolver=target_band)

    return ddmc
