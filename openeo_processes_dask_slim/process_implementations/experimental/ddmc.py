import xarray as xr

from openeo_processes_dask_slim.process_implementations.data_model import RasterCube

__all__ = ["ddmc"]


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
    def _band_sel(data, band_name):
        if isinstance(data, xr.Dataset):
            return data[band_name]
        return data.sel(bands=band_name)

    midcl = (_band_sel(data, nir08) - _band_sel(data, nir09)) * gain
    dc = (_band_sel(data, cirrus) - _band_sel(data, swir22)) * gain
    lowcl = (_band_sel(data, swir16) - _band_sel(data, cirrus)) * gain

    result = xr.Dataset({"midcl": midcl, "dc": dc, "lowcl": lowcl}, attrs=data.attrs)
    if data.odc.crs is not None:
        try:
            import odc.geo.xr

            result = odc.geo.xr.assign_crs(result, crs=data.odc.crs)
        except (ValueError, AttributeError):
            pass
    return result
