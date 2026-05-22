from openeo_processes_dask_slim.process_implementations.arrays import array_element
from openeo_processes_dask_slim.process_implementations.cubes.general import (
    add_dimension,
)
from openeo_processes_dask_slim.process_implementations.cubes.merge import merge_cubes
from openeo_processes_dask_slim.process_implementations.cubes.reduce import (
    reduce_dimension,
)
from openeo_processes_dask_slim.process_implementations.data_model import (
    RasterCube,
    band_names,
)

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
    dimension = data.openeo.band_dims[0]
    if target_band is None:
        target_band = dimension

    def _band(data, name):
        if name in band_names(data):
            return data[name]
        try:
            return data.sel(**{dimension: name})
        except (KeyError, ValueError):
            pass
        raise KeyError(f"Band '{name}' not found in {band_names(data)}")

    # Mid-Level Clouds
    def MIDCL(data):
        B08 = _band(data, nir08)
        B09 = _band(data, nir09)
        MIDCL = B08 - B09
        MIDCL_result = MIDCL * gain
        return MIDCL_result

    # Deep moist convection
    def DC(data):
        B10 = _band(data, cirrus)
        B12 = _band(data, swir22)
        DC = B10 - B12
        DC_result = DC * gain
        return DC_result

    # low-level cloudiness
    def LOWCL(data):
        B10 = _band(data, cirrus)
        B11 = _band(data, swir16)
        LOWCL = B11 - B10
        LOWCL_result = LOWCL * gain
        return LOWCL_result

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
