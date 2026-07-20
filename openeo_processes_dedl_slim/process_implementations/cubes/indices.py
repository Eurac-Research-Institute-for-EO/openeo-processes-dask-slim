import xarray as xr

from openeo_processes_dedl_slim.process_implementations.cubes.utils import (
    ensure_raster_cube,
)
from openeo_processes_dedl_slim.process_implementations.data_model import RasterCube
from openeo_processes_dedl_slim.process_implementations.exceptions import (
    BandExists,
    NirBandAmbiguous,
    RedBandAmbiguous,
)
from openeo_processes_dedl_slim.process_implementations.math import (
    normalized_difference,
)

__all__ = ["ndvi"]


def ndvi(data: RasterCube, nir="nir", red="red", target_band=None):
    ensure_raster_cube(data, "ndvi")
    nir_name = nir
    red_name = red
    if nir not in data.data_vars:
        for var_name, var_data in data.data_vars.items():
            if var_data.attrs.get("common_name") == nir:
                nir_name = var_name
                break
    if red not in data.data_vars:
        for var_name, var_data in data.data_vars.items():
            if var_data.attrs.get("common_name") == red:
                red_name = var_name
                break
    if nir_name not in data.data_vars:
        raise NirBandAmbiguous(
            "The NIR band can't be resolved, please specify the specific NIR band name."
        )
    if red_name not in data.data_vars:
        raise RedBandAmbiguous(
            "The Red band can't be resolved, please specify the specific Red band name."
        )
    nir_band = data[nir_name]
    red_band = data[red_name]
    nd = normalized_difference(nir_band, red_band)
    if target_band is not None:
        if target_band in data.data_vars:
            raise BandExists("A band with the specified target name exists.")
        nd = data.assign({target_band: nd})
    else:
        nd = nd.to_dataset(name="ndvi")
    nd.attrs = data.attrs
    return nd
