from typing import Optional

import dask.array as da
import xarray as xr
from openeo.udf import UdfData
from openeo.udf.run_code import run_udf_code
from openeo.udf.xarraydatacube import XarrayDataCube

from openeo_processes_dedl_slim.process_implementations.cubes.dataset_bridge import (
    dataset_to_virtual_bands,
    virtual_bands_to_dataset,
)
from openeo_processes_dedl_slim.process_implementations.data_model import RasterCube

__all__ = ["run_udf"]


def run_udf(
    data: da.Array, udf: str, runtime: str, context: Optional[dict] = None
) -> RasterCube:
    input_attrs = data.attrs if isinstance(data, (xr.DataArray, xr.Dataset)) else {}
    was_dataset = isinstance(data, xr.Dataset)
    if was_dataset:
        data, meta = dataset_to_virtual_bands(data, dim="bands")
    else:
        meta = None
    udf_input = XarrayDataCube(xr.DataArray(data))
    udf_data = UdfData(datacube_list=[udf_input], user_context=context)
    result = run_udf_code(code=udf, data=udf_data)
    cubes = result.get_datacube_list()
    if len(cubes) != 1:
        raise ValueError(
            f"The provided UDF should return one datacube, but got: {result}"
        )
    result_array: xr.DataArray = cubes[0].array
    if was_dataset:
        result_array = virtual_bands_to_dataset(result_array, meta, dim="bands")
    if not result_array.attrs:
        result_array.attrs = input_attrs
    return result_array
