try:
    import dask
except ImportError:
    dask = None

import numpy as np
import xarray as xr
from xarray.core.duck_array_ops import isnull as xr_isnull


def _has_dask():
    return dask is not None


def _is_dask_array(arr):
    return _has_dask() and isinstance(arr, dask.array.Array)


def isnull(data):
    if _is_dask_array(data):
        return dask.array.map_blocks(xr_isnull, data)
    else:
        return xr_isnull(data)


def notnull(data):
    return ~isnull(data)


def ensure_raster_cube(data, process_name=None):
    if not isinstance(data, xr.Dataset):
        msg = (
            f"RasterCube must be an xr.Dataset"
            f"{' in ' + process_name if process_name else ''}"
            f", got {type(data).__name__}. "
            f"DataArray inputs are no longer supported."
        )
        raise TypeError(msg)


from openeo_processes_dedl_slim.process_implementations.cubes.dataset_bridge import (
    _capture_var_metadata as _capture_var_metadata,
)
from openeo_processes_dedl_slim.process_implementations.cubes.dataset_bridge import (
    _detect_band_permutation as _detect_band_permutation,
)
from openeo_processes_dedl_slim.process_implementations.cubes.dataset_bridge import (
    _restore_var_metadata as _restore_var_metadata,
)
