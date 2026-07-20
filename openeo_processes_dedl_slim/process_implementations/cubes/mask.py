import logging
from typing import Callable

import numpy as np
import xarray as xr

from openeo_processes_dedl_slim.process_implementations.cubes.resample import (
    resample_cube_spatial,
)
from openeo_processes_dedl_slim.process_implementations.cubes.utils import (
    ensure_raster_cube,
    notnull,
)
from openeo_processes_dedl_slim.process_implementations.data_model import RasterCube
from openeo_processes_dedl_slim.process_implementations.exceptions import (
    DimensionLabelCountMismatch,
    DimensionMismatch,
    LabelMismatch,
)
from openeo_processes_dedl_slim.process_implementations.logic import _not

logger = logging.getLogger(__name__)

__all__ = ["mask"]


def _mask_dataset(data: xr.Dataset, mask: RasterCube, replacement) -> xr.Dataset:
    from openeo_processes_dedl_slim.process_implementations.cubes.resample import (
        resample_cube_spatial,
    )

    if isinstance(mask, xr.Dataset):
        data_temporal_dims = data.openeo.temporal_dims
        mask_temporal_dims = mask.openeo.temporal_dims
        check_temporal_labels = True
        if not set(data_temporal_dims) == set(mask_temporal_dims):
            check_temporal_labels = False
            if len(mask_temporal_dims) != 0:
                from openeo_processes_dedl_slim.process_implementations.exceptions import (
                    DimensionMismatch,
                )

                raise DimensionMismatch(
                    f"data and mask temporal dimensions do no match: data has temporal dimensions ({data_temporal_dims}) and mask {mask_temporal_dims}."
                )
        if check_temporal_labels:
            for n in data_temporal_dims:
                if not all(data[n].values == mask[n].values):
                    from openeo_processes_dedl_slim.process_implementations.exceptions import (
                        LabelMismatch,
                    )

                    raise LabelMismatch(
                        f"data and mask temporal dimension labels don't match for dimension {n}."
                    )

        data_spatial_dims = data.openeo.spatial_dims
        mask_spatial_dims = mask.openeo.spatial_dims
        apply_resample = False
        if not set(data_spatial_dims) == set(mask_spatial_dims):
            from openeo_processes_dedl_slim.process_implementations.exceptions import (
                DimensionMismatch,
            )

            raise DimensionMismatch(
                f"data and mask spatial dimensions do no match: data has spatial dimensions ({data_spatial_dims}) and mask {mask_spatial_dims}"
            )
        for n in data_spatial_dims:
            if len(data[n]) != len(mask[n]) or not all(
                data[n].values == mask[n].values
            ):
                apply_resample = True
                break
        if apply_resample:
            mask = resample_cube_spatial(data=mask, target=data)

        mask_vars = list(mask.data_vars)
        is_single_mask = len(mask_vars) == 1 and mask_vars[0] not in data.data_vars
    else:
        is_single_mask = True
        mask_vars = []

    result_vars = {}
    for var_name in data.data_vars:
        if is_single_mask:
            if isinstance(mask, xr.Dataset):
                mask_var = mask[mask_vars[0]]
            else:
                mask_var = mask
            result_vars[var_name] = data[var_name].where(~mask_var, replacement)
        elif var_name in mask.data_vars:
            result_vars[var_name] = data[var_name].where(~mask[var_name], replacement)
        else:
            result_vars[var_name] = data[var_name]
    return xr.Dataset(result_vars, attrs=data.attrs)


def mask(data: RasterCube, mask: RasterCube, replacement=None) -> RasterCube:
    ensure_raster_cube(data, "mask")
    if replacement is None:
        replacement = np.nan
    return _mask_dataset(data, mask, replacement)
