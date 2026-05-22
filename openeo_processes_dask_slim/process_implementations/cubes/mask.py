import logging
from typing import Callable

import numpy as np
import xarray as xr

from openeo_processes_dask_slim.process_implementations.cubes.resample import (
    resample_cube_spatial,
)
from openeo_processes_dask_slim.process_implementations.cubes.utils import notnull
from openeo_processes_dask_slim.process_implementations.data_model import RasterCube
from openeo_processes_dask_slim.process_implementations.exceptions import (
    DimensionLabelCountMismatch,
    DimensionMismatch,
    LabelMismatch,
)
from openeo_processes_dask_slim.process_implementations.logic import _not

logger = logging.getLogger(__name__)

__all__ = ["mask"]


def mask(data: RasterCube, mask: RasterCube, replacement=None) -> RasterCube:
    if replacement is None:
        replacement = np.nan

    data_temporal_dims = data.openeo.temporal_dims
    mask_temporal_dims = mask.openeo.temporal_dims

    check_temporal_labels = True
    if not set(data_temporal_dims) == set(mask_temporal_dims):
        check_temporal_labels = False
        if len(mask_temporal_dims) != 0:
            raise DimensionMismatch(
                f"data and mask temporal dimensions do no match: data has temporal dimensions ({data_temporal_dims}) and mask {mask_temporal_dims}."
            )
    if check_temporal_labels:
        for n in data_temporal_dims:
            data_temporal_labels = data[n].values
            mask_temporal_labels = mask[n].values
            data_n_labels = len(data_temporal_labels)
            mask_n_labels = len(mask_temporal_labels)

            if not data_n_labels == mask_n_labels:
                raise DimensionLabelCountMismatch(
                    f"data and mask temporal dimensions do no match: data has {data_n_labels} temporal dimensions labels and mask {mask_n_labels}."
                )
            elif not all(data_temporal_labels == mask_temporal_labels):
                raise LabelMismatch(
                    f"data and mask temporal dimension labels don't match for dimension {n}."
                )

    apply_resample_cube_spatial = False

    data_spatial_dims = data.openeo.spatial_dims
    mask_spatial_dims = mask.openeo.spatial_dims
    if not set(data_spatial_dims) == set(mask_spatial_dims):
        raise DimensionMismatch(
            f"data and mask spatial dimensions do no match: data has spatial dimensions ({data_spatial_dims}) and mask {mask_spatial_dims}"
        )

    for n in data_spatial_dims:
        data_spatial_labels = data[n].values
        mask_spatial_labels = mask[n].values
        data_n_labels = len(data_spatial_labels)
        mask_n_labels = len(mask_spatial_labels)

        if not data_n_labels == mask_n_labels:
            apply_resample_cube_spatial = True
            logger.info(
                f"data and mask spatial dimension labels don't match: data has ({data_n_labels}) labels and mask has {mask_n_labels} for dimension {n}."
            )
        elif not all(data_spatial_labels == mask_spatial_labels):
            apply_resample_cube_spatial = True
            logger.info(
                f"data and mask spatial dimension labels don't match for dimension {n}, i.e. the coordinate values are different."
            )

    if apply_resample_cube_spatial:
        logger.info(f"mask is aligned to data using resample_cube_spatial.")
        mask = resample_cube_spatial(data=mask, target=data)

    if isinstance(data, xr.Dataset) and isinstance(mask, xr.Dataset):
        mask_vars = list(mask.data_vars)
        data_vars = list(data.data_vars)
        if len(mask_vars) == 1:
            mask = mask[mask_vars[0]]
        elif set(mask_vars) != set(data_vars):
            raise Exception(
                f"Mask has variables {mask_vars} that don't match data variables {data_vars}. "
                "A mask Dataset must have either a single variable (applied to all) or variables matching the data variables."
            )

    if isinstance(mask, xr.Dataset):
        data = data.where(mask == 0, replacement)
    else:
        data = data.where(_not(mask), replacement)
    return data
