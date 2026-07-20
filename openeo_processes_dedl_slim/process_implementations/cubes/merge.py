from typing import Callable, Optional

import numpy as np
import odc.geo.xr
import xarray as xr

from openeo_processes_dedl_slim.process_implementations.cubes.utils import (
    ensure_raster_cube,
)
from openeo_processes_dedl_slim.process_implementations.data_model import RasterCube
from openeo_processes_dedl_slim.process_implementations.exceptions import (
    OverlapResolverMissing,
)

__all__ = ["merge_cubes"]

NEW_DIM_NAME = "__cubes__"
NEW_DIM_COORDS = ["cube1", "cube2"]

FLOAT_TOLERANCE = 1e-6


from collections import namedtuple

Overlap = namedtuple("Overlap", ["only_in_cube1", "only_in_cube2", "in_both"])


def _align_coordinates(
    cube1: RasterCube, cube2: RasterCube
) -> tuple[RasterCube, RasterCube]:
    shared_dims = set(cube1.dims).intersection(set(cube2.dims))
    coords_to_align = {}

    for dim in shared_dims:
        coords1 = cube1[dim].values
        coords2 = cube2[dim].values

        if not (
            np.issubdtype(coords1.dtype, np.floating)
            and np.issubdtype(coords2.dtype, np.floating)
        ):
            continue

        if coords1.shape != coords2.shape:
            continue

        max_diff = np.max(np.abs(coords1 - coords2))
        if max_diff < FLOAT_TOLERANCE:
            coords_to_align[dim] = cube1[dim]

    if coords_to_align:
        cube2 = cube2.assign_coords(coords_to_align)

    return cube1, cube2


def merge_dataset_variable_conflict(
    var_name: str,
    var1: xr.DataArray,
    var2: xr.DataArray,
    overlap_resolver: Callable = None,
    context: Optional[dict] = None,
) -> xr.DataArray:
    return merge_dataarray_cubes(var1, var2, overlap_resolver, context)


def merge_dataset_cubes(
    cube1: xr.Dataset,
    cube2: xr.Dataset,
    overlap_resolver: Callable = None,
    context: Optional[dict] = None,
) -> xr.Dataset:
    if context is None:
        context = {}
    cube1, cube2 = _align_coordinates(cube1, cube2)
    vars1 = set(cube1.data_vars)
    vars2 = set(cube2.data_vars)
    in_both = vars1 & vars2
    only_in1 = vars1 - vars2
    only_in2 = vars2 - vars1
    var_order = list(cube1.data_vars) + [
        v for v in cube2.data_vars if v not in cube1.data_vars
    ]

    var_attrs = {v: cube1[v].attrs for v in cube1.data_vars}
    var_attrs.update({v: cube2[v].attrs for v in cube2.data_vars if v not in var_attrs})

    if in_both and (only_in1 or only_in2) and overlap_resolver is None:
        raise OverlapResolverMissing(
            "Overlapping data cubes, but no overlap resolver has been specified."
        )

    result_vars = {}
    for var in in_both:
        result_vars[var] = merge_dataset_variable_conflict(
            var,
            cube1[var],
            cube2[var],
            overlap_resolver=overlap_resolver,
            context=context,
        )
    for var in only_in1:
        result_vars[var] = cube1[var]
    for var in only_in2:
        result_vars[var] = cube2[var]

    if result_vars:
        non_dim_coords = {c: cube1[c] for c in cube1.coords if c not in cube1.dims}
        if non_dim_coords:
            clean_vars = {
                k: v.drop_vars([c for c in v.coords if c in non_dim_coords])
                for k, v in result_vars.items()
            }
        else:
            clean_vars = result_vars
        result = xr.merge(
            list(clean_vars.values()),
            combine_attrs="drop_conflicts",
        )
        result.attrs = cube1.attrs
        for c_name, c_data in non_dim_coords.items():
            result.coords[c_name] = c_data
    else:
        result = xr.Dataset({}, coords=cube1.coords, attrs=cube1.attrs)
    for v, attrs in var_attrs.items():
        if v in result.data_vars:
            result[v].attrs = attrs
    if cube1.odc.crs is not None:
        try:
            result = odc.geo.xr.assign_crs(result, crs=cube1.odc.crs)
        except ValueError:
            pass
    result = result[var_order]
    return result


def merge_dataarray_cubes(
    cube1: RasterCube,
    cube2: RasterCube,
    overlap_resolver: Callable = None,
    context: Optional[dict] = None,
) -> RasterCube:
    if context is None:
        context = {}
    cube1, cube2 = _align_coordinates(cube1, cube2)

    overlap_per_shared_dim = {
        dim: Overlap(
            only_in_cube1=np.setdiff1d(cube1[dim].data, cube2[dim].data),
            only_in_cube2=np.setdiff1d(cube2[dim].data, cube1[dim].data),
            in_both=np.intersect1d(cube1[dim].data, cube2[dim].data),
        )
        for dim in set(cube1.dims).intersection(set(cube2.dims))
    }

    differing_dims = set(cube1.dims).symmetric_difference(set(cube2.dims))

    if len(differing_dims) == 0:
        dims_have_no_label_diff = all(
            [
                len(overlap.only_in_cube1) == 0 and len(overlap.only_in_cube2) == 0
                for overlap in overlap_per_shared_dim.values()
            ]
        )
        if dims_have_no_label_diff:
            concat_both_cubes = xr.concat([cube1, cube2], dim=NEW_DIM_NAME).reindex(
                {NEW_DIM_NAME: NEW_DIM_COORDS}
            )

            concat_both_cubes_rechunked = concat_both_cubes.chunk(
                {NEW_DIM_NAME: -1}
                | {dim: "auto" for dim in cube1.dims if dim != NEW_DIM_NAME}
            )
            if overlap_resolver is None:
                merged_cube = concat_both_cubes_rechunked
            else:
                positional_parameters = {}
                named_parameters = {
                    "x": cube1.data,
                    "y": cube2.data,
                    "context": context,
                }

                merged_cube = concat_both_cubes_rechunked.reduce(
                    overlap_resolver,
                    dim=NEW_DIM_NAME,
                    keep_attrs=True,
                    positional_parameters=positional_parameters,
                    named_parameters=named_parameters,
                )
        else:
            dims_requiring_resolve = [
                dim
                for dim, overlap in overlap_per_shared_dim.items()
                if len(overlap.in_both) > 0
                and (len(overlap.only_in_cube1) > 0 or len(overlap.only_in_cube2) > 0)
            ]

            if len(dims_requiring_resolve) == 0:
                previous_dim_order = list(cube1.dims) + [
                    dim for dim in cube2.dims if dim not in cube1.dims
                ]
                has_band_dim = (
                    len(cube1.openeo.band_dims) > 0 and len(cube2.openeo.band_dims) > 0
                )
                if has_band_dim:
                    band_dim = cube1.openeo.band_dims[0]
                    previous_band_order = list(cube1[band_dim].values) + [
                        band
                        for band in list(cube2[band_dim].values)
                        if band not in list(cube1[band_dim].values)
                    ]
                    cube1 = cube1.to_dataset(band_dim)
                    cube2 = cube2.to_dataset(band_dim)

                merged_cube = xr.combine_by_coords(
                    [cube1, cube2],
                    combine_attrs="drop_conflicts",
                    compat="override",
                    coords="minimal",
                )
                if has_band_dim and isinstance(merged_cube, xr.Dataset):
                    merged_cube = merged_cube.to_array(dim=band_dim)
                    merged_cube = merged_cube.reindex({band_dim: previous_band_order})

                merged_cube = merged_cube.transpose(*previous_dim_order)

            elif len(dims_requiring_resolve) == 1:
                if overlap_resolver is None or not callable(overlap_resolver):
                    raise OverlapResolverMissing(
                        "Overlapping data cubes, but no overlap resolver has been specified."
                    )

                overlapping_dim = dims_requiring_resolve[0]

                stacked_conflicts = xr.concat(
                    [
                        cube1.sel(
                            **{
                                overlapping_dim: overlap_per_shared_dim[
                                    overlapping_dim
                                ].in_both
                            }
                        ),
                        cube2.sel(
                            **{
                                overlapping_dim: overlap_per_shared_dim[
                                    overlapping_dim
                                ].in_both
                            }
                        ),
                    ],
                    dim=NEW_DIM_NAME,
                ).reindex({NEW_DIM_NAME: NEW_DIM_COORDS})

                stacked_conflicts_rechunked = stacked_conflicts.chunk(
                    {NEW_DIM_NAME: -1}
                    | {dim: "auto" for dim in cube1.dims if dim != NEW_DIM_NAME}
                )

                conflicts_cube_1 = cube1.sel(
                    **{overlapping_dim: overlap_per_shared_dim[overlapping_dim].in_both}
                )

                conflicts_cube_2 = cube2.sel(
                    **{overlapping_dim: overlap_per_shared_dim[overlapping_dim].in_both}
                )

                positional_parameters = {}
                named_parameters = {
                    "x": conflicts_cube_1.data,
                    "y": conflicts_cube_2.data,
                    "context": context,
                }

                merge_conflicts = stacked_conflicts_rechunked.reduce(
                    overlap_resolver,
                    dim=NEW_DIM_NAME,
                    keep_attrs=True,
                    positional_parameters=positional_parameters,
                    named_parameters=named_parameters,
                )

                rest_of_cube_1 = cube1.sel(
                    **{
                        overlapping_dim: overlap_per_shared_dim[
                            overlapping_dim
                        ].only_in_cube1
                    }
                )
                rest_of_cube_2 = cube2.sel(
                    **{
                        overlapping_dim: overlap_per_shared_dim[
                            overlapping_dim
                        ].only_in_cube2
                    }
                )
                merged_cube = xr.combine_by_coords(
                    [merge_conflicts, rest_of_cube_1, rest_of_cube_2],
                    combine_attrs="drop_conflicts",
                )

            else:
                raise ValueError(
                    "More than one overlapping dimension, merge not possible."
                )

    elif len(differing_dims) <= 2:
        if overlap_resolver is None or not callable(overlap_resolver):
            raise OverlapResolverMissing(
                "Overlapping data cubes, but no overlap resolver has been specified."
            )

        if len(cube1.dims) < len(cube2.dims):
            lower_dim_cube = cube1
            higher_dim_cube = cube2
            is_cube1_lower_dim = True
        else:
            lower_dim_cube = cube2
            higher_dim_cube = cube1
            is_cube1_lower_dim = False

        lower_dim_cube_broadcast = lower_dim_cube.broadcast_like(higher_dim_cube)

        both_stacked = xr.concat(
            [higher_dim_cube, lower_dim_cube_broadcast], dim=NEW_DIM_NAME
        ).reindex({NEW_DIM_NAME: NEW_DIM_COORDS})

        both_stacked_rechunked = both_stacked.chunk(
            {NEW_DIM_NAME: -1}
            | {dim: "auto" for dim in cube1.dims if dim != NEW_DIM_NAME}
        )

        positional_parameters = {}

        named_parameters = {"context": context}
        if is_cube1_lower_dim:
            named_parameters["x"] = lower_dim_cube_broadcast.data
            named_parameters["y"] = higher_dim_cube.data
        else:
            named_parameters["x"] = higher_dim_cube.data
            named_parameters["y"] = lower_dim_cube_broadcast.data

        merged_cube = both_stacked_rechunked.reduce(
            overlap_resolver,
            dim=NEW_DIM_NAME,
            keep_attrs=True,
            positional_parameters=positional_parameters,
            named_parameters=named_parameters,
        )
    else:
        raise ValueError("Number of differing dimensions is >2, merge not possible.")

    return merged_cube


def merge_cubes(
    cube1: RasterCube,
    cube2: RasterCube,
    overlap_resolver: Callable = None,
    context: Optional[dict] = None,
) -> RasterCube:
    if context is None:
        context = {}
    if not isinstance(cube1, type(cube2)):
        raise Exception(
            f"Provided cubes have incompatible types. cube1: {type(cube1)}, cube2: {type(cube2)}"
        )

    if isinstance(cube1, xr.Dataset):
        return merge_dataset_cubes(cube1, cube2, overlap_resolver, context)
    raise TypeError(
        "RasterCube must be an xr.Dataset in merge_cubes, "
        f"got {type(cube1).__name__}. "
        "DataArray inputs are no longer supported."
    )
