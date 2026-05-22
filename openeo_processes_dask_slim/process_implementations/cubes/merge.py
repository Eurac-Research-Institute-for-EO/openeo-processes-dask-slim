from typing import Callable, Optional

import numpy as np
import xarray as xr

from openeo_processes_dask_slim.process_implementations.data_model import (
    RasterCube,
    _stack_bands,
    _unstack_bands,
)
from openeo_processes_dask_slim.process_implementations.exceptions import (
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
    """Align coordinates between two cubes if they're very close numerically."""
    shared_dims = set(cube1.dims).intersection(set(cube2.dims))

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
            cube2[dim] = cube1[dim]

    return cube1, cube2


def _get_data_for_resolver(cube):
    """Get per-variable data for passing to overlap resolver."""
    if isinstance(cube, xr.Dataset):
        return cube.to_dataarray(dim="__bands__").data
    return cube.data


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

    cube1, cube2 = _align_coordinates(cube1, cube2)

    overlap_per_shared_dim = {
        dim: Overlap(
            only_in_cube1=np.setdiff1d(cube1[dim].values, cube2[dim].values),
            only_in_cube2=np.setdiff1d(cube2[dim].values, cube1[dim].values),
            in_both=np.intersect1d(cube1[dim].values, cube2[dim].values),
        )
        for dim in set(cube1.dims).intersection(set(cube2.dims))
    }

    differing_dims = set(cube1.dims).symmetric_difference(set(cube2.dims))

    if isinstance(cube1, xr.Dataset) and isinstance(cube2, xr.Dataset):
        cube1_vars = set(cube1.data_vars)
        cube2_vars = set(cube2.data_vars)
        common_vars = cube1_vars & cube2_vars
        disjoint_vars = cube1_vars ^ cube2_vars
    else:
        common_vars = None

    if len(differing_dims) == 0:
        dims_have_no_label_diff = all(
            [
                len(overlap.only_in_cube1) == 0 and len(overlap.only_in_cube2) == 0
                for overlap in overlap_per_shared_dim.values()
            ]
        )
        if dims_have_no_label_diff:
            if (
                common_vars is not None
                and len(common_vars) == 0
                and len(disjoint_vars) > 0
            ):
                previous_dim_order = list(cube1.dims) + [
                    dim for dim in cube2.dims if dim not in cube1.dims
                ]
                merged_cube = xr.combine_by_coords(
                    [cube1, cube2], combine_attrs="drop_conflicts", compat="override"
                )
                merged_cube = merged_cube.transpose(*previous_dim_order)
            elif (
                common_vars is not None
                and len(common_vars) > 0
                and len(disjoint_vars) > 0
                and overlap_resolver is None
            ):
                raise OverlapResolverMissing(
                    "Overlapping data cubes, but no overlap resolver has been specified."
                )
            elif overlap_resolver is None:
                concat_both_cubes = xr.concat([cube1, cube2], dim=NEW_DIM_NAME).reindex(
                    {NEW_DIM_NAME: NEW_DIM_COORDS}
                )
                merged_cube = concat_both_cubes.chunk(
                    {NEW_DIM_NAME: -1}
                    | {dim: "auto" for dim in cube1.dims if dim != NEW_DIM_NAME}
                )
            else:
                is_dataset = isinstance(cube1, xr.Dataset)
                if is_dataset and common_vars is not None and len(disjoint_vars) > 0:
                    c1_common = cube1[list(sorted(common_vars))]
                    c2_common = cube2[list(sorted(common_vars))]
                    c1_stacked = _stack_bands(c1_common)
                    c2_stacked = _stack_bands(c2_common)
                elif is_dataset:
                    c1_stacked = _stack_bands(cube1)
                    c2_stacked = _stack_bands(cube2)
                else:
                    c1_stacked = cube1
                    c2_stacked = cube2
                concat_cubes = xr.concat(
                    [c1_stacked, c2_stacked], dim=NEW_DIM_NAME
                ).reindex({NEW_DIM_NAME: NEW_DIM_COORDS})
                concat_cubes_rechunked = concat_cubes.chunk(
                    {NEW_DIM_NAME: -1}
                    | {dim: "auto" for dim in c1_stacked.dims if dim != NEW_DIM_NAME}
                )
                named_parameters = {
                    "x": _get_data_for_resolver(c1_stacked),
                    "y": _get_data_for_resolver(c2_stacked),
                    "context": context,
                }
                merged_stacked = concat_cubes_rechunked.reduce(
                    overlap_resolver,
                    dim=NEW_DIM_NAME,
                    keep_attrs=True,
                    positional_parameters={},
                    named_parameters=named_parameters,
                )
                if is_dataset:
                    merged_common = _unstack_bands(merged_stacked)
                    if len(disjoint_vars) > 0:
                        unique_cube1 = cube1[list(sorted(cube1_vars - common_vars))]
                        unique_cube2 = cube2[list(sorted(cube2_vars - common_vars))]
                        to_merge = [merged_common]
                        if len(unique_cube1.data_vars) > 0:
                            to_merge.append(unique_cube1)
                        if len(unique_cube2.data_vars) > 0:
                            to_merge.append(unique_cube2)
                        merged_cube = xr.merge(to_merge, combine_attrs="drop_conflicts")
                    else:
                        merged_cube = merged_common
                else:
                    merged_cube = merged_stacked
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
                merged_cube = xr.combine_by_coords(
                    [cube1, cube2], combine_attrs="drop_conflicts", compat="override"
                )
                merged_cube = merged_cube.transpose(*previous_dim_order)

            elif len(dims_requiring_resolve) == 1:
                if overlap_resolver is None or not callable(overlap_resolver):
                    raise OverlapResolverMissing(
                        "Overlapping data cubes, but no overlap resolver has been specified."
                    )

                overlapping_dim = dims_requiring_resolve[0]

                conflicts_cube_1 = cube1.sel(
                    **{overlapping_dim: overlap_per_shared_dim[overlapping_dim].in_both}
                )
                conflicts_cube_2 = cube2.sel(
                    **{overlapping_dim: overlap_per_shared_dim[overlapping_dim].in_both}
                )

                is_dataset = isinstance(cube1, xr.Dataset)
                if is_dataset:
                    c1_conflict_stacked = _stack_bands(conflicts_cube_1)
                    c2_conflict_stacked = _stack_bands(conflicts_cube_2)
                else:
                    c1_conflict_stacked = conflicts_cube_1
                    c2_conflict_stacked = conflicts_cube_2

                stacked_conflicts = xr.concat(
                    [c1_conflict_stacked, c2_conflict_stacked],
                    dim=NEW_DIM_NAME,
                ).reindex({NEW_DIM_NAME: NEW_DIM_COORDS})

                stacked_conflicts_rechunked = stacked_conflicts.chunk(
                    {NEW_DIM_NAME: -1}
                    | {
                        dim: "auto"
                        for dim in c1_conflict_stacked.dims
                        if dim != NEW_DIM_NAME
                    }
                )

                positional_parameters = {}
                named_parameters = {
                    "x": _get_data_for_resolver(c1_conflict_stacked),
                    "y": _get_data_for_resolver(c2_conflict_stacked),
                    "context": context,
                }

                merge_conflicts_stacked = stacked_conflicts_rechunked.reduce(
                    overlap_resolver,
                    dim=NEW_DIM_NAME,
                    keep_attrs=True,
                    positional_parameters=positional_parameters,
                    named_parameters=named_parameters,
                )

                if is_dataset:
                    merge_conflicts = _unstack_bands(merge_conflicts_stacked)
                else:
                    merge_conflicts = merge_conflicts_stacked

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

        is_dataset = isinstance(cube1, xr.Dataset) or isinstance(cube2, xr.Dataset)
        if is_dataset:
            h_stacked = _stack_bands(higher_dim_cube)
            l_stacked = _stack_bands(lower_dim_cube_broadcast)
        else:
            h_stacked = higher_dim_cube
            l_stacked = lower_dim_cube_broadcast

        both_stacked = xr.concat([h_stacked, l_stacked], dim=NEW_DIM_NAME).reindex(
            {NEW_DIM_NAME: NEW_DIM_COORDS}
        )

        both_stacked_rechunked = both_stacked.chunk(
            {NEW_DIM_NAME: -1}
            | {dim: "auto" for dim in h_stacked.dims if dim != NEW_DIM_NAME}
        )

        positional_parameters = {}
        named_parameters = {"context": context}
        if is_cube1_lower_dim:
            named_parameters["x"] = _get_data_for_resolver(l_stacked)
            named_parameters["y"] = _get_data_for_resolver(h_stacked)
        else:
            named_parameters["x"] = _get_data_for_resolver(h_stacked)
            named_parameters["y"] = _get_data_for_resolver(l_stacked)

        merged_stacked = both_stacked_rechunked.reduce(
            overlap_resolver,
            dim=NEW_DIM_NAME,
            keep_attrs=True,
            positional_parameters=positional_parameters,
            named_parameters=named_parameters,
        )

        if is_dataset:
            merged_cube = _unstack_bands(merged_stacked)
        else:
            merged_cube = merged_stacked
    else:
        raise ValueError("Number of differing dimensions is >2, merge not possible.")

    return merged_cube
