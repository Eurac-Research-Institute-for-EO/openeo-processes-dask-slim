from __future__ import annotations

from math import isqrt

import numpy as np
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import BoundingBox

from openeo_processes_dedl_slim.process_implementations.data_model import RasterCube
from openeo_processes_dedl_slim.process_implementations.exceptions import (
    DimensionNotAvailable,
)

MAX_LAZY_HEALPIX_COORD_CELLS = 10_000_000


def is_healpix_dim(data: xr.Dataset | xr.DataArray, dim: str) -> bool:
    if "healpix" in str(dim).casefold():
        return True

    coord = data.coords.get(dim)
    if coord is None:
        return False
    return str(coord.attrs.get("dggs:grid_name", "")).casefold() == "healpix"


def get_healpix_dim(data: xr.Dataset | xr.DataArray) -> str | None:
    return next((dim for dim in data.dims if is_healpix_dim(data, dim)), None)


def get_healpix_nside(
    data: xr.Dataset | xr.DataArray, dim: str | None = None
) -> int | None:
    if dim is None:
        dim = get_healpix_dim(data)

    nside = data.attrs.get("healpix_nside")
    coord = data.coords.get(dim) if dim is not None else None

    if nside is None and coord is not None:
        level = coord.attrs.get("dggs:level")
        if level is not None:
            try:
                nside = 2 ** int(level)
            except (TypeError, ValueError):
                pass

    if nside is None:
        crs = str(data.attrs.get("crs", "")).strip().casefold()
        if crs.startswith("healpix:"):
            try:
                nside = int(crs.removeprefix("healpix:"))
            except ValueError:
                pass

    if nside is None and dim is not None:
        # A complete HEALPix grid always contains exactly 12 * NSIDE**2 cells.
        npix = data.sizes[dim]
        if npix % 12 == 0:
            candidate = isqrt(npix // 12)
            if (
                candidate > 0
                and candidate & (candidate - 1) == 0
                and 12 * candidate**2 == npix
            ):
                nside = candidate

    return int(nside) if nside is not None else None


def get_healpix_order(
    data: xr.Dataset | xr.DataArray, dim: str | None = None
) -> str:
    if dim is None:
        dim = get_healpix_dim(data)

    order = data.attrs.get("healpix_order")
    coord = data.coords.get(dim) if dim is not None else None
    if order is None and coord is not None:
        order = coord.attrs.get("dggs:indexing_scheme")
    return str(order or "ring").casefold()


def filter_healpix_bbox(data: RasterCube, extent: BoundingBox) -> RasterCube:
    dim = get_healpix_dim(data)
    if dim is None:
        raise DimensionNotAvailable(
            "No HEALPix spatial dimension available, can't apply filter_bbox."
        )

    indexer = healpix_lat_lon_bbox_indexer(data, dim, extent)
    if indexer is None:
        indexer = healpix_bbox_indexer(data, dim, extent)
    if indexer is None:
        raise DimensionNotAvailable(
            "HEALPix filter_bbox requires either lat/lon coordinates over the "
            "HEALPix dimension or HEALPix nside metadata with healpy installed."
        )

    return data.isel({dim: indexer})


def healpix_bbox_indexer(
    data: xr.Dataset | xr.DataArray,
    dim: str,
    extent: BoundingBox,
) -> np.ndarray | None:
    """Return integer positions for a HEALPix bbox crop.

    This mirrors the DEDL cube-load HEALPix crop: select cells by their centre
    lon/lat rather than using great-circle polygon edges.
    """
    if not is_healpix_dim(data, dim):
        return None

    try:
        import healpy as hp
    except ImportError:
        return None

    nside = get_healpix_nside(data, dim)
    if nside is None:
        return None

    nest = get_healpix_order(data, dim) != "ring"

    def bbox_mask(pixels: np.ndarray) -> np.ndarray:
        theta, phi = hp.pix2ang(nside, np.asarray(pixels, dtype=np.int64), nest=nest)
        lat = 90.0 - np.degrees(theta)
        lon = ((np.degrees(phi) + 180.0) % 360.0) - 180.0
        return _center_bbox_mask(lon=lon, lat=lat, extent=extent)

    try:
        full_grid_size = int(hp.nside2npix(nside))
    except (TypeError, ValueError):
        full_grid_size = -1

    if data.sizes[dim] == full_grid_size:
        return np.flatnonzero(bbox_mask(np.arange(full_grid_size)))

    coord = data.coords.get(dim)
    if coord is None:
        return None
    if (
        hasattr(coord.data, "dask") and coord.size > MAX_LAZY_HEALPIX_COORD_CELLS
    ):
        return None
    return np.flatnonzero(bbox_mask(np.asarray(coord.values)))


def healpix_lat_lon_bbox_indexer(
    data: xr.Dataset | xr.DataArray,
    dim: str,
    extent: BoundingBox,
) -> np.ndarray | None:
    lat = _get_named_variable(data, "lat", "latitude")
    lon = _get_named_variable(data, "lon", "longitude")
    if lat is None or lon is None:
        return None
    if lat.dims != (dim,) or lon.dims != (dim,):
        return None
    if (hasattr(lat.data, "dask") or hasattr(lon.data, "dask")) and (
        lat.size > MAX_LAZY_HEALPIX_COORD_CELLS
    ):
        return None

    lat_data = lat.data.compute() if hasattr(lat.data, "dask") else lat.data
    lon_data = lon.data.compute() if hasattr(lon.data, "dask") else lon.data
    return np.flatnonzero(
        _center_bbox_mask(
            lon=np.asarray(lon_data),
            lat=np.asarray(lat_data),
            extent=extent,
        )
    )


def _get_named_variable(
    data: xr.Dataset | xr.DataArray, *names: str
) -> xr.DataArray | None:
    for name in names:
        if name in data.coords:
            return data.coords[name]
        if name in data:
            return data[name]
    return None


def _center_bbox_mask(
    *,
    lon: np.ndarray,
    lat: np.ndarray,
    extent: BoundingBox,
) -> np.ndarray:
    lon = ((lon + 180.0) % 360.0) - 180.0
    width = float(extent.east) - float(extent.west)

    if width >= 360.0:
        lon_mask = np.ones_like(lon, dtype=bool)
    else:
        west = _normalize_longitude(float(extent.west))
        east = _normalize_longitude(float(extent.east))
        if float(extent.west) <= float(extent.east) and west <= east:
            lon_mask = (lon >= west) & (lon <= east)
        else:
            lon_mask = (lon >= west) | (lon <= east)

    return lon_mask & (lat >= float(extent.south)) & (lat <= float(extent.north))


def _normalize_longitude(value: float) -> float:
    return ((value + 180.0) % 360.0) - 180.0
