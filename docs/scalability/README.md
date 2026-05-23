# Scalability notes

Edge cases and design tradeoffs that affect scalability in `openeo-processes-dask`.

## Dataset RasterCube: memory and laziness

The migration from `xr.DataArray` to `xr.Dataset` for RasterCube changes the memory profile:

**Per-variable storage.** Each band is its own data variable with shape `(t, y, x)`. This is beneficial when processes touch only a subset of bands — the untouched variables stay as dask task graphs without being computed.

**Virtual bands stacking.** Cross-band processes (`reduce_dimension(dimension="bands")`, `apply_dimension(dimension="bands")`) stack variables into a temporary DataArray via `to_array(dim="bands")`. For cubes with many bands (100+), this temporarily creates a 4D array in the task graph. Mitigations:

- The stacking is lazy (dask graph nodes, not memory).
- If the reducer operates per-pixel (like `mean`), the dask scheduler handles chunking.
- For extreme band counts, consider processing bands in batches.

**Dask chunking.** Dataset variables inherit their chunks from the source DataArray's "bands" dimension (which becomes the variable count). Each variable's spatial-temporal chunks are preserved independently. This means:

- `(t: 100, y: 1000, x: 1000, bands: 20)` → 20 variables each chunked `(t: 100, y: 500, x: 500)`
- Temporal reduction on one band does not trigger computation on other bands.

## Known patterns that break laziness

- Calling `.values` or `.compute()` in process hot paths. The static audit rejects these in migrated raster modules unless justified.
- `np.nanmean` on a Dataset directly — use per-variable iteration or `to_array` stacking instead.
- `data.reduce` along real dimensions (t, y, x) preserves dask; only the virtual bands path introduces stacking overhead.

## DataArray fallback in non-migrated paths

Some processes (e.g., `vector_buffer`, `load_geojson`) operate on `VectorCube`, not `RasterCube`, and are unaffected by the Dataset migration. Other non-raster processes (`dates.py`, `text.py`, `math.py` scalars) work on plain arrays or scalars.

Processes that accept `RasterCube` but have not been migrated to Dataset-native patterns will fail with `TypeError: expects an xarray.Dataset RasterCube`. The `ensure_raster_cube` guard at the public boundary catches these early.
