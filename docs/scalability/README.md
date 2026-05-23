# Scalability notes

Edge cases and design tradeoffs that affect scalability in `openeo-processes-dask-slim`.

## Modernization baseline

The `dev_remodel` branch first modernized the repository before changing the RasterCube data model. This matters for scalability because the Dataset migration depends on current xarray, dask, NumPy, and XGBoost behavior.

Relevant commits:

| Commit | Scalability relevance |
|---|---|
| `2fc230a` | Widened Python and dependency support, enabled NumPy 2 compatibility, refreshed CI/release tooling, updated XGBoost Dask imports, and adjusted array helpers to avoid unnecessary eager computation. |
| `93e72e6` | Removed private NumPy exception usage and updated Pydantic fixture validation. |
| `d880d90` | Merged the modernization checkpoint into `dev_remodel` before Dataset migration phases. |

Modernization does not by itself change RasterCube semantics. It establishes the runtime baseline used by the Dataset migration.

## Dataset RasterCube: memory and laziness

The migration from `xr.DataArray` to `xr.Dataset` for RasterCube changes the memory profile:

**Per-variable storage.** Each band is its own data variable with shape `(t, y, x)`. This is beneficial when processes touch only a subset of bands — the untouched variables stay as dask task graphs without being computed.

**Dataset-native hot paths.** Real-dimension operations should use Dataset-aware xarray APIs such as `xr.apply_ufunc(..., dask="allowed")`, `Dataset.reduce`, `Dataset.where`, `xr.merge`, or explicit per-variable iteration. These paths keep data chunked by variable and avoid constructing a full 4D band stack.

**Virtual bands stacking.** Cross-band processes (`reduce_dimension(dimension="bands")`, `apply_dimension(dimension="bands")`) stack variables into a temporary DataArray via `to_array(dim="bands")`. For cubes with many bands (100+), this temporarily creates a 4D array in the task graph. Mitigations:

- The stacking is lazy (dask graph nodes, not memory).
- If the reducer operates per-pixel (like `mean`), the dask scheduler handles chunking.
- For extreme band counts, consider processing bands in batches.
- Keep virtual-band conversion scoped to the process that needs it, and convert back to Dataset immediately.

**Dask chunking.** Dataset variables inherit their chunks from the source DataArray's "bands" dimension (which becomes the variable count). Each variable's spatial-temporal chunks are preserved independently. This means:

- `(t: 100, y: 1000, x: 1000, bands: 20)` → 20 variables each chunked `(t: 100, y: 500, x: 500)`
- Temporal reduction on one band does not trigger computation on other bands.

## Dataset migration record

The migration was completed on `dev_remodel` in reviewable phases:

| Commit | Scalability relevance |
|---|---|
| `30ad4c5` | Added Dataset-aware L1 apply/reduce paths and Dataset comparisons in shared test helpers. |
| `56f9feb`, `47481a0`, `d1172dc` | Added Dataset-native paths for filters, NDVI, temporal aggregation, mask, kernel handling, and merge behavior. |
| `022a912` | Added Dataset support for random forest prediction inputs. |
| `2f4ec02`, `4711606` | Enforced Dataset at public RasterCube boundaries and made Dataset the default raster test fixture. |
| `fd075db` | Fixed `Dataset.dims` mapping compatibility in `apply_neighborhood_intertwin`. |

## Known patterns that break laziness

- Calling `.values` or `.compute()` on raster payloads in process hot paths. Coordinate-label inspection may still use `.values`, but array payloads should stay lazy.
- `np.nanmean` on a Dataset directly — use per-variable iteration or `to_array` stacking instead.
- `data.reduce` along real dimensions (t, y, x) preserves dask; only the virtual bands path introduces stacking overhead.

## Bounded DataArray bridges

The migration rejects `xr.DataArray` as a public RasterCube input, but some internals still need a DataArray-shaped representation:

- Virtual band operations use `to_array(dim="bands")` because xarray has no real Dataset dimension for data variables.
- `run_udf` adapts Dataset input to the openEO UDF `XarrayDataCube` interface and converts the result back to Dataset.
- `fit_curve` bridges Dataset input through a band axis while fitting, then restores the Dataset shape.
- `merge_cubes` can bridge through DataArray when an overlap resolver is expressed along a band dimension.

These bridges are acceptable only when they are local, lazy for dask-backed arrays, preserve variable order and attributes where possible, and return Dataset at the public process boundary.

## DataArray fallback in non-migrated paths

Some processes (e.g., `vector_buffer`, `load_geojson`) operate on `VectorCube`, not `RasterCube`, and are unaffected by the Dataset migration. Other non-raster processes (`dates.py`, `text.py`, `math.py` scalars) work on plain arrays or scalars.

Processes that accept `RasterCube` but have not been migrated to Dataset-native patterns will fail with `TypeError: expects an xarray.Dataset RasterCube`. The `ensure_raster_cube` guard at the public boundary catches these early.
