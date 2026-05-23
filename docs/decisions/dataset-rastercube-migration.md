# RasterCube: xr.Dataset as the native raster type

Date: 2026-05-23

Status: accepted and implemented on `dev_remodel`

## Context

openEO raster cubes are logically multidimensional arrays with axes `(t, bands, y, x)`. The codebase historically represented them as `xr.DataArray` with a "bands" dimension. This worked, but had a fundamental mismatch: openEO bands are named labels with semantics, not just another axis. Every band carries its own metadata, CRS, and nodata handling - things a flat DataArray dimension struggles to preserve.

The upstream reference PR ([openeo-processes-dask#372](https://github.com/Open-EO/openeo-processes-dask/pull/372)) adopted `xr.Dataset` to fix this. This repo (`openeo-processes-dask-slim`) follows suit.

The remodel plan also called out an implementation risk: a blanket `Dataset -> DataArray -> old implementation -> Dataset` wrapper hides Dataset semantics and can drop metadata or variables. The migration therefore makes `xr.Dataset` the public RasterCube contract and keeps any DataArray bridge local to process paths that structurally require a band axis, such as virtual-band reducers or UDF adapters.

## Decision

**`RasterCube = xr.Dataset`**, with bands stored as named data variables:

```
Dataset
  B02: (t, y, x)
  B03: (t, y, x)
  B04: (t, y, x)
  coords: {t, y, x}
  attrs: {crs, ...}
```

Key rules:

- **Bands are data variables**, not a named xarray dimension. The logical `(t, bands, y, x)` order is virtual: `bands` = `list(dataset.data_vars)`.
- **Public process boundaries assume `xr.Dataset`**. The type alias `RasterCube = xr.Dataset` provides static enforcement for type checkers. No runtime guard is needed — all raster inputs are expected to be Dataset already.
- **Virtual dimension `"bands"`** is handled explicitly in `apply_dimension`, `reduce_dimension`, `filter_labels`, and band-aware helpers. Cross-band operations may create a lazy temporary band axis, but the public result remains a Dataset.
- **Real dimensions** (`t`, `y`, `x`) use Dataset-aware xarray operations: `xr.apply_ufunc`, `Dataset.reduce`, `where`, `xr.merge`, or per-variable iteration.
- **Tests default to Dataset** through `create_fake_rastercube(..., as_dataset=True)`, so new raster process coverage exercises the target model by default.

## What changed

| Area | Before | After |
|---|---|---|
| Type alias | `Union[xr.DataArray, xr.Dataset]` | `xr.Dataset` |
| Test default | `as_dataset=False` | `as_dataset=True` |
| Band access | `.sel(bands="B02")` | `dataset["B02"]` |
| `.data` checks | `isinstance(cube.data, da.Array)` | per-variable `isinstance(var.data, da.Array)` |
| Cross-band reduce | `data.reduce(reducer, dim="bands")` | `to_array("bands")` → reduce → `to_dataset("bands")` with per-variable attrs preserved |
| Per-variable attrs | Dropped across `to_array` bridges | Preserved via `_capture_var_metadata`/`_restore_var_metadata` |
| Test fixtures | DataArray unless explicitly converted | Dataset by default |

## Implementation record

The relevant `dev_remodel` commits are:

| Commit | Purpose |
|---|---|
| `30ad4c5` | L1 minimal Dataset migration: `ensure_raster_cube` helper, Dataset-aware `apply_dimension` and `reduce_dimension`, Dataset-aware test comparisons. |
| `b6608dd`, `56f9feb`, `47481a0` | L2 and L2A process coverage: dimension helpers, filters, NDVI, temporal aggregation, mask, and kernel handling. |
| `d1172dc`, `022a912` | L3 and ML coverage: `merge_cubes`, mask tests, and `predict_random_forest` Dataset support. |
| `8e162fa`, `c31e490` | Final enforcement plan added to `plan_document.md` and kept current during the migration. |
| `b5ed758`, `2f4ec02`, `9c90626`, `4711606` | Final enforcement: `RasterCube = xr.Dataset`, `ensure_raster_cube` rejects DataArray, dead L1 imports removed, Dataset test default enabled. |
| `fd075db` | Dataset `dims` compatibility fix for `apply_neighborhood_intertwin` and plan status cleanup. |
| `3e3ad49` | Phase 0 review fixes: `create_data_cube()` returns `Dataset`, tie-breaking restored in `resample_cube_temporal`, `ensure_raster_cube` removed, Python 3.13/3.14 classifiers dropped. |
| `c30d8c9` | Phase 1: native Dataset `merge_cubes` — replaces `to_array` bridge, preserves variable order/attrs/CRS, fixes `_align_coordinates` in-place mutation. |
| `2f1c2bc` | Phase 2: remove `.compute()` from `predict_random_forest` — dask arrays flow through `dxgb.inplace_predict` natively. |
| `cc36e01` | Phase 3: per-variable attrs preserved across all `to_array` bridges via `_capture_var_metadata`/`_restore_var_metadata`. |
| `11612fd` | Phase 4: `array_find` return type documented (filled arrays, not masked). |

As of the completed review-fix cycle on `dev_remodel`:

- `RasterCube = xr.Dataset` with no runtime rejection guard.
- `merge_cubes` uses native Dataset merge.
- `predict_random_forest` does not call `.compute()` on raster payloads.
- All `to_array` bridges preserve per-variable attrs and variable order.
- Test fixture defaults and remaining process paths were migrated to Dataset compatibility.

## Consequences

### Positive

- **Semantic correctness**: Bands carry their own metadata, CRS, dtype — no more implicit assumptions from a shared dimension.
- **Multi-variable safety**: Impossible to accidentally drop or misalign bands. Each variable is independently addressable.
- **Future-proofing**: Aligns with upstream openeo-processes-dask direction.
- **Better test pressure**: Dataset is the default test cube shape, so new process tests are less likely to pass only through legacy DataArray behavior.

### Negative

- **Band operations need extra ceremony**: Cross-band processes (`reduce_dimension(dimension="bands")`, `apply_dimension(dimension="bands")`) need a temporary virtual band axis, adding graph overhead for wide band collections.
- **Some xarray APIs behave differently on Dataset**: `data.dims` returns a frozen mapping (not a tuple), `data.transpose()` works per-variable, etc. These surface as `TypeError` at runtime.
- **`.data` / `.values` on the cube itself no longer exists**: Callers must iterate `data_vars`. This required rewriting ~50 test assertions.
- **Compatibility adapters remain visible**: UDF, curve fitting still bridge through a DataArray-shaped representation because the external or existing API expects one. These paths preserve band order, attrs, and laziness via `_capture_var_metadata`/`_restore_var_metadata` helpers.
- **`merge_cubes` uses native Dataset merge**: The original `to_array`→`to_dataset` bridge was replaced with a per-variable merge that preserves variable order, per-variable attrs, and CRS.

### Scalability

- Dask laziness is preserved for Dataset-native process paths and for `to_array`/`to_dataset` conversions as long as the underlying arrays are dask-backed.
- Per-variable iteration does not materialize data: `for var in dataset.data_vars.values()` yields lazy DataArrays.
- Memory footprint per variable is smaller than a full 4D DataArray, which helps when individual bands are processed independently.
- The virtual bands path (`to_array` then reduce/apply) creates a temporary stacked DataArray in the task graph. For wide band collections (100+ bands), this can increase graph size even when it does not immediately load data into memory. Mitigate by processing bands in groups if needed.
