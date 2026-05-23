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
- **Public process boundaries reject `xr.DataArray`** via `ensure_raster_cube()`. Users pass Dataset, get Dataset back.
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
| Cross-band reduce | `data.reduce(reducer, dim="bands")` | `to_array("bands")` → reduce → `to_dataset("bands")` |
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

As of the final enforcement plan in `plan_document.md`, Phases A-D are complete:

- `RasterCube = xr.Dataset`.
- `ensure_raster_cube` rejects `xr.DataArray`.
- No runtime downstream call chain required the removed L1 imports.
- Test fixture defaults and remaining process paths were migrated to Dataset compatibility.

## Consequences

### Positive

- **Semantic correctness**: Bands carry their own metadata, CRS, dtype — no more implicit assumptions from a shared dimension.
- **Multi-variable safety**: Impossible to accidentally drop or misalign bands. Each variable is independently addressable.
- **Clearer contracts**: `ensure_raster_cube` fails fast on DataArray inputs, catching mismatches at the boundary instead of silently converting.
- **Future-proofing**: Aligns with upstream openeo-processes-dask direction.
- **Better test pressure**: Dataset is the default test cube shape, so new process tests are less likely to pass only through legacy DataArray behavior.

### Negative

- **Band operations need extra ceremony**: Cross-band processes (`reduce_dimension(dimension="bands")`, `apply_dimension(dimension="bands")`) need a temporary virtual band axis, adding graph overhead for wide band collections.
- **Some xarray APIs behave differently on Dataset**: `data.dims` returns a frozen mapping (not a tuple), `data.transpose()` works per-variable, etc. These surface as `TypeError` at runtime.
- **`.data` / `.values` on the cube itself no longer exists**: Callers must iterate `data_vars`. This required rewriting ~50 test assertions.
- **Backward-incompatible for callers**: Any code passing DataArray to a migrated process now gets `TypeError`.
- **Compatibility adapters remain visible**: UDF, curve fitting, and some overlap-resolver paths still bridge through a DataArray-shaped representation because the external or existing API expects one. These paths must preserve band order, attrs, and laziness where possible.

### Scalability

- Dask laziness is preserved for Dataset-native process paths and for `to_array`/`to_dataset` conversions as long as the underlying arrays are dask-backed.
- Per-variable iteration does not materialize data: `for var in dataset.data_vars.values()` yields lazy DataArrays.
- Memory footprint per variable is smaller than a full 4D DataArray, which helps when individual bands are processed independently.
- The virtual bands path (`to_array` then reduce/apply) creates a temporary stacked DataArray in the task graph. For wide band collections (100+ bands), this can increase graph size even when it does not immediately load data into memory. Mitigate by processing bands in groups if needed.
