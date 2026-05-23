# RasterCube: xr.Dataset as the native raster type

Date: 2026-05-23

## Context

openEO raster cubes are logically multidimensional arrays with axes `(t, bands, y, x)`. The codebase historically represented them as `xr.DataArray` with a "bands" dimension. This worked, but had a fundamental mismatch: openEO bands are named labels with semantics, not just another axis. Every band carries its own metadata, CRS, andnodata handling — things a flat DataArray dimension struggles to preserve.

The upstream `openeo-processes-dask` reference PR [#372](https://github.com/Open-EO/openeo-processes-dask/pull/372) adopted `xr.Dataset` to fix this. This repo follows suit.

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
- **Virtual dimension `"bands"`** is handled specially in `apply_dimension`, `reduce_dimension`, and `filter_labels` — they convert to/from DataArray internally when cross-band operations are needed.
- **Real dimensions** (`t`, `y`, `x`) work identically to DataArray mode: per-variable `apply_ufunc` or `data.reduce()`.

## What changed

| Area | Before | After |
|---|---|---|
| Type alias | `Union[xr.DataArray, xr.Dataset]` | `xr.Dataset` |
| Test default | `as_dataset=False` | `as_dataset=True` |
| Band access | `.sel(bands="B02")` | `dataset["B02"]` |
| `.data` checks | `isinstance(cube.data, da.Array)` | per-variable `isinstance(var.data, da.Array)` |
| Cross-band reduce | `data.reduce(reducer, dim="bands")` | `to_array("bands")` → reduce → `to_dataset("bands")` |

## Consequences

### Positive

- **Semantic correctness**: Bands carry their own metadata, CRS, dtype — no more implicit assumptions from a shared dimension.
- **Multi-variable safety**: Impossible to accidentally drop or misalign bands. Each variable is independently addressable.
- **Clearer contracts**: `ensure_raster_cube` fails fast on DataArray inputs, catching mismatches at the boundary instead of silently converting.
- **Future-proofing**: Aligns with upstream openeo-processes-dask direction.

### Negative

- **Band operations need extra ceremony**: Cross-band processes (`reduce_dimension(dimension="bands")`) must stack variables via `to_array`, reducing laziness slightly.
- **Some xarray APIs behave differently on Dataset**: `data.dims` returns a frozen mapping (not a tuple), `data.transpose()` works per-variable, etc. These surface as `TypeError` at runtime.
- **`.data` / `.values` on the cube itself no longer exists**: Callers must iterate `data_vars`. This required rewriting ~50 test assertions.
- **Backward-incompatible for callers**: Any code passing DataArray to a migrated process now gets `TypeError`.

### Scalability

- Dask laziness is preserved: all `to_array`/`to_dataset` conversions are lazy as long as the underlying arrays are dask-backed.
- Per-variable iteration does not materialize data: `for var in dataset.data_vars.values()` yields lazy DataArrays.
- Memory footprint per variable is smaller than a full 4D DataArray, which helps when individual bands are processed independently.
- The virtual bands path (`to_array` then reduce) creates a temporary stacked DataArray. For wide band collections (100+ bands), this allocates a temporary 4D array. Mitigate by processing bands in groups if needed.
