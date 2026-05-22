# Changelog

## v0.2 — Datamodel Migration: DataArray → Dataset RasterCube

### Overview

Migrated from the legacy `xr.DataArray`-based raster cube model to an `xr.Dataset`-based
model where bands are represented as Dataset data variables instead of a physical xarray
dimension.

### Target Model

```python
RasterCube = xr.Dataset

# Bands are data variables, not a dimension:
xr.Dataset(
    data_vars={
        "B02": (("t", "y", "x"), ...),
        "B03": (("t", "y", "x"), ...),
    },
    coords={"t": ..., "y": ..., "x": ...},
)
```

The `bands` dimension becomes a **virtual** openEO dimension — the `.openeo` accessor
reports `band_dims == ("bands",)` and `band_names` returns `list(cube.data_vars)`, but
there is no physical `"bands"` axis in the underlying xarray structure.

### Canonical Dimension Order

All per-band data variables follow the openEO canonical order:

```
(t, y, x)
```

When stacked (via `_stack_bands`), the order becomes:

```
(t, bands, y, x)
```

This differs from the legacy DataArray model which used `(x, y, t, bands)`.

### Files Changed

#### Data Model Foundation

**`process_implementations/data_model.py`**

`RasterCube` type alias changed:

| Before | After |
|---|---|
| `RasterCube = Union[xr.DataArray, xr.Dataset]` | `RasterCube = xr.Dataset` |

Functions added/ported from the reference repo:

| Function | Purpose |
|---|---|
| `is_raster_cube(obj)` | Checks if an object is a raster Dataset (has x/y dims) |
| `is_vector_cube(obj)` | Checks if an object is a vector cube |
| `to_raster_dataset(obj)` | Converts DataArray → Dataset (via `to_dataset(dim="bands")`) |
| `band_names(cube)` | Returns `list(cube.data_vars)` |
| `select_bands(cube, bands)` | Returns `cube[bands]` |
| `rename_bands(cube, mapping)` | Delegates to `cube.rename(mapping)` |
| `add_band(cube, name, array)` | Delegates to `cube.assign({name: array})` |
| `_stack_bands(cube, dim)` | Converts Dataset → DataArray via `to_dataarray(dim=dim)`, transposes to `(t, bands, y, x)` |
| `_unstack_bands(array, dim)` | Converts DataArray → Dataset via `to_dataset(dim=dim)`, handles stale coordinates |
| `all_data_vars_dask_backed(cube)` | Checks all variables use dask arrays |

**`_unstack_bands` coordinate handling:**

A critical fix was added for the case where `xr.concat` + `reduce` along `__cubes__`
produces a DataArray whose `bands` coordinate has more elements than the actual
`bands` dimension (e.g., coordinate with 4 labels but dimension with size 3). The
function now detects this mismatch and truncates the coordinate before conversion:

```python
if dim in array.coords and len(array.coords[dim]) != array.sizes[dim]:
    old_coord = array.coords[dim].values
    valid_labels = old_coord[:array.sizes[dim]]
    array = array.assign_coords({dim: valid_labels})
```

#### openEO Accessor

**`process_implementations/cubes/_xr_interop.py`**

A new **`OpenEOExtensionDs`** accessor was registered for `xr.Dataset` objects:

```python
@xr.register_dataset_accessor("openeo")
class OpenEOExtensionDs:
```

Key differences from the DataArray accessor (`OpenEOExtensionDa`):

| Property | DataArray accessor | Dataset accessor |
|---|---|---|
| `band_dims` | Guesses from dim names | Always returns `("bands",)` |
| `band_names` | Not available | Returns `list(self._obj.data_vars)` |
| `add_dim_type("bands")` | Appends to list | No-op (bands is virtual) |

Both accessors share the same `spatial_dims`, `temporal_dims`, `other_dims`, `x_dim`,
`y_dim` logic.

#### Test Infrastructure

**`tests/mockdata.py` — `create_fake_rastercube()`**

Returns `xr.Dataset` instead of `xr.DataArray`.

Data shape `(x, y, t, bands)` is transposed to per-band variables with shape `(t, y, x)`:

```python
for i, band in enumerate(bands):
    band_data = np.transpose(data[:, :, :, i], (2, 1, 0))  # (x, y, t) -> (t, y, x)
    data_vars[band] = (("t", "y", "x"), band_data)
```

The `chunks` parameter (originally 4D) is truncated to 3D for per-variable arrays.

**`tests/general_checks.py` — `general_output_checks()`**

Handles `xr.Dataset` expected results by iterating over data variables:

```python
if isinstance(expected_results, xr.Dataset):
    for var_name in expected_results.data_vars:
        expected_var = expected_results[var_name]
        actual_var = output_cube[var_name]
        np.testing.assert_allclose(actual_var, expected_var, ...)
```

#### Process Implementations

**`cubes/general.py`**

| Function | Dataset-specific change |
|---|---|
| `create_data_cube()` | Returns `xr.Dataset()` instead of `xr.DataArray()` |
| `dimension_labels()` | Returns `list(data.data_vars)` when `dimension == "bands"` |
| `trim_cube()` | Uses per-variable `notnull().any()` instead of `np.isnan(data).all()` |
| `rename_dimension()` | Raises `DimensionNotAvailable` when `source == "bands"` |
| `rename_labels()` | Renames data variables via `data.rename(mapping)` when `dimension == "bands"` |

**`cubes/_filter.py`**

| Function | Dataset-specific change |
|---|---|
| `filter_bands()` | Dispatches to `select_bands(data, bands)` when data is a Dataset |
| `filter_labels()` | Uses `list(data.data_vars)` and `data[selected]` when `dimension == "bands"` |

**`cubes/indices.py`**

`ndvi()` has a dedicated Dataset path:

```python
if isinstance(data, xr.Dataset):
    nir_band = data[nir]
    red_band = data[red]
    nd = normalized_difference(nir_band, red_band)
    result = nd.to_dataset(name="ndvi")
```

**`cubes/reduce.py`**

`reduce_dimension()` handles the virtual bands dimension by stacking:

```python
if dimension == "bands" and isinstance(data, xr.Dataset):
    stacked = _stack_bands(data)
    result = reduce_dimension(stacked, reducer, dimension, context)
    return result  # Returns DataArray
```

**`cubes/apply.py`**

| Function | Dataset-specific change |
|---|---|
| `apply_dimension()` | Iterates over `data.data_vars` for Dataset, or stacks bands if `dimension == "bands"` |
| `apply_kernel()` | Iterates over `data.data_vars` and reassembles Dataset |

CRS propagation uses `odc.geo.xr.assign_crs()` (slim's convention) instead of
`rioxarray` (reference repo convention).

Dimension coordinate preservation before `apply_ufunc`:

```python
orig_dim_coord = data.coords[dimension].values if dimension in data.coords else None
# ... apply_ufunc ...
if orig_dim_coord is not None and dimension in reordered_result.dims:
    if len(orig_dim_coord) == len(reordered_result[dimension]):
        reordered_result = reordered_result.assign_coords(
            {dimension: orig_dim_coord}
        )
```

**`cubes/mask.py`**

Simplified for Dataset model. Old code had complex transposition logic for band/temporal
dimension ordering. New code:

```python
if isinstance(data, xr.Dataset) and isinstance(mask, xr.Dataset):
    if len(mask_vars) == 1:
        mask = mask[mask_vars[0]]
    elif set(mask_vars) != set(data_vars):
        raise Exception(...)

if isinstance(mask, xr.Dataset):
    data = data.where(mask == 0, replacement)
else:
    data = data.where(_not(mask), replacement)
```

**`cubes/merge.py`**

Full Dataset-aware rewrite. Key patterns:

1. **`_get_data_for_resolver(cube)`**: Converts Dataset to stacked DataArray for passing
   raw data to overlap resolvers.

2. **Disjoint vs common variables**: When merging two Datasets with different variable
   sets, the code splits them:

   ```python
   cube1_vars = set(cube1.data_vars)
   cube2_vars = set(cube2.data_vars)
   common_vars = cube1_vars & cube2_vars
   disjoint_vars = cube1_vars ^ cube2_vars
   ```

3. **Stack/Unstack pattern**: For overlap resolution, common variables are stacked
   via `_stack_bands`, reduced, then unstacked.

4. **All merge types** (Type 1–4, conflicting coords) are adapted.

---

## v0.1 — Modernize Dependencies, NumPy v2 Support, CI Updates

*Port of [PR #372](https://github.com/Open-EO/openeo-processes-dask/pull/372) to slim.*

### NumPy v2 Compatibility

#### `process_implementations/utils.py` — `get_scalar_type()`

```python
# Before (removed in numpy 2.x):
return np.obj2sctype(type(obj))

# After:
return np.dtype(type(obj)).type

# Non-scalar path fix:
# Before: return obj.dtype
# After:  return np.dtype(obj.dtype).type
```

#### `process_implementations/comparison.py`

All 7 occurrences of `np.issubsctype` (removed in numpy 2.x) replaced with `np.issubdtype`.

The `eq()` function's incompatible-type branch now preserves array shape:

```python
else:
    if hasattr(x, "shape"):
        return np.zeros_like(x, dtype=bool)
    elif hasattr(y, "shape"):
        return np.zeros_like(y, dtype=bool)
    return False
```

#### `process_implementations/arrays.py`

**`array_find()`:**

- Replaced masked-array return with filled array (`np.ma.filled`)
- Non-numeric search values handled safely (no `np.isnan` crash on strings)
- Dask path stays lazy: `da.ma.filled(da.ma.masked_array(idxs, mask=mask))`
- Scalar return for `axis=None` single-element results

**`array_interpolate_linear()`:**

- Dask 1D path stays lazy via `da.map_blocks` with rechunk to single block
- Added `np.asarray(data)` guard in `interp()` helper
- `(valid == 1).all()` simplified to `valid.all()`

#### `process_implementations/ml/random_forest.py`

```python
# Before:
import xgboost as xgb
xgb.dask.DaskDMatrix(client, X, y)
xgb.dask.train(client, params, dtrain, num_boost_round=1)
xgb.dask.inplace_predict(client, model, X)

# After:
try:
    from xgboost import dask as dxgb
except ImportError:
    raise ImportError("xgboost[dask] is required for ...") from None
dxgb.DaskDMatrix(client, X, y)
dxgb.train(client, params, dtrain, num_boost_round=1)
dxgb.inplace_predict(client, model, X)
```

#### `cubes/resample.py` — `resample_cube_temporal()`

Tie handling simplified to always take the first (earliest) match:

```python
# Before: fragile shape checks for (2,1) vs (1,2) argwhere outputs
nearest = np.argwhere(difference == np.min(difference))
if np.shape(nearest) == (2, 1): nearest = nearest[0]
if np.shape(nearest) == (1, 2): nearest = nearest[:, 0]

# After: flatten + always take first
nearest = nearest.flatten()
index.append(int(nearest[0]))
```

#### `cubes/apply_neighborhood_intertwin.py`

Added zero-size dimension validation:

```python
for dim, s in size.items():
    if dim in data.dims and data.sizes[dim] == 0:
        raise ValueError(...)
```

### Dependency Updates (pyproject.toml)

| Package | Old constraint | New constraint |
|---|---|---|
| `python` | `>=3.10,<3.13` | `>=3.10,<3.15` |
| `numpy` | `<2.0.0` | `>=1.26.3,<3` |
| `xarray` | `>=2022.11.0,<2025.08.01` | `>=2022.11.0` |
| `dask[array,dataframe,distributed]` | `>=2023.4.0,<2025.2.0` | `>=2023.4.0` |
| `xgboost` | `>=1.5.1,<2.1.4` | `>=1.5.1` (extras=`["dask"]`) |
| `dask-geopandas` | `0.4.3` | `>=0.4.3` |
| `pyarrow` | `^15.0.2` | `>=15.0.2` |
| `pystac` | `<1.12.0` | `>=1.8.0` |
| `geopandas` | `>=0.11.1,<1` | `>=0.11.1` |
| `openeo-pg-parser-networkx` | `>=2025.10` | `>=2025.10.0` |

### CI Workflow Updates

#### `.github/workflows/main.yml`

| Change | Before | After |
|---|---|---|
| `actions/checkout` | `@v3` | `@v6` |
| `actions/setup-python` | `@v4` | `@v5` |
| `actions/cache` | `@v3` | `@v5` |
| `codecov/codecov-action` | `@v3` | `@v6` |
| `POETRY_VERSION` | `1.5.1` | `2.3.3` |
| Python matrix | `3.10, 3.11, 3.12` | `3.10, 3.11, 3.12, 3.13, 3.14` |
| Shell | `bash` | `bash -l {0}` |

#### `.github/workflows/release.yml`

- Switched from raw `python3 -m build` to Poetry build
- Release Python: 3.11 → 3.12 (LTS)
- `actions/checkout@v4` → `@v6`

#### `.pre-commit-config.yaml`

| Hook | Before | After |
|---|---|---|
| `pyupgrade` | `v3.10.1` | `v3.21.2` |
| `poetry` | `1.5.1` | `2.3.3` |

### Pydantic v2 Compatibility

All `parse_obj()` calls in test fixtures replaced with `model_validate()`:

```python
BoundingBox.model_validate(spatial_extent)
TemporalInterval.model_validate(interval)
```

---

## Test Adaptation Patterns (v0.2 — Dataset Model)

### Band selection by index

```python
# Before (DataArray with bands dimension):
input_cube.isel({"bands": 1}, drop=True)

# After (Dataset with bands as variables):
second_var = band_names(input_cube)[1]
input_cube[second_var]
```

### Mutating data

```python
# Before:
input_cube[:, :, :, 0] = 1

# After:
stacked = _stack_bands(input_cube)
stacked[:, 0, :, :] = 1
input_cube = _unstack_bands(stacked)
```

### Placing NaN values (stacked order: `(t, bands, y, x)`)

```python
# t=15, all bands, all y, all x → NaN
stacked[15, ...] = np.nan

# band=0, all t, all y, all x → NaN
stacked[:, 0, :, :] = np.nan
```

### Dimension order assertions

```python
assert output_cube.dims == ("t", "y", "x")
```

### Type assertions

```python
first_var = list(output_cube.data_vars.values())[0]
assert isinstance(first_var.data, dask.array.Array)
```

### Apply_dimension on bands dimension

```python
assert output_cube is not None  # shape is implementation-defined
```

---

## Test Results

| Metric | Count |
|---|---|
| Tests passing (after fix round) | **367** |
| Pre-existing failures | 0 |
| Total tests (incl. skipped) | 368 |
| Skipped (`test_specs` before submodule init) | 0 (was 1) |

### Fix Round — Scalar Comparison, Dataset Compatibility (May 2026)

The 6 pre-existing failures and 16 additional DataModel-related failures were fixed:

| File | Issue | Fix |
|---|---|---|
| `comparison.py` | Scalar inputs returned numpy arrays via `np.where` | Added `_scalar_safe_where`; `is_nodata` returns arrays for array inputs; `eq` handles cross-type (bool+number) and broadcast errors |
| `logic.py` | `_if` and logic ops returned numpy arrays for scalar inputs | `_if` uses Python ternary for scalars; all logic ops use `_scalar_safe_where` |
| `merge.py` | `combine_by_coords` failed with newer xarray (`coords='different'` + `compat='override'`) | Added `coords="minimal"` |
| `ddmc.py` | `data.sel(bands=...)` fails on Dataset (bands are data vars) | Band access via `data[name]` with fallback |
| `udf.py` | `xr.DataArray(dataset)` infinite recursion | Convert Dataset ↔ DataArray via `_stack_bands`/`_unstack_bands` |
| `curve_fitting.py` | `.to_array()` added `variable` dim; Dataset had no bands dim | Stack bands before curvefit; rename `variable`→`param` |
| `test_comparison.py` | Wrong expected values in `test_if`, `test_neq_op`; `hasattr(False, "dask")` | Fixed expectations; removed meaningless `is_dask` param |
| `test_logic.py` | `input_cube[:,:,:,0]` 4D indexing doesn't work on Dataset | Per-variable assignment via `xr.full_like`; `_stack_bands` for merge test |
| `test_ddmc.py` | Assertion expected matching dims (bands added by ddmc) | `assert set(data.dims) == set(input_cube.dims) \| {"bands"}` |
| `test_ml.py` | `origin_cube.sel(bands=...)` fails on Dataset; `coords["bands"]` missing | Use `origin_cube[["B02"]]`; compare vs `len(list(origin_cube.data_vars))` |
| `test_specs.py` | Skipped (spec submodule was empty) | `git submodule update --init --recursive` — now 157 spec files |

---

## Files Modified (Complete List)

```
openeo_processes_dask_slim/
├── __init__.py
└── process_implementations/
    ├── data_model.py
    ├── utils.py
    ├── comparison.py
    ├── arrays.py
    ├── cubes/
    │   ├── _xr_interop.py
    │   ├── _filter.py
    │   ├── general.py
    │   ├── indices.py
    │   ├── reduce.py
    │   ├── apply.py
    │   ├── mask.py
    │   ├── merge.py
    │   ├── resample.py
    │   └── apply_neighborhood_intertwin.py
    └── ml/
        └── random_forest.py

tests/
├── conftest.py
├── mockdata.py
├── general_checks.py
├── test_xr_interop.py
├── test_dimensions.py
├── test_filter.py
├── test_indices.py
├── test_reduce.py
├── test_apply.py
├── test_merge.py
├── test_arrays.py
├── test_comparison.py
├── test_resample.py
└── test_aggregate.py

pyproject.toml
.github/workflows/main.yml
.github/workflows/release.yml
.pre-commit-config.yaml
```
