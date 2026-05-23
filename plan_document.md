# Remodel Plan: Modernization And Native Dataset RasterCube Migration

## 1. Purpose

This document defines the redo plan for modernizing `openeo-processes-dask-slim` and migrating raster cube process implementations from the legacy `xarray.DataArray` model to a native `xarray.Dataset` model.

The migration must be conservative. The codebase should remain recognizable from `main`; this is not a rewrite. The objective is correctness, reviewability, and optimized Dataset-native execution without shortcut conversions through the old DataArray implementation.

Primary references:

- openEO process profiles: https://openeo.org/documentation/1.0/developers/profiles/processes.html
- Modernization reference: https://github.com/Open-EO/openeo-processes-dask/pull/372

## 2. Current Repository State

The repository currently has:

- `main`: baseline branch whose structure, style, and process organization should be preserved.
- `dataset-rastercube-refactor`: previous attempted Dataset refactor.
- `dev_remodel`: integration branch for the redo work.

The previous attempted refactor changed multiple raster cube modules, tests, UDF, ML, and data model code. It showed useful direction, but also introduced risky patterns that should not become the redo architecture.

The primary anti-pattern to avoid is:

```text
xr.Dataset -> xr.DataArray -> old DataArray process implementation -> xr.Dataset
```

This shortcut can appear to work for simple single-variable cases, but it hides real Dataset semantics, drops or corrupts metadata, and often fails for multi-variable cubes.

## 3. Development Environment

All development, testing, and checkpoint verification will be performed using the `slim_dataset` micromamba environment:

```bash
micromamba create -n slim_dataset python=3.12
micromamba activate slim_dataset
pip install poetry
poetry install --with dev --all-extras
```

This environment must be used consistently across all phases to ensure reproducible results. Activate it before any `poetry install`, `poetry run pytest`, or related development commands.

## 4. Branch And Review Model

Use `dev_remodel` as the long-lived integration branch.

Use short-lived service branches for reviewable checkpoint work:

```text
main
  -> dev_remodel
       <- refactor_remodel/01-modernization
       <- refactor_remodel/02-l1-dataset
       <- refactor_remodel/03-l2-recommended
       <- refactor_remodel/04-l2a-raster
       <- later profile branches
```

Rules:

- Do not create one massive migration branch.
- Each service branch targets `dev_remodel`.
- Each checkpoint must be reviewable and rollback-friendly.
- Do not mix modernization and Dataset migration in the same checkpoint.

## 5. Core Migration Contract

The target RasterCube model is:

```python
RasterCube = xr.Dataset
```

All raster inputs are assumed to be `xr.Dataset` at the point of entry. No runtime guard is needed — the type alias provides static enforcement for type checkers. DataArray may still appear as individual xarray variables during implementation (e.g., extracting a single band), but public RasterCube APIs expect Dataset.

## 6. Canonical Dimension Model

The canonical logical openEO raster cube order is:

```text
(t, bands, y, x)
```

For native `xr.Dataset`, `bands` is not a real xarray dimension. Bands are represented as Dataset data variables. Therefore each band variable should normally use:

```text
(t, y, x)
```

The logical view is:

```text
Dataset data variables:
  B02: (t, y, x)
  B03: (t, y, x)

Logical openEO cube:
  (t, bands, y, x)
```

Dimension rules:

- Temporal dimension is first when present.
- Spatial dimensions are ordered as `(y, x)`.
- Dataset variables preserve real dimension order `(t, y, x)` when those dimensions exist.
- Virtual `bands` order is `list(dataset.data_vars)`.
- If the temporal dimension is named `time` instead of `t`, preserve the actual name but keep it in the temporal slot.
- Extra dimensions should be preserved after known canonical dimensions unless a process explicitly changes them.
- Do not use `to_array(dim="bands")` to enforce logical band order.

Expected examples:

```text
Dataset with time:
  B02: (t, y, x)
  B03: (t, y, x)
  logical order: (t, bands, y, x)

Dataset without time:
  B02: (y, x)
  B03: (y, x)
  logical order: (bands, y, x)

Single-band Dataset:
  ndvi: (t, y, x)
  logical order: (t, bands, y, x), bands = ["ndvi"]
```

## 7. Refactor Principles

Preserve:

- Existing module layout.
- Existing public function names.
- Existing process implementation structure where practical.
- Existing behavior unless it conflicts with native Dataset semantics.
- Existing tests where the semantics remain valid.

Avoid:

- Large file moves.
- Broad deletion.
- Rewriting whole modules when focused local edits are enough.
- Formatting unrelated files.
- Adding generic abstraction layers without repeated correctness need.
- Calling `.values`, `.to_numpy()`, or `.compute()` in raster process hot paths.
- Deep-copying large Datasets unless required.

Acceptable changes:

- Small Dataset-native helpers for repeated invariants.
- Dataset-native replacement of DataArray-specific assumptions.
- Removal of proven dead or harmful DataArray raster cube branches.
- Focused tests for each migrated process profile.

## 8. Phase 1: Modernization

Modernization must happen before the Dataset migration.

Use Open-EO/openeo-processes-dask PR #372 as the reference, but port only changes that apply to this slim repo.

Modernization tasks:

- Widen dependency constraints conservatively, especially for NumPy 2 compatibility.
- Replace deprecated NumPy APIs such as `np.obj2sctype` and `np.issubsctype`.
- Preserve lazy behavior for dask-backed arrays.
- Update XGBoost Dask imports to `from xgboost import dask as dxgb` where relevant.
- Update Pydantic-related test usage if deprecated APIs are present.
- Update CI only where applicable to this slim repo, including conda-forge/GDAL handling if needed.

Modernization constraints:

- Do not change the RasterCube data model in this phase.
- Do not include Dataset migration work.
- Do not import unrelated upstream changelog content unless this repo explicitly needs it.
- Keep the diff tightly scoped to dependency, CI, compatibility, and targeted bug fixes.

Modernization acceptance criteria:

- Existing tests pass.
- Modernization-specific tests pass.
- Deprecated NumPy API audit passes.
- Dask laziness is preserved in changed array paths.
- Diff is reviewable and focused.

## 9. Phase 2: L1 Minimal Dataset Migration

After modernization is merged into `dev_remodel`, migrate the L1 Minimal profile first.

Base profile scope on the openEO process profile documentation. Start with raster-relevant L1 behavior before later profiles.

Primary L1 raster targets:

- `apply`
- `apply_dimension`
- `reduce_dimension`
- L1 math, comparison, logic, and array reducers used inside callbacks.

### 9.1 `apply`

Expected behavior:

- Apply element-wise callbacks to every Dataset data variable.
- Preserve coordinates, Dataset attrs, variable attrs where possible, CRS/geobox metadata, variable names, and dimension order.
- Preserve dask laziness.

Implementation direction:

- Prefer `xr.apply_ufunc` directly on `xr.Dataset` for element-wise operations.
- Use explicit per-variable loops only when Dataset-level behavior is ambiguous or metadata-sensitive.
- Preserve resulting DataArrays directly when rebuilding a Dataset.
- Normalize real variable dimension order to `(t, y, x)` when xarray reorders dimensions.

### 9.2 `apply_dimension`

Expected behavior:

- If `dimension in data.dims`, treat it as a real xarray dimension and apply the callback per variable along that dimension.
- If `dimension == "bands"`, treat it as the virtual band dimension over `data.data_vars`.
- Otherwise raise the existing appropriate dimension error.
- Output remains `xr.Dataset`.
- Remaining dimensions follow canonical order.

Virtual bands behavior:

- Variables are stacked into a temporary DataArray via `to_array(dim="bands")` and processed with `xr.apply_ufunc`.
- Per-variable attributes and variable order are captured before stacking and restored after unstacking via `_capture_var_metadata` / `_restore_var_metadata` helpers.
- Callback output is normalized to `xr.Dataset`.
- The stacking is lazy for dask-backed arrays.

### 9.3 `reduce_dimension`

Expected behavior:

- Real dimension reduction reduces that dimension per variable.
- Virtual `bands` reduction combines variables cross-band.
- Result remains `xr.Dataset`.
- Metadata remains deterministic and meaningful.
- Remaining dimensions follow canonical order.

Virtual bands behavior:

- Variables are stacked into a temporary DataArray via `to_array(dim="bands")` and reduced with `DataArray.reduce`.
- Per-variable attributes and variable order are captured before stacking and restored after unstacking.
- If process semantics require a scalar-like raster result, wrap it in a deterministic Dataset variable name defined by the process behavior.

## 10. Dataset-Native Implementation Patterns

Prefer xarray Dataset-native operations:

```python
result = data.where(condition, replacement)
result = data.map(lambda variable: process_variable(variable))
result = xr.apply_ufunc(func, data, dask="allowed")
result = data.reduce(func, dim=dimension)
```

Use explicit per-variable loops when required:

```python
result_vars = {}
for name, variable in data.data_vars.items():
    result_vars[name] = process_variable(variable)

result = xr.Dataset(result_vars, coords=data.coords, attrs=data.attrs)
```

After operations that may reorder dimensions, restore canonical real dimension order:

```text
known dimensions first: temporal, y, x
extra dimensions after known dimensions
```

Dataset-to-DataArray shortcut patterns to avoid:

```python
data.values
data.to_numpy()
data.compute()
old_dataarray_process(...)
```

`to_array(dim="bands")` and `to_dataset(dim="bands")` are permitted in virtual bands paths and compatibility adapters, provided per-variable attributes are preserved via `_capture_var_metadata`/`_restore_var_metadata`.

`.compute()` must not be called on raster payloads in process hot paths. It was removed from `predict_random_forest` in Phase 2.

## 11. Later Process Profiles

After L1 is complete and merged into `dev_remodel`, proceed in profile order:

1. L2 Recommended
2. L2A Recommended Raster
3. L2B Recommended Vector
4. L2-Date
5. L2-Text
6. L3 Advanced
7. L3-ML
8. L3-UDF
9. L3-Clim
10. L3-ARD
11. L4

Each profile phase must:

- Identify processes in that profile.
- Determine which processes touch RasterCube behavior.
- Migrate only relevant behavior for that profile.
- Preserve logical order `(t, bands, y, x)`.
- Preserve Dataset variable order `(t, y, x)`.
- Add comprehensive Dataset tests.
- Pass static shortcut audit before PR review.

Do not begin the next profile until the previous one is reviewed and merged into `dev_remodel`.

## 12. Test Plan

Add reusable Dataset fixtures for:

- Single-band Dataset.
- Multi-band Dataset.
- NumPy-backed Dataset.
- Dask-backed Dataset.
- Dataset with dimensions `(t, y, x)`.
- Dataset with dimensions `(time, y, x)`.
- Dataset without temporal dimension `(y, x)`.
- Dataset with attrs and per-variable attrs.
- Dataset with CRS/geobox metadata where supported.

For every migrated raster process, test:

- Accepts `xr.Dataset`.
- Rejects `xr.DataArray`.
- Preserves variable names.
- Preserves coordinates.
- Preserves attrs where expected.
- Preserves CRS/geobox where expected.
- Preserves dask laziness.
- Handles nodata and NaN values.
- Handles single-variable Dataset.
- Handles multi-variable Dataset.
- Preserves real variable dimension order `(t, y, x)`.
- Preserves logical band order `list(data.data_vars)`.

L1-specific tests:

- `apply` applies an element-wise process to all variables.
- `apply_dimension` over temporal dimension preserves spatial coordinates.
- `apply_dimension(..., dimension="bands")` supports native cross-band callbacks.
- `reduce_dimension` over temporal dimension returns Dataset.
- `reduce_dimension(..., dimension="bands")` combines variables natively.
- L1 reducer callback chains work on NumPy-backed and dask-backed Datasets.
- Dimension order is correct after each operation.

Static audit searches for migrated raster modules:

```text
.to_array(
.to_dataset(
xr.DataArray
.values
.to_numpy()
.compute()
```

Each remaining occurrence must be outside migrated raster paths or explicitly justified.

## 13. Optimized Code Guidance

Implementation should be smart, local, and optimized:

- Prefer minimal local edits over rewrites.
- Prefer xarray alignment and broadcasting over manual shape handling.
- Prefer Dataset-native arithmetic and `xr.apply_ufunc`.
- Prefer preserving DataArrays directly when constructing Datasets.
- Add helpers only for repeated correctness rules.
- Avoid clever generic dispatch when explicit xarray code is clearer.
- Do not materialize dask-backed raster data.
- Keep errors explicit and process-specific.
- Keep each PR focused on modernization or one process-profile migration.

Useful helper categories:

- RasterCube validation.
- Virtual band handling.
- Canonical dimension ordering.
- Dataset metadata preservation.
- Dataset test assertions.

## 14. Acceptance Criteria

A phase is complete only when:

- All existing tests pass.
- New profile-specific tests pass.
- Static shortcut audit passes.
- Migrated RasterCube APIs are Dataset-native.
- No hidden Dataset-to-DataArray fallback exists without metadata preservation.
- Dask-backed tests prove laziness is preserved.
- Multi-variable Dataset tests prove no variable is dropped.
- Canonical order is preserved:
  - logical openEO order: `(t, bands, y, x)`
  - Dataset variable order: `(t, y, x)`
- The code still resembles `main` and avoids unnecessary deletion.

## 15. Final Enforcement Plan

The following phases complete the migration by switching `RasterCube` from `Union[xr.DataArray, xr.Dataset]` to strict `xr.Dataset`.

### Status Overview

| Phase | Description | Status |
|---|---|---|---|
| A | Change `RasterCube = xr.Dataset` type alias | ✅ Done |
| B | Enable `ensure_raster_cube` to reject DataArray | ✅ Done |
| C | Fix downstream call chains | ✅ Done (no-op) |
| D | Make Dataset the test default | ✅ Done |

### Phase A — Change the type alias (✅ Done)

`RasterCube = xr.Dataset` set in `data_model.py`. No runtime impact — Python type aliases are hints, not enforced. Type checkers (mypy, pyright) will now flag DataArray usage in annotated code.

All 301 tests pass with this change alone.

### Phase B — Enable enforcement process-by-process (✅ Done → removed in Phase 0)

`ensure_raster_cube` was added to raise `TypeError` for `xr.DataArray` inputs, then removed in Phase 0 of the review-fix cycle. Under the Dataset-only assumption (all inputs are `xr.Dataset`), the runtime guard is unnecessary ceremony and was deleted. The function and its calls were removed from `utils.py`, `apply.py`, and `reduce.py`. Virtual bands paths in `apply_dimension` and `reduce_dimension` were fixed during Phase 3 to preserve per-variable attributes.

### Phase C — Fix the call chain (✅ Done — no-op)

No processes call L1 processes (`reduce_dimension`, `apply`, `apply_dimension`) internally. Removed dead imports of these functions from `ddmc.py` and `curve_fitting.py`.

### Phase D — Make Dataset the test default (✅ Done)

1. Changed `create_fake_rastercube` default `as_dataset` to `True`.
2. Migrated all remaining processes for Dataset compatibility: `merge_cubes`, `filter_labels`, `ddmc`, `fit_curve`/`predict_curve`, `run_udf`, `resample_cube_temporal`, `aggregate_temporal`, `trim_cube`.
3. Updated all test files to work with Dataset as the default cube type.

### Acceptance Gates

- `RasterCube = xr.Dataset` with no remaining Union.
- All 300+ tests pass with Dataset test data by default.
- Static audit finds no hidden DataArray fallback in migrated raster paths without metadata preservation.
- `predict_random_forest` does not call `.compute()` on raster payloads.
- `merge_cubes` uses a native Dataset merge path that preserves variable order, per-variable attrs, and CRS.

## 16. Explicit Assumptions

- `main` remains the style and structure baseline.
- `dev_remodel` is the long-lived integration branch.
- `refactor_remodel/*` branches are checkpoint PR branches.
- Bands are Dataset data variables, not a real xarray dimension.
- Canonical dimension order is mandatory unless a process explicitly removes or creates dimensions.
- Native Dataset support means real Dataset-native logic, not wrapper conversion around old DataArray logic.
