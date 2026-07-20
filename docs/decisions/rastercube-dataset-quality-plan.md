# RasterCube Dataset Migration Quality Plan

This document is an implementation plan for improving the quality and correctness of the `RasterCube: xr.Dataset` migration. It is written for a code generator or implementation agent to follow in small, verifiable phases.

The goal is to make `xr.Dataset` the real raster process contract, not only the type alias, while preserving dask laziness, per-variable metadata, CRS, and multi-resolution behavior.

## Current Baseline

- `RasterCube = xr.Dataset` in `openeo_processes_dedl_slim/process_implementations/data_model.py`.
- Test fixtures default to Dataset through `create_fake_rastercube(..., as_dataset=True)`.
- Many raster process implementations already support Dataset inputs.
- Some operations still use bounded `Dataset -> DataArray -> Dataset` bridges for virtual band logic.
- Some legacy `xr.DataArray` fallback paths remain and are not covered by the default test path.
- Full `tests/test_ml.py` is unstable when multiple xgboost/dask training tests run in one pytest process, although affected tests pass individually.

## Principles

1. Public raster process inputs and outputs must be `xr.Dataset`.
2. Use Dataset-native xarray APIs where possible.
3. Use `DataArray` bridges only when a process truly needs a virtual band axis.
4. Every bridge must preserve data variable order, attrs, CRS, coordinates, and dask laziness.
5. Multi-resolution behavior must be explicit: either supported, harmonized, or rejected with a clear error.
6. Tests must assert Dataset shape and semantic correctness, not just absence of exceptions.

## Phase 0: Add Migration Inventory and Guards

### Objective

Create a clear inventory of process implementation status and make accidental RasterCube `DataArray` use visible.

### Tasks

1. Add a markdown inventory table, for example `docs/decisions/rastercube-process-status.md`.
2. For every process in `openeo_processes_dedl_slim/process_implementations/cubes`, classify it as:
   - `dataset-native`
   - `bounded-bridge`
   - `legacy-fallback`
   - `non-raster`
3. Include columns for:
   - process name
   - implementation file
   - Dataset input support
   - Dataset output support
   - uses `to_array`
   - preserves dask laziness
   - multi-resolution status
   - test file
4. Add or update a small static check script that reports:
   - `RasterCube = Union[...]`
   - `isinstance(..., xr.DataArray)` in raster process files
   - `to_array(` bridge sites
   - direct `.data` access inside raster process implementations

### Acceptance Criteria

- The inventory exists and covers all raster processes.
- The static check can be run locally.
- The static check is informational in this phase and does not fail CI yet.

### Suggested Verification

```bash
/home/sdhinakaran/micromamba/envs/openeo-dataset-py312/bin/python scripts/check_rastercube_migration.py
```

## Phase 1: Enforce Dataset at Public Raster Boundaries

### Objective

Remove ambiguity at public raster process boundaries.

### Tasks

1. Decide whether `ensure_raster_cube` should be used or removed.
2. If kept, call it at the top of public raster processes that accept `RasterCube`.
3. Do not call it inside helper paths that intentionally receive `xr.DataArray` after a bounded bridge.
4. Remove or isolate legacy `DataArray` public fallback paths in:
   - `cubes/mask.py`
   - `cubes/_filter.py`
   - `cubes/apply.py`
   - `cubes/reduce.py`
   - `cubes/general.py`
   - `cubes/indices.py`
5. For functions that are array utilities rather than RasterCube processes, keep `xr.DataArray` support if it is part of their real contract, but document that they are not RasterCube boundaries.
6. Add tests that passing a RasterCube-shaped `xr.DataArray` to public raster processes fails clearly.

### Acceptance Criteria

- Public RasterCube processes consistently accept `xr.Dataset`.
- Public RasterCube processes reject RasterCube-shaped `xr.DataArray` with a clear error.
- Array utility functions remain unaffected unless they are exported as raster processes.

### Suggested Tests

```bash
/home/sdhinakaran/micromamba/envs/openeo-dataset-py312/bin/python -m pytest \
  tests/test_apply.py \
  tests/test_reduce.py \
  tests/test_filter.py \
  tests/test_dimensions.py \
  tests/test_mask.py \
  -q
```

## Phase 2: Centralize Dataset/DataArray Bridge Logic

### Objective

Make all virtual-band bridges explicit, reusable, and tested.

### Tasks

1. Create a dedicated module, for example:
   - `openeo_processes_dedl_slim/process_implementations/cubes/dataset_bridge.py`
2. Move bridge helpers into that module:
   - `_capture_var_metadata`
   - `_restore_var_metadata`
   - `_detect_band_permutation`
3. Replace private helper names with clearer public/internal names:
   - `capture_dataset_metadata`
   - `restore_dataset_metadata`
   - `dataset_to_virtual_bands`
   - `virtual_bands_to_dataset`
   - `detect_band_permutation`
4. `dataset_to_virtual_bands(dataset, dim="bands")` must:
   - preserve variable order
   - preserve per-variable attrs
   - preserve Dataset attrs
   - preserve CRS
   - return bridge metadata required for restoration
5. `virtual_bands_to_dataset(array, bridge_metadata, dim="bands")` must:
   - restore variable order where possible
   - restore per-variable attrs where matching variables exist
   - restore Dataset attrs
   - restore CRS where possible
   - preserve dask arrays without computing
6. Replace direct `data.to_array(dim="bands")` calls in:
   - `cubes/apply.py`
   - `cubes/reduce.py`
   - `ml/curve_fitting.py`
   - `udf/udf.py`
7. Add dedicated bridge tests in `tests/test_dataset_bridge.py`.

### Acceptance Criteria

- All RasterCube bridge sites go through the central bridge module.
- Bridge tests cover:
   - variable order
   - per-variable attrs
   - Dataset attrs
   - CRS
   - dask laziness
   - heterogeneous dtype behavior
   - missing or extra result bands

### Suggested Tests

```bash
/home/sdhinakaran/micromamba/envs/openeo-dataset-py312/bin/python -m pytest \
  tests/test_dataset_bridge.py \
  tests/test_apply.py \
  tests/test_reduce.py \
  tests/test_ml.py::test_curve_fitting \
  tests/test_udf.py \
  -q
```

## Phase 3: Add Multi-Resolution Dataset Fixtures

### Objective

Add test pressure for the main reason to use Dataset: variables can have different grids, dimensions, dtypes, and metadata.

### Tasks

1. Add a fixture builder, for example:
   - `create_multiresolution_rastercube`
2. It should produce an `xr.Dataset` with:
   - at least two variables with different spatial resolutions
   - shared temporal dimension
   - different dtypes, for example `int16` and `float32`
   - per-variable attrs
   - Dataset attrs
   - CRS assigned through `odc.geo.xr.assign_crs`
   - dask-backed option
3. Add tests for processes that should work without harmonizing variables:
   - `filter_bands`
   - `dimension_labels`
   - `rename_labels`
   - `mask` with per-variable same-grid masks
   - `merge_cubes` with non-overlapping variables
4. Add tests for processes that require harmonization or virtual band stacking:
   - `apply_dimension(dimension="bands")`
   - `reduce_dimension(dimension="bands")`
   - `run_udf`
   - `fit_curve`
5. For each virtual-band process, choose and document behavior:
   - fail with clear error on multi-resolution input
   - harmonize first
   - allow xarray alignment and document consequences

### Acceptance Criteria

- Multi-resolution fixtures exist and are reused across process tests.
- Process behavior on multi-resolution Dataset input is explicit.
- No virtual-band process silently broadcasts/resamples multi-resolution variables without a test.

### Suggested Tests

```bash
/home/sdhinakaran/micromamba/envs/openeo-dataset-py312/bin/python -m pytest \
  tests/test_multiresolution.py \
  tests/test_filter.py \
  tests/test_dimensions.py \
  tests/test_merge.py \
  tests/test_mask.py \
  -q
```

## Phase 4: Refactor `merge_cubes` Dataset Logic

### Objective

Make `merge_cubes` robust for Dataset-first and multi-resolution workflows.

### Tasks

1. Split `merge_cubes` into explicit helpers:
   - `merge_dataset_cubes`
   - `merge_dataarray_cubes`
   - `merge_dataset_variable_conflict`
   - `align_close_coordinates`
2. Keep the public `merge_cubes` function small:
   - validate same public type
   - dispatch to Dataset or helper path
3. For Dataset input:
   - preserve variable order
   - preserve per-variable attrs
   - preserve Dataset attrs
   - preserve CRS
   - preserve dask arrays
   - avoid whole-Dataset `to_array`
4. For same-name variable conflicts:
   - define whether each variable is merged with DataArray logic or Dataset-aware logic
   - make that recursion explicit in a named helper
   - add tests for one conflicting variable and one non-conflicting variable in the same merge
5. For multi-resolution input:
   - merging non-overlapping variables should preserve each variable's own grid
   - merging same-name variables on different grids should fail, align, or resample according to a documented rule
6. Add tests for:
   - no overlapping variable names
   - overlapping variable names with equal coordinates
   - overlapping variable names with close float coordinates
   - overlapping variable names with different spatial resolution
   - CRS preservation
   - attrs preservation
   - dask laziness

### Acceptance Criteria

- Dataset merge behavior is documented by helper names and tests.
- Multi-resolution variable-preserving merge is supported for non-overlapping variables.
- Same-name conflict behavior is deterministic and tested.

### Suggested Tests

```bash
/home/sdhinakaran/micromamba/envs/openeo-dataset-py312/bin/python -m pytest \
  tests/test_merge.py \
  tests/test_multiresolution.py \
  -q
```

## Phase 5: Harden `predict_random_forest`

### Objective

Prevent silent wrong predictions due to Dataset variable order mismatch.

### Tasks

1. Change `predict_random_forest` Dataset path so that:
   - if `model.feature_names` is present, `set(data.data_vars)` must exactly match it
   - output feature order must be `model.feature_names`
   - mismatch raises a clear exception
2. If feature names are missing from the model:
   - require explicit `context["feature_order"]`, or
   - fall back to Dataset order only with a warning and test
3. Consider adding an explicit parameter or context key:
   - `feature_order`
   - `feature_mapping`
4. Add tests for:
   - matching names in different order
   - missing variable
   - extra variable
   - no feature names
   - dask-backed Dataset prediction remains lazy

### Acceptance Criteria

- `predict_random_forest` never silently predicts with mismatched features.
- Dataset variable order does not affect predictions when names match.
- Mismatch errors are clear and actionable.

### Suggested Tests

```bash
/home/sdhinakaran/micromamba/envs/openeo-dataset-py312/bin/python -m pytest \
  tests/test_ml.py::test_predict_random_forest_dask \
  tests/test_ml.py::test_predict_random_forest_feature_order \
  -q
```

## Phase 6: Stabilize xgboost/dask ML Tests

### Objective

Make ML tests reliable and separate Dataset migration tests from xgboost integration instability.

### Tasks

1. Replace the generic `dask_client` fixture for xgboost tests with a smaller deterministic fixture:
   - one or two workers
   - one thread per worker
   - explicit dashboard disabled
   - explicit close/shutdown with timeout
2. Avoid running multiple xgboost training jobs against reused worker state.
3. Split tests into:
   - fast Dataset shape/unit tests
   - xgboost integration tests
4. For Dataset prediction unit tests, use a minimal fake model object with `feature_names` where possible.
5. Keep only one or two true `xgboost.dask.train` integration tests.
6. Add marks:
   - `@pytest.mark.integration`
   - `@pytest.mark.xgboost`
7. Update CI to run unit tests always and integration tests deliberately.

### Acceptance Criteria

- `tests/test_ml.py` passes as a module.
- Random-forest Dataset prediction tests do not require training a real xgboost model unless the test is explicitly integration-scoped.
- No test hangs under normal local execution.

### Suggested Tests

```bash
timeout 240 /home/sdhinakaran/micromamba/envs/openeo-dataset-py312/bin/python -m pytest tests/test_ml.py -q
```

## Phase 7: Reduce Per-Variable Graph Overhead

### Objective

Improve performance for Dataset processes that iterate over many variables.

### Tasks

1. Benchmark current per-variable implementations for:
   - `apply_kernel`
   - `mask`
   - selected real-dimension reductions
2. Add a benchmark or lightweight graph-size test:
   - number of variables
   - number of dask tasks
   - no eager compute
3. For `mask`, prefer Dataset-native `where` when mask structure allows it.
4. For `apply_kernel`, evaluate:
   - per-variable apply remains safest for heterogeneous grids
   - grouped execution for variables with identical grid/dtype/chunks
5. Do not sacrifice correctness for graph compactness.

### Acceptance Criteria

- Existing behavior is preserved.
- Dask graph size is measured for wide-band Dataset cases.
- Any optimization has before/after tests or benchmarks.

## Phase 8: Turn Static Checks Into CI Guardrails

### Objective

Prevent regressions back to implicit DataArray RasterCube behavior.

### Tasks

1. Promote the Phase 0 static check to CI failure for selected rules:
   - `RasterCube` must remain `xr.Dataset`
   - no new public raster `xr.DataArray` fallback paths
   - bridge calls must go through `dataset_bridge.py`
2. Allow explicit exceptions through a small allowlist with comments.
3. Add the check to the test workflow.

### Acceptance Criteria

- New DataArray bridge sites cannot be added silently.
- Exceptions are explicit and reviewed.

## Suggested Execution Order

1. Phase 0: inventory and static report.
2. Phase 2: central bridge module and bridge tests.
3. Phase 3: multi-resolution fixtures.
4. Phase 4: `merge_cubes` refactor.
5. Phase 5: `predict_random_forest` hardening.
6. Phase 6: ML test stability.
7. Phase 1: remove or enforce legacy public fallback paths after tests are strong.
8. Phase 7 and Phase 8: performance and CI hardening.

Phase 1 is intentionally not first. Removing fallback paths before bridge and multi-resolution tests exist can hide behavior changes. Build test pressure first, then remove dead or unsupported behavior.

## Completion Definition

The migration quality work is complete when:

- Public raster processes consistently accept and return `xr.Dataset`.
- All required `DataArray` bridges are centralized, tested, and documented.
- Multi-resolution Dataset behavior is covered by fixtures and process tests.
- `merge_cubes` has deterministic Dataset-first behavior for multi-resolution cases.
- `predict_random_forest` cannot silently misorder features.
- `tests/test_ml.py` is stable as a module or integration tests are isolated and marked.
- Static CI checks prevent accidental reintroduction of unreviewed RasterCube `DataArray` assumptions.
