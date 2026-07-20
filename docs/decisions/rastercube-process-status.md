# RasterCube Migration Process Inventory

This inventory classifies every process in
`openeo_processes_dedl_slim/process_implementations/cubes` for the Dataset
migration. See `rastercube-dataset-quality-plan.md` for the full plan.

Status categories:

- **dataset-native**: Process naturally works with `xr.Dataset`; no
  `DataArray` bridge needed.
- **bounded-bridge**: Process uses `to_array` + `to_dataset` to get a virtual
  band axis, then restores metadata.
- **legacy-fallback**: Process has a `xr.DataArray` code path that is not the
  primary path.
- **non-raster**: Process operates on `VectorCube` or is not a RasterCube
  process.

| Process | File | Dataset Input | Dataset Output | `to_array` | Dask Lazy | Multi-Res Status | Test File | Classification |
|---|---|---|---|---|---|---|---|---|
| `filter_temporal` | `_filter.py` | Yes | Yes | No | Yes | Supported | `tests/test_filter.py` | dataset-native |
| `filter_labels` | `_filter.py` | Yes | Yes | No | Yes | Supported | `tests/test_filter.py` | dataset-native |
| `filter_bands` | `_filter.py` | Yes | Yes | No | Yes | Supported | `tests/test_filter.py` | dataset-native |
| `filter_bbox` | `_filter.py` | Yes | Yes | No | Yes | Supported | `tests/test_filter.py` | dataset-native |
| `apply` | `apply.py` | Yes | Yes | No | Yes | Supported | `tests/test_apply.py` | dataset-native |
| `apply_dimension` | `apply.py` | Yes | Yes | Yes (bands) | Yes | Fails silently on multi-res | `tests/test_apply.py` | bounded-bridge |
| `apply_kernel` | `apply.py` | Yes | Yes | No | Yes | Per-variable iteration | `tests/test_apply.py` | dataset-native |
| `apply_neighborhood_intertwin` | `apply_neighborhood_intertwin.py` | Yes | Yes | No | Yes | Unknown | — | dataset-native |
| `reduce_dimension` | `reduce.py` | Yes | Yes | Yes (bands) | Yes | Fails silently on multi-res | `tests/test_reduce.py` | bounded-bridge |
| `reduce_spatial` | `reduce.py` | Yes | Yes | No | Yes | Supported | `tests/test_reduce.py` | dataset-native |
| `aggregate_temporal` | `aggregate.py` | Yes | Yes | No | Yes | Supported | `tests/test_aggregate.py` | dataset-native |
| `aggregate_temporal_period` | `aggregate.py` | Yes | Yes | No | Yes | Supported | `tests/test_aggregate.py` | dataset-native |
| `mask` | `mask.py` | Yes (via `_mask_dataset`) | Yes | No | Yes | `_mask_dataset` explicit per-var | `tests/test_mask.py` | legacy-fallback |
| `merge_cubes` | `merge.py` | Yes | Yes | Yes (line 229, DataArray path) | Yes | Partial via Dataset path | `tests/test_merge.py` | bounded-bridge |
| `ndvi` | `indices.py` | Yes (preferred) | Yes | No | Yes | N/A (band math) | `tests/test_indices.py` | legacy-fallback |
| `drop_dimension` | `general.py` | Yes | Yes | No | Yes | Supported | `tests/test_dimensions.py` | dataset-native |
| `create_data_cube` | `general.py` | N/A | Yes | No | N/A | N/A | — | dataset-native |
| `trim_cube` | `general.py` | Yes | Yes | Yes (line 54, NaN mask) | Yes | Supported (per-var NaN) | — | bounded-bridge |
| `dimension_labels` | `general.py` | Yes | N/A (returns array) | No | N/A | Supported | `tests/test_dimensions.py` | dataset-native |
| `add_dimension` | `general.py` | Yes | Yes | No | Yes | Supported | `tests/test_dimensions.py` | dataset-native |
| `rename_dimension` | `general.py` | Yes | Yes | No | Yes | Supported | `tests/test_dimensions.py` | dataset-native |
| `rename_labels` | `general.py` | Yes | Yes | No | Yes | Supported | `tests/test_dimensions.py` | dataset-native |
| `resample_spatial` | `resample.py` | Yes | Yes | No | Yes | Supported (per-var) | `tests/test_resample.py` | dataset-native |
| `resample_cube_spatial` | `resample.py` | Yes | Yes | No | Yes | Supported (per-var) | `tests/test_resample.py` | dataset-native |
| `resample_cube_temporal` | `resample.py` | Yes | Yes | No | Yes | Supported (per-var) | `tests/test_resample.py` | dataset-native |
| `load_geojson` | `geometries.py` | N/A (VectorCube) | No (DataArray) | No | N/A | N/A | — | non-raster |
| `vector_buffer` | `geometries.py` | N/A (VectorCube) | No (DataArray) | No | N/A | N/A | — | non-raster |
| `vector_reproject` | `geometries.py` | N/A (VectorCube) | No (DataArray) | No | N/A | N/A | — | non-raster |
| `load_vector_cube` | `experimental.py` | N/A (VectorCube) | No (dask_geopandas) | No | N/A | N/A | — | non-raster |

## Non-Cubes Processes (ML / UDF)

| Process | File | Dataset Input | Dataset Output | `to_array` | Classification |
|---|---|---|---|---|---|
| `fit_curve` | `ml/curve_fitting.py` | N/A (uses DataArray) | N/A | Unknown | bounded-bridge (likely) |
| `predict_random_forest` | `ml/random_forest.py` | Yes | Yes | No | dataset-native (needs hardening) |
