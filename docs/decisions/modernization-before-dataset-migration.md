# Modernization before Dataset migration

Date: 2026-05-23

## Context

The remodel plan defines `dev_remodel` as the integration branch for two related but different bodies of work:

1. Modernize the slim repository for newer runtimes and dependency versions.
2. Migrate RasterCube process implementations from legacy `xr.DataArray` behavior to native `xr.Dataset` behavior.

These changes were intentionally split. The Dataset migration already changes the most important public data model contract in the project. Mixing dependency, CI, NumPy, Pydantic, and XGBoost compatibility changes into the same checkpoint would make regressions difficult to isolate.

The plan document therefore made modernization Phase 1 and explicitly said not to change the RasterCube data model during that phase.

## Decision

Modernization is a prerequisite phase for the Dataset migration, not part of the data model change itself.

The modernization scope is limited to:

- Python and packaging metadata updates.
- CI and release workflow refreshes.
- Dependency range updates needed for current xarray, dask, NumPy, and related packages.
- NumPy 2 compatibility fixes, including replacing removed or private NumPy APIs.
- Pydantic API updates in tests and fixtures.
- XGBoost Dask import compatibility.
- Small laziness-preserving fixes in existing array helpers.

The modernization phase must not:

- Change `RasterCube` away from the existing model.
- Introduce Dataset-specific process behavior.
- Restructure process modules or rewrite implementations unrelated to compatibility.

## Implementation record

The relevant `dev_remodel` commits are:

| Commit | Purpose |
|---|---|
| `2fc230a` | Phase 1 modernization: dependency ranges, Python 3.13/3.14 metadata, Poetry and GitHub Actions refresh, NumPy 2 replacements, XGBoost Dask import update. |
| `93e72e6` | Pydantic `parse_obj` to `model_validate`, and replacement of private `np.core` exception handling. |
| `d880d90` | Merge of `refactor_remodel/01-modernization` into `dev_remodel`. |

## Consequences

### Positive

- The Dataset migration starts from a current dependency and CI baseline.
- NumPy 2 and newer Python compatibility problems are separated from Dataset behavior changes.
- Reviewers can reason about modernization regressions independently from data model regressions.
- The branch remains close to the original module layout while removing deprecated API usage.

### Negative

- Wider dependency ranges place more responsibility on CI to catch upstream compatibility changes.
- The modernization phase updates packaging and workflow files that are not directly related to RasterCube behavior.
- Some array helper behavior changed to preserve dask laziness, so existing tests remain the source of truth for intended edge-case behavior.
