import numpy as np
import odc.geo.xr
import pandas as pd
import pytest
import xarray as xr

from openeo_processes_dedl_slim.process_implementations.cubes.dataset_bridge import (
    capture_dataset_metadata,
    dataset_to_virtual_bands,
    detect_band_permutation,
    restore_dataset_metadata,
    virtual_bands_to_dataset,
)


@pytest.fixture
def band_dataset():
    rng = np.random.default_rng(42)
    x_coords = np.arange(10.45, 10.5, 0.01)
    y_coords = np.arange(46.1, 46.2, 0.02)
    t_coords = pd.date_range("2018-05-01", "2018-06-01", periods=4).values
    bands = ["B02", "B03", "B04", "B08"]
    data = rng.integers(
        -100, 100, size=(len(y_coords), len(x_coords), 4, len(bands))
    ).astype(np.float32)
    ds = xr.Dataset(
        {
            band: xr.DataArray(
                data[:, :, :, i],
                dims=["y", "x", "t"],
                coords={"y": y_coords, "x": x_coords, "t": t_coords},
            )
            for i, band in enumerate(bands)
        },
        attrs={"crs": "EPSG:4326"},
    )
    return odc.geo.xr.assign_crs(ds, crs="EPSG:4326")


class TestCaptureDatasetMetadata:
    def test_captures_variable_order(self, band_dataset):
        meta = capture_dataset_metadata(band_dataset)
        assert meta["order"] == ["B02", "B03", "B04", "B08"]

    def test_captures_per_variable_attrs(self, band_dataset):
        band_dataset["B02"].attrs["common_name"] = "blue"
        band_dataset["B03"].attrs["common_name"] = "green"
        meta = capture_dataset_metadata(band_dataset)
        assert meta["attrs"]["B02"]["common_name"] == "blue"
        assert meta["attrs"]["B03"]["common_name"] == "green"

    def test_captures_dataset_attrs(self, band_dataset):
        band_dataset.attrs["title"] = "test"
        meta = capture_dataset_metadata(band_dataset)
        assert meta["dataset_attrs"]["title"] == "test"

    def test_captures_crs(self, band_dataset):
        meta = capture_dataset_metadata(band_dataset)
        assert meta["crs"] is not None
        assert "EPSG" in str(meta["crs"]).upper()


class TestRestoreDatasetMetadata:
    def test_restores_variable_order(self, band_dataset):
        meta = capture_dataset_metadata(band_dataset)
        rearranged = band_dataset[["B08", "B04", "B03", "B02"]]
        restored = restore_dataset_metadata(rearranged, meta)
        assert list(restored.data_vars) == ["B02", "B03", "B04", "B08"]

    def test_restores_per_variable_attrs(self, band_dataset):
        band_dataset["B02"].attrs["common_name"] = "blue"
        meta = capture_dataset_metadata(band_dataset)
        cleared = band_dataset.copy()
        for v in cleared.data_vars:
            cleared[v].attrs = {}
        restored = restore_dataset_metadata(cleared, meta)
        assert restored["B02"].attrs["common_name"] == "blue"

    def test_ignores_missing_variables(self, band_dataset):
        meta = capture_dataset_metadata(band_dataset)
        subset = band_dataset[["B02", "B03"]]
        restored = restore_dataset_metadata(subset, meta)
        assert list(restored.data_vars) == ["B02", "B03"]


class TestDatasetToVirtualBands:
    def test_returns_dataarray_and_metadata(self, band_dataset):
        array, meta = dataset_to_virtual_bands(band_dataset, dim="bands")
        assert isinstance(array, xr.DataArray)
        assert "bands" in array.dims
        assert meta["order"] == ["B02", "B03", "B04", "B08"]

    def test_preserves_variable_order(self, band_dataset):
        array, meta = dataset_to_virtual_bands(band_dataset, dim="bands")
        assert list(array.coords["bands"].values) == ["B02", "B03", "B04", "B08"]

    def test_preserves_dataset_attrs(self, band_dataset):
        band_dataset.attrs["title"] = "test"
        array, meta = dataset_to_virtual_bands(band_dataset, dim="bands")
        assert meta["dataset_attrs"]["title"] == "test"

    def test_preserves_crs(self, band_dataset):
        _, meta = dataset_to_virtual_bands(band_dataset, dim="bands")
        assert meta["crs"] is not None

    @pytest.mark.parametrize("backend", ["numpy", "dask"])
    def test_preserves_dask_laziness(self, band_dataset, backend):
        if backend == "dask":
            pytest.importorskip("dask")
            band_dataset = band_dataset.chunk({"y": 3, "x": 3, "t": 2})
        array, meta = dataset_to_virtual_bands(band_dataset, dim="bands")
        if backend == "dask":
            assert hasattr(array.data, "dask") or hasattr(array.data, "compute")
        else:
            assert isinstance(array.data, np.ndarray)


class TestVirtualBandsToDataset:
    def test_round_trip_preserves_structure(self, band_dataset):
        array, meta = dataset_to_virtual_bands(band_dataset, dim="bands")
        result = virtual_bands_to_dataset(array, meta, dim="bands")
        assert list(result.data_vars) == list(band_dataset.data_vars)
        for v in band_dataset.data_vars:
            assert v in result.data_vars

    def test_round_trip_preserves_attrs(self, band_dataset):
        band_dataset["B02"].attrs["common_name"] = "blue"
        band_dataset.attrs["title"] = "test"
        array, meta = dataset_to_virtual_bands(band_dataset, dim="bands")
        result = virtual_bands_to_dataset(array, meta, dim="bands")
        assert result["B02"].attrs.get("common_name") == "blue"
        assert result.attrs.get("title") == "test"

    def test_round_trip_preserves_crs(self, band_dataset):
        array, meta = dataset_to_virtual_bands(band_dataset, dim="bands")
        result = virtual_bands_to_dataset(array, meta, dim="bands")
        assert result.odc.crs == band_dataset.odc.crs

    @pytest.mark.parametrize("backend", ["numpy", "dask"])
    def test_round_trip_preserves_dask(self, band_dataset, backend):
        if backend == "dask":
            pytest.importorskip("dask")
            band_dataset = band_dataset.chunk({"y": 3, "x": 3, "t": 2})
        array, meta = dataset_to_virtual_bands(band_dataset, dim="bands")
        result = virtual_bands_to_dataset(array, meta, dim="bands")
        if backend == "dask":
            for v in result.data_vars:
                assert hasattr(result[v].data, "dask") or hasattr(
                    result[v].data, "compute"
                )

    def test_restores_variable_order_from_permuted(self, band_dataset):
        array, meta = dataset_to_virtual_bands(band_dataset, dim="bands")
        permuted = array.sel(bands=["B08", "B04", "B03", "B02"])
        result = virtual_bands_to_dataset(permuted, meta, dim="bands")
        assert list(result.data_vars) == ["B02", "B03", "B04", "B08"]

    def test_heterogeneous_dtype_round_trip(self, band_dataset):
        band_dataset["B02"] = band_dataset["B02"].astype(np.int16)
        band_dataset["B08"] = band_dataset["B08"].astype(np.float32)
        array, meta = dataset_to_virtual_bands(band_dataset, dim="bands")
        result = virtual_bands_to_dataset(array, meta, dim="bands")
        assert list(result.data_vars) == ["B02", "B03", "B04", "B08"]
        assert result["B02"].dtype == result["B08"].dtype


class TestDetectBandPermutation:
    def test_no_permutation(self):
        da = xr.DataArray(
            np.arange(12).reshape(3, 4),
            dims=["bands", "x"],
            coords={"bands": ["a", "b", "c"]},
        )
        result = da.copy()
        perm = detect_band_permutation(result, da, "bands")
        assert perm == ["a", "b", "c"]

    def test_reversed_bands(self):
        da = xr.DataArray(
            np.arange(12).reshape(3, 4),
            dims=["bands", "x"],
            coords={"bands": ["a", "b", "c"]},
        )
        result = da.isel(bands=[2, 1, 0])
        perm = detect_band_permutation(result, da, "bands")
        assert perm == ["c", "b", "a"]

    def test_returns_none_for_size_mismatch(self):
        da = xr.DataArray(
            np.arange(12).reshape(3, 4),
            dims=["bands", "x"],
            coords={"bands": ["a", "b", "c"]},
        )
        result = da.isel(bands=[0, 1])
        perm = detect_band_permutation(result, da, "bands")
        assert perm is None

    def test_returns_none_for_count_mismatch(self):
        da = xr.DataArray(
            np.arange(12).reshape(3, 4),
            dims=["bands", "x"],
            coords={"bands": ["a", "b", "c"]},
        )
        result = xr.DataArray(
            np.arange(8).reshape(2, 4),
            dims=["bands", "x"],
            coords={"bands": ["a", "b"]},
        )
        perm = detect_band_permutation(result, da, "bands")
        assert perm is None
