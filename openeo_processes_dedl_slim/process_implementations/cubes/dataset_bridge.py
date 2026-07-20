try:
    import dask
except ImportError:
    dask = None

import numpy as np
import xarray as xr


def capture_dataset_metadata(dataset):
    crs = None
    try:
        crs = dataset.odc.crs
    except Exception:
        pass
    return {
        "attrs": {v: dataset[v].attrs for v in dataset.data_vars},
        "order": list(dataset.data_vars),
        "dataset_attrs": dataset.attrs,
        "crs": crs,
    }


def restore_dataset_metadata(result, metadata):
    for v, attrs in metadata["attrs"].items():
        if v in result.data_vars:
            result[v].attrs = attrs
    if metadata.get("dataset_attrs"):
        result.attrs = metadata["dataset_attrs"]
    try:
        result = result[metadata["order"]]
    except KeyError:
        pass
    crs = metadata.get("crs")
    if crs is not None:
        import odc.geo.xr

        try:
            result = odc.geo.xr.assign_crs(result, crs=crs)
        except (ValueError, ImportError):
            pass
    return result


def dataset_to_virtual_bands(dataset, dim="bands"):
    meta = capture_dataset_metadata(dataset)
    return dataset.to_array(dim=dim), meta


def virtual_bands_to_dataset(array, metadata, dim="bands"):
    result = array.to_dataset(dim=dim)
    return restore_dataset_metadata(result, metadata)


def detect_band_permutation(result_da, input_da, dimension):
    n_out = len(result_da[dimension])
    n_in = len(input_da[dimension])
    if n_out != n_in or n_out == 0:
        return None

    idx = {d: 0 for d in result_da.dims if d != dimension}
    if not idx:
        out_data = result_da.values
        in_data = input_da.values
    else:
        out_data = result_da.isel(idx).values
        in_data = input_da.isel(idx).values

    out_data = out_data.ravel()
    in_data = in_data.ravel()

    if len(out_data) != len(in_data):
        return None

    permutation = [-1] * n_out
    used = [False] * n_in

    for out_pos in range(n_out):
        for in_pos in range(n_in):
            if used[in_pos]:
                continue
            match = np.allclose(
                out_data[out_pos], in_data[in_pos], atol=0, rtol=0, equal_nan=True
            )
            if match:
                permutation[out_pos] = in_pos
                used[in_pos] = True
                break

    if -1 in permutation or not all(used):
        return None

    original_labels = list(input_da[dimension].values)
    return [original_labels[p] for p in permutation]


# Backward-compat aliases (used by utils.py re-exports)
_capture_var_metadata = capture_dataset_metadata
_restore_var_metadata = restore_dataset_metadata
_detect_band_permutation = detect_band_permutation
