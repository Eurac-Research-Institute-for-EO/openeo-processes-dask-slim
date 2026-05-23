try:
    import dask
except ImportError:
    dask = None

from xarray.core.duck_array_ops import isnull as xr_isnull


def _has_dask():
    return dask is not None


def _is_dask_array(arr):
    return _has_dask() and isinstance(arr, dask.array.Array)


def isnull(data):
    if _is_dask_array(data):
        return dask.array.map_blocks(xr_isnull, data)
    else:
        return xr_isnull(data)


def notnull(data):
    return ~isnull(data)


def _capture_var_metadata(dataset):
    """Capture per-variable attrs and variable order before to_array()."""
    return {
        "attrs": {v: dataset[v].attrs for v in dataset.data_vars},
        "order": list(dataset.data_vars),
    }


def _restore_var_metadata(result, metadata):
    """Restore per-variable attrs and variable order after to_dataset()."""
    for v, attrs in metadata["attrs"].items():
        if v in result.data_vars:
            result[v].attrs = attrs
    try:
        result = result[metadata["order"]]
    except KeyError:
        pass
    return result
