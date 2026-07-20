from functools import partial
from unittest.mock import patch

import dask
import dask.array as da
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
import xgboost as xgb
from openeo_pg_parser_networkx.pg_schema import DEFAULT_CRS, ParameterReference

from openeo_processes_dedl_slim.process_implementations.core import process
from openeo_processes_dedl_slim.process_implementations.cubes.apply import (
    apply_dimension,
)
from openeo_processes_dedl_slim.process_implementations.cubes.general import (
    dimension_labels,
)
from openeo_processes_dedl_slim.process_implementations.ml import (
    fit_curve,
    fit_regr_random_forest,
    predict_curve,
    predict_random_forest,
)
from tests.mockdata import create_fake_rastercube

# ── xgboost integration tests ─────────────────────────────────────────────


@pytest.mark.xgboost
def test_fit_regr_random_forest(vector_data_cube, xgboost_client):
    predictors_vars = ["value2"]
    target_var = "value1"

    model = fit_regr_random_forest(
        predictors=vector_data_cube,
        target=vector_data_cube,
        target_var=target_var,
        predictors_vars=predictors_vars,
    )

    assert isinstance(model, xgb.core.Booster)


@pytest.mark.xgboost
def test_fit_regr_random_forest_inline_geojson(
    vector_data_cube: gpd.GeoDataFrame, xgboost_client
):
    predictors_vars = ["value2"]
    target_var = "value1"

    model = fit_regr_random_forest(
        predictors=vector_data_cube,
        target=vector_data_cube.compute().to_json(),
        target_var=target_var,
        predictors_vars=predictors_vars,
    )

    assert isinstance(model, xgb.core.Booster)


@pytest.mark.xgboost
def test_predict_random_forest_dask(xgboost_client, vector_data_cube):
    predictors_vars = ["value2"]
    target_var = "value1"

    model = fit_regr_random_forest(
        predictors=vector_data_cube,
        target=vector_data_cube,
        target_var=target_var,
        predictors_vars=predictors_vars,
    )

    feature_names = model.feature_names
    n_features = len(feature_names)
    shape = (10, n_features)
    data = np.random.default_rng(42).random(shape).astype(np.float64)
    ds = xr.Dataset(
        {
            name: xr.DataArray(dask.array.from_array(data, chunks=-1), dims=["y", "x"])
            for name in feature_names
        }
    )

    result = predict_random_forest(data=ds, model=model)
    assert isinstance(result, xr.Dataset)
    assert "result" in result.data_vars
    assert isinstance(result["result"].data, dask.array.Array)


# ── prediction unit tests (mock-based, no training needed) ─────────────────


def _make_mock_model(feature_names=None):
    model = _make_mock_model
    model.feature_names = feature_names
    return model


@pytest.mark.xgboost
def test_predict_random_forest_feature_order_mismatch(xgboost_client, vector_data_cube):
    predictors_vars = ["value2"]
    target_var = "value1"
    model = fit_regr_random_forest(
        predictors=vector_data_cube,
        target=vector_data_cube,
        target_var=target_var,
        predictors_vars=predictors_vars,
    )

    feature_names = list(model.feature_names)
    n = len(feature_names)
    shape = (10, n)
    data = np.random.default_rng(42).random(shape).astype(np.float64)
    reversed_names = list(reversed(feature_names))
    ds = xr.Dataset(
        {
            name: xr.DataArray(data[:, i], dims=["y"])
            for i, name in enumerate(reversed_names)
        }
    )
    result = predict_random_forest(data=ds, model=model)
    assert isinstance(result, xr.Dataset)


@pytest.mark.xgboost
def test_predict_random_forest_missing_variable(xgboost_client, vector_data_cube):
    predictors_vars = ["value2"]
    target_var = "value1"
    model = fit_regr_random_forest(
        predictors=vector_data_cube,
        target=vector_data_cube,
        target_var=target_var,
        predictors_vars=predictors_vars,
    )

    ds = xr.Dataset({"wrong_name": xr.DataArray(np.ones(10), dims=["y"])})
    with pytest.raises(Exception, match="not present"):
        predict_random_forest(data=ds, model=model)


@pytest.mark.xgboost
def test_predict_random_forest_extra_variable(xgboost_client, vector_data_cube):
    predictors_vars = ["value2"]
    target_var = "value1"
    model = fit_regr_random_forest(
        predictors=vector_data_cube,
        target=vector_data_cube,
        target_var=target_var,
        predictors_vars=predictors_vars,
    )

    feature_names = list(model.feature_names)
    ds = xr.Dataset(
        {
            **{name: xr.DataArray(np.ones(10), dims=["y"]) for name in feature_names},
            "extra_var": xr.DataArray(np.ones(10), dims=["y"]),
        }
    )
    with pytest.raises(Exception, match="not in the model"):
        predict_random_forest(data=ds, model=model)


@pytest.mark.xgboost
def test_predict_random_forest_no_feature_names(xgboost_client, vector_data_cube):
    predictors_vars = ["value2"]
    target_var = "value1"
    model = fit_regr_random_forest(
        predictors=vector_data_cube,
        target=vector_data_cube,
        target_var=target_var,
        predictors_vars=predictors_vars,
    )
    model.feature_names = None

    ds = xr.Dataset({"B02": xr.DataArray(np.ones(10), dims=["y"])})
    with pytest.warns(UserWarning, match="no feature_names"):
        predict_random_forest(data=ds, model=model)


@pytest.mark.xgboost
def test_predict_random_forest_feature_order_context(xgboost_client, vector_data_cube):
    predictors_vars = ["value2"]
    target_var = "value1"
    model = fit_regr_random_forest(
        predictors=vector_data_cube,
        target=vector_data_cube,
        target_var=target_var,
        predictors_vars=predictors_vars,
    )
    model.feature_names = None

    ds = xr.Dataset({"B02": xr.DataArray(np.ones(10), dims=["y"])})
    result = predict_random_forest(
        data=ds, model=model, context={"feature_order": ["B02"]}
    )
    assert isinstance(result, xr.Dataset)
    assert "result" in result.data_vars


# ── curve fitting tests (no xgboost) ──────────────────────────────────────


@pytest.mark.parametrize("size", [(6, 5, 4, 3)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_curve_fitting(temporal_interval, bounding_box, random_raster_data):
    origin_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04"],
        backend="dask",
    )

    @process
    def fitFunction(x, parameters):
        t0 = 2 * np.pi / 31557600 * x
        return parameters[0] + parameters[1] * np.cos(t0) + parameters[2] * np.sin(t0)

    _process = partial(
        fitFunction,
        x=ParameterReference(from_parameter="x"),
        parameters=ParameterReference(from_parameter="parameters"),
    )

    parameters = [1, 0, 0]
    result = fit_curve(
        origin_cube, parameters=parameters, function=_process, dimension="t"
    )
    assert len(result.param) == 3

    assert isinstance(result, xr.Dataset)
    for var in result.data_vars:
        assert isinstance(result[var].data, dask.array.Array)
    assert result.odc.crs == origin_cube.odc.crs

    assert list(result.data_vars) == list(origin_cube.data_vars)
    assert len(result.coords["x"]) == len(origin_cube.coords["x"])
    assert len(result.coords["y"]) == len(origin_cube.coords["y"])
    assert len(result.coords["param"]) == len(parameters)

    origin_cube_B02 = origin_cube[["B02"]]
    result_B02 = fit_curve(
        origin_cube_B02, parameters=parameters, function=_process, dimension="t"
    )
    assert list(result_B02.data_vars) == ["B02"]

    labels = dimension_labels(origin_cube, origin_cube.openeo.temporal_dims[0])
    predictions = predict_curve(
        result,
        _process,
        origin_cube.openeo.temporal_dims[0],
        labels=labels,
    ).compute()

    assert isinstance(predictions, xr.Dataset)
    assert len(predictions.coords[origin_cube.openeo.temporal_dims[0]]) == len(labels)
    assert "param" not in predictions.dims
    assert result.odc.crs == predictions.odc.crs

    labels = [0, 1, 2, 3]
    predictions = predict_curve(
        result,
        _process,
        origin_cube.openeo.temporal_dims[0],
        labels=labels,
    ).compute()

    assert len(predictions.coords[origin_cube.openeo.temporal_dims[0]]) == len(labels)
    assert "param" not in predictions.dims
    assert result.odc.crs == predictions.odc.crs


def test_fit_curve_preserves_dask_laziness(temporal_interval, bounding_box):
    """P1.1: fit_curve preserves dask laziness (no eager .persist())."""
    np.random.seed(42)
    data = np.random.rand(6, 5, 10, 3).astype(np.float64)
    origin_cube = create_fake_rastercube(
        data=data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04"],
        backend="dask",
        as_dataset=True,
    )

    def fitFunction(x, parameters):
        t0 = 2 * np.pi / 31557600 * x
        return parameters[0] + parameters[1] * np.cos(t0)

    parameters = [1, 0]
    result = fit_curve(
        origin_cube, parameters=parameters, function=fitFunction, dimension="t"
    )
    assert isinstance(result, xr.Dataset)
    for var_name in result.data_vars:
        var = result[var_name]
        assert isinstance(var.data, dask.array.Array)

        graph = var.data.__dask_graph__()
        assert len(graph) > 0, f"Dask graph for {var_name} is empty (was computed)"
