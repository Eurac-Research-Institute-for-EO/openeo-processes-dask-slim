"""L2 (recommended profile) process applicability against the real MSG SEVIRI HEALPix datacube.

Loads the msg-hrseviri-healpix-datacube once through dedl `load_stac` (real
S3/icechunk data), then runs every L2 process through the same ProcessRegistry
path the executor uses, using semantically correct process graphs (built with
the openEO client).

The `load_stac` node is registered as a function that returns the pre-loaded
cube, so each process graph exercises full callback wiring on the real HEALPix
cube without re-loading S3 per process.

Prerequisites (live test - S3/icechunk credentials and the
``openeo-processes-dedl-cube-load`` package are required):

* ``AWS_ENDPOINT_URL``, ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``,
  ``AWS_DEFAULT_REGION`` (or a ``.env`` file, see ``DEDL_ENV_PATH`` below).
* ``openeo-processes-dedl-cube-load`` installed.

Usage::

    DEDL_ENV_PATH=/path/to/openeo-processes-dedl-cube-load/.env \\
        python live_tests/l2_healpix_applicability.py [PROCESS ...]
"""
import importlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import openeo
from dotenv import load_dotenv
from openeo_pg_parser_networkx import OpenEOProcessGraph, Process, ProcessRegistry

from openeo_processes_dedl_slim.process_implementations.core import process

_DEFAULT_ENV_PATH = (
    Path(__file__).resolve().parents[2] / "openeo-processes-dedl-cube-load" / ".env"
)
_ENV_PATH = Path(os.environ.get("DEDL_ENV_PATH", _DEFAULT_ENV_PATH))
load_dotenv(_ENV_PATH)

RESULTS_OUT = Path(
    os.environ.get("L2_RESULTS_OUT", Path(__file__).parent / "l2_healpix_results.json")
)

STAC_API_URL = "https://stac.datalakecube.eumetsat.data.destination-earth.eu"
CUBE_COLLECTION = "msg-hrseviri-healpix-datacube"
STAC_COLLECTION_URL = f"{STAC_API_URL}/collections/{CUBE_COLLECTION}"
BANDS = ["ir_10.8", "vis_0.6"]
BBOX_AREA = {"west": 7, "east": 8, "south": 49, "north": 50, "crs": "EPSG:4326"}
FILTER_BBOX_SUBAREA = {
    "west": BBOX_AREA["west"],
    "east": (BBOX_AREA["west"] + BBOX_AREA["east"]) / 2,
    "south": BBOX_AREA["south"],
    "north": (BBOX_AREA["south"] + BBOX_AREA["north"]) / 2,
    "crs": BBOX_AREA["crs"],
}
TEMPORAL = ["2023-07-21T10:30:00Z", "2023-07-21T16:30:00Z"]
DIMS = {"band": "healpix_index", "temporal": "t"}
NEW_DIM_NAME = "l2_check_dim"
NEW_DIM_LABEL = "1"

# L2 (recommended) profile process list, per TC_CUBE_0021.py
L2_PROCESSES = [
    "add_dimension",
    "aggregate_temporal",
    "aggregate_temporal_period",
    "all",
    "any",
    "apply_dimension",
    "arcosh",
    "arctan2",
    "array_element",
    "array_find",
    "arsinh",
    "artanh",
    "cosh",
    "count",
    "dimension_labels",
    "drop_dimension",
    "extrema",
    "filter_bbox",
    "filter_temporal",
    "if",
    "inspect",
    "is_nan",
    "is_nodata",
    "is_valid",
    "linear_scale_range",
    "nan",
    "normalized_difference",
    "reduce_dimension",
    "rename_dimension",
    "rename_labels",
    "sinh",
    "sort",
    "tanh",
    "xor",
]


def load_cube():
    from openeo_processes_dedl_cube_load import load_stac

    t0 = time.perf_counter()
    cube = load_stac(
        url=STAC_COLLECTION_URL,
        bands=BANDS,
        spatial_extent=BBOX_AREA,
        temporal_extent=TEMPORAL,
    )
    print(
        f"loaded {CUBE_COLLECTION}: {dict(cube.sizes)} in {time.perf_counter() - t0:.1f}s"
    )
    return cube


def build_registry(loaded_cube):
    reg = ProcessRegistry(wrap_funcs=[process])
    impls = [
        func
        for _, func in inspect.getmembers(
            importlib.import_module(
                "openeo_processes_dedl_slim.process_implementations"
            ),
            inspect.isfunction,
        )
    ]
    specs = importlib.import_module("openeo_processes_dedl_slim.specs")
    for func in impls:
        reg[func.__name__] = Process(
            spec=getattr(specs, func.__name__, None), implementation=func
        )
    # `except ImportError as e:` shadows the `e` process; register it explicitly.
    from openeo_processes_dedl_slim.process_implementations.math import e as _e

    reg["e"] = Process(spec=getattr(specs, "e", None), implementation=_e)

    # load_stac returns the pre-loaded cube (avoids re-loading S3 per process)
    reg["load_stac"] = Process(
        spec=getattr(specs, "load_stac", None), implementation=lambda **kw: loaded_cube
    )
    return reg


def make_graph(conn, process_id):
    """Build a semantically correct openEO graph for `process_id` against the MSG cube."""
    cube = conn.load_stac(
        STAC_COLLECTION_URL,
        bands=BANDS,
        temporal_extent=TEMPORAL,
        spatial_extent=BBOX_AREA,
    )
    eop = openeo.processes

    def unary(name):
        return cube.apply(lambda x: getattr(x, name)())

    def binary(name):
        return cube.apply(lambda x: getattr(x, name)(x))

    def reducer_t(name):
        return cube.reduce_dimension(
            dimension=DIMS["temporal"], reducer=lambda data: getattr(data, name)()
        )

    def reducer_band(name):
        return cube.reduce_dimension(
            dimension=DIMS["band"], reducer=lambda data: getattr(data, name)()
        )

    handlers = {
        "add_dimension": lambda: cube.add_dimension(
            name=NEW_DIM_NAME, label=NEW_DIM_LABEL, type="other"
        ),
        "aggregate_temporal": lambda: cube.aggregate_temporal(
            intervals=[TEMPORAL],
            reducer=lambda data: data.mean(),
            dimension=DIMS["temporal"],
        ),
        "aggregate_temporal_period": lambda: cube.aggregate_temporal_period(
            period="day", reducer=lambda data: data.mean(), dimension=DIMS["temporal"]
        ),
        "all": lambda: reducer_band("all"),
        "any": lambda: reducer_band("any"),
        "apply_dimension": lambda: cube.apply_dimension(
            dimension=DIMS["temporal"], process=lambda data: data.mean()
        ),
        "arcosh": lambda: unary("arcosh"),
        "arctan2": lambda: binary("arctan2"),
        "array_element": lambda: cube.reduce_dimension(
            dimension=DIMS["band"],
            reducer=lambda data: data.array_element(index=0),
        ),
        "array_find": lambda: cube.reduce_dimension(
            dimension=DIMS["band"],
            reducer=lambda data: data.array_find(value=0),
        ),
        "arsinh": lambda: unary("arsinh"),
        "artanh": lambda: unary("artanh"),
        "cosh": lambda: unary("cosh"),
        "count": lambda: reducer_band("count"),
        "dimension_labels": lambda: cube.dimension_labels(dimension=DIMS["temporal"]),
        "drop_dimension": lambda: cube.add_dimension(
            name=NEW_DIM_NAME, label=NEW_DIM_LABEL, type="other"
        ).drop_dimension(name=NEW_DIM_NAME),
        "extrema": lambda: cube.apply_dimension(
            dimension=DIMS["temporal"],
            process=lambda data: data.extrema(),
        ),
        "filter_bbox": lambda: cube.filter_bbox(
            west=FILTER_BBOX_SUBAREA["west"],
            south=FILTER_BBOX_SUBAREA["south"],
            east=FILTER_BBOX_SUBAREA["east"],
            north=FILTER_BBOX_SUBAREA["north"],
            crs=FILTER_BBOX_SUBAREA["crs"],
        ),
        "filter_temporal": lambda: cube.filter_temporal(extent=TEMPORAL),
        "if": lambda: cube.apply(lambda x: x.gt(0).if_(x, 0)),
        "inspect": lambda: cube.apply(
            lambda x: x.inspect(message="TC-CUBE-0021 L2 check")
        ),
        "is_nan": lambda: unary("is_nan"),
        "is_nodata": lambda: unary("is_nodata"),
        "is_valid": lambda: unary("is_valid"),
        "linear_scale_range": lambda: cube.apply(
            lambda x: x.linear_scale_range(
                inputMin=0, inputMax=1, outputMin=0, outputMax=255
            )
        ),
        "nan": lambda: cube.apply(lambda x: x * eop.nan()),
        "normalized_difference": lambda: binary("normalized_difference"),
        "reduce_dimension": lambda: cube.reduce_dimension(
            dimension=DIMS["temporal"], reducer=lambda data: data.mean()
        ),
        "rename_dimension": lambda: cube.rename_dimension(
            source=DIMS["band"], target="renamed_dimension"
        ),
        "rename_labels": lambda: cube.add_dimension(
            name=NEW_DIM_NAME, label=NEW_DIM_LABEL, type="other"
        ).rename_labels(
            dimension=NEW_DIM_NAME,
            target=[f"{NEW_DIM_LABEL}_renamed"],
            source=[NEW_DIM_LABEL],
        ),
        "sinh": lambda: unary("sinh"),
        "sort": lambda: cube.apply_dimension(
            dimension=DIMS["temporal"],
            process=lambda data: data.sort(nodata=True),
        ),
        "tanh": lambda: unary("tanh"),
        "xor": lambda: binary("xor"),
    }
    return handlers[process_id]().flat_graph()


def run_one(reg, graph, process_id):
    t0 = time.perf_counter()
    parsed = OpenEOProcessGraph(pg_data=graph)
    callable_ = parsed.to_callable(process_registry=reg, results_cache={})
    result = callable_()
    # Materialize dask/xarray output so lazy compute errors surface here
    if hasattr(result, "compute"):
        result.compute()
    return time.perf_counter() - t0


def main():
    targets = sys.argv[1:] or None
    conn = openeo.connect(
        "https://openeo-staging.datalakecube.eumetsat.data.destination-earth.eu/openeo/1.1.0/"
    )

    cube = load_cube()
    reg = build_registry(cube)

    processes = L2_PROCESSES
    if targets:
        processes = [p for p in processes if p in targets]

    results = []
    for pid in processes:
        print("=" * 100)
        print(f"PROCESS: {pid}")
        try:
            graph = make_graph(conn, pid)
            dur = run_one(reg, graph, pid)
            results.append({"process": pid, "status": "ok", "time_s": round(dur, 2)})
            print(f"  OK ({dur:.2f}s)")
        except Exception as exc:
            results.append(
                {
                    "process": pid,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                }
            )
            print(f"  ERROR: {type(exc).__name__}: {str(exc)[:200]}")
        print()

    print("\n" + "=" * 100)
    print("SUMMARY")
    for r in results:
        status = r["status"]
        mark = "PASS" if status == "ok" else "FAIL"
        extra = f"  {r['time_s']}s" if status == "ok" else f"  {r['error']}"
        print(f"  [{mark}] {r['process']:<28}{extra}")
    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{len(results)} processes: {n_ok} OK, {len(results) - n_ok} failed")

    out = RESULTS_OUT
    json.dump(results, open(out, "w"), indent=2)
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
