"""L2 (recommended profile) process applicability against the real DEDL HEALPix datacubes.

Loads a HEALPix datacube once through dedl `load_stac` (real S3/icechunk data),
then runs every L2 process through the same ProcessRegistry path the executor
uses, using semantically correct process graphs (built with the openEO client).

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
        python live_tests/l2_healpix_applicability.py [--datacube MSG] [--region small] [PROCESS ...]
"""
import argparse
import importlib
import inspect
import json
import os
import time
from pathlib import Path

import openeo
from _datacubes import DATACUBES, REGIONS
from dotenv import load_dotenv
from openeo_pg_parser_networkx import OpenEOProcessGraph, Process, ProcessRegistry

from openeo_processes_dedl_slim.process_implementations.core import process

_DEFAULT_ENV_PATH = (
    Path(__file__).resolve().parents[2] / "openeo-processes-dedl-cube-load" / ".env"
)
_ENV_PATH = Path(os.environ.get("DEDL_ENV_PATH", _DEFAULT_ENV_PATH))
load_dotenv(_ENV_PATH)

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


def load_cube(datacube: str, region: str):
    from openeo_processes_dedl_cube_load import load_stac

    cfg = DATACUBES[datacube]
    temporal = cfg["global_temporal"] if region == "global" else cfg["temporal"]
    t0 = time.perf_counter()
    cube = load_stac(
        url=cfg["url"],
        bands=[cfg["band"]],
        spatial_extent=REGIONS[region],
        temporal_extent=temporal,
    )
    print(
        f"loaded {datacube} ({region}): {dict(cube.sizes)} in {time.perf_counter() - t0:.1f}s"
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


def make_graph(conn, process_id, datacube: str, region: str):
    """Build a semantically correct openEO graph for `process_id` against the cube."""
    cfg = DATACUBES[datacube]
    temporal = cfg["global_temporal"] if region == "global" else cfg["temporal"]
    cube = conn.load_stac(
        cfg["url"],
        bands=[cfg["band"]],
        temporal_extent=temporal,
        spatial_extent=REGIONS[region],
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

    region_bbox = REGIONS[region]
    subarea = None
    if region_bbox is not None:
        subarea = {
            "west": region_bbox["west"],
            "east": (region_bbox["west"] + region_bbox["east"]) / 2,
            "south": region_bbox["south"],
            "north": (region_bbox["south"] + region_bbox["north"]) / 2,
            "crs": region_bbox["crs"],
        }

    handlers = {
        "add_dimension": lambda: cube.add_dimension(
            name=NEW_DIM_NAME, label=NEW_DIM_LABEL, type="other"
        ),
        "aggregate_temporal": lambda: cube.aggregate_temporal(
            intervals=[temporal],
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
            west=subarea["west"],
            south=subarea["south"],
            east=subarea["east"],
            north=subarea["north"],
            crs=subarea["crs"],
        ),
        "filter_temporal": lambda: cube.filter_temporal(extent=temporal),
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datacube", choices=list(DATACUBES), default="MSG")
    parser.add_argument("--region", choices=list(REGIONS), default="small")
    parser.add_argument("processes", nargs="*", help="process ids to run (default: all)")
    args = parser.parse_args(argv)

    conn = openeo.connect(
        "https://openeo-staging.datalakecube.eumetsat.data.destination-earth.eu/openeo/1.1.0/"
    )

    cube = load_cube(args.datacube, args.region)
    reg = build_registry(cube)

    processes = L2_PROCESSES
    if args.processes:
        processes = [p for p in processes if p in args.processes]

    results = []
    for pid in processes:
        print("=" * 100)
        print(f"PROCESS: {pid}")
        try:
            graph = make_graph(conn, pid, args.datacube, args.region)
            dur = run_one(reg, graph, pid)
            results.append({"process": pid, "status": "ok", "time_s": round(dur, 2)})
            print(f"  OK ({dur:.2f}s)")
        except Exception as exc:  # noqa: BLE001
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
    print(f"SUMMARY ({args.datacube}, {args.region})")
    for r in results:
        status = r["status"]
        mark = "PASS" if status == "ok" else "FAIL"
        extra = f"  {r['time_s']}s" if status == "ok" else f"  {r['error']}"
        print(f"  [{mark}] {r['process']:<28}{extra}")
    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{len(results)} processes: {n_ok} OK, {len(results) - n_ok} failed")

    out = Path(
        os.environ.get(
            "L2_RESULTS_OUT",
            Path(__file__).parent
            / f"l2_{args.datacube.lower()}_{args.region}_results.json",
        )
    )
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
