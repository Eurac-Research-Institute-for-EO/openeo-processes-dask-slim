"""L1 process applicability against the real DEDL HEALPix datacubes.

Loads a HEALPix datacube once through dedl `load_stac` (real S3/icechunk data),
then runs every L1(ish) process through the same ProcessRegistry path the
executor uses, using semantically correct process graphs (built with the
openEO client).

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
        python live_tests/l1_healpix_applicability.py [--datacube MSG] [--region small] [PROCESS ...]
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

DIMS = {"band": "bands", "temporal": "t"}
PROBABILITIES = [0.25, 0.5, 0.75]


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
        py = {"not": "not_"}.get(name, name)
        return cube.apply(lambda x: getattr(x, py)())

    def binary(name):
        py = {"and": "and_", "or": "or_"}.get(name, name)
        return cube.apply(lambda x: getattr(x, py)(x))

    def reducer_t(name):
        return cube.reduce_dimension(
            dimension=DIMS["temporal"], reducer=lambda data: getattr(data, name)()
        )

    handlers = {
        "absolute": lambda: unary("absolute"),
        "add": lambda: binary("add"),
        "and": lambda: binary("and"),
        "apply": lambda: cube.apply(lambda x: x.absolute()),
        "apply_dimension": lambda: cube.apply_dimension(
            dimension=DIMS["temporal"], process=lambda data: data.mean()
        ),
        "arccos": lambda: unary("arccos"),
        "arcsin": lambda: unary("arcsin"),
        "arctan": lambda: unary("arctan"),
        "array_concat": lambda: cube.apply_dimension(
            dimension=DIMS["temporal"],
            process=lambda data: eop.array_concat(array1=data, array2=data),
        ),
        "array_create": lambda: cube.apply_dimension(
            dimension=DIMS["temporal"],
            process=lambda data: eop.array_create(data=[1, 2, 3]),
        ),
        "array_element": lambda: cube.reduce_dimension(
            dimension=DIMS["temporal"],
            reducer=lambda data: data.array_element(index=0),
        ),
        "between": lambda: cube.apply(lambda x: x.between(min=0, max=1)),
        "ceil": lambda: unary("ceil"),
        "clip": lambda: cube.apply(lambda x: x.clip(min=0, max=1)),
        "constant": lambda: cube.apply(lambda x: x + eop.constant(1)),
        "cos": lambda: unary("cos"),
        "divide": lambda: binary("divide"),
        "e": lambda: cube.apply(lambda x: x * eop.e()),
        "eq": lambda: binary("eq"),
        "exp": lambda: unary("exp"),
        "first": lambda: reducer_t("first"),
        "floor": lambda: unary("floor"),
        "gt": lambda: binary("gt"),
        "gte": lambda: binary("gte"),
        "int": lambda: unary("int"),
        "last": lambda: reducer_t("last"),
        "ln": lambda: unary("ln"),
        "log": lambda: cube.apply(lambda x: x.log(base=10)),
        "lt": lambda: binary("lt"),
        "lte": lambda: binary("lte"),
        "max": lambda: reducer_t("max"),
        "mean": lambda: reducer_t("mean"),
        "median": lambda: reducer_t("median"),
        "min": lambda: reducer_t("min"),
        "mod": lambda: binary("mod"),
        "multiply": lambda: binary("multiply"),
        "neq": lambda: binary("neq"),
        "not": lambda: unary("not"),
        "or": lambda: binary("or"),
        "pi": lambda: cube.apply(lambda x: x * eop.pi()),
        "power": lambda: cube.apply(lambda x: x.power(p=2)),
        "product": lambda: reducer_t("product"),
        "quantiles": lambda: cube.apply_dimension(
            dimension=DIMS["temporal"],
            process=lambda data: data.quantiles(probabilities=PROBABILITIES),
        ),
        "reduce_dimension": lambda: cube.reduce_dimension(
            dimension=DIMS["temporal"], reducer=lambda data: data.mean()
        ),
        "round": lambda: unary("round"),
        "sd": lambda: reducer_t("sd"),
        "sgn": lambda: unary("sgn"),
        "sin": lambda: unary("sin"),
        "sqrt": lambda: unary("sqrt"),
        "subtract": lambda: binary("subtract"),
        "sum": lambda: reducer_t("sum"),
        "tan": lambda: unary("tan"),
        "variance": lambda: reducer_t("variance"),
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
    parser.add_argument(
        "processes", nargs="*", help="process ids to run (default: all)"
    )
    args = parser.parse_args(argv)

    conn = openeo.connect(
        "https://openeo-staging.datalakecube.eumetsat.data.destination-earth.eu/openeo/1.1.0/"
    )

    cube = load_cube(args.datacube, args.region)
    reg = build_registry(cube)

    processes = [
        "absolute",
        "add",
        "and",
        "apply",
        "apply_dimension",
        "arccos",
        "arcsin",
        "arctan",
        "array_concat",
        "array_create",
        "array_element",
        "between",
        "ceil",
        "clip",
        "constant",
        "cos",
        "divide",
        "e",
        "eq",
        "exp",
        "first",
        "floor",
        "gt",
        "gte",
        "int",
        "last",
        "ln",
        "log",
        "lt",
        "lte",
        "max",
        "mean",
        "median",
        "min",
        "mod",
        "multiply",
        "neq",
        "not",
        "or",
        "pi",
        "power",
        "product",
        "quantiles",
        "reduce_dimension",
        "round",
        "sd",
        "sgn",
        "sin",
        "sqrt",
        "subtract",
        "sum",
        "tan",
        "variance",
    ]
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
        print(f"  [{mark}] {r['process']:<18}{extra}")
    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{len(results)} processes: {n_ok} OK, {len(results) - n_ok} failed")

    out = Path(
        os.environ.get(
            "L1_RESULTS_OUT",
            Path(__file__).parent
            / f"l1_{args.datacube.lower()}_{args.region}_results.json",
        )
    )
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
