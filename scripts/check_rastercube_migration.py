#!/usr/bin/env python3
"""
Static check script for RasterCube Dataset migration.

Reports on patterns that indicate incomplete migration from xr.DataArray
to xr.Dataset as the RasterCube contract.

Usage:
    python scripts/check_rastercube_migration.py          # informational
    python scripts/check_rastercube_migration.py --ci     # fail on ERROR/WARN
"""

import argparse
import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RASTER_DIR = (
    PROJECT_ROOT / "openeo_processes_dedl_slim" / "process_implementations" / "cubes"
)
DATA_MODEL = (
    PROJECT_ROOT
    / "openeo_processes_dedl_slim"
    / "process_implementations"
    / "data_model.py"
)

ALLOWLIST = [
    # _xr_interop.py: registers the openeo accessor on both DataArray and Dataset
    "openeo_processes_dedl_slim/process_implementations/cubes/_xr_interop.py:23:register_dataarray_accessor",
    # geometries.py: VectorCube functions, not RasterCube
    "openeo_processes_dedl_slim/process_implementations/cubes/geometries.py:114:xr.DataArray",
    "openeo_processes_dedl_slim/process_implementations/cubes/geometries.py:126:isinstance(..., xr.DataArray)",
    "openeo_processes_dedl_slim/process_implementations/cubes/geometries.py:157:isinstance(..., xr.DataArray)",
    # apply.py: checks if apply_ufunc returned DataArray (bounded bridge output)
    "openeo_processes_dedl_slim/process_implementations/cubes/apply.py:167:isinstance(..., xr.DataArray)",
    # reduce.py: checks if reduce returned DataArray (bounded bridge output)
    "openeo_processes_dedl_slim/process_implementations/cubes/reduce.py:41:isinstance(..., xr.DataArray)",
    # dataset_bridge.py: the centralized bridge module itself
    "openeo_processes_dedl_slim/process_implementations/cubes/dataset_bridge.py:47:to_array(",
    # general.py: to_array for NaN mask merge (not a band bridge)
    "openeo_processes_dedl_slim/process_implementations/cubes/general.py:58:to_array(",
    # merge.py: to_array in DataArray path for band reordering after combine_by_coords
    "openeo_processes_dedl_slim/process_implementations/cubes/merge.py:223:to_array(",
    # merge.py: .data access in DataArray merge path (not RasterCube input)
    "openeo_processes_dedl_slim/process_implementations/cubes/merge.py:149:.data access",
    "openeo_processes_dedl_slim/process_implementations/cubes/merge.py:150:.data access",
    "openeo_processes_dedl_slim/process_implementations/cubes/merge.py:151:.data access",
    "openeo_processes_dedl_slim/process_implementations/cubes/merge.py:179:.data access",
    "openeo_processes_dedl_slim/process_implementations/cubes/merge.py:180:.data access",
    "openeo_processes_dedl_slim/process_implementations/cubes/merge.py:271:.data access",
    "openeo_processes_dedl_slim/process_implementations/cubes/merge.py:272:.data access",
    "openeo_processes_dedl_slim/process_implementations/cubes/merge.py:338:.data access",
    "openeo_processes_dedl_slim/process_implementations/cubes/merge.py:339:.data access",
    "openeo_processes_dedl_slim/process_implementations/cubes/merge.py:341:.data access",
    "openeo_processes_dedl_slim/process_implementations/cubes/merge.py:342:.data access",
]

findings = []


def is_allowlisted(file_rel, line, pattern):
    key = f"{file_rel}:{line}:{pattern}"
    return key in ALLOWLIST


def check_rastercube_type():
    src = DATA_MODEL.read_text()
    tree = ast.parse(src, filename=str(DATA_MODEL))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "RasterCube":
                    if (
                        isinstance(node.value, ast.Attribute)
                        and isinstance(node.value.value, ast.Name)
                        and node.value.value.id == "xr"
                        and node.value.attr == "Dataset"
                    ):
                        return
                    findings.append(
                        {
                            "file": str(DATA_MODEL.relative_to(PROJECT_ROOT)),
                            "line": node.lineno,
                            "pattern": "RasterCube type is not xr.Dataset",
                            "severity": "ERROR",
                        }
                    )
    findings.append(
        {
            "file": str(DATA_MODEL.relative_to(PROJECT_ROOT)),
            "line": 0,
            "pattern": "RasterCube assignment not found",
            "severity": "ERROR",
        }
    )


def check_isinstance_dataarray():
    for pyfile in sorted(RASTER_DIR.rglob("*.py")):
        src = pyfile.read_text()
        tree = ast.parse(src, filename=str(pyfile))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "isinstance"
                    or isinstance(func, ast.Name)
                    and func.id == "isinstance"
                ):
                    if len(node.args) >= 2:
                        arg2 = node.args[1]
                        if (
                            isinstance(arg2, ast.Attribute)
                            and isinstance(arg2.value, ast.Name)
                            and arg2.value.id == "xr"
                            and arg2.attr == "DataArray"
                        ):
                            file_rel = str(pyfile.relative_to(PROJECT_ROOT))
                            if not is_allowlisted(
                                file_rel, node.lineno, "isinstance(..., xr.DataArray)"
                            ):
                                findings.append(
                                    {
                                        "file": file_rel,
                                        "line": node.lineno,
                                        "pattern": "isinstance(..., xr.DataArray)",
                                        "severity": "WARN",
                                    }
                                )


def check_to_array_bridge():
    for pyfile in sorted(RASTER_DIR.rglob("*.py")):
        src = pyfile.read_text()
        tree = ast.parse(src, filename=str(pyfile))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "to_array":
                    file_rel = str(pyfile.relative_to(PROJECT_ROOT))
                    if not is_allowlisted(file_rel, node.lineno, "to_array("):
                        findings.append(
                            {
                                "file": file_rel,
                                "line": node.lineno,
                                "pattern": "to_array(",
                                "severity": "WARN",
                            }
                        )


def check_direct_data_access():
    for pyfile in sorted(RASTER_DIR.rglob("*.py")):
        src = pyfile.read_text()
        tree = ast.parse(src, filename=str(pyfile))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "data":
                file_rel = str(pyfile.relative_to(PROJECT_ROOT))
                if not is_allowlisted(file_rel, node.lineno, ".data access"):
                    findings.append(
                        {
                            "file": file_rel,
                            "line": node.lineno,
                            "pattern": ".data access",
                            "severity": "INFO",
                        }
                    )


def check_dataarray_register():
    for pyfile in sorted(RASTER_DIR.rglob("*.py")):
        src = pyfile.read_text()
        if "register_dataarray_accessor" in src:
            tree = ast.parse(src, filename=str(pyfile))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if (
                        isinstance(func, ast.Attribute)
                        and func.attr == "register_dataarray_accessor"
                    ):
                        file_rel = str(pyfile.relative_to(PROJECT_ROOT))
                        if not is_allowlisted(
                            file_rel, node.lineno, "register_dataarray_accessor"
                        ):
                            findings.append(
                                {
                                    "file": file_rel,
                                    "line": node.lineno,
                                    "pattern": "register_dataarray_accessor",
                                    "severity": "INFO",
                                }
                            )


def main():
    parser = argparse.ArgumentParser(description="Check RasterCube migration status")
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Fail on ERROR or WARN findings (for CI use)",
    )
    args = parser.parse_args()

    check_rastercube_type()
    check_isinstance_dataarray()
    check_to_array_bridge()
    check_direct_data_access()
    check_dataarray_register()

    seen = set()
    unique_findings = []
    for f in findings:
        key = (f["file"], f["line"], f["pattern"])
        if key not in seen:
            seen.add(key)
            unique_findings.append(f)

    print("=" * 72)
    print("  RasterCube Migration Static Check")
    print("=" * 72)

    if not unique_findings:
        print("\n  No issues found.")
        sys.exit(0)

    print()

    for f in sorted(
        unique_findings, key=lambda x: (x["severity"], x["file"], x["line"])
    ):
        print(f"  [{f['severity']:5s}] " f"{f['file']}:{f['line']}  " f"{f['pattern']}")

    info_count = sum(1 for f in unique_findings if f["severity"] == "INFO")
    warn_count = sum(1 for f in unique_findings if f["severity"] == "WARN")
    error_count = sum(1 for f in unique_findings if f["severity"] == "ERROR")

    print(
        f"\n  Summary: {len(unique_findings)} total — "
        f"{error_count} ERROR, {warn_count} WARN, {info_count} INFO"
    )

    if args.ci and (error_count > 0 or warn_count > 0):
        print("\n  FAILED: --ci mode and ERROR/WARN findings exist.\n")
        sys.exit(1)

    print()


if __name__ == "__main__":
    main()
