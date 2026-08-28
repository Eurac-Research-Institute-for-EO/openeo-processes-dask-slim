"""Shared config for the L1/L2 HEALPix applicability harnesses.

Datacube parameters mirror the load_stac timing benchmark in
``openeo-processes-dedl-cube-load/benchmarks/load-stac-timing``.
"""
from __future__ import annotations

STAC_API_URL = "https://stac.datalakecube.eumetsat.data.destination-earth.eu"
STAC_BASE = f"{STAC_API_URL}/collections"

DATACUBES = {
    "ASCAT": {
        "url": f"{STAC_BASE}/ascat-ssm-datacube",
        "band": "surface_soil_moisture",
        "temporal": ["2011-01-01T00:00:00Z", "2011-01-01T06:00:00Z"],
        "global_temporal": ["2011-01-01T00:30:00Z", "2011-01-01T01:30:00Z"],
    },
    "MSG": {
        "url": f"{STAC_BASE}/msg-hrseviri-healpix-datacube",
        "band": "ir_10.8",
        "temporal": ["2023-07-21T10:30:00Z", "2023-07-21T11:30:00Z"],
        "global_temporal": ["2023-07-21T10:30:00Z", "2023-07-21T10:45:00Z"],
    },
    "FDHSI": {
        "url": f"{STAC_BASE}/mtg-fci-fdhsi-datacube",
        "band": "ir_3.8",
        "temporal": ["2025-12-06T15:00:00Z", "2025-12-06T15:40:00Z"],
        "global_temporal": ["2025-12-06T15:00:00Z", "2025-12-06T15:10:00Z"],
    },
    "HRFI": {
        "url": f"{STAC_BASE}/mtg-fci-hrfi-datacube",
        "band": "ir_10.5",
        "temporal": ["2025-10-22T13:30:00Z", "2025-10-22T13:40:00Z"],
        "global_temporal": ["2025-10-22T13:30:00Z", "2025-10-22T13:40:00Z"],
    },
    "S3R": {
        "url": f"{STAC_BASE}/sentinel3-olci-wfr-reprocessed-datacube",
        "band": "CHL_NN",
        "temporal": ["2020-04-28T00:00:00Z", "2020-04-28T01:00:00Z"],
        "global_temporal": ["2020-04-28T00:05:00Z", "2020-04-28T00:10:00Z"],
    },
    "S3L": {
        "url": f"{STAC_BASE}/sentinel3-olci-wfr-live-datacube",
        "band": "CHL_NN",
        "temporal": ["2021-04-29T00:00:00Z", "2021-04-29T01:00:00Z"],
        "global_temporal": ["2021-04-29T00:05:00Z", "2021-04-29T00:10:00Z"],
    },
}

REGIONS = {
    "small": {"west": 0.0, "south": 0.0, "east": 1.0, "north": 1.0, "crs": "EPSG:4326"},
    "large": {"west": -100.0, "south": -20.0, "east": 100.0, "north": 20.0, "crs": "EPSG:4326"},
    "global": None,
}
