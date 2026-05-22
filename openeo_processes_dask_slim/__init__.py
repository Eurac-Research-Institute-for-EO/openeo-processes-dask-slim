import importlib.metadata

try:
    __version__ = importlib.metadata.version("openeo_processes_dask_slim")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0.dev0"
