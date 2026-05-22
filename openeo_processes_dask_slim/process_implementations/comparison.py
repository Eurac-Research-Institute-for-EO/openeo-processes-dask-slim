from typing import Optional

import dask.array as da
import numpy as np
from numpy.typing import ArrayLike

from openeo_processes_dask_slim.process_implementations.cubes.utils import notnull
from openeo_processes_dask_slim.process_implementations.utils import get_scalar_type

__all__ = [
    "is_infinite",
    "is_valid",
    "is_nan",
    "is_nodata",
    "eq",
    "neq",
    "gt",
    "gte",
    "lt",
    "lte",
    "between",
]


def is_infinite(x: ArrayLike):
    if np.issubdtype(get_scalar_type(x), np.number):
        return np.isinf(x)
    else:
        return False


def is_valid(x: ArrayLike):
    finite = np.logical_not(is_infinite(x))
    return np.logical_and(notnull(x), finite)


def is_nodata(x: ArrayLike):
    if isinstance(x, da.Array):
        return da.zeros_like(x, dtype=bool)
    if isinstance(x, np.ndarray):
        return np.zeros_like(x, dtype=bool)
    return x is None


def is_nan(x: ArrayLike):
    if is_nodata(x):
        return is_nodata(x)
    return np.isnan(x)


def eq(
    x: ArrayLike,
    y: ArrayLike,
    delta: Optional[float] = None,
    case_sensitive: Optional[bool] = True,
):
    x_dtype = get_scalar_type(x)
    y_dtype = get_scalar_type(y)

    try:
        both_number = np.issubdtype(x_dtype, np.number) and np.issubdtype(y_dtype, np.number)
        both_bool = np.issubdtype(x_dtype, np.bool_) and np.issubdtype(y_dtype, np.bool_)
        both_flexible = np.issubdtype(x_dtype, np.flexible) and np.issubdtype(y_dtype, np.flexible)

        if both_number or both_bool:
            if both_number and delta:
                ar_eq = np.isclose(x, y, atol=delta)
            else:
                ar_eq = x == y
        elif both_flexible:
            if not case_sensitive:
                if np.issubdtype(get_scalar_type(x), np.character):
                    x = np.char.lower(x)
                if np.issubdtype(get_scalar_type(y), np.character):
                    y = np.char.lower(y)
            ar_eq = x == y
        else:
            ar_eq = x == y

        null_mask = np.logical_and(notnull(x), notnull(y))

        result = _scalar_safe_where(null_mask, ar_eq, np.nan)
        return result
    except ValueError:
        if hasattr(x, "shape"):
            return np.zeros_like(x, dtype=bool)
        elif hasattr(y, "shape"):
            return np.zeros_like(y, dtype=bool)
        return False


def _scalar_safe_where(condition, x, y):
    if not hasattr(condition, 'shape') or isinstance(condition, (np.bool_, np.integer, np.floating)):
        return x if condition else y
    return np.where(condition, x, y)


def neq(
    x: ArrayLike,
    y: ArrayLike,
    delta: Optional[float] = None,
    case_sensitive: Optional[bool] = True,
):
    eq_val = eq(x, y, delta=delta, case_sensitive=case_sensitive)
    nulls = np.logical_and(notnull(x), notnull(y))
    result = _scalar_safe_where(nulls, np.logical_not(eq_val), np.nan)
    return result


def _cmp_op(x, y, op):
    try:
        result = op(x, y)
    except ValueError:
        if hasattr(x, "shape"):
            return np.zeros_like(x, dtype=bool)
        elif hasattr(y, "shape"):
            return np.zeros_like(y, dtype=bool)
        return False
    nulls = np.logical_and(notnull(x), notnull(y))
    return _scalar_safe_where(nulls, result, np.nan)


def gt(x: ArrayLike, y: ArrayLike):
    return _cmp_op(x, y, lambda a, b: a > b)


def gte(x: ArrayLike, y: ArrayLike):
    return _cmp_op(x, y, lambda a, b: (a - b) >= 0)


def lt(x: ArrayLike, y: ArrayLike):
    return _cmp_op(x, y, lambda a, b: a < b)


def lte(x: ArrayLike, y: ArrayLike):
    return _cmp_op(x, y, lambda a, b: a <= b)


def between(
    x: ArrayLike,
    min: float,
    max: float,
    exclude_max: Optional[bool] = False,
):
    if not notnull(min) or not notnull(max):
        return np.nan
    if exclude_max:
        bet = np.logical_and(gte(x, y=min), lt(x, y=max))
    else:
        bet = np.logical_and(gte(x, y=min), lte(x, y=max))
    if not hasattr(x, 'shape'):
        return bool(bet) if not isinstance(bet, np.floating) else bet
    return np.where(notnull(x), bet, np.nan)
