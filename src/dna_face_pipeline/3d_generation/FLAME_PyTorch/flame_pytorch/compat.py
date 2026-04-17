import inspect
import numpy as np


def apply_compat_patches():
    # Python 3.11+ compatibility for older dependencies
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec

    # Use np.__dict__ to avoid FutureWarning triggered by hasattr(np, ...)
    alias_map = {
        "bool": np.bool_,
        "int": np.int_,
        "float": np.float64,
        "complex": np.complex128,
        "object": np.object_,
        "str": np.str_,
        "infty": np.inf,
    }

    for name, value in alias_map.items():
        if name not in np.__dict__:
            setattr(np, name, value)
