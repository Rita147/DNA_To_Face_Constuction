import inspect
import io
import pickle
import sys
import types
import numpy as np


def _patch_chumpy() -> None:
    """
    Inject a minimal chumpy stub so that FLAME's generic_model.pkl (which
    was saved with chumpy arrays) can be unpickled without the real chumpy
    package (which is broken on Python >= 3.10 because it imports `pip`).

    Uses a custom Unpickler that intercepts chumpy class lookups and
    returns a numpy-compatible proxy class that can accept chumpy's pickle
    constructor arguments (data array, not shape).
    """
    if "chumpy" in sys.modules:
        return

    class _ChumProxy(np.ndarray):
        """
        Numpy ndarray subclass that accepts chumpy's constructor signature:
          chumpy.ch.array(x)  where x is the underlying data array.
        Chumpy's __reduce__ stores (cls, (x_array,), ...) so pickle calls
        cls(x_array) — we must accept an ndarray-like as the first arg.
        """
        def __new__(cls, x=None, *args, **kwargs):
            if x is None:
                return np.empty(0).view(cls)
            arr = np.asarray(x)
            obj = arr.view(cls)
            return obj

        def __array_finalize__(self, obj):
            pass

        def __setstate__(self, state):
            # chumpy stores a dict with key 'x' holding the real ndarray;
            # numpy stores a 5-tuple.  Reconstruct from whichever we get.
            if isinstance(state, dict) and "x" in state:
                arr = np.asarray(state["x"])
                # Resize self in-place by reinitialising through __new__
                # (numpy doesn't support __setitem__ on shape), so we use
                # the ndarray state tuple from the underlying array.
                nd_state = arr.__reduce__()[2]
                super().__setstate__(nd_state)
            elif isinstance(state, tuple) and len(state) == 5:
                super().__setstate__(state)

    # Placeholder module so 'import chumpy' doesn't raise ImportError.
    # The FLAME pickle references chumpy.ch.Ch (capital C) as the array class.
    ch_mod = types.ModuleType("chumpy")
    ch_sub = types.ModuleType("chumpy.ch")
    ch_sub.Ch    = _ChumProxy                         # type: ignore[attr-defined]
    ch_sub.array = _ChumProxy                         # type: ignore[attr-defined]
    ch_mod.Ch    = _ChumProxy                         # type: ignore[attr-defined]
    ch_mod.array = _ChumProxy                         # type: ignore[attr-defined]
    ch_mod.ch    = ch_sub                             # type: ignore[attr-defined]
    sys.modules["chumpy"]      = ch_mod
    sys.modules["chumpy.ch"]   = ch_sub
    sys.modules["chumpy.reqs"] = types.ModuleType("chumpy.reqs")


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

    _patch_chumpy()
