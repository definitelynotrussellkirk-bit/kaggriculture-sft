"""Primitive-BC Kaggle agent: lazy CPU policy load, PASS on any exception, one stderr line per step."""
import os as _os
import sys as _sys
import time as _time
from pathlib import Path as _Path

_THREADS = max(1, int(_os.environ.get("PBC_THREADS", "1")))
_os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ[_key] = str(_THREADS)
_ROOT = _Path("/kaggle_simulations/agent")
if not (_ROOT / "MANIFEST.json").is_file():
    _ROOT = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
for _p in (_ROOT / "work/runs/maintainer/n-flow-deploy", _ROOT / "src"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
_PBC_SETTINGS = {"seed": 0, "temperature": 0.0, "weights": "policy.pt", "decode": {'mode': 'greedy'}}
_PBC_STATUS = {"calls": 0, "fallbacks": 0, "last_fallback": False, "last_error": None,
               "load_ms": None, "import_error": None}
_PBC_HOLDER = {}


def agent(observation, configuration=None):
    begun = _time.perf_counter()
    status = _PBC_STATUS
    status["calls"] += 1
    fallback, error, loading = False, None, "agent" not in _PBC_HOLDER
    try:
        if loading:
            if status["import_error"] is not None:
                raise RuntimeError(status["import_error"])
            try:
                import importlib.util as _util
                for _name in ("model", "codec"):
                    _mod = _sys.modules.get(_name)
                    if _mod is not None and not str(getattr(_mod, "__file__", "")).startswith(str(_ROOT)):
                        del _sys.modules[_name]
                _spec = _util.spec_from_file_location("pbc_agent", str(_ROOT / "work/runs/maintainer/n-flow-deploy" / "agent.py"))
                _module = _util.module_from_spec(_spec)
                _sys.modules["pbc_agent"] = _module
                _spec.loader.exec_module(_module)
                _PBC_HOLDER["agent"] = _module.make_agent(
                    str(_ROOT / _PBC_SETTINGS["weights"]), seed=_PBC_SETTINGS["seed"],
                    temperature=_PBC_SETTINGS["temperature"], threads=_THREADS, log=False,
                    decode=_PBC_SETTINGS["decode"])
            except Exception as exc:
                status["import_error"] = type(exc).__name__ + ": " + str(exc)
                raise
        actor = _PBC_HOLDER["agent"]
        action = actor(observation, configuration)
        fallback, error = actor.last_fallback, actor.last_error
        if loading:
            status["load_ms"] = actor.load_ms
    except Exception as exc:
        action = {"farmer": ["PASS"], "hands": [], "market": []}
        fallback, error = True, type(exc).__name__ + ": " + str(exc)
    status["last_fallback"], status["last_error"] = fallback, error
    status["fallbacks"] += int(bool(fallback))
    try:
        ms = (_time.perf_counter() - begun) * 1000.0
        step = observation.get("step") if hasattr(observation, "get") else None
        extra = ""
        if loading and status["load_ms"] is not None:
            extra += " load_ms=%.0f" % status["load_ms"]
        if fallback:
            extra += " err=" + str(error).split(":", 1)[0][:40]
        print("[pbc] step=%s ms=%.1f fb=%d%s" % (step, ms, int(bool(fallback)), extra),
              file=_sys.stderr, flush=True)
    except Exception:
        pass
    return action
