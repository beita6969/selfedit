# selfedit 1109c: harness package. Vendored SkillOpt lives under harness/vendor (put on sys.path as top-level "skillopt").
import os as _os, sys as _sys
_v = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "vendor")
if _v not in _sys.path:
    _sys.path.insert(0, _v)
