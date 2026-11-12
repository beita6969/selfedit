"""Run a script with flash-linear-attention blocked (transformers then uses its torch fallback): python nofla.py SCRIPT ARGS..."""
import runpy, sys
sys.modules["fla"] = None
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
