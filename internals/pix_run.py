"""pix_run (fable_int): run an unchanged fable_int script with the two-lattice lab. python pix_run.py SCRIPT ARGS..."""
import os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import intlib, pixlib
intlib.Lab = pixlib.LabVP
sys.argv = sys.argv[1:]
runpy.run_path(os.path.join(HERE, sys.argv[0]), run_name="__main__")
