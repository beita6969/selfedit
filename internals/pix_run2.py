"""pix_run2 (fable_int): run an unchanged fable_int script with the lab of pixlib2 (all contexts on a visual-interface edit when FI_PIX_ALL=1). python pix_run2.py SCRIPT ARGS..."""
import os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import intlib, pixlib2
intlib.Lab = pixlib2.LabVP2
sys.argv = sys.argv[1:]
runpy.run_path(os.path.join(HERE, sys.argv[0]), run_name="__main__")
