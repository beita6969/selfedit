"""pixlib3 (pixlib2 with the base-class call fixed: pixlib2 recursed once intlib.Lab had been replaced) (fable_int, stack check): the two-lattice lab of pixlib with one more switch. FI_PIX_ALL=1: EVERY context of the run (base and condition alike) is tokenised by the
visual-interface edit named in FI_VP_MODEL (its preprocessor bounds and its visual tower) on the original image: the language model then sees what it sees on the stacked base
(for example the accepted edit 1, .../v5/wckpt/vp2m512: patch 8, 524,288 to 1,048,576 px). Used to ask, teacher-forced, whether a cell fitted under the 4,096-token floor view keeps
its dose on the stacked base. Use: python pix_run3.py SCRIPT ARGS..."""
import os
import intlib, pixlib


BASE = pixlib.LabVP.__bases__[0]            # the original intlib.Lab (pix_run3 replaces intlib.Lab afterwards)


class LabVP2(pixlib.LabVP):
    def prep(self, rid, effort, gen, system=None):
        if os.environ.get("FI_PIX_ALL") != "1":
            return super().prep(rid, effort, gen, system)
        pixlib.NATIVE[0] = True; self.proc = self.proc_vp; self.model.model.visual = self.vis_vp
        try:
            return BASE.prep(self, rid, effort, gen, system)
        finally:
            pixlib.NATIVE[0] = False; self.proc = self.proc_stock; self.model.model.visual = self.vis_stock
