"""pixlib (fable_int, law-table row C_pix): intlib.Lab with a second image tokenisation in the same process.
Mode "vp2" = the same prompt and the same ORIGINAL image, tokenised by the VP2 checkpoint (its preprocessor: patch 8, merge 2, at most 1,048,576 px; its visual tower, which differs from
stock in the patch-embedding kernel only). Mode "medium" (the base of every script) = the stock lattice on the original image (the stock request without client floor), unless
FI_PIX_NATIVE=0, in which case it is the row's own view (t005r: 4,096-token floor). Mode "floor" = the row's own view, stock lattice. The language model is the stock one in every mode.
Use: python pix_run.py SCRIPT ARGS...  (pix_run replaces intlib.Lab by LabVP and runs SCRIPT unchanged)."""
import json, os, time
import torch
import intlib

VP = os.environ.get("FI_VP_MODEL", "/data/fd/mingdazhang/selfedit/models/vh/8fab893f70b1")
NATIVE = [False]; _vi = intlib.view_image


def _view(path, mn=None, mx=None):
    return _vi(path, None, None) if NATIVE[0] else _vi(path, mn, mx)


intlib.view_image = _view


class LabVP(intlib.Lab):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        from transformers import AutoConfig, AutoProcessor
        from safetensors import safe_open
        t0 = time.time()
        self.proc_stock = self.proc; self.proc_vp = AutoProcessor.from_pretrained(VP)
        cfg = AutoConfig.from_pretrained(VP); v0 = self.model.model.visual; v1 = type(v0)(cfg.vision_config)
        idx = json.load(open(os.path.join(VP, "model.safetensors.index.json")))["weight_map"]; new = {}
        for f in sorted({idx[k] for k in idx if k.startswith("model.visual.")}):
            with safe_open(os.path.join(VP, f), "pt") as g:
                for k in g.keys():
                    if k.startswith("model.visual."):
                        new[k[len("model.visual."):]] = g.get_tensor(k)
        sd = v0.state_dict()
        dif = [k for k in new if k not in sd or sd[k].shape != new[k].shape or not torch.equal(sd[k].float().cpu(), new[k].float().cpu())]
        v1.load_state_dict(new, strict=True)
        self.vis_stock = v0; self.vis_vp = v1.to(self.dev, torch.float32).eval().requires_grad_(False)
        print(f"[pix] second visual tower from {VP}: {len(new)} tensors, differing from stock: {dif}; patch {cfg.vision_config.patch_size} against {self.model.config.vision_config.patch_size}; {time.time() - t0:.0f}s", flush=True)

    def prep(self, rid, effort, gen, system=None):
        vp = effort == "vp2"; native = os.environ.get("FI_PIX_NATIVE", "1") == "1"
        if effort == "floor":
            native = False
        if effort in ("vp2", "floor", "stock"):
            effort = "medium"
        NATIVE[0] = native
        if vp:
            self.proc = self.proc_vp; self.model.model.visual = self.vis_vp
        try:
            return super().prep(rid, effort, gen, system)
        finally:
            NATIVE[0] = False; self.proc = self.proc_stock; self.model.model.visual = self.vis_stock
