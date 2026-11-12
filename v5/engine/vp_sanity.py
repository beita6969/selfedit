"""Sanity of the edited vision front-ends (coordinator 11-12): vision tower only (HF transformers Qwen3_5VisionModel), one GPU.
For 20 signal rows (first image), at the edited processor's own size:
  VP2  (patch 8 box, merge 2): edited tower on image I  vs  stock tower on nearest-x2(I)          -> expected ~ identical (cosine ~1)
  VP11 (patch 11 bicubic):     edited tower on image I  vs  stock tower on bicubic-(16/11)(I)       -> approximate
  VPM  (patch 8 box, merge 4): edited tower on image I  vs  stock tower on I (same LLM token count)  -> token count equal; cosine reported
Also: patches processed by the tower and visual tokens handed to the LLM per image, per arm.
usage: vp_sanity.py   (writes /data/fd/mingdazhang/selfedit/v5/vp/sanity.json)"""
import json, os, sys, math
import torch, torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoImageProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
V4 = "/data/fd/mingdazhang/selfedit/v4"; OUT = "/data/fd/mingdazhang/selfedit/v5/vp"; os.makedirs(OUT, exist_ok=True)
ST = "/data/fd/mingdazhang/selfedit/models/Qwen3.8-27B"
ARMS = {"vp2": "/data/fd/mingdazhang/selfedit/models/vh/8fab893f70b1", "vpm": "/data/fd/mingdazhang/selfedit/v5/wckpt/vpm",
        "vp11": "/data/fd/mingdazhang/selfedit/v5/wckpt/vp11"}
dev = "cuda"


def tower(d):
    cfg = AutoConfig.from_pretrained(d).vision_config
    cfg._attn_implementation = "sdpa"
    m = Qwen3_5VisionModel(cfg).to(torch.bfloat16)
    wm = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
    sd = {}
    for f in sorted({v for k, v in wm.items() if k.startswith("model.visual.")}):
        with safe_open(os.path.join(d, f), "pt") as fh:
            for k in fh.keys():
                if k.startswith("model.visual."): sd[k[len("model.visual."):]] = fh.get_tensor(k)
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not unexp and not [k for k in miss if "rotary" not in k], (miss[:5], unexp[:5])
    return m.to(dev).eval(), AutoImageProcessor.from_pretrained(d), cfg


def run(m, proc, im):
    x = proc(images=[im], return_tensors="pt")
    pv, g = x["pixel_values"].to(dev, torch.bfloat16), x["image_grid_thw"].to(dev)
    with torch.no_grad():
        o = m(pv, grid_thw=g)
    o = o.pooler_output if hasattr(o, "pooler_output") and o.pooler_output is not None else (o[0] if isinstance(o, (tuple, list)) else o)
    return o.float(), g[0].tolist()


def cos(a, b):
    if a.shape != b.shape: return None
    c = F.cosine_similarity(a, b, dim=-1)
    return dict(mean=round(c.mean().item(), 4), min=round(c.min().item(), 4))


rows = [r for r in map(json.loads, open(f"{V4}/rows/learn.jsonl")) if r.get("v4_role") in ("show", "monitor")]
pick = rows[::16][:20]
S, Sp, Scfg = tower(ST)
res = []
for name, d in ARMS.items():
    E, Ep, Ecfg = tower(d)
    p, mg = Ecfg.patch_size, Ecfg.spatial_merge_size
    for r in pick:
        path = r["image_paths"][0] if isinstance(r["image_paths"], list) else r["image_paths"]
        im = Image.open(path).convert("RGB")
        # the edited processor's own resize, done once here so both towers see the same pixels
        e_out, eg = run(E, Ep, im)
        h, w = eg[1] * p, eg[2] * p
        base = im.resize((w, h), Image.BICUBIC)
        e_out, eg = run(E, Ep, base)
        if name == "vp2":
            ref = base.resize((2 * w, 2 * h), Image.NEAREST)
        elif name == "vp11":
            ref = base.resize((round(w * 16 / 11 / 32) * 32, round(h * 16 / 11 / 32) * 32), Image.BICUBIC)
        else:
            ref = base
        s_out, sg = run(S, Sp, ref)
        res.append(dict(arm=name, row=r["id"], bench=r["bench"], orig=list(im.size), edited_grid=eg, stock_grid=sg,
                        tower_patches_edited=eg[0] * eg[1] * eg[2], tower_patches_stock_on_orig=None,
                        llm_tokens_edited=e_out.shape[0], llm_tokens_stock_ref=s_out.shape[0], cos=cos(e_out, s_out)))
        so, sog = run(S, Sp, im)                       # stock tower on the original image: the stock token count
        res[-1].update(tower_patches_stock_on_orig=sog[0] * sog[1] * sog[2], llm_tokens_stock_on_orig=so.shape[0])
        print(json.dumps(res[-1]), flush=True)
    del E; torch.cuda.empty_cache()
json.dump(res, open(f"{OUT}/sanity.json", "w"), indent=1)
for name in ARMS:
    R = [x for x in res if x["arm"] == name]
    cs = [x["cos"]["mean"] for x in R if x["cos"]]
    print("SUMMARY", name, "n", len(R), "cos mean of means", round(sum(cs) / max(1, len(cs)), 4) if cs else None, "min", min((x["cos"]["min"] for x in R if x["cos"]), default=None),
          "shape mismatches", sum(1 for x in R if not x["cos"]),
          "LLM tokens edited/stock_on_orig", round(sum(x["llm_tokens_edited"] for x in R) / sum(x["llm_tokens_stock_on_orig"] for x in R), 2),
          "tower patches edited/stock_on_orig", round(sum(x["tower_patches_edited"] for x in R) / sum(x["tower_patches_stock_on_orig"] for x in R), 2), flush=True)
