"""Finer eyes as a checkpoint edit: a new vision patch size with the patch kernel derived from the old one, so that a raw patch of p pixels gets what the old kernel would give
on that patch upsampled to 16 pixels: new kernel = B^T old kernel, B the upsampling. box with p = 8 is exactly nearest-neighbour x2. Nothing else in the tower changes; every token now covers fewer pixels.
Weights other than the patch kernel are links to the source. Runs on stock vLLM and transformers, both read the patch size from the config.

Usage: python rsi/vision_patch.py --model SRC --out DST --patch 8 --kernel box|bilinear|bicubic [--max-tokens 4096] [--min-tokens 3600] [--merge 4]
--max-tokens caps the merged tokens of one image, the processor shrinks larger images to fit. --min-tokens is the same from below.
--merge makes the merger take a larger square of patches, a multiple of the old one: each old input becomes the mean of the patches in its part of the square and the old merger is kept
inside the new one, its unused outputs zero. --patch 8 --merge 4 gives the tower four times the patches and the text the old number of tokens, which no resizing of the image can do.
"""
import argparse, json, os
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--patch", type=int, required=True)
ap.add_argument("--kernel", default="box")
ap.add_argument("--max-tokens", type=int, default=4096)
ap.add_argument("--merge", type=int, default=0)
ap.add_argument("--min-tokens", type=int, default=0)                    # smaller images are enlarged to this many tokens
args = ap.parse_args()

src, dst = os.path.abspath(args.model), os.path.abspath(args.out)
os.makedirs(dst, exist_ok=True)
index = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
KEY = "model.visual.patch_embed.proj.weight"
shard = index[KEY]
for name in os.listdir(src):
    if name not in (shard, "config.json", "preprocessor_config.json", "video_preprocessor_config.json") and not os.path.exists(os.path.join(dst, name)):
        os.symlink(os.path.join(src, name), os.path.join(dst, name))
cfg = json.load(open(os.path.join(src, "config.json")))
old = cfg["vision_config"]["patch_size"]
merge = cfg["vision_config"]["spatial_merge_size"]
tensors = load_file(os.path.join(src, shard))
w = tensors[KEY].float()                                                # [out, channels, temporal, old, old]
p = args.patch
if args.kernel == "box":
    if old % p:
        raise SystemExit(f"box needs a patch size dividing {old}")
    f = old // p
    new = F.avg_pool3d(w, (1, f, f)) * f * f
else:
    # B: p x p -> old x old, one column per pixel of the new patch; new kernel = B^T w
    eye = torch.eye(p * p).view(p * p, 1, p, p)
    B = F.interpolate(eye, size=(old, old), mode=args.kernel, align_corners=False).view(p * p, -1)
    new = (w.flatten(3) @ B.T).reshape(*w.shape[:3], p, p)
print(f"patch {old} -> {p}, kernel {args.kernel}: response to a flat patch changes by at most {(new.sum((3, 4)) - w.sum((3, 4))).abs().max().item():.4f}", flush=True)
tensors[KEY] = new.to(tensors[KEY].dtype)
if args.merge and args.merge != merge:
    if args.merge % merge:
        raise SystemExit(f"--merge needs a multiple of {merge}")
    dim, m, f = cfg["vision_config"]["hidden_size"], args.merge, args.merge // merge
    A, B = "model.visual.merger.linear_fc1.", "model.visual.merger.linear_fc2."
    if any(index[k + x] != shard for k in (A, B) for x in ("weight", "bias")):
        raise SystemExit("merger and patch kernel are in different shards")
    w1, b1, w2 = tensors[A + "weight"], tensors[A + "bias"], tensors[B + "weight"]
    new1 = torch.zeros(dim * m * m, dim * m * m, dtype=w1.dtype)
    for r in range(m):                                                  # the tower hands the merger each square row by row
        for c in range(m):
            i, o = (r * m + c) * dim, (r // f * merge + c // f) * dim
            new1[: w1.size(0), i: i + dim] = w1[:, o: o + dim].float().div(f * f).to(w1.dtype)
    new2 = torch.zeros(w2.size(0), dim * m * m, dtype=w2.dtype)
    new2[:, : w2.size(1)] = w2
    bias = torch.zeros(dim * m * m, dtype=b1.dtype)
    bias[: b1.size(0)] = b1
    tensors.update({A + "weight": new1, A + "bias": bias, B + "weight": new2})
    cfg["vision_config"]["spatial_merge_size"] = m
    merge = m
save_file(tensors, os.path.join(dst, shard), metadata={"format": "pt"})
cfg["vision_config"]["patch_size"] = p
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
pp = json.load(open(os.path.join(src, "preprocessor_config.json")))
pp["patch_size"] = p
pp["merge_size"] = merge
pp["size"]["longest_edge"] = args.max_tokens * (p * merge) ** 2
pp["size"]["shortest_edge"] = args.min_tokens * (p * merge) ** 2 if args.min_tokens else min(pp["size"]["shortest_edge"], pp["size"]["longest_edge"])
json.dump(pp, open(os.path.join(dst, "preprocessor_config.json"), "w"), indent=4)
# Video goes through the same patch kernel, so its processor must cut the same patch, or the tower gets 4 times the sequence the grid says and stops (p8box before this fix: 2048 against 512).
# The largest video keeps its old token count; a smaller one gets more, as an image does. Without this file in the source there is nothing to fix.
vp = os.path.join(src, "video_preprocessor_config.json")
if os.path.exists(vp):
    v = json.load(open(vp))
    oldside = v["patch_size"] * v["merge_size"]
    v["patch_size"], v["merge_size"] = p, merge
    v["size"]["longest_edge"] = int(v["size"]["longest_edge"] * (p * merge) ** 2 / oldside ** 2)
    v["size"]["shortest_edge"] = min(v["size"]["shortest_edge"], v["size"]["longest_edge"])
    json.dump(v, open(os.path.join(dst, "video_preprocessor_config.json"), "w"), indent=4)
print(f"wrote {dst}: image limit {pp['size']['longest_edge']} pixels, {args.max_tokens} tokens", flush=True)
