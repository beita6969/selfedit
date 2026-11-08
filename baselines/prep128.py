"""Baseline data for the rebuilt main-system pool (owner 11-08): each bench has exactly 256 rows, 128 training side (show 40 / score 56 / select 32)
and 128 IID evaluation. Baselines train only on the training-side rows, repeated as needed, and are evaluated on the IID rows (plus OOD at the end).
Writes to OUT_DIR:
  train.jsonl                 the training rows, gold included: all 768 training-side rows (default), or show + score only (--holdout-select)
  val.jsonl                   the select rows (192). Default: report-only training-fit curve (they are trained on). --holdout-select: held-out selection set
  paper_select.node.jsonl     the select rows; paper_learn.node.jsonl the show + score rows
  iid_eval.node.jsonl         the IID evaluation rows, read by bl_train.py only at the end (--final-iid)
  counts.json                 per bench and role, overlaps, file sha
Usage: prep128.py XFER_DIR OUT_DIR [--holdout-select]. XFER_DIR from bl_pexport128.py: paper_learn.jsonl, paper_select.jsonl, paper_iid.jsonl, paper_rest_idonly.json.
Images are remapped to the node and sha checked. Overlaps between training and IID rows (id, image sha, near dhash <= 8 of 256 bits, pic) are counted,
not removed: the split is the main system's, and the main system has the same exposure. Gold of the IID rows is never printed."""
import argparse, collections, hashlib, json, os
ap = argparse.ArgumentParser(); ap.add_argument("xfer"); ap.add_argument("out"); ap.add_argument("--holdout-select", action="store_true")
a = ap.parse_args(); X, O = a.xfer, a.out; os.makedirs(O, exist_ok=True)
N = "/home/bedicloud/sharestore2/zxc/selfedit"
rd = lambda p: [json.loads(l) for l in open(p)]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
C = lambda rows: dict(sorted(collections.Counter(r["bench"] for r in rows).items()))
part = dict(learn=rd(f"{X}/paper_learn.jsonl"), select=rd(f"{X}/paper_select.jsonl"), iid=rd(f"{X}/paper_iid.jsonl"))
rest = json.load(open(f"{X}/paper_rest_idonly.json")) if os.path.exists(f"{X}/paper_rest_idonly.json") else []
bad = collections.Counter()
for name, rows in part.items():
    for r in rows:
        r["image_paths"] = [p.replace("/data/fd/mingdazhang/selfedit", N) for p in r["image_paths"]]
        sh = r.get("img_sha256"); sh = sh if isinstance(sh, list) else [sh] * len(r["image_paths"])
        for p, h in zip(r["image_paths"], sh):
            if not os.path.exists(p):
                bad[name + " missing"] += 1
            elif h and sha(p) != h:
                bad[name + " sha differs"] += 1
print("image check", dict(bad) or "all images present, sha equal")
ids = {k: [r["id"] for r in v] for k, v in part.items()}
for k, v in ids.items():
    assert len(v) == len(set(v)), f"{k}: repeated ids"
L = lambda v: [x for x in (v if isinstance(v, list) else [v]) if x]
index = lambda rows: {k: {x for r in rows for x in L(r.get(k))} for k in ("id", "img_sha256", "img_dhash", "pic")}
near = lambda r, ds, b=8: any(bin(int(x, 16) ^ d).count("1") <= b for x in L(r.get("img_dhash")) for d in ds)
train = part["learn"] + ([] if a.holdout_select else part["select"])
over = {}
for m, oth in (("iid", part["iid"]), ("rest", rest)) + ((("select", part["select"]),) if a.holdout_select else ()):
    I, J = index(train), index(oth); ds = {int(x, 16) for x in J["img_dhash"]}
    over[f"train & {m}"] = dict({k: len(I[k] & J[k]) for k in I}, near8_rows=sum(near(r, ds) for r in train))
print("overlaps (counted, not removed):", json.dumps(over))
roles = collections.Counter((r["bench"], r.get("paper_role") or r.get("hb_role") or r.get("role") or "?") for r in part["learn"] + part["select"])
out = {"train.jsonl": train, "val.jsonl": part["select"], "paper_select.node.jsonl": part["select"], "paper_learn.node.jsonl": part["learn"], "iid_eval.node.jsonl": part["iid"]}
for f, rows in out.items():
    open(f"{O}/{f}", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
counts = dict(mode="holdout-select" if a.holdout_select else "train-all-768", train=C(train), train_total=len(train), val_select=C(part["select"]), iid=C(part["iid"]),
              roles={f"{b}/{x}": n for (b, x), n in sorted(roles.items())}, overlaps=over, images=dict(bad), sha={f: sha(f"{O}/{f}") for f in out})
json.dump(counts, open(f"{O}/counts.json", "w"), indent=1)
print("train by bench", counts["train"], "total", len(train)); print("val (select) by bench", counts["val_select"]); print("IID by bench", counts["iid"])
print("roles", counts["roles"])
for f in out:
    print(f, "sha", counts["sha"][f][:16])
