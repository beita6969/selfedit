"""PILOT op "zoom" (coordinator 11-11), pilot-only code (engine_v4 untouched). Two E-path calls per read:
 loc: tip view of the row + one fixed sentence asking for the one region to inspect as a box (multi-image: also the image number);
      max_tokens LOC_MAX (thinking capped by the total budget); seed = crn(row, sample).
 ans: crop from the ORIGINAL file (pad 20 %/side, min side 1/8 of the image side, no zoom if the box exceeds 2/3 of a side or is
      unparsable or the request does not fit 46000), crop appended as an extra image (min_image_tokens as the tip view: 4096) and one
      fixed sentence appended; normal budget; same seed. No zoom -> plain tip read (same seed).
 Twin rows (zloc/<row>, zans/<row>/s<k>) are appended to v4/rows/learn.jsonl (backup learn.jsonl.pre_zoom_1111 before the first append).
usage: zoom.py loc NAME SIDE SAMPLES PRIO [ROWS.json]      (SIDE ood|sig; tip view from scratch/ext/tip_<side>.jsonl)
       zoom.py ans LOCNAME NAME FRAME PRIO_SINGLE PRIO_MULTI   (FRAME norm|abs)
       zoom.py draw LOCNAME OUTDIR N"""
import hashlib, json, math, os, re, shutil, sys, time, zlib
sys.path.insert(0, "/workspace/fudan/mingdazhang/selfedit-v4/engine_v4")
import common
V4 = "/data/fd/mingdazhang/selfedit/v4"; F = f"{V4}/rows/learn.jsonl"; X = f"{V4}/scratch/ext"; Z = f"{V4}/scratch/zoom"
os.makedirs(f"{Z}/crops", exist_ok=True)
MAXLEN, RMAX, LOC_MAX = 46000, 32768, int(os.environ.get('ZOOM_LOC_MAX', 3072))
LOC1 = "Before answering, give the one region of the image that must be inspected most closely to answer this question, as a bounding box [x1, y1, x2, y2]. Reply with the box only."
LOCN = "Before answering, give the one region of the images that must be inspected most closely to answer this question, as the image number and a bounding box: image k, [x1, y1, x2, y2]. Reply with these only."
crn = lambda row, s: zlib.crc32(f"101|0|{row}".encode()) % (2 ** 31 - 1024) + int(s)
lst = lambda v: v if isinstance(v, list) else ([v] if v else [])
def rows_all(): return {r["id"]: r for r in map(json.loads, open(F))}
def append(twins):
    B = f"{V4}/rows/learn.jsonl.pre_zoom_1111"
    if not os.path.exists(B): shutil.copy2(F, B)
    L = open(F).read().splitlines(); have = {json.loads(l)["id"] for l in L}
    new = [t for t in twins if t["id"] not in have]
    open(F + ".tmp", "w").write("\n".join(L + [json.dumps(t) for t in new]) + "\n"); os.replace(F + ".tmp", F)
    print(f"appended {len(new)} twin rows; learn.jsonl now {len(L) + len(new)} rows", flush=True); time.sleep(3)
def put(name, Q):
    assert len({q["key"] for q in Q}) == len(Q)
    p = f"{V4}/q/in/{name}.jsonl"; open(p + ".tmp", "w").write("".join(json.dumps(x) + "\n" for x in Q)); os.replace(p + ".tmp", p)
    open(f"{X}/submit_times.jsonl", "a").write(json.dumps(dict(batch=name, t_submit=time.time(), n=len(Q))) + "\n"); print(name, len(Q), flush=True)
def interleaved(r): return r["prompt"].count("<image>") == len(lst(r["image_paths"])) > 0
def sent_dims(path, st):
    from PIL import Image
    with Image.open(path) as im: w, h = im.size
    mn, mx = common.view_pixels(st.get("min_image_tokens"), st.get("max_image_tokens"))
    hb, wb = common.smart_resize(h, w, min_pixels=mn, max_pixels=mx); return (w, h), (wb, hb)

def loc(name, side, samples, prio, rowsel=None):
    R = rows_all(); T = [json.loads(l) for l in open(f"{X}/tip_{side}.jsonl")]
    T = [q for q in T if q["sample"] in samples and (rowsel is None or q["row"] in rowsel)]
    twins, Q = {}, []
    for q in T:
        r = R[q["row"]]; multi = len(lst(r["image_paths"])) > 1
        twins[q["row"]] = dict(r, id="zloc/" + r["id"], v4_role="pilot_zoom", hb_role="pilot_zoom", zoom_src=r["id"], prompt=r["prompt"] + "\n\n" + (LOCN if multi else LOC1))
        Q.append(dict(q, key=f"zl|{name}|{q['row']}|{q['sample']}", row="zloc/" + q["row"], engine="E", priority=prio if not multi else prio + 1,
                      seed=crn(q["row"], q["sample"]), max_tokens=LOC_MAX, zoom_src=q["row"]))
    append(list(twins.values())); put(name, sorted(Q, key=lambda q: q["priority"]))
    json.dump(dict(side=side, samples=samples, tipfile=f"{X}/tip_{side}.jsonl"), open(f"{Z}/{name}.cfg.json", "w"))

NUM = r"(-?\d+(?:\.\d+)?)"
BOX = re.compile(r"\[\s*" + r"\s*,\s*".join([NUM] * 4) + r"\s*\]")
def parse(x, multi):
    for txt, src in ((x.get("content") or "", "content"), ((x.get("reasoning") or "")[-1500:], "reasoning")):
        m = list(BOX.finditer(txt))
        if m:
            b = [float(v) for v in m[-1].groups()]; k = None
            if multi:
                mk = re.findall(r"image\s*(\d+)", txt[: m[-1].start()], re.I); k = int(mk[-1]) if mk else None
            return b, k, src
    return None, None, None

def geometry(b, frame, orig, sent):
    W, H = orig
    if frame == "auto":   # smoke 11-11: the checkpoint answers in 0-1000 units; a few boxes exceed 1000 (pixels of the image as sent) or are 0-1 fractions
        frame = "abs" if max(b) > 1000 else ("frac" if max(b) <= 1.0 else "norm")
    if frame == "frac": sx, sy = W, H
    elif frame == "norm": sx, sy = W / 1000, H / 1000
    else: sx, sy = W / sent[0], H / sent[1]
    x1, y1, x2, y2 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
    x1, x2 = sorted((max(0, x1), min(W, x2))); y1, y2 = sorted((max(0, y1), min(H, y2)))
    return x1, y1, x2, y2

def ans(locname, name, frame, ps, pm):
    from PIL import Image
    cfg = json.load(open(f"{Z}/{locname}.cfg.json")); R = rows_all()
    T = {(q["row"], q["sample"]): q for q in map(json.loads, open(cfg["tipfile"]))}
    out = {}
    for l in open(f"{V4}/q/out/{locname}.jsonl"):
        x = json.loads(l); k = x["key"].split("|"); out[(k[2], int(k[3]))] = x
    twins, Q, meta = [], [], []
    for (row, s), x in sorted(out.items()):
        r = R[row]; ims = lst(r["image_paths"]); multi = len(ims) > 1; q = T[(row, s)]; st = q.get("settings") or {}
        m = dict(row=row, sample=s, loc_tokens=x.get("tokens"), loc_prompt_tokens=x.get("prompt_tokens"), loc_finish=x.get("finish"), frame=frame)
        b, kimg, src = (None, None, None) if x.get("error") else parse(x, multi); m.update(box=b, img=kimg, src=src)
        why = None
        if b is None: why = "unparsable"
        elif multi and (kimg is None or not 1 <= kimg <= len(ims)): why = "no image number"
        if why is None:
            path = ims[(kimg or 1) - 1]; orig, sent = sent_dims(path, st); W, H = orig
            x1, y1, x2, y2 = geometry(b, frame, orig, sent); bw, bh = x2 - x1, y2 - y1
            m.update(area=(bw * bh) / (W * H) if bw > 0 and bh > 0 else 0.0, orig=orig, sent=sent)
            if bw <= 0 or bh <= 0: why = "degenerate box"
            elif bw > 2 * W / 3 or bh > 2 * H / 3: why = "box > 2/3"
        if why is None:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; pw, ph = max(bw * 1.4, W / 8), max(bh * 1.4, H / 8)
            c = [max(0, cx - pw / 2), max(0, cy - ph / 2), min(W, cx + pw / 2), min(H, cy + ph / 2)]
            cp = f"{Z}/crops/{name}__{row.replace('/', '__')}__s{s}.png"
            common.load_rgb(path).crop(tuple(int(round(v)) for v in c)).save(cp)
            ntok = sum(common.image_url_for(p, st.get("min_image_tokens"), st.get("max_image_tokens"))[1] for p in ims + [cp])
            need = ntok + len(r["prompt"]) // 3 + 200 + len(q.get("prefix") or "") // 3 + 64 + RMAX
            m.update(crop=c, crop_path=cp, need=need)
            if need > MAXLEN: why = "does not fit"
        bs = ", ".join(f"{v:g}" for v in b) if b else ""
        if why is None:
            sent_txt = f"The last image is an enlarged crop of the region [{bs}] of " + (f"image {kimg} above." if multi else "the image above.")
            tid = f"zans/{row}/s{s}/{name}"
            p2 = r["prompt"] + ("\n\n<image>\n" if interleaved(r) else "\n\n") + sent_txt
            twins.append(dict(r, id=tid, v4_role="pilot_zoom", hb_role="pilot_zoom", zoom_src=row, prompt=p2, image_paths=ims + [cp]))
            Q.append(dict(q, key=f"za|{name}|{row}|{s}", row=tid, engine="E", priority=pm if multi else ps, seed=crn(row, s), zoom=True))
        else:
            Q.append(dict(q, key=f"za|{name}|{row}|{s}", row=row, engine="E", priority=pm if multi else ps, seed=crn(row, s), zoom=False))
        m.update(zoom=why is None, why=why); meta.append(m)
    with open(f"{Z}/{name}_meta.jsonl", "w") as f:
        for m in meta: f.write(json.dumps(m) + "\n")
    import collections
    print(f"{name}: reads {len(meta)} zoom {sum(m['zoom'] for m in meta)}; no zoom {dict(collections.Counter(m['why'] for m in meta if not m['zoom']))}", flush=True)
    append(twins); put(name, sorted(Q, key=lambda q: q["priority"]))

def draw(locname, outdir, n):
    from PIL import Image, ImageDraw
    cfg = json.load(open(f"{Z}/{locname}.cfg.json")); R = rows_all(); os.makedirs(outdir, exist_ok=True)
    T = {(q["row"], q["sample"]): q for q in map(json.loads, open(cfg["tipfile"]))}
    k = 0
    for l in open(f"{V4}/q/out/{locname}.jsonl"):
        x = json.loads(l); kk = x["key"].split("|"); row, s = kk[2], int(kk[3]); r = R[row]; ims = lst(r["image_paths"])
        b, kimg, src = parse(x, len(ims) > 1)
        if b is None: print(row, "unparsable", repr((x.get("content") or "")[:120])); continue
        path = ims[(kimg or 1) - 1]; orig, sent = sent_dims(path, T[(row, s)].get("settings") or {})
        im = common.load_rgb(path).copy(); d = ImageDraw.Draw(im); lw = max(2, orig[0] // 200)
        for frame, col in (("auto", (255, 0, 0)),):
            g = geometry(b, frame, orig, sent); d.rectangle(g, outline=col, width=lw)
        sc = 768 / max(orig); im = im.resize((max(1, int(orig[0] * sc)), max(1, int(orig[1] * sc))))
        fn = f"{outdir}/{k:02d}_{row.replace('/', '__')}.jpg"; im.save(fn, quality=85)
        print(f"{fn}: box {b} img {kimg} src {src} orig {orig} sent {sent} tokens {x.get('tokens')} finish {x.get('finish')} q: {r['prompt'][:110]!r}")
        k += 1
        if k >= n: break

if __name__ == "__main__":
    a = sys.argv
    if a[1] == "loc": loc(a[2], a[3], [int(v) for v in a[4].split(",")], int(a[5]), set(json.load(open(a[6]))) if len(a) > 6 else None)
    elif a[1] == "ans": ans(a[2], a[3], a[4], int(a[5]), int(a[6]))
    elif a[1] == "draw": draw(a[2], a[3], int(a[4]))
