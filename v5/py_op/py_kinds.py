"""What the model does with Python (from the compact PY reads): kinds of code by keyword, outcome per kind, per benchmark category, and
the row-level movement against the base (8 cached samples).   usage: py_kinds.py PY_COMPACT.jsonl[.gz] BASE.jsonl.gz LEARN.jsonl"""
import collections
import gzip
import json
import re
import sys

KINDS = [
    ("show / crop (look again)", r"\bshow\s*\(|\.show\s*\("),
    ("colour or grey threshold mask", r"[<>]=?\s*\d{2,3}\b|inRange|threshold\(|\bmask\b"),
    ("connected components / contours", r"ndimage\.label|\blabel\(|connectedComponents|findContours|find_objects"),
    ("row / column profile", r"\.(?:sum|mean|any|max|min|all)\(\s*axis|axis\s*=\s*[01]\)"),
    ("line / circle / angle fit", r"polyfit|HoughLines|HoughCircles|arctan|atan2|minAreaRect|fitLine|fitEllipse|slope"),
    ("edges / gradients", r"Canny|sobel|Sobel|FIND_EDGES|laplace|Laplacian|gradient\("),
    ("flip / rotate / compare shapes", r"\[\s*:\s*,\s*::-1\s*\]|\[::-1\]|fliplr|flipud|FLIP_|matchTemplate|absdiff|\biou\b|\.rotate\(|rot90|transpose\("),
    ("morphology / distance / fill", r"binary_(?:opening|closing|dilation|erosion|fill_holes)|morphologyEx|distance_transform|dilate\(|erode\(|floodFill"),
    ("draws on the image", r"ImageDraw|\.line\(|\.rectangle\(|\.ellipse\(|cv2\.(?:line|rectangle|circle|putText)\("),
    ("pixel sampling (getpixel / indexing a point)", r"getpixel|\ba\[\s*\d+\s*,\s*\d+\s*\]|\barr\[\s*\d+\s*,\s*\d+\s*\]"),
    ("counts unique colours / histogram", r"np\.unique|histogram|getcolors|Counter\("),
]


def opn(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def main():
    py_p, base_p, rows_p = sys.argv[1:4]
    rows = {r["id"]: r for r in map(json.loads, open(rows_p)) if r.get("v4_role") in ("show", "monitor")}
    base = collections.defaultdict(list)
    for l in opn(base_p):
        v = json.loads(l)
        if v["row"] in rows:
            base[v["row"]].append(v)
    b_ok = {r: sum(bool(v["ok"]) for v in b) / len(b) for r, b in base.items()}
    b_maj = {r: collections.Counter(str(v.get("pred")) for v in b).most_common(1)[0][0] for r, b in base.items()}
    R = [x for x in map(json.loads, opn(py_p)) if not x.get("error") and x.get("ok") is not None and x["row"] in rows]
    R = list({(x["row"], x["sample"]): x for x in R}.values())
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    line = lambda lab, rr: print(f"{lab:48s} reads {len(rr):4d} rows {len({x['row'] for x in rr}):3d}  PY {100 * mean([float(x['ok']) for x in rr]):6.1f}  "
                                 f"base8 {100 * mean([b_ok[x['row']] for x in rr]):6.1f}  diff {100 * (mean([float(x['ok']) for x in rr]) - mean([b_ok[x['row']] for x in rr])):+6.1f}") if rr else None

    print("== kinds of code (a read counts for every kind any of its blocks matches)")
    code_of = lambda x: "\n".join(u["code"] for u in x.get("runs", []))
    ran = [x for x in R if x.get("n_runs")]
    for name, pat in KINDS:
        line(name, [x for x in ran if re.search(pat, code_of(x))])
    size_only = [x for x in ran if all(len(u["code"]) < 260 and re.search(r"\.size|\.shape", u["code"]) and not re.search(r"crop|show|where|label|sum\(|mean\(", u["code"]) for u in x["runs"])]
    line("only prints sizes / shapes", size_only)
    no_img = [x for x in ran if not re.search(r"\bimgs?\b|\bim\b|\barr\b|\ba\b", code_of(x))]
    line("code never touches the image", no_img)
    line("no code at all", [x for x in R if not x.get("n_runs")])
    print("\n== lines of code and output per run")
    runs = [u for x in ran for u in x["runs"]]
    if runs:
        ln = sorted(len(u["code"].splitlines()) for u in runs)
        so = sorted(len(u["stdout"]) for u in runs)
        print(f"runs {len(runs)}: code lines median {ln[len(ln) // 2]}, p90 {ln[int(.9 * len(ln))]}; printed chars median {so[len(so) // 2]}, p90 {so[int(.9 * len(so))]}; "
              f"runs with no printed text and no image {100 * mean([float(not u['stdout'].strip() and not u.get('n_sent')) for u in runs]):.1f}%; "
              f"cpu s median {sorted((u.get('cpu') or 0) for u in runs)[len(runs) // 2]}, max {max((u.get('cpu') or 0) for u in runs)}; wall max {max((u.get('wall') or 0) for u in runs)}")
        print("status:", dict(collections.Counter(u["status"] for u in runs)))
        print("refusal texts:", dict(collections.Counter(re.sub(r"line \d+: ", "", (u.get("error") or ""))[:80] for u in runs if u["status"] == "refused").most_common(10)))
        print("error types:", dict(collections.Counter((u.get("error") or "").split(":")[0][:40] for u in runs if u["status"] == "error").most_common(12)))
    print("\n== by benchmark category (top level)")
    cats = collections.defaultdict(list)
    for x in R:
        r = rows[x["row"]]
        c = str(r.get("category") or "").split("/")[0]
        cats[(r["bench"], c)].append(x)
    for (b, c), rr in sorted(cats.items()):
        line(f"{b}: {c}"[:48], rr)
    print("\n== row-level movement (PY reads against the base's 8 samples of the same row)")
    for lab, f in (("base8 = 0 (always wrong)", lambda v: v == 0), ("base8 in (0, .5)", lambda v: 0 < v < .5), ("base8 in [.5, 1)", lambda v: .5 <= v < 1), ("base8 = 1 (always right)", lambda v: v == 1)):
        line(lab, [x for x in R if f(b_ok[x["row"]])])
    same = [x for x in R if str(x.get("pred")) == b_maj[x["row"]]]
    print(f"PY answer = the base's majority answer: {100 * len(same) / len(R):.1f}% of reads; when different: PY right {100 * mean([float(x['ok']) for x in R if str(x.get('pred')) != b_maj[x['row']]]):.1f}%, "
          f"base majority right on those rows {100 * mean([float(any(v['ok'] and str(v.get('pred')) == b_maj[x['row']] for v in base[x['row']])) for x in R if str(x.get('pred')) != b_maj[x['row']]]):.1f}%")
    byrow = collections.defaultdict(list)
    for x in R:
        byrow[x["row"]].append(x)
    gain = sorted(((mean([float(x["ok"]) for x in v]) - b_ok[r], r) for r, v in byrow.items()), reverse=True)
    print("rows with the largest gain:", [(r, round(d, 2)) for d, r in gain[:14]])
    print("rows with the largest loss:", [(r, round(d, 2)) for d, r in gain[-14:]])
    print(f"rows up by >= .5: {sum(d >= .5 for d, _ in gain)}; rows down by >= .5: {sum(d <= -.5 for d, _ in gain)}; of {len(gain)}")


if __name__ == "__main__":
    main()
