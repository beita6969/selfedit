"""Build the multimodal test sets (DATA/sets) and disjoint edit-selection pools (DATA/pool) with images saved under DATA/images.

Every item: {id, bench, kind (mc|num|yesno|short), prompt, image_paths, answer, options, category, group, img_sha256, img_dhash, ...}.
`prompt` may contain "<image>" placeholders (one per image, ERQA interleaving); otherwise images go before the text.
Split rule: items are grouped into connected components by shared source group (pair / figure set / sample), image
sha256 and 256-bit dHash; whole components go to test or pool (seeded, stratified); a final global pass drops any pool
item whose image sha256 or dHash appears in ANY test set. Test files are written in stratified interleaved order, so
the first N lines are a stratified N-item subset (gen_mm.py --limit N).
Usage: python prep_mm.py [DATA_DIR] [--only bench1,bench2]
"""
import argparse, base64, collections, concurrent.futures as cf, hashlib, io, json, os, random, re, time, urllib.error, urllib.request
import pandas as pd
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("data", nargs="?", default="/data/fd/mingdazhang/selfedit/data/mm")
ap.add_argument("--only", default="")
args = ap.parse_args()
DATA = args.data
OUT = f"{DATA}/debug" if args.only else DATA  # partial (--only) builds never overwrite the official sets/pools
for d in ("sets", "pool"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
for d in ("images", "raw"):
    os.makedirs(f"{DATA}/{d}", exist_ok=True)
SEED = 20261103
Image.MAX_IMAGE_PIXELS = None

SUF = {"mc": "Answer with the letter of the correct option. Put only that letter in \\boxed{}.",
       "num": "Answer with a number. Put the final number in \\boxed{}.",
       "yesno": "Answer yes or no. Put your final answer (yes or no) in \\boxed{}.",
       "short": "Put your final answer in \\boxed{}, using the answer format the question asks for."}
LET = "ABCDEFGHIJ"
REPOS = {  # repo -> files to fetch (None = whole repo with allow_patterns)
    "UnipatAI/BabyVision": ["data/train-00000-of-00001.parquet"],
    "FlagEval/ERQA": ["data/test-00000-of-00001.parquet"],
    "BLINK-Benchmark/BLINK": ["*/val-00000-of-00001.parquet"],
    "nyu-visionx/CV-Bench": ["test_2d.parquet", "test_3d.parquet"],
    "MMVP/MMVP": ["Questions.csv", "MMVP Images/*"],
    "craigwu/vstar_bench": ["test_questions.jsonl", "direct_attributes/*", "relative_position/*"],
    "lmms-lab-encoder/HallusionBench": ["data/image-00000-of-00001.parquet"],
    "vikhyatk/CountBenchQA": ["data/test-00000-of-00001.parquet"],
    "allenai/pixmo-count": ["data/validation-00000-of-00001.parquet"],
    "xai-org/RealworldQA": ["data/*.parquet"],
    "Lin-Chen/MMStar": ["mmstar.parquet"],
    "AI4Math/MathVista": ["data/testmini-*.parquet"],
    "princeton-nlp/CharXiv": ["val.parquet"],
    "DreamMr/HR-Bench": ["hr_bench_4k.parquet"],
    "BaiqiL/NaturalBench": ["data/train-*.parquet"],
}
MANIFEST = {"seed": SEED, "revisions": {}, "benchmarks": {}, "dropped": {}}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fetch(repo):
    api = HfApi()
    sha = api.dataset_info(repo).sha
    MANIFEST["revisions"][repo] = sha
    d = snapshot_download(repo, repo_type="dataset", revision=sha, allow_patterns=REPOS[repo],
                          local_dir=f"{DATA}/raw/{repo.replace('/', '__')}", max_workers=8)
    return d


# ---------------- images ----------------
def dhash256(im):
    g = im.convert("L").resize((17, 16), Image.LANCZOS)
    px = list(g.get_flattened_data()) if hasattr(g, "get_flattened_data") else list(g.getdata())
    bits = [px[r * 17 + c] > px[r * 17 + c + 1] for r in range(16) for c in range(16)]
    n = sum(bits)
    if n < 16 or n > 240:  # near-blank / degenerate images: dHash would collide spuriously
        return None
    return "%064x" % int("".join("1" if b else "0" for b in bits), 2)


def save_img(bench, key, data):
    """Write image bytes (JPEG/PNG kept as is, anything else re-encoded to PNG); return (path, sha256, dhash)."""
    if isinstance(data, dict):
        data = data.get("bytes")
    im = Image.open(io.BytesIO(data))
    fmt = (im.format or "").upper()
    if fmt not in ("JPEG", "PNG"):
        im.seek(0)
        im2 = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
        buf = io.BytesIO(); im2.save(buf, "PNG"); data = buf.getvalue(); fmt = "PNG"
    ext = "jpg" if fmt == "JPEG" else "png"
    os.makedirs(f"{DATA}/images/{bench}", exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9_.-]", "_", str(key))
    path = f"{DATA}/images/{bench}/{key}.{ext}"
    sha = hashlib.sha256(data).hexdigest()
    if not (os.path.exists(path) and hashlib.sha256(open(path, "rb").read()).hexdigest() == sha):
        tmp = path + ".tmp"  # write-then-rename: a reader (running gen_mm.py) never sees a half-written image
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    im.seek(0)
    return path, sha, dhash256(im)


def opts_block(options):
    return "\n\nOptions:\n" + "\n".join(f"({LET[i]}) {o}" for i, o in enumerate(options))


def mk(bench, sid, kind, text, imgs, answer, options=None, category=None, group=None, **extra):
    """imgs: list of raw image bytes (or {'bytes':...}); text: question (+options), the kind suffix is appended."""
    paths, shas, dh = [], [], []
    for k, b in enumerate(imgs):
        p, s, d = save_img(bench, f"{sid}_{k}", b)
        paths.append(p); shas.append(s); dh.append(d)
    prompt = text.rstrip() + "\n\n" + extra.pop("suffix", SUF[kind])
    return {"id": f"{bench}/{sid}", "bench": bench, "kind": kind, "prompt": prompt, "image_paths": paths,
            "answer": str(answer), "options": options or [], "category": category, "group": group,
            "img_sha256": shas, "img_dhash": dh, **extra}


# ---------------- split ----------------
def components(items):
    par = list(range(len(items)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    first = {}
    for i, it in enumerate(items):
        keys = [("g", it["group"])] if it.get("group") is not None else []
        keys += [("s", s) for s in it["img_sha256"]] + [("d", d) for d in it["img_dhash"] if d]
        for k in keys:
            if k in first:
                a, b = find(i), find(first[k])
                if a != b:
                    par[a] = b
            else:
                first[k] = i
    comp = collections.defaultdict(list)
    for i in range(len(items)):
        comp[find(i)].append(items[i])
    return list(comp.values())


def interleave(items, key, rng):
    """Stratified order: each stratum shuffled, then merged by fractional position so any prefix is ~stratified."""
    by = collections.defaultdict(list)
    for it in items:
        by[key(it)].append(it)
    out = []
    for s, lst in sorted(by.items(), key=lambda kv: str(kv[0])):
        rng.shuffle(lst)
        out += [((j + rng.random() * 0.01 + 0.5) / len(lst), it) for j, it in enumerate(lst)]
    return [it for _, it in sorted(out, key=lambda t: t[0])]


def split(items, key, n_test, n_pool, equal=False, rng=None):
    """Assign whole components to test/pool. equal=True: same quota per stratum (BLINK, CV-Bench); else proportional."""
    rng = rng or random.Random(SEED)
    comps = components(items)
    by = collections.defaultdict(list)
    for c in comps:
        by[key(c[0])].append(c)
    strata = sorted(by, key=str)
    tot = sum(len(c) for c in comps)

    def quotas(n, avail):
        if equal:
            q = {s: n // len(strata) for s in strata}
        else:
            raw = {s: n * avail[s] / max(1, sum(avail.values())) for s in strata}
            q = {s: int(raw[s]) for s in strata}
            for s in sorted(strata, key=lambda s: raw[s] - q[s], reverse=True)[: n - sum(q.values())]:
                q[s] += 1
        return q
    for s in strata:
        rng.shuffle(by[s])
    test, pool, rest = [], [], []
    avail = {s: sum(len(c) for c in by[s]) for s in strata}
    qt = quotas(min(n_test, tot), avail)
    left = {}
    for s in strata:
        got, rem = 0, []
        for c in by[s]:
            if got < qt[s]:
                test += c; got += len(c)
            else:
                rem.append(c)
        left[s] = rem
    avail = {s: sum(len(c) for c in left[s]) for s in strata}
    qp = quotas(min(n_pool, sum(avail.values())), avail)
    for s in strata:
        got = 0
        for c in left[s]:
            if got < qp[s]:
                pool += c; got += len(c)
            else:
                rest += c
    return interleave(test, key, rng), interleave(pool, key, rng), rest


# ---------------- answer helpers ----------------
def canon_gold(s):
    s = s.strip().replace("−", "-")
    if "=" in s:
        s = s.split("=")[-1].strip()
    return s.rstrip(".").strip()


def blank_kind(g):
    """BabyVision blank answers: numbers -> num; short structured strings -> short; long free text -> None (not auto-gradable)."""
    if re.fullmatch(r"-?\d+(\.\d+)?", g):
        return "num"
    words = re.findall(r"[A-Za-z]{2,}", g)
    filler = {"row", "column", "col", "and", "the", "th", "st", "nd", "rd"}
    if len([w for w in words if w.lower() not in filler]) > 6:
        return None
    return "short"


def num_str(x):
    try:
        f = float(str(x).replace(",", ""))
        return str(int(f)) if f == int(f) and "." not in str(x) else str(x).replace(",", "")
    except ValueError:
        return None


# ---------------- loaders ----------------
def rp(repo, f):
    return f"{DATA}/raw/{repo.replace('/', '__')}/{f}"


def load_babyvision():
    df = pd.read_parquet(rp("UnipatAI/BabyVision", "data/train-00000-of-00001.parquet"))
    out, drop = [], 0
    for r in df.itertuples(index=False):
        cat = f"{r.type}/{r.subtype}"
        if r.ansType == "choice":
            opts = list(r.options)
            it = mk("babyvision", r.taskId, "mc", r.question + opts_block(opts), [r.image], LET[int(r.choiceAns)], opts, cat)
        else:
            g = canon_gold(r.blankAns)
            k = blank_kind(g)
            if k is None:
                drop += 1; continue
            it = mk("babyvision", r.taskId, k, r.question, [r.image], g, None, cat, gold_raw=r.blankAns)
        out.append(it)
    MANIFEST["dropped"]["babyvision_free_text_blank"] = drop
    return out


def load_erqa():
    df = pd.read_parquet(rp("FlagEval/ERQA", "data/test-00000-of-00001.parquet"))
    out = []
    for r in df.itertuples(index=False):
        q = re.sub(r"\s*Please answer directly with only the letter of the correct option and nothing else\.?\s*$", "", r.question)
        imgs = list(r.images)
        idx = list(r.visual_indices) if r.visual_indices is not None else []
        if len(idx) != len(imgs):  # 183/400 rows have empty visual_indices: images go before the text
            idx = [0] * len(imgs)
        order = sorted(range(len(imgs)), key=lambda k: (idx[k], k))
        text, last = "", 0
        for k in order:
            p = min(int(idx[k]), len(q))
            text += q[last:p] + "<image>"; last = p
        text += q[last:]
        out.append(mk("erqa", r.question_id, "mc", text, [imgs[k] for k in order], r.answer.strip(), None, r.question_type))
    return out


BLINK_CFGS = ["Art_Style", "Counting", "Forensic_Detection", "Functional_Correspondence", "IQ_Test", "Jigsaw",
              "Multi-view_Reasoning", "Object_Localization", "Relative_Depth", "Relative_Reflectance",
              "Semantic_Correspondence", "Spatial_Relation", "Visual_Correspondence", "Visual_Similarity"]


def mc_prompt_strip(p):
    return re.sub(r"\s*Answer with the option'?s letter from the given choices directly\.?\s*$", "", p.strip())


def load_blink():
    out = []
    for c in BLINK_CFGS:
        df = pd.read_parquet(rp("BLINK-Benchmark/BLINK", f"{c}/val-00000-of-00001.parquet"))
        for r in df.itertuples(index=False):
            imgs = [x for x in (r.image_1, r.image_2, r.image_3, r.image_4) if x is not None and x.get("bytes")]
            out.append(mk("blink", r.idx, "mc", mc_prompt_strip(r.prompt), imgs, r.answer.strip("() "), list(r.choices), c))
    return out


def load_cvbench():
    out = []
    for f in ("test_2d.parquet", "test_3d.parquet"):
        df = pd.read_parquet(rp("nyu-visionx/CV-Bench", f))
        for r in df.itertuples(index=False):
            out.append(mk("cvbench", f"{r.type}_{r.idx}", "mc", mc_prompt_strip(r.prompt), [r.image], r.answer.strip("() "),
                          list(r.choices), f"{r.type}/{r.task}", source=r.source))
    return out


def load_mmvp():
    df = pd.read_csv(rp("MMVP/MMVP", "Questions.csv"))
    df.columns = ["idx", "question", "options", "answer"]
    out = []
    for r in df.itertuples(index=False):
        opts = [o.strip() for o in re.split(r"\([a-z]\)", r.options) if o.strip()]
        ans = LET["abcdefghij".index(r.answer.strip("() ").lower())]
        img = open(rp("MMVP/MMVP", f"MMVP Images/{int(r.idx)}.jpg"), "rb").read()
        pair = (int(r.idx) - 1) // 2
        out.append(mk("mmvp", int(r.idx), "mc", r.question + opts_block(opts), [img], ans, opts, None, group=f"pair{pair}", pair=pair))
    return out


def load_vstar():
    out = []
    for l in open(rp("craigwu/vstar_bench", "test_questions.jsonl")):
        r = json.loads(l)
        lines = mc_prompt_strip(r["text"]).split("\n")
        opts = [re.sub(r"^\([A-J]\)\s*", "", x) for x in lines if re.match(r"^\([A-J]\)", x)]
        q = "\n".join(x for x in lines if not re.match(r"^\([A-J]\)", x))
        img = open(rp("craigwu/vstar_bench", r["image"]), "rb").read()
        out.append(mk("vstar", r["question_id"], "mc", q + opts_block(opts), [img], r["label"], opts, r["category"]))
    return out


def load_hrbench():
    df = pd.read_parquet(rp("DreamMr/HR-Bench", "hr_bench_4k.parquet"))
    rows, seen = [], set()
    for r in df.itertuples(index=False):
        key = (r.question, hashlib.md5(r.image.encode()).hexdigest())
        if key in seen:  # drop option-cycled duplicates of the same question/image
            continue
        seen.add(key); rows.append(r)
    MANIFEST["dropped"]["hrbench4k_duplicate_question_image"] = len(df) - len(rows)
    out = []
    for r in random.Random(f"{SEED}-hr").sample(rows, min(400, len(rows))):  # only the pool-sized sample is decoded/saved
        opts = [str(getattr(r, x)) for x in "ABCD"]
        img = base64.b64decode(r.image)
        out.append(mk("hrbench4k", r.index, "mc", r.question + opts_block(opts), [img], r.answer.strip(), opts, r.category))
    return out


def load_hallusion():
    df = pd.read_parquet(rp("lmms-lab-encoder/HallusionBench", "data/image-00000-of-00001.parquet"))
    out = []
    for r in df.itertuples(index=False):
        if r.image is None or not r.image.get("bytes"):
            continue
        sid = f"{r.category}_{r.subcategory}_{r.set_id}_{r.figure_id}_{r.question_id}"
        out.append(mk("hallusionbench", sid, "yesno", r.question, [r.image], "yes" if str(r.gt_answer) == "1" else "no",
                      None, f"{r.category}/{r.subcategory}", group=f"{r.category}_{r.subcategory}_{r.set_id}"))
    return out


def load_countbench():
    df = pd.read_parquet(rp("vikhyatk/CountBenchQA", "data/test-00000-of-00001.parquet"))
    return [mk("countbench", i, "num", r.question, [r.image], int(r.number), None, str(int(r.number)))
            for i, r in enumerate(df.itertuples(index=False))]


def load_pixmo():
    """PixmoCount val images come from Flickr URLs. Most bytes differ from the published sha256 (Flickr re-encodes the
    same photo), so any image that decodes is kept and `sha_match` is recorded; 429s are retried slowly."""
    df = pd.read_parquet(rp("allenai/pixmo-count", "data/validation-00000-of-00001.parquet"))

    def get(r):
        for attempt in range(4):
            try:
                req = urllib.request.Request(r.image_url, headers={"User-Agent": "Mozilla/5.0"})
                b = urllib.request.urlopen(req, timeout=30).read()
                Image.open(io.BytesIO(b)).verify()
                return b
            except urllib.error.HTTPError as e:
                if e.code != 429:
                    return None
                time.sleep(5 * (attempt + 1))
            except Exception:
                return None
        return None
    rows = list(df.itertuples(index=False))
    with cf.ThreadPoolExecutor(4) as ex:
        blobs = list(ex.map(get, rows))
    out, bad = [], 0
    for i, (r, b) in enumerate(zip(rows, blobs)):
        if b is None:
            bad += 1; continue
        out.append(mk("pixmocount", i, "num", f"How many {r.label} are there in the image?", [b], int(r.count), None,
                      str(min(int(r.count), 11)), sha_match=hashlib.sha256(b).hexdigest() == r.image_sha256))
    MANIFEST["dropped"]["pixmocount_url_fail"] = bad
    return out


def load_rwqa():
    out = []
    for f in sorted(os.listdir(rp("xai-org/RealworldQA", "data"))):
        df = pd.read_parquet(rp("xai-org/RealworldQA", f"data/{f}"))
        for r in df.itertuples(index=False):
            q = re.sub(r"\s*Please answer directly with (only the letter of the correct option and nothing else|a single word or number)\.?\s*$",
                       "", r.question.strip())
            a = r.answer.strip().rstrip(".")
            has_opts = bool(re.search(r"^[A-F][.)]\s", q, re.M))
            if has_opts and re.fullmatch(r"[A-F]", a):
                kind = "mc"
            elif a.lower() in ("yes", "no"):
                kind, a = "yesno", a.lower()
            elif num_str(a) is not None:
                kind, a = "num", num_str(a)
            else:
                kind = "short"
            sid = len(out)
            out.append(mk("realworldqa", sid, kind, q, [r.image], a, None, kind))
    return out


def load_mmstar():
    df = pd.read_parquet(rp("Lin-Chen/MMStar", "mmstar.parquet"))
    return [mk("mmstar", r.index, "mc", r.question, [r.image], r.answer.strip(), None, r.category,
               source=(r.meta_info or {}).get("source")) for r in df.itertuples(index=False)]


def load_mathvista():
    import glob
    out, drop = [], 0
    for f in sorted(glob.glob(rp("AI4Math/MathVista", "data/testmini-*.parquet"))):
        df = pd.read_parquet(f)
        for r in df.itertuples(index=False):
            unit = r.unit if isinstance(r.unit, str) and r.unit.strip() else None  # 950/1000 units are NaN
            q = r.question + (f" (Unit: {unit})" if unit and "(Unit:" not in r.question else "")
            cat = (r.metadata or {}).get("category")
            if r.question_type == "multi_choice":
                opts = list(r.choices)
                if r.answer not in opts:
                    drop += 1; continue
                out.append(mk("mathvista", r.pid, "mc", q + opts_block(opts), [r.decoded_image], LET[opts.index(r.answer)], opts, cat))
            elif r.answer_type == "integer":
                out.append(mk("mathvista", r.pid, "num", q, [r.decoded_image], r.answer, None, cat,
                              suffix="Answer with an integer. Put the final number in \\boxed{}."))
            elif r.answer_type == "float" and r.precision is not None and r.precision == r.precision:
                p = int(r.precision)
                out.append(mk("mathvista", r.pid, "num", q, [r.decoded_image], r.answer, None, cat, precision=p,
                              suffix=f"Answer with a number with {p} decimal place{'s' if p != 1 else ''}. Put the final number in \\boxed{{}}."))
            else:
                drop += 1
    MANIFEST["dropped"]["mathvista_list_or_text"] = drop
    return out


def load_charxiv():
    df = pd.read_parquet(rp("princeton-nlp/CharXiv", "val.parquet"))
    out = []
    for n, r in enumerate(df.itertuples(index=False)):
        if int(r.reasoning_a_type) not in (3, 4):  # 3 number-in-chart, 4 number-in-general; text answers need an LLM judge
            continue
        a = num_str(r.reasoning_a.strip())
        if a is None:
            continue
        out.append(mk("charxiv", r.original_id if r.original_id else n, "num", r.reasoning_q.strip(), [r.image], a, None,
                      {3: "number-in-chart", 4: "number-in-general"}[int(r.reasoning_a_type)], group=r.original_figure_path))
    return out


def load_naturalbench():
    import glob
    out = []
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(rp("BaiqiL/NaturalBench", "data/train-*.parquet")))])
    pick = set(random.Random(f"{SEED}-nb").sample(sorted(df["Index"].tolist()), 110))  # 110 samples x 4 questions
    for r in df.itertuples(index=False):
        if r.Index not in pick:
            continue
        if True:
            d = r._asdict()
            for i in (0, 1):
                for j in (0, 1):
                    a = str(d[f"Image_{i}_Question_{j}"]).strip()
                    q = d[f"Question_{j}"]
                    if r.Question_Type == "yes_no":
                        kind, a = "yesno", a.lower()
                    else:
                        kind = "mc"
                        if not re.fullmatch(r"[A-D]", a):
                            continue
                    out.append(mk("naturalbench", f"{r.Index}_i{i}q{j}", kind, q, [d[f"Image_{i}"]], a, None, r.Question_Type,
                                  group=f"nb{r.Index}"))
    return out


def dump(kind, name, rows):
    with open(f"{OUT}/{kind}/{name}.jsonl.tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(f"{OUT}/{kind}/{name}.jsonl.tmp", f"{OUT}/{kind}/{name}.jsonl")
    log(kind, name, len(rows), dict(collections.Counter(r["category"] for r in rows).most_common(20)))


# bench -> (loader, role, n_test, n_pool, stratum key, equal quotas)
PLAN = {
    "babyvision": (load_babyvision, "split", 200, 400, lambda it: it["category"], False),
    "erqa": (load_erqa, "split", 200, 200, lambda it: it["category"], False),
    "blink": (load_blink, "split", 420, 420, lambda it: it["category"], True),
    "cvbench": (load_cvbench, "split", 400, 400, lambda it: it["category"], True),
    "mmvp": (load_mmvp, "test", 300, 0, None, False),
    "naturalbench": (load_naturalbench, "pool_for:mmvp", 0, 400, lambda it: it["category"], False),
    "vstar": (load_vstar, "test", 191, 0, None, False),
    "hrbench4k": (load_hrbench, "pool_for:vstar", 0, 400, lambda it: it["category"], False),
    "hallusionbench": (load_hallusion, "split", 450, 450, lambda it: it["category"], False),
    "countbench": (load_countbench, "test", 491, 0, None, False),
    "pixmocount": (load_pixmo, "pool_for:countbench", 0, 540, lambda it: it["category"], False),
    "realworldqa": (load_rwqa, "split", 400, 365, lambda it: it["category"], False),
    "mmstar": (load_mmstar, "split", 400, 400, lambda it: it["category"], False),
    "mathvista": (load_mathvista, "split", 500, 500, lambda it: it["category"], False),
    "charxiv": (load_charxiv, "split", 231, 230, lambda it: it["category"], False),
}
REPO_OF = {"babyvision": "UnipatAI/BabyVision", "erqa": "FlagEval/ERQA", "blink": "BLINK-Benchmark/BLINK",
           "cvbench": "nyu-visionx/CV-Bench", "mmvp": "MMVP/MMVP", "naturalbench": "BaiqiL/NaturalBench",
           "vstar": "craigwu/vstar_bench", "hrbench4k": "DreamMr/HR-Bench", "hallusionbench": "lmms-lab-encoder/HallusionBench",
           "countbench": "vikhyatk/CountBenchQA", "pixmocount": "allenai/pixmo-count", "realworldqa": "xai-org/RealworldQA",
           "mmstar": "Lin-Chen/MMStar", "mathvista": "AI4Math/MathVista", "charxiv": "princeton-nlp/CharXiv"}

only = [b for b in args.only.split(",") if b] or list(PLAN)
with cf.ThreadPoolExecutor(6) as ex:
    futs = {ex.submit(fetch, REPO_OF[b]): b for b in only}
    for fu in cf.as_completed(futs):
        try:
            fu.result(); log("fetched", futs[fu])
        except Exception as e:
            log("FETCH FAILED", futs[fu], repr(e)[:300])

tests, pools = {}, {}
for b in only:
    loader, role, nt, npl, key, eq = PLAN[b]
    try:
        items = loader()
    except Exception as e:
        log("BUILD FAILED", b, repr(e)[:500]); continue
    log("loaded", b, len(items))
    rng = random.Random(f"{SEED}-{b}")
    if role == "split":
        t, p, rest = split(items, key, nt, npl, eq, rng)
        tests[b], pools[b] = t, p
        MANIFEST["benchmarks"][b] = {"repo": REPO_OF[b], "rule": f"held-out split by image/group components, stratified by category"
                                     f"{' (equal quota)' if eq else ''}", "loaded": len(items), "unused": len(rest)}
    elif role == "test":
        tests[b] = interleave(items, lambda it: it["category"], rng)
        MANIFEST["benchmarks"][b] = {"repo": REPO_OF[b], "rule": "full benchmark as test", "loaded": len(items)}
    else:
        fam = role.split(":")[1]
        _, p, rest = split(items, key, 0, npl, False, rng)
        pools[b] = p
        MANIFEST["benchmarks"][b] = {"repo": REPO_OF[b], "rule": f"pool for the {fam} family (different source dataset)",
                                     "loaded": len(items), "unused": len(rest)}

# global disjointness: drop pool items sharing any image hash with ANY test item (also across benchmarks)
if True:
    tsha = {s for t in tests.values() for it in t for s in it["img_sha256"]}
    tdh = {d for t in tests.values() for it in t for d in it["img_dhash"] if d}
    tids = {it["id"] for t in tests.values() for it in t}
    for b, p in pools.items():
        keep = [it for it in p if it["id"] not in tids and not (set(it["img_sha256"]) & tsha)
                and not ({d for d in it["img_dhash"] if d} & tdh)]
        MANIFEST["dropped"][f"pool_{b}_hash_overlap_with_tests"] = len(p) - len(keep)
        pools[b] = keep
for b, t in tests.items():
    dump("sets", b, t); MANIFEST["benchmarks"][b]["test"] = len(t)
for b, p in pools.items():
    dump("pool", b, p); MANIFEST["benchmarks"][b]["pool"] = len(p)
mf = f"{OUT}/manifest.json"
json.dump(MANIFEST, open(mf, "w"), indent=1)
log("manifest", json.dumps(MANIFEST)[:3000])
