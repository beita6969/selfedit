"""Stage 1 inputs that are on the Mac (11-11): the JUDGE tables of the hand judge (tip E-cache samples 0-3 as base_*, W-null, the FL cell,
XS: 444 rows x 4 samples each, with pred) and the signal-row tables. Exports of the engine agent (~/selfedit/v5_exports) replace / extend
these when they arrive (load_exports)."""
import glob
import os

import reads5 as R

HOME = os.path.expanduser("~/selfedit")
LEARN = os.path.join(HOME, "code-v4", "data", "learn.jsonl")
W_JUDGE = os.path.join(HOME, "v4_reports", "w_judge")
W_SIGNAL = os.path.join(HOME, "v4_reports", "w_signal")
EXPORTS = os.path.join(HOME, "v5_exports")
OUT = os.path.join(HOME, "b5_restart", "stage1")
FL, XS, WNULL = "FLs8_L24-34_+0.75", "XS_bias_all_+0.75", "W-null"


def rows():
    return R.load_rows(LEARN)


def judge_tables(rows_):
    """-> arms {cell: arm}, tipE (4 E-cache samples), tip8 (E samples 0-3 + W-null as samples 4-7: the same tip on the two engines)."""
    arms, base = R.load_tsv(W_JUDGE, rows_)
    J = R.rows_of(rows_, "judge")
    tip8 = {r: list(base[r]) + [dict(x, s=x["s"] + 4, engine="W") for x in arms[WNULL][r]] for r in J}
    return arms, base, tip8


# ------------------------------------------------------------------------------------------------ exports of 11-11 (engine agent)
import gzip
import json

import policy5 as _P

E_SHA = {"e1_tip_t007c_learn_s0-7.jsonl.gz": "002bcad6", "e1_tip_fired_learn.jsonl.gz": "cc9babea", "e3_lrn_wo_and_tip_refs.jsonl.gz": "56567242",
         "e2_floor_t001_and_stock_learn.jsonl.gz": "82b4bd28", "e6_tip_entry_sources.jsonl.gz": "f299fd48", "e4_W_arms_signal.jsonl.gz": "19a4d769",
         "e5_signal_E_arms.jsonl.gz": "a72e0347", "e8_floor_t001_learn_full.jsonl.gz": "fb801d57", "e1b_tip_fix_31keys.jsonl.gz": "1a15b4aa",
         "e9_settings_signal.jsonl.gz": "c1c50eef", "e10_settings_judge.jsonl.gz": "af8dbc85"}
GROUPS = {"E1": [1], "E2": [2], "E5": [5], "E6": [6], "E9": [9], "E11": [11], "H": [3, 4, 7, 8, 10]}


def _lines(name):
    import hashlib
    p = os.path.join(EXPORTS, name)
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if name in E_SHA and not sha.startswith(E_SHA[name]):
        raise ValueError(f"{name}: sha256 {sha[:8]} is not the announced {E_SHA[name]}")
    return [json.loads(l) for l in gzip.open(p, "rt") if l.strip()]


def _rd(v, rows_):
    """one export line -> a read (None for an abort / error line: not a read)."""
    if v["row"] not in rows_:
        raise R.RowError(f"export row {v['row']} is not a learn row")
    if v.get("finish") == "abort" or v.get("error") or v.get("ok") is None:
        return None
    c = v.get("cls")
    return dict(ok=bool(v["ok"]), tokens=float(v.get("tokens") or 0.0), pred=v.get("pred"), cls=(_P.NONE if c in (None, "'none'", "none") else c),
                guess=None, finish=v.get("finish"), s=int(v["s"]), prompt_tokens=float(v.get("prompt_tokens") or 0.0), tail=v.get("tail"), key=v.get("key"))


def _arms(lines, rows_):
    arms = {}
    for v in lines:
        if "row" not in v or "s" not in v:
            continue
        x = _rd(v, rows_)
        if x is not None:
            arms.setdefault(v["arm"], {}).setdefault(v["row"], {}).setdefault(x["s"], x)
    return {a: {r: [d[s] for s in sorted(d)] for r, d in t.items()} for a, t in arms.items()}


def load_e1(rows_):
    """-> tip arm (768 rows, <= 8 reads each with tail), fired {row: [entry index]}, entries."""
    L1 = _lines("e1_tip_t007c_learn_s0-7.jsonl.gz")
    if os.path.exists(os.path.join(EXPORTS, "e1b_tip_fix_31keys.jsonl.gz")):       # the 31 keys whose first export line was an abort record
        fix = {v["key"]: v for v in _lines("e1b_tip_fix_31keys.jsonl.gz")}
        L1 = [dict(fix[v["key"]], arm="tip_t007c") if (v.get("finish") == "abort" and v["key"] in fix) else v for v in L1]
    tip = _arms(L1, rows_)["tip_t007c"]
    L = _lines("e1_tip_fired_learn.jsonl.gz")
    ents = L[0]["entries"]
    fired = {v["row"]: list(v["fired"]) for v in L[1:]}
    return tip, fired, ents


def load_e2(rows_):
    """-> floor-only arm (e8: 8 reads on every learn row, with tails, when the file is there; else e2), stock arm."""
    a = _arms(_lines("e2_floor_t001_and_stock_learn.jsonl.gz"), rows_)
    floor = a["floor_t001"]
    if os.path.exists(os.path.join(EXPORTS, "e8_floor_t001_learn_full.jsonl.gz")):
        floor = _arms(_lines("e8_floor_t001_learn_full.jsonl.gz"), rows_)["floor_t001"]
    return floor, a["stock_p000"]


def load_e3(rows_):
    """-> {group: wo arm}, tip reference arm (the cached tip reads paired by tip_key), {group: routed rows}, {group: {(row, s): tip_key}}."""
    L = _lines("e3_lrn_wo_and_tip_refs.jsonl.gz")
    arms = _arms([v for v in L if v.get("arm", "").startswith("lrn_wo_")], rows_)
    wo = {a[len("lrn_wo_"):]: t for a, t in arms.items()}
    ref = {}
    for v in L:
        if v.get("arm") == "tip_ref_cache":
            x = _rd(v, rows_)
            if x is not None:
                ref[v["key"]] = dict(x, row=v["row"])
    routed = {v["arm"][len("rows_wo_"):]: list(v["rows"]) for v in L if v.get("arm", "").startswith("rows_wo_")}
    pair = {}
    for v in L:
        if v.get("arm", "").startswith("lrn_wo_"):
            pair.setdefault(v["arm"][len("lrn_wo_"):], {})[(v["row"], int(v["s"]))] = v["tip_key"]
    return wo, ref, routed, pair


def load_e4(rows_):
    return _arms(_lines("e4_W_arms_signal.jsonl.gz"), rows_)


def load_e6():
    """-> {entry index: source benchmark} from the ledger (card_source of the proposer's card)."""
    src = {}
    for v in _lines("e6_tip_entry_sources.jsonl.gz"):
        if v.get("card_source"):
            src[int(v["entry"])] = v["card_source"]
    return src


# ------------------------------------------------------------------------------------------------ answer classes and guess flags
def full_rows():
    out = {}
    for l in open(LEARN):
        if l.strip():
            r = json.loads(l)
            out[r["id"]] = r
    return out


_EC = []


def _ecomp():
    if not _EC:
        import sys
        sys.path.insert(0, os.path.join(HOME, "code-v5", "loop"))
        import ecomp
        _EC.append(ecomp)
    return _EC[0]


def classify(arm, full):
    """set read["cls"] = the grader's answer class (ecomp.answer_class of the pred and the row; hashable, the same for every arm and
    process: the export's repr strings are not stable for set-valued answers). A tsv pred of a num row is a string: parsed first."""
    ec = _ecomp()
    for r, xs in arm.items():
        row = full[r]
        for x in xs:
            p = x.get("pred")
            if p in ("", "None"):
                p = None
            if p is not None and row.get("kind") == "num" and isinstance(p, str):
                try:
                    p = float(p)
                except ValueError:
                    pass
            c = ec.answer_class(p, row)
            x["cls"] = _P.NONE if c == ec.NONE_CLASS else c
    return arm


def set_guess(arm, det, windows=None):
    """read["guess"] = {window: flag} from the reasoning tail (guess5 detector: the seed stem and the detector's patterns)."""
    import guess5
    ws = windows or [det["window"]]
    for xs in arm.values():
        for x in xs:
            x["guess"] = {w: guess5.flag(dict(det, window=w), x.get("tail")) for w in ws}
    return arm
