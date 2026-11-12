"""selfedit v5, read tables for the offline judge (stage 1). Standard library + numpy.

A READ is dict(ok: bool, tokens: float (generated tokens), pred: str | None, guess: bool | None, finish: str | None, s: int (sample index),
prompt_tokens: float | None). An ARM is {row id: [read, ...]} (errors / aborts are dropped; a read cut at the budget, finish "length", stays
and is wrong). Rows come from the clean 768-row learn.jsonl (sha256 b1263f834ed1d2db...); any other row raises.

Loaders: tsv tables of the engine agent (per-read: batch, cell, row, sample, ok, tokens, finish, ..., pred, base_* = the tip's cached read at
the same row and sample), the compact tip table (row -> samples {idx: ok, tokens, finish[, pred, guess]}), and the v5 export form
(jsonl: {"arm", "row", "s", "ok", "tokens", "pred", "guess", "finish", "prompt_tokens", "fired": [...]}).
"""
import collections
import glob
import hashlib
import json
import os

LEARN_SHA = "b1263f834ed1d2db1c1d55442556118a29853e667f342991d2befb346d14e50e"
SIGNAL_ROLES = ("show", "monitor")
BENCHES = ("babyvision", "blink", "charxiv", "lego", "visulogic", "vlmsbiased")


class RowError(ValueError):
    pass


def load_rows(path, check_sha=True):
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if check_sha and sha != LEARN_SHA:
        raise RowError(f"{path}: sha256 {sha[:16]}... is not the clean 768-row learn.jsonl ({LEARN_SHA[:16]}...)")
    rows = {}
    for l in raw.decode().splitlines():
        if l.strip():
            r = json.loads(l)
            role = r.get("v4_role")
            if role not in ("show", "monitor", "judge"):
                raise RowError(f"{path}: row {r.get('id')} has role {role!r}: only show / monitor / judge rows may be loaded")
            rows[r["id"]] = dict(id=r["id"], bench=r["bench"], role=role, pic=r.get("pic") or r["id"], kind=r.get("kind"),
                                 category=r.get("category"))
    return rows


def rows_of(rows, roles):
    roles = (roles,) if isinstance(roles, str) else tuple(roles)
    return sorted(r for r, x in rows.items() if x["role"] in roles)


def _f(v, default=None):
    return default if v in (None, "") else float(v)


def _read(ok, tokens, pred, finish, s, guess=None, prompt_tokens=None):
    p = pred if pred not in (None, "", "None") else None
    return dict(ok=bool(int(float(ok))) and p is not None if pred is not False else bool(int(float(ok))), tokens=float(tokens or 0.0), pred=p,
                guess=guess, finish=finish or None, s=int(s), prompt_tokens=prompt_tokens)


def load_tsv(paths, rows, with_base=True):
    """-> ({cell: arm}, base arm or None). The base_* columns of every table are merged into one arm (the tip's cached E reads); a (row,
    sample) seen twice must agree. Tables without a pred column give pred False-marked reads (ok from the table, pred None)."""
    files = []
    for p in ([paths] if isinstance(paths, str) else paths):
        files += sorted(glob.glob(os.path.join(p, "*.tsv"))) if os.path.isdir(p) else sorted(glob.glob(p))
    arms = collections.defaultdict(lambda: collections.defaultdict(dict))
    base = collections.defaultdict(dict)
    for f in files:
        with open(f) as fh:
            head = fh.readline().rstrip("\n").split("\t")
            hp = "pred" in head
            for l in fh:
                v = dict(zip(head, l.rstrip("\n").split("\t")))
                r = v.get("row")
                if not r or not v.get("cell"):
                    continue
                if r not in rows:
                    raise RowError(f"{f}: row {r} is not a learn row")
                s = int(v.get("sample") or 0)
                if with_base and v.get("base_ok") not in (None, ""):
                    b = dict(ok=bool(int(float(v["base_ok"]))), tokens=_f(v.get("base_tokens"), 0.0),
                             pred=(v.get("base_pred") or None) if "base_pred" in head else None, has_pred="base_pred" in head,
                             guess=None, finish=v.get("base_finish") or None, s=s, prompt_tokens=_f(v.get("prompt_tokens")))
                    old = base[r].get(s)
                    if old is not None and (old["ok"], old["tokens"]) != (b["ok"], b["tokens"]):
                        raise RowError(f"{f}: base read of {r} sample {s} differs between tables")
                    if old is None or (b["has_pred"] and not old["has_pred"]):
                        base[r][s] = b
                if v.get("error") or v.get("finish") == "abort" or v.get("ok") in (None, ""):
                    continue
                arms[v["cell"]][r].setdefault(s, dict(ok=bool(int(float(v["ok"]))), tokens=_f(v.get("tokens"), 0.0),
                                                      pred=(v.get("pred") or None) if hp else None, has_pred=hp, guess=None,
                                                      finish=v.get("finish") or None, s=s, prompt_tokens=_f(v.get("prompt_tokens"))))
    out = {c: {r: [d[s] for s in sorted(d)] for r, d in a.items()} for c, a in arms.items()}
    return out, ({r: [d[s] for s in sorted(d)] for r, d in base.items()} if base else None)


def load_compact(path, rows):
    """t007c_signal_compact.json: row -> {bench, role, samples {idx: {ok, tokens, finish[, pred, guess]}}}."""
    d = json.load(open(path))
    arm = {}
    for r, x in d.items():
        if r not in rows:
            raise RowError(f"{path}: row {r} is not a learn row")
        arm[r] = [dict(ok=bool(v["ok"]), tokens=float(v.get("tokens") or 0.0), pred=v.get("pred"), has_pred="pred" in v, guess=v.get("guess"),
                       finish=v.get("finish"), s=int(s), prompt_tokens=v.get("prompt_tokens")) for s, v in sorted(x["samples"].items(), key=lambda kv: int(kv[0]))]
    return arm


def load_export(paths, rows):
    """v5 export (jsonl, one read per line: arm, row, s, ok, tokens, pred, guess, finish, prompt_tokens, fired) -> {arm: arm table},
    {arm: {row: fired entry ids}}."""
    arms = collections.defaultdict(lambda: collections.defaultdict(dict))
    fired = collections.defaultdict(dict)
    for p in ([paths] if isinstance(paths, str) else paths):
        for f in sorted(glob.glob(p)):
            for l in open(f):
                if not l.strip():
                    continue
                v = json.loads(l)
                r = v["row"]
                if r not in rows:
                    raise RowError(f"{f}: row {r} is not a learn row (OOD / test rows are never loaded)")
                if v.get("error") or v.get("finish") == "abort" or v.get("ok") is None:
                    continue
                s = int(v["s"])
                arms[v["arm"]][r].setdefault(s, dict(ok=bool(v["ok"]), tokens=float(v.get("tokens") or 0.0), pred=v.get("pred"), has_pred=True,
                                                     guess=v.get("guess"), finish=v.get("finish"), s=s, prompt_tokens=v.get("prompt_tokens")))
                if v.get("fired") is not None:
                    fired[v["arm"]][r] = list(v["fired"])
    return {a: {r: [d[s] for s in sorted(d)] for r, d in t.items()} for a, t in arms.items()}, dict(fired)
