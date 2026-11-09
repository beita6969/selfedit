# selfedit 1109c: NEW (design_1109c §7). Sets ops/registry.json 'admitted' from the probe verdict files only, at a round boundary.
# A verdict file: {family, pass, prediction, measured, kill_reason, sha}. A family is admitted iff every probe its rule names has a file with pass true
# AND the family is wired in this tree (a family without a build path is never given slots, whatever its probe says; logged).
import glob, json, os, sys

STEP0 = os.environ.get("SELFEDIT_STEP0", "/data/fd/mingdazhang/selfedit/step0")
HERE = os.path.dirname(os.path.abspath(__file__))


def verdicts(step0=STEP0):
    out = {}
    for f in sorted(glob.glob(f"{step0}/*/verdict.json")):
        pid = os.path.basename(os.path.dirname(f))
        try:
            v = json.load(open(f))
        except (OSError, ValueError) as e:
            out[pid] = dict(pass_=False, error=str(e)[:200], file=f)
            continue
        if isinstance(v, dict) and "arms" in v:            # a probe with arms (Q1 a/b, Q2 n/s): one entry per arm, id = probe + arm
            for arm, x in v["arms"].items():
                out[pid + arm] = dict(pass_=bool(x.get("pass")), file=f, measured=x.get("measured"), kill_reason=x.get("kill_reason"))
        else:
            out[pid] = dict(pass_=bool(v.get("pass")), file=f, measured=v.get("measured"), kill_reason=v.get("kill_reason"), family=v.get("family"))
    return out


def admit(registry_path=os.path.join(HERE, "registry.json"), step0=STEP0, write=True):
    reg = json.load(open(registry_path))
    v = verdicts(step0)
    reg["verdicts"] = {k: {kk: vv for kk, vv in x.items()} for k, x in v.items()}
    for name, fam in reg["families"].items():
        ok = all(v.get(p, {}).get("pass_") for p in fam["probe"])
        fam["admitted"] = bool(ok and fam["wired"])
        fam["why"] = ("admitted" if fam["admitted"] else "probe passed but no build path in this tree" if ok else
                      "missing verdict: " + ",".join(p for p in fam["probe"] if p not in v) if any(p not in v for p in fam["probe"]) else
                      "probe failed: " + ",".join(p for p in fam["probe"] if not v[p]["pass_"]))
    if write:
        json.dump(reg, open(registry_path + ".new", "w"), indent=1)
        os.replace(registry_path + ".new", registry_path)
    return reg


def admitted_ops(reg):
    return sorted(f["op"] for f in reg["families"].values() if f["admitted"])


if __name__ == "__main__":
    r = admit(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "registry.json"))
    for n, f in r["families"].items():
        print(f"{n:16s} tier {f['tier']:3s} admitted {f['admitted']!s:5s} {f['why']}")
