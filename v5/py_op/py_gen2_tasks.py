"""PY generation 2 (coordinator 11-11 23:32): the escalation reads AS DEPLOYED. Per learn row, the fixed replicate pairs of cached base
reads (s0,s1), (s2,s3), (s4,s5), (s6,s7); a pair whose answer classes differ gets ONE escalation read. Task files:
  V1   {row, sample j, rep k, hint null}        j = order of the replicate among the row's disagreeing replicates (seed = crn(row, j))
  PYA  the same reads with hint = the pair's two final answers verbatim (grader-extracted answer; order by a seeded shuffle); a pair in
       which one read has no answer gets no hint (the V1 form).
usage: py_gen2_tasks.py signal|judge OUT_PREFIX"""
import json, os, random, sys, zlib
V5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v5")
sys.path.insert(0, V5)
import policy5 as P, reads5 as R, stage1_data as SD, stage3_settings as S3   # noqa: E401,E402


def show(p):
    if isinstance(p, float) and p == int(p):
        return str(int(p))
    return str(p)


def replicates(xs):
    by = {x["s"]: x for x in xs}
    return [(k, by[2 * k], by[2 * k + 1]) for k in range(4) if 2 * k in by and 2 * k + 1 in by]


def main():
    role, out = sys.argv[1], sys.argv[2]
    rows, full = SD.rows(), SD.full_rows()
    floor = S3.base(rows, full)
    ids = R.rows_of(rows, R.SIGNAL_ROLES if role == "signal" else "judge")
    v1, pya, nohint = [], [], 0
    for r in ids:
        j = 0
        for k, a, b in replicates(floor[r]):
            if P.cls_of(a) == P.cls_of(b):
                continue
            hint = None
            if a.get("pred") is not None and b.get("pred") is not None and show(a["pred"]) != show(b["pred"]):
                hint = [show(a["pred"]), show(b["pred"])]
                random.Random(zlib.crc32(f"pya|{r}|{k}".encode())).shuffle(hint)
            else:
                nohint += 1
            v1.append(dict(row=r, sample=j, rep=k, hint=None))
            pya.append(dict(row=r, sample=j, rep=k, hint=hint))
            j += 1
    json.dump(v1, open(out + "_v1.json", "w"))
    json.dump(pya, open(out + "_pya.json", "w"))
    import collections
    print(role, "rows", len(ids), "rows with a disagreeing replicate", len({t["row"] for t in v1}), "escalation reads", len(v1), "without hint", nohint,
          "by order index", dict(collections.Counter(t["sample"] for t in v1)), "by bench", dict(collections.Counter(rows[t["row"]]["bench"] for t in v1)))
    print("examples:", [t["hint"] for t in pya[:6]])


if __name__ == "__main__":
    main()
