"""PROBE hdr_probe (coordinator 11-11; hand-drafted presentation probe, NOT a system proposal): tip t007c view unchanged except the routed
prefix is wrapped: HDR line + blank line + prefix (byte-identical, same order) + blank line + END line. Only rows with a routed procedure
(prefix not None); samples 0-1; E path; prio -7 (after e1two_probe). ood_hdr_probe / sig_hdr_probe."""
import json, os, time
V4 = "/data/fd/mingdazhang/selfedit/v4"; X = f"{V4}/scratch/ext"
HDR = "General working notes. The same notes are attached to many different questions; they are not part of this question and carry no information about its answer."
END = "End of notes. The question follows."
for side, tag in (("ood", "ohdr"), ("sig", "shdr")):
    T = [q for q in map(json.loads, open(f"{X}/tip_{side}.jsonl")) if q["sample"] in (0, 1) and q.get("prefix")]
    L = [dict(q, key=f"{tag}|" + q["key"], engine="E", priority=-7, probe="hdr_probe", prefix=HDR + "\n\n" + q["prefix"] + "\n\n" + END) for q in T]
    assert all(q["prefix"].startswith(HDR) and t["prefix"] in q["prefix"] for q, t in zip(L, T)) and len({q["key"] for q in L}) == len(L)
    name = f"{side}_hdr_probe"; p = f"{V4}/q/in/{name}.jsonl"
    open(p + ".tmp", "w").write("".join(json.dumps(x) + "\n" for x in L)); os.replace(p + ".tmp", p)
    open(f"{X}/submit_times.jsonl", "a").write(json.dumps(dict(batch=name, t_submit=time.time(), n=len(L))) + "\n")
    print(name, len(L), "rows", len({q["row"] for q in L})); time.sleep(1.1)
