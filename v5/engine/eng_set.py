"""Update engines.json entries from a JSON spec file: {"<port>": {field: value, ...}, ...}. usage: python3 eng_set.py SPEC.json"""
import json, os, sys, time
V4 = "/data/fd/mingdazhang/selfedit/v4"
spec = json.load(open(sys.argv[1]))
e = json.load(open(f"{V4}/engines.json"))
for x in e["servers"]:
    if str(x["port"]) in spec:
        x.update(spec[str(x["port"])])
        if x.get("role") == "T" and x.get("weights") != "stock":
            x.pop("note", None)
        print(x["port"], {k: x.get(k) for k in ("role", "state", "cap", "weights", "edit_sha", "model", "job")})
e["updated"] = time.time()
json.dump(e, open(f"{V4}/engines.json.tmp", "w"), indent=1)
os.replace(f"{V4}/engines.json.tmp", f"{V4}/engines.json")
