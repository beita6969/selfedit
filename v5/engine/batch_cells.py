"""Update ~/selfedit/v4_reports/w_signal/batch_cells.json from the exported TSVs there (Mac side): batch -> {cell, edit}.
Cell name = edit spec file name without .json; batches with no edit (W-null) -> "W-null"."""
import csv, glob, json, os
D = os.path.expanduser("~/selfedit/v4_reports/w_signal")
EDIT_DIRS = {"XB_": "fable_f1", "XS_": "fable_f1", "RAND": "fable_f1", "FL": "fable_int"}
p = f"{D}/batch_cells.json"
m = json.load(open(p)) if os.path.exists(p) else {}
for k in [k for k in m if k.endswith(tuple(f"_s{i}" for i in range(8))) and isinstance(m[k], dict) and "cell" in m[k] and k != "cells"]:
    m.pop(k)                               # entries belong under "cells" (the cell-maker's schema)
C = m.setdefault("cells", {})
for f in sorted(glob.glob(f"{D}/*.tsv")):
    b = os.path.basename(f)[:-4]
    with open(f) as fh:
        r = next(csv.DictReader(fh, delimiter="\t"), None)
    cell = (r or {}).get("cell") or ""
    if not cell or cell in ("W-null", "None"):
        C.setdefault(b, {}).update(cell="W-null", edit=None)
    else:
        sub = next((v for k, v in EDIT_DIRS.items() if cell.startswith(k)), "fable_f1")
        C.setdefault(b, {}).update(cell=cell, edit=f"/data/fd/mingdazhang/selfedit/v4/cells/{sub}/edits/{cell}.json")
json.dump(m, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
print(json.dumps(C, indent=1))
