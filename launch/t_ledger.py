"""Read-only CPU check for the paper128b card ledger.

Usage on the server::

    python c1109deploy/t_ledger.py
"""

import json
import os
import re
import sys


N = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [N + "/rsi", N + "/mm", N]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import hboost as H
from harness.hw.experience import build_ledger


W = "/data/fd/mingdazhang/selfedit/rsi/mm/hb/paper128b"
H.set_run(W)
H.HB = W
am = json.load(open(f"{W}/amendment.json"))
data = H.Data(None, am["iid"], am.get("ood"), am.get("exclude"))
S = json.load(open(f"{W}/state.json"))
members = S["members"]
H.REG.update({m["name"]: m for m in members})
votes = {m["name"]: H.load_member(m) for m in members}

show_ids = data.of("show")
assert all(data.role.get(row_id) == "show" for row_id in show_ids)
ledger = build_ledger(data, votes, show_ids)
text = ledger["text"]

assert all(data.role.get(row_id) == "show" for row_id in show_ids)
assert not re.search(r"[a-z0-9_]+/[A-Za-z0-9_.:-]*\d", text)
assert re.search(r"^### Bench: lego$", text, re.M), "lego bench section missing"

print(text)
lego = ledger["fields"].get("lego", {}).get("categories", {})
print("decision_words[lego]:")
for category in sorted(lego):
    for entry in lego[category].get("decision_words", []):
        print(category, entry)
