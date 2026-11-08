"""CPU end-to-end check of rsi/loop.py's two-look confirm with a fake engine (no vLLM).
Each fake edit has a known true effect: layers.10 +0.07 accuracy, layers.11 -0.10, layers.12 0 accuracy / -10% tokens.
Usage: python3 t_confirm.py CODE_DIR WORK_DIR"""
import json, os, random, sys, types

CODE, W = sys.argv[1], sys.argv[2]
os.makedirs(W, exist_ok=True)
rng = random.Random(7)
P = {f"p{i}": rng.uniform(0.25, 0.75) for i in range(300)}
G = {f"g{i}": rng.uniform(0.6, 0.95) for i in range(200)}
with open(f"{W}/pool.jsonl", "w") as f:
    for i in P:
        f.write(json.dumps({"id": i, "kind": "mc", "answer": "A", "prompt": "Q " + i}) + "\n")
with open(f"{W}/guard.jsonl", "w") as f:
    for i in G:
        f.write(json.dumps({"id": i, "kind": "mc", "answer": "A", "prompt": "Q " + i}) + "\n")
os.makedirs(f"{W}/lib", exist_ok=True)

EFFECT = {"layers.10.mlp.down_proj": (0.07, 1.0), "layers.11.mlp.down_proj": (-0.10, 1.0),
          "layers.12.mlp.down_proj": (0.0, 0.9)}
EDITS = [{"op": "scale", "param": p, "alpha": 1.2, "why": "test"} for p in EFFECT]


class FakeSurgeon:
    def __init__(self, *a, **k):
        self.active, self.r, self.n_gen = {}, random.Random(11), 0

    def _eff(self):
        acc, tok = 0.0, 1.0
        for e in self.active.values():
            for x in (e if isinstance(e, list) else [e]):
                a, t = EFFECT.get(x.get("param", ""), (0.0, 1.0))
                acc, tok = acc + a, tok * t
        return acc, tok

    def generate(self, prompts, n=1, sampling=None, max_tokens=0, reasoning_effort=None, seeds=None):
        out, (acc, tok) = [], self._eff()
        for p in prompts:
            if not p.startswith("Q "):                                      # the surgeon dossier
                s = {"reasoning": "", "content": json.dumps({"edits": EDITS}), "finish_reason": "stop",
                     "num_tokens": 100, "token_ids": [5]}
                out.append({"prompt_token_ids": [1], "samples": [s]})
                continue
            pid = p[2:]
            base = P.get(pid, G.get(pid))
            samples = []
            for _ in range(n):
                ok = self.r.random() < min(1, max(0, base + acc))
                nt = int(self.r.gauss(1000, 150) * tok)
                samples.append({"reasoning": "r", "content": "the answer is (A)" if ok else "the answer is (B)",
                                "finish_reason": "stop", "num_tokens": nt, "token_ids": [4, 4]})
            out.append({"prompt_token_ids": [1, 2], "samples": samples})
            self.n_gen += n
        return out

    def render(self, prompt, reasoning_effort=None, **k):
        return prompt

    def tokenizer(self, text, add_special_tokens=False):                    # about 4 characters per token
        return {"input_ids": [0] * (len(text) // 4)}

    def score(self, seqs, starts, **k):
        return [{"mean": self.r.gauss(0, 1e-3)} for _ in seqs]

    def apply(self, e):
        h = types.SimpleNamespace(key=object(), changed=1)
        self.active[h.key] = e
        return h

    def revert(self, h):
        del self.active[h.key]


fake = types.ModuleType("surgery")
fake.Surgeon = FakeSurgeon
sys.modules["surgery"] = fake
mv = types.ModuleType("math_verify")
mv.parse, mv.verify = (lambda s: s), (lambda a, b: False)
sys.modules["math_verify"] = mv
sys.argv = ["loop.py", "--run-dir", f"{W}/run", "--model", "x", "--pool", f"{W}/pool.jsonl", "--guard",
            f"{W}/guard.jsonl", "--dirlib", f"{W}/lib", "--mentor", "1-99:auto", "--gens", "8", "--efficiency",
            "--doses", "1", "--null-controls", "0", "--gen-screen", "3", "--max-edits", "3", "--seed", "3",
            "--calib-samples", "4"] + sys.argv[3:]
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "engine"))
import rsi.loop as L
L.main()
recs = [json.loads(l) for l in open(f"{W}/run/log.jsonl")]
for r in recs:
    if r.get("confirm"):
        c, g = r["confirm"], r["guard"]
        print(f"gen {r['gen']} {r['target']}: looks {r['looks']} confirm n={c['n']} diff {c['diff']:+.3f} "
              f"[{c['lo90']:+.3f},{c['hi90']:+.3f}] tok {c['tok']['rel']:+.3f} guard n={g['n']} {g['diff']:+.3f} -> {r['verdict']}"
              + (f" | look1 n={r['confirm_look1']['n']} diff {r['confirm_look1']['diff']:+.3f}" if r.get("confirm_look1") else ""))
acc = [json.loads(l) for l in open(f"{W}/run/accepted.jsonl")] if os.path.exists(f"{W}/run/accepted.jsonl") else []
print("accepted:", [(a["gen"], a["edit"]["param"]) for a in acc])
