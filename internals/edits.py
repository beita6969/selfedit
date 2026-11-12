"""fable_int: apply Surgeon-format edit specs (set / scale / add on mlp.gate_proj / up_proj / down_proj neuron selections) in place to the HF model,
with the engine's arithmetic (fp32, one rounding to the parameter dtype per op), LIFO revert."""
import re
import numpy as np, torch


class Editor:
    def __init__(self, lab, remap=None):
        self.lab, self.stack, self.npz, self.remap = lab, [], {}, remap or {}

    def _w(self, name):
        m = re.match(r"(?:model\.language_model\.)?layers\.(\d+)\.(.+)", name)
        mod = self.lab.layers[int(m.group(1))]; sub = m.group(2).replace(".weight", "")
        for part in sub.split("."):
            mod = getattr(mod, part)
        return mod.weight, sub.split(".")[-1]

    def _val(self, v):
        if isinstance(v, dict) and "npz" in v:
            path = self.remap.get(v["npz"], v["npz"])
            z = self.npz.setdefault(path, np.load(path))
            return torch.tensor(np.asarray(z[v["key"]], dtype=np.float32))
        return torch.tensor(np.asarray(v, dtype=np.float32))

    @torch.no_grad()
    def apply(self, spec):
        for op in spec:
            W, leaf = self._w(op["param"]); sel = op.get("sel", op)
            ns = sel.get("neurons", sel.get("neuron")); ns = [ns] if isinstance(ns, int) else list(ns)
            idx = torch.tensor(ns, device=W.device)
            if leaf in ("gate_proj", "up_proj"):
                old = W[idx].clone()
            elif leaf == "down_proj":
                old = W[:, idx].T.clone()
            else:
                raise NotImplementedError(op["param"])
            x = old.float()
            if op["op"] == "set":
                new = self._val(op["value"]).to(W.device).reshape(len(ns), -1)
            elif op["op"] == "scale":
                al = op["alpha"]
                new = x * (float(al) if not isinstance(al, (list, dict)) else self._val(al).to(W.device).reshape(len(ns), 1))
            elif op["op"] == "add":
                new = x + self._val(op["value"]).to(W.device).reshape(len(ns), -1)
            else:
                raise NotImplementedError(op["op"])
            new = new.to(W.dtype)
            if leaf == "down_proj":
                W[:, idx] = new.T
            else:
                W[idx] = new
            self.stack.append((W, leaf, idx, old))

    @torch.no_grad()
    def revert(self):
        while self.stack:
            W, leaf, idx, old = self.stack.pop()
            if leaf == "down_proj":
                W[:, idx] = old.T
            else:
                W[idx] = old
