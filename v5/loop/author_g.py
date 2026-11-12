"""B2 (coordinator 11-11, Track B): author "G", signal-driven cells.

Input  RUN/signal/<tip>/cells.json, written by the signal worker for the current tip (nothing of JUDGE goes into it):
         {"tip": "<tip id>", "scale": "accuracy per row, show+monitor rows", "cells": [ITEM, ...]}   (cells in rank order)
       ITEM = {"kind": "weight", "name": str, "edit": {W edit form}, "pred": float, "se": float[, "lb", "pred_tokens_ratio", "family", ...]}
            | {"kind": "setting", "name": str, "settings": {knob: value}, "pred": float, "se": float[, ...]}
            | {"kind": "routed", "base_kind": "weight" | "setting", "route": {"type": "cond", "condition": <a router condition of the tip>},
               "edit" | "settings", ...}   (B4, signal_cells.py: the cell applied only where the condition holds; proposed only when
               RUN/cards.json says "g_routed": true; a route of another type needs an engine op and is skipped)
       Any other kind is logged and skipped (a later spec may add kinds; nothing here assumes weight only).
       B4 fields carried from the item into c["g"] and realised.jsonl when present: prior_family (cells of the item's library read before
       on sets holding JUDGE rows; an integer >= 0, else the item is skipped), library (the library the count refers to; default its
       family); the judge's BH family of (G, generation) = its candidates + the SUM over the distinct libraries among them of each
       library's prior_family (g_prior / g_prior_pad, B4b), pred_raw, shrink, p_pass, K.
Props  author "G", slot numbers 0.. like B; a weight cell is a global candidate on the W cards (engine W, AM17 quarter, LITE), a setting cell
       a global candidate on the E cards like an arm-B setting cell; both judged in BH family <tip>|G|g with LOBO, never bundle members.
       G takes its ranked weight cells first from the W budget (sl["BW"], or RUN/cards.json "g_cells"), arm B's weight cells get the rest;
       setting cells up to RUN/cards.json "g_settings" (default G_SETTINGS). A cell already read on this tip (same edit / settings, any
       author) is not proposed again. No cells.json for the current tip, a file for another tip, or W off for weight cells: logged, no slot.
Output RUN/signal/<tip>/realised.jsonl, one line per decided G candidate after the generation's report: predicted gain / se / scale /
       tokens ratio against the loop's pooled d / se / p (JUDGE rows, routed scale = all rows for a global candidate), rows, samples per row,
       measured tokens ratio, decision, accepted / merged. No row ids, no per-row numbers. A G cell that duplicates an earlier candidate
       of the generation (invalid by the duplicate check; proposers come first) gets that candidate's numbers with "duplicate_of": <cid>
       (coordinator 11-11). Proposer memory labels G candidates "blind" (versions._rel); the ledger and reports keep author "G".
Mixed into gen.Run."""
import hashlib, json, os, time

GKINDS = ("weight", "setting", "routed")
G_META = ("id", "prior_family", "library", "pred_raw", "se_raw", "shrink", "p_pass", "K", "base_kind", "share")   # B4(b): copied from the item into c["g"]
G_SETTINGS = 4                 # setting cells per generation unless RUN/cards.json says "g_settings"


def _sha16(b):
    return hashlib.sha256(b).hexdigest()[:16]


class AuthorGMixin:
    def g_cells_path(self, tid=None):
        return os.path.join(self.rd, "signal", tid or self.state["tip"]["id"], "cells.json")

    def g_read_cells(self):
        """(cells, info): the current tip's cells.json items that can be proposed, in rank order, each {kind, name, edit | settings, pred,
        se, rank, ...}; info says what was skipped and why."""
        tid = self.state["tip"]["id"]
        p = self.g_cells_path(tid)
        if not os.path.exists(p):
            return [], dict(state="no cells.json", path=p)
        try:
            raw = open(p, "rb").read()
            x = json.loads(raw)
        except (OSError, ValueError) as ex:
            return [], dict(state=f"unreadable: {type(ex).__name__}: {ex}", path=p)
        items = x.get("cells") if isinstance(x, dict) else x
        if isinstance(x, dict) and x.get("tip") not in (None, tid):
            return [], dict(state=f"cells.json is for tip {x.get('tip')}, the tip is {tid}", path=p)
        if not isinstance(items, list):
            return [], dict(state=f"cells is not a list ({type(items).__name__})", path=p)
        items = [it if isinstance(it, dict) else {"kind": f"not an object: {str(it)[:40]}"} for it in items]
        info = dict(state="read", path=p, sha=_sha16(raw), scale=(x.get("scale") if isinstance(x, dict) else None), n=len(items or []), skipped=[])
        tried_e = {json.dumps(c["entry"]["edit"], sort_keys=True) for c in self.tip_cands() if (c.get("entry") or {}).get("edit")}
        tried_s = {json.dumps(c["entry"].get("settings") or {}, sort_keys=True) for c in self.tip_cands()
                   if (c.get("entry") or {}).get("cond") == "always" and not c["entry"].get("edit") and c["entry"].get("settings")}
        tried_r = {(c["entry"].get("condition"), json.dumps(c["entry"].get("edit") or c["entry"].get("settings") or {}, sort_keys=True))
                   for c in self.tip_cands() if (c.get("entry") or {}).get("cond") not in (None, "always")}
        routed_on = bool((self._cards() or {}).get("g_routed"))
        out = []
        for i, it in enumerate(items or []):
            kind = (it or {}).get("kind")
            why = None
            pf, lib = (it or {}).get("prior_family"), (it or {}).get("library")
            if kind not in GKINDS:
                why = f"kind {kind!r} not handled (weight / setting)"
            elif pf is not None and (isinstance(pf, bool) or not isinstance(pf, int) or pf < 0):   # B4b (coordinator 11-11): never reaches the judge
                why = f"prior_family must be an integer >= 0 (got {pf!r})"
            elif lib is not None and (not isinstance(lib, str) or not lib.strip()):
                why = f"library must be a non-empty string (got {lib!r})"
            elif not isinstance(it.get("pred"), (int, float)):
                why = "no numeric pred"
            elif kind == "routed":                               # B4: the cell on the rows where a router condition of the tip holds
                rt, bk = it.get("route") if isinstance(it.get("route"), dict) else {}, it.get("base_kind")
                pay = it.get("edit") if bk == "weight" else it.get("settings")
                if not routed_on:
                    why = "routed cells wait for RUN/cards.json g_routed"
                elif rt.get("type") != "cond" or not isinstance(rt.get("condition"), str) or not rt["condition"].strip() or rt["condition"] == "always":
                    why = f"route type {rt.get('type')!r} cannot be proposed (only a router condition; a length class needs an engine op)"
                elif bk == "weight":
                    why = "; ".join(self.edit_problems(pay)) or None
                elif bk == "setting":
                    why = "; ".join(["settings must be a non-empty object"] if not isinstance(pay, dict) or not pay else self.b_setting_problems(dict(settings=pay))) or None
                else:
                    why = f"base kind {bk!r} not handled (weight / setting)"
                if why is None and (rt["condition"], json.dumps(pay, sort_keys=True)) in tried_r:
                    why = "read on this tip already"
            elif kind == "weight":
                pr = self.edit_problems(it.get("edit"))
                why = "; ".join(pr) if pr else ("read on this tip already" if json.dumps(it["edit"], sort_keys=True) in tried_e else None)
            else:
                st = it.get("settings")
                pr = ["settings must be a non-empty object"] if not isinstance(st, dict) or not st else self.b_setting_problems(dict(settings=st))
                why = "; ".join(pr) if pr else ("read on this tip already" if json.dumps(st, sort_keys=True) in tried_s else None)
            if why:
                info["skipped"].append(dict(rank=i, name=(it or {}).get("name"), why=why))
                continue
            out.append(dict(it, rank=i))
        return out, info

    def add_g_props(self, props, sl):
        """G's props (before arm B's weight cells); returns the number of W cells G took from the W budget."""
        G = self.G()
        if getattr(self.a, "no_g", False):
            return 0
        cells, info = self.g_read_cells()
        c = self._cards() or {}
        w_on = self.w_on()
        bw = int(sl.get("BW") or 0) if self.a.am16 else int(sl.get("BW", self.a.w_cells) or 0)
        cap_w = int(c["g_cells"]) if c.get("g_cells") is not None else bw
        cap_s = int(c["g_settings"]) if c.get("g_settings") is not None else G_SETTINGS
        is_wc = lambda x: x["kind"] == "weight" or (x["kind"] == "routed" and x.get("base_kind") == "weight")   # noqa: E731 (B4: routed cells too)
        wc = [x for x in cells if is_wc(x)]
        sc = [x for x in cells if not is_wc(x)]
        take_w = wc[:cap_w] if w_on else []
        take_s = sc[:cap_s]
        k = 0
        for x in take_w + take_s:
            g = dict(rank=x["rank"], kind=x["kind"], name=x.get("name"), pred=float(x["pred"]), se=x.get("se"), lb=x.get("lb"),
                     pred_tokens_ratio=x.get("pred_tokens_ratio"), family=x.get("family"), scale=info.get("scale"), cells_sha=info.get("sha"),
                     tip=self.state["tip"]["id"])
            g.update({k_: x[k_] for k_ in G_META if x.get(k_) is not None})          # B4
            cond = x["route"]["condition"] if x["kind"] == "routed" else "always"
            if x["kind"] == "routed":
                g["route"] = "cond"
            cand = dict(why=f"signal cell ({x['kind']}) rank {x['rank']}: {x.get('name')}; predicted {g['pred']:+.4f}"
                            + (f" (se {float(g['se']):.4f})" if isinstance(g["se"], (int, float)) else "") + (f", {g['scale']}" if g["scale"] else ""),
                        condition=cond, procedure=None, settings=dict(x.get("settings") or {}), ops=[], cell=x.get("name"), cell_group="signal", g=g)
            if is_wc(x):
                cand["edit"] = x["edit"]
            props.append(dict(author="G", slot=k, cand=cand))
            k += 1
        used_w = len(take_w)
        G["g"] = dict(info, weight_cells=len(wc), setting_cells=len(sc), proposed=k, w_cells=used_w, w_budget=bw, w_cap=cap_w, s_cap=cap_s,
                      w_on=w_on, waiting_for_w=(len(wc) if not w_on else max(0, len(wc) - used_w)), t=time.time())
        self._signal_log(f"  G (signal cells, {info['state']}): {k} props ({used_w} weight on W of budget {bw}"
                         + (f", cap {cap_w}" if c.get("g_cells") is not None else "") + f"; {len(take_s)} setting)"
                         + (f"; {len(wc)} weight cells wait for W cards" if wc and not w_on else "")
                         + (f"; skipped {len(info.get('skipped') or [])}" if info.get("skipped") else ""))
        return used_w

    # ------------------------------------------------------------------ realised (after the report)
    def g_realised(self):
        """Append RUN/signal/<parent tip>/realised.jsonl lines for this generation's decided G candidates (once each). A G cell found to be a
        duplicate of an earlier candidate of the generation (validation, proposers first) gets the line of that candidate's measured numbers
        with "duplicate_of": <cid> (coordinator 11-11); a duplicate of the tip gets "duplicate_of": "tip" and no measured numbers."""
        G = self.G()
        done = set(G.get("g_realised") or [])
        byc = {c["cid"]: c for c in G.get("cands", [])}
        out = []
        for c in G.get("cands", []):
            if c.get("author") != "G" or c["cid"] in done or not c.get("g"):
                continue
            dup = c.get("duplicate_of")
            m = byc.get(dup) if dup and dup != "tip" else (None if dup else c)
            if dup != "tip" and (m is None or not m.get("decided")):
                continue
            m = m or {}
            j = m.get("judge") or {}
            dec = m.get("decision") or {}
            a21 = m.get("am21") or {}
            tc, tp = m.get("tok_cand"), m.get("tok_parent")
            g = c["g"]
            x = dict(gen=self.g, cid=c["cid"], tip=c.get("parent"), rank=g.get("rank"), kind=g.get("kind"), name=g.get("name"),
                     family=g.get("family"), cells_sha=g.get("cells_sha"), scale=g.get("scale"),
                     pred=g.get("pred"), pred_se=g.get("se"), pred_lb=g.get("lb"), pred_tokens_ratio=g.get("pred_tokens_ratio"),
                     d=j.get("D", j.get("d")), se=j.get("se"), se_used=j.get("se_used"), p=j.get("p"), n_rows=j.get("n_rows") or j.get("n"),
                     samples_per_row=a21.get("n"), left_step1=bool(a21.get("left")), w_stage=m.get("w_stage"),
                     tokens_ratio=(float(tc) / float(tp)) if tc and tp else None, basis=dec.get("basis"), bh=j.get("bh"), lobo=j.get("lobo"),
                     accepted=bool(m.get("accepted")), merged=bool(m.get("merged")), t=time.time())
            x.update({k_: g[k_] for k_ in ("prior_family", "library", "pred_raw", "se_raw", "shrink", "p_pass", "route", "share") if g.get(k_) is not None})   # B4(b)
            if g.get("K") is not None or dec.get("K") is not None:
                x.update(K_planned=g.get("K"), K=dec.get("K"))
            if dup:
                x.update(duplicate_of=dup, duplicate_author=(m.get("author") if dup != "tip" else None))
            out.append(x)
            done.add(c["cid"])
        if out:
            d = os.path.join(self.rd, "signal", out[0]["tip"] or self.state["tip"]["id"])
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "realised.jsonl"), "a") as fh:
                for x in out:
                    fh.write(json.dumps(x) + "\n")
            G["g_realised"] = sorted(done)
            self._signal_log(f"  G realised: {len(out)} lines appended to {d}/realised.jsonl"
                             + (f" ({sum(1 for x in out if x.get('duplicate_of'))} from the candidate they duplicate)" if any(x.get("duplicate_of") for x in out) else ""))
        return out

    # ------------------------------------------------------------------ the judge's family (B4, coordinator 11-11)
    def g_prior(self, ids, bycid):
        """(n, {library: prior}): the cells read before on sets holding JUDGE rows, summed over the DISTINCT libraries of the candidates
        (B4b, coordinator 11-11), within one library its largest stated prior_family. library = c["g"]["library"], else its family."""
        lib = {}
        for i in ids:
            g = (bycid.get(i) or {}).get("g") or {}
            pf = g.get("prior_family")
            if pf is None:
                continue
            k = g.get("library") or g.get("family") or "?"
            lib[k] = max(lib.get(k, 0), int(pf))
        return sum(lib.values()), lib

    def g_prior_pad(self, key, ids, bycid):
        """Honest family size for cells whose library was screened before on sets holding JUDGE rows: the BH family of (G, generation)
        = its candidates + the prior cells of every distinct library among them (g_prior: sum over libraries, B4b). Returned as that many
        extra judge_bh inputs with p = 1 (d < 0 with no error), appended to the family in am15_numbers and interval: BH's bar becomes
        alpha x k / (K + prior); a p = 1 input can only make acceptance harder. [] for other authors."""
        if key[0] != "G":
            return []
        n, _ = self.g_prior(ids, bycid)
        return [dict(name=f"__prior{j}", d=-1.0, se=0.0, n=10 ** 9, nu=1.0, share=0.0, is_global=False) for j in range(n)]
