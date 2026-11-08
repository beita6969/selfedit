"""Fast causal read of an edit on the decision to stop thinking, without generating. On the model's own traces: the change in
log p(end of thinking | prefix) at cut points late in the reasoning and at the natural end of finished traces, and in the
log-probs of its own tokens (all of them; the revisit words that reopen a line of thought). The same prefixes under base and
edited weights, so the differences carry no sampling noise. On Gemma the stop read tracks the measured token change across a
dose series and takes the opposite sign for an edit that lengthens traces; a random direction reads near zero.
post and pre split the cut points by whether the correct answer has already appeared in the trace: a stop raised before it is
a stop that gives up, not one that skips re-checking. ans reads the close right after lines that state the correct answer, the
decision on a model like Qwen, whose stop token is near 0 nats at a natural end and -20 at an arbitrary line break, and which
goes on after 60 % of its concluding lines."""
import bisect, re
import numpy as np

REVISIT = ("Wait", "Hmm", "Actually", "Alternatively", "But")


def family_tokens(S):
    """(tokens that open the thinking text after the prompt, tokens that close it where the model stops, revisit word tokens)."""
    tk, stop = S.tokenizer, S.think_end_id
    assert stop is not None and stop != tk.unk_token_id, "this family has no single end-of-thinking token"
    gemma = getattr(S, "family", None) == "gemma4"
    lead = tk("<|channel>thought\n", add_special_tokens=False).input_ids if gemma else []
    # Qwen writes a newline then </think>; Gemma's marker follows the last sentence.
    closing = [stop] if gemma else tk("\n", add_special_tokens=False).input_ids + [stop]
    revisit = {t for w in REVISIT for t in (tk(w, add_special_tokens=False).input_ids[-1], tk(" " + w, add_special_tokens=False).input_ids[-1])}
    return lead, closing, revisit


def answer_marks(tk, th, answer):
    """(token index where the correct answer first appears in the thinking, line-break tokens of the lines stating it). First is -1
    if it never appears, None if the answer cannot be searched for (short or not a plain number: "7" turns up in any arithmetic).
    First mention is a proxy for having reached it."""
    a = str(answer).strip()
    if len(a) < 2 or not a.isdigit():
        return None, []
    text, ends, breaks = "", [], []
    for k, t in enumerate(th):
        piece = tk.decode([t])
        text += piece
        ends.append(len(text))
        if "\n" in piece:
            breaks.append(k)
    found = [m.end() for m in re.finditer(rf"(?<!\d){a}(?!\d)", text)]
    if not found:
        return -1, []
    lines = sorted({breaks[i] for i in (bisect.bisect_left(breaks, bisect.bisect_left(ends, x)) for x in found) if i < len(breaks)})
    return min(bisect.bisect_left(ends, found[0]), len(th) - 1), lines


def pick(recs, k, rng):
    """Up to k traces: half finished, half truncated, one per problem."""
    recs = list(recs)
    rng.shuffle(recs)
    picked, seen = [], set()
    for want in (lambda r: r["finish"] != "length", lambda r: r["finish"] == "length"):
        n = 0
        for r in recs:
            if n < k // 2 and r["id"] not in seen and r["reasoning"] and want(r):
                picked.append(r)
                seen.add(r["id"])
                n += 1
    return picked


def own_traces(S, recs, k, rng):
    """Traces for Hazard from evaluate() records, which carry the exact token sequences."""
    tk, stop = S.tokenizer, S.think_end_id
    lead, _, _ = family_tokens(S)
    out = []
    for r in pick(recs, k, rng):
        prompt, gen = r["seq"][: r["start"]], r["seq"][r["start"]:]
        if stop in gen:
            gen = gen[: gen.index(stop)]
        while gen and not tk.decode([gen[-1]]).strip():
            gen = gen[:-1]
        if lead and gen[: len(lead)] == lead:
            prompt, gen = prompt + lead, gen[len(lead):]
        out.append((prompt, gen, r["finish"] != "length", *answer_marks(tk, gen, r["answer"])))
    return out


class EndSplit:
    """The close at the natural end of finished traces, right and wrong apart: an edit that stops on right conclusions and goes on after wrong ones raises the first and lowers the second."""
    def __init__(self, S, right, wrong):
        """right, wrong: (token ids up to where the thinking text starts, thinking token ids)."""
        self.S = S
        _, closing, _ = family_tokens(S)
        self.jobs, self.starts, self.kinds = [], [], []
        for kind, group in (("right", right), ("wrong", wrong)):
            for head, th in group:
                self.jobs.append(head + th + closing)
                self.starts.append(len(head) + len(th))
                self.kinds.append(kind)
        self.base = self._read()

    def _read(self):
        return np.array([float(np.asarray(x["logprobs"], dtype=np.float64).sum()) for x in self.S.score(self.jobs, self.starts)])

    def effect(self):
        d, kinds, row = self._read() - self.base, np.array(self.kinds), {}
        for kind in ("right", "wrong"):
            x = d[kinds == kind]
            row[f"d_end_{kind}"], row[f"se_end_{kind}"] = float(x.mean()), float(x.std() / max(1, np.sqrt(len(x))))
        return row


class StyleRead:
    """Toward which of the model's own ways of working does an edit lean? Same problem, the fast and the slow correct trace, the same early span of each (so position is equal): the
    change in mean own-token log-prob on the fast one minus on the slow one. Common drift cancels."""
    def __init__(self, S, pairs):
        """pairs: (token ids up to where the thinking text starts, fast thinking tokens, slow thinking tokens)."""
        self.S = S
        self.jobs, self.starts = [], []
        for head, fast, slow in pairs:
            self.jobs += [head + fast, head + slow]
            self.starts += [len(head), len(head)]
        self.base = self._read()

    def _read(self):
        return np.array([float(np.mean(x["logprobs"])) for x in self.S.score(self.jobs, self.starts)])

    def effect(self):
        d = self._read() - self.base
        x = d[0::2] - d[1::2]
        return {"d_style": float(x.mean()), "se_style": float(x.std() / max(1, np.sqrt(len(x)))), "d_fast": float(d[0::2].mean()), "d_slow": float(d[1::2].mean())}


class Hazard:
    def __init__(self, S, traces, cuts=(0.3, 0.6, 0.8, 0.97), late=0.6):
        """traces: (token ids up to where the thinking text starts, thinking token ids, finished, *answer_marks(...)).
        Cuts at or after `late` make up the stop read; all of them are split into post and pre."""
        self.S = S
        tk = S.tokenizer
        _, closing, revisit = family_tokens(S)
        self.jobs, self.starts, self.meta = [], [], []      # full traces read their own tokens; cuts read the closing tokens alone
        for i, (head, th, finished, first, lines) in enumerate(traces):
            self._add(head + th, len(head), (i, ("full",), [k for k, t in enumerate(th) if t in revisit]))
            bounds = [k for k, t in enumerate(th) if "\n" in tk.decode([t])]     # each cut ends a sentence, before its line break
            for q in cuts:
                if bounds:
                    k = bounds[min(len(bounds) - 1, int(q * len(bounds)))]
                    side = () if first is None else ("post",) if 0 <= first < k else ("pre",)
                    self._add(head + th[:k] + closing, len(head) + k, (i, side + (("cut",) if q >= late else ()), None))
            lines = [k for k in lines if k < len(th) - 2]                        # the last line of a finished trace is its natural end
            for k in sorted({lines[j] for j in (0, len(lines) // 2, len(lines) - 1)}) if lines else []:
                self._add(head + th[:k] + closing, len(head) + k, (i, ("ans",), None))
            if finished:
                self._add(head + th + closing, len(head) + len(th), (i, ("end",), None))
        self.tokens = sum(len(j) for j in self.jobs)
        self.base = self._read()

    def _add(self, seq, start, meta):
        self.jobs.append(seq)
        self.starts.append(start)
        self.meta.append(meta)

    def _read(self):
        out = {k: {} for k in ("cut", "end", "ans", "post", "pre", "all", "rev")}
        for (i, kinds, where), x in zip(self.meta, self.S.score(self.jobs, self.starts)):
            lp = np.asarray(x["logprobs"], dtype=np.float64)
            if kinds == ("full",):
                out["all"][i] = lp
                out["rev"][i] = lp[where]
            else:
                for kind in kinds:
                    out[kind].setdefault(i, []).append(float(lp.sum()))
        return out

    def base_stop(self):
        return tuple(float(np.mean([np.mean(v) for v in self.base[k].values()])) for k in ("cut", "end", "ans") if self.base[k])

    def effect(self):
        """Paired change from the base read under the weights now applied: d_<kind> and se_<kind>, for cut, end, ans, post, pre, all, rev."""
        cand = self._read()
        row = {}
        for kind in cand:
            d = [np.mean(cand[kind][i]) - np.mean(self.base[kind][i]) for i in self.base[kind] if np.size(self.base[kind][i])]
            row[f"d_{kind}"], row[f"se_{kind}"] = (float(np.mean(d)), float(np.std(d) / max(1, np.sqrt(len(d))))) if d else (0.0, 0.0)
        return row
