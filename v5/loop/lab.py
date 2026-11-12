"""selfedit v4, proposer lab (coordinator item S2-A): a proposer slot becomes an episode instead of one shot.

  turn 1   card -> draft entry (the existing P0 chat of step_propose, key p["key"]);
  lab      the draft is routed on the card's shown questions and on the other SHOW rows of the card's source, and read LAB_SAMPLES
           times (samples 0 .. LAB_SAMPLES - 1, priority 1) with tip entries + draft on every routed row (AM18). The reads close
           when CLOSE_ALL of them are in, or CLOSE_AFTER s after CLOSE_MOST of them are in, whichever is first (never after the
           deadline); a read with an error line counts as in. The proposer gets back, per shown question ("Question k" as numbered
           in its card): router yes / no with p, fractions only ("Tip right 3 of 8 samples; with your entry right 2 of 4 samples";
           a row with fewer samples in shows what it has, e.g. "2 of 3"), and the end of the executor's reasoning and final text of
           one sample with the entry. Per group (shown questions; the source's other SHOW rows, never shown to it) one summary line:
           the mean over routed rows with tip samples and >= 1 entry sample of (entry accuracy - tip accuracy), its standard error
           (sd / sqrt(n), n >= 2) and n. No transition counts and no verdict words (AM18: one sample misleads);
  turn 2   the proposer keeps, revises or replaces the entry; with lab_turns = 2 the lab runs again on the turn-2 entry and a
           turn 3 follows. The last answered turn's key is p["key_final"]; step_validate reads final_key(p). A replacement after
           the last lab turn gets no further lab; the candidate records lab_tested (True: its final entry is one a lab turn ran;
           False: the lab ran, the final entry is not one it ran; None: no lab turn ran).

lab_turns in {0, 1, 2} is a per-slot option (p["lab_turns"], a proposer version's card option; P0 stays at 0). All routes of one
lab turn go out as ONE batch for all slots, then all reads as ONE batch, then all chat turns as ONE batch. Lab rows are SHOW rows
only; JUDGE and MONITOR rows are never routed, read or shown by the lab, and lab reads never enter the judge's D. The feedback is
data, not advice (AM4): a neutral description of what was run, the numbers, the executor's own text tails, the request to keep,
revise or replace, and the answer format. Every turn's full message list passes proposer.guard_messages before it is sent.

Resume: the turn requests of all slots are written to <run>/prompts_lab_g###.jsonl before they are submitted (one line per
(turn, slot), the whole file replaced atomically per turn). On re-entry a turn that is in the file is never rebuilt: its stored
requests are resubmitted (idempotent by key) and waited for. Keys: <previous key>|lab<t>|<sha16(messages)[:8]>.

Mixed into gen.Run by the builder (class Run(LabMixin, ...) or method assignment). Uses only Run's attributes and methods:
rows, role, state, tip_view, io, g, deadline, rd, route_reqs, load_routes, fires, read_req, samples, check, G. Defines no method
that gen.Run has. Does not import gen (gen will import this module)."""
import hashlib
import json
import math
import os
import re
import time

import proposer as PR

LAB_MAX = 2                         # lab_turns is clamped to 0..2
LAB_CTX = 46000 - 512               # engine max_model_len minus a margin: prompt estimate + max_tokens of a lab turn must fit
QHEAD = re.compile(r"^### Question (\d+): ", re.M)
LAB_SAMPLES = 4                     # AM18 (a choice): samples of the draft read on every routed lab row, s = 0 .. LAB_SAMPLES - 1
CLOSE_ALL = 0.99                    # AM18 (a choice): the lab reads close when this share of them is in ...
CLOSE_MOST = 0.90                   # AM18 (a choice): ... or CLOSE_AFTER seconds after this share of them is in, whichever is first
CLOSE_AFTER = 300.0                 # AM18 (a choice): 5 minutes; never later than the generation's deadline


def _sha16(x):
    return hashlib.sha256(x.encode()).hexdigest()[:16]


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _now():
    return time.time()


def _sleep(s):
    time.sleep(s)


def wait_close(io, keys, deadline=None, every=5.0, label="", tick=300, log=_log):
    """AM18 close rule for the lab reads (polls io instead of io.wait): returns when every key is in, when at least CLOSE_ALL of them
    are in, CLOSE_AFTER s after at least CLOSE_MOST of them were first seen in, or at the deadline, whichever comes first. A key is in
    when io has its result or an error line for it. Returns (done = keys with a result, missing = keys not in, info)."""
    keys = list(dict.fromkeys(keys))
    n = len(keys)
    t0 = last = _now()
    t_most = None
    while True:
        io.poll()
        err = getattr(io, "err", None) or {}
        miss = [k for k in keys if io.get(k) is None and k not in err]
        k_in = n - len(miss)
        t = _now()
        if t_most is None and k_in >= CLOSE_MOST * n:
            t_most = t
        why = ("all in" if not miss else
               f"{CLOSE_ALL:.0%} in" if k_in >= CLOSE_ALL * n else
               f"{CLOSE_AFTER:.0f} s after {CLOSE_MOST:.0%} in" if t_most is not None and t - t_most >= CLOSE_AFTER else
               "deadline" if deadline is not None and t >= deadline else None)
        if why:
            info = dict(n=n, k_in=k_in, errors=sum(1 for k in keys if io.get(k) is None and k in err), why=why, waited_s=round(t - t0, 1),
                        most_in_s=None if t_most is None else round(t_most - t0, 1))
            return [k for k in keys if io.get(k) is not None], miss, info
        if t - last >= tick:
            last = t
            log(f"  waiting {label}: {k_in}/{n} in after {t - t0:.0f}s")
        _sleep(every)


def final_key(p):
    """The proposer answer step_validate uses for slot p: the last answered lab turn, else the turn-1 (P0 chat) answer."""
    return p.get("key_final") or p["key"]


def _turns(p):
    try:
        return max(0, min(LAB_MAX, int(p.get("lab_turns") or 0)))
    except (TypeError, ValueError):
        return 0


def _text(content):
    return PR._flatten(content, [])


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _acc(vs):
    """Share of graded samples that are right (None without samples)."""
    return sum(bool(v["ok"]) for v in vs) / len(vs) if vs else None


def _msg_tokens(m):
    """Prompt-token estimate of one chat message: text at proposer.CHARS_PER_TOKEN, each image part at its max (16384 if null)."""
    c = m.get("content")
    if isinstance(c, str):
        return len(c) / PR.CHARS_PER_TOKEN + 8
    t = 8.0
    for x in c or []:
        if x.get("type") == "image":
            t += x.get("max_image_tokens") or 16384
        else:
            t += len(str(x.get("text", ""))) / PR.CHARS_PER_TOKEN
    return t


class LabMixin:
    LAB_CTX = LAB_CTX
    lab_every = 5.0                     # poll interval of the lab reads' close rule (s); dry runs with fast polling set 0.2
    final_key = staticmethod(final_key)

    # ------------------------------------------------------------------ files
    def lab_file(self):
        return os.path.join(self.rd, f"prompts_lab_g{self.g:03d}.jsonl")

    def lab_load(self):
        """{(turn, slot): record} of this generation's lab file ({} if none)."""
        f, out = self.lab_file(), {}
        if os.path.exists(f):
            for l in open(f):
                if l.strip():
                    r = json.loads(l)
                    out[(r["turn"], r["slot"])] = r
        return out

    def lab_write(self, recs):
        f = self.lab_file()
        with open(f + ".tmp", "w") as fh:
            for k in sorted(recs):
                fh.write(json.dumps(recs[k], ensure_ascii=False) + "\n")
        os.replace(f + ".tmp", f)

    def lab_request(self, key):
        """The stored chat request (key, messages, ...) of a proposer key: a P0 key from prompts_g###.jsonl or a lab key from
        prompts_lab_g###.jsonl; None if neither has it. gen.Run.repair needs this for a lab-finished slot (its final key is a lab key)."""
        if "|lab" in key:
            for r in self.lab_load().values():
                if r.get("key") == key:
                    return r.get("req")
            return None
        return self.lab_p0_requests().get(key)

    def lab_p0_requests(self):
        """{key: request} of this generation's prompts_g###.jsonl (written once by step_propose; cached per generation)."""
        c = getattr(self, "_lab_p0", None)
        if c is None or c[0] != (self.rd, self.g):
            pf, d = os.path.join(self.rd, f"prompts_g{self.g:03d}.jsonl"), {}
            if os.path.exists(pf):
                for l in open(pf):
                    if l.strip():
                        q = json.loads(l)
                        d[q["key"]] = q
            c = self._lab_p0 = ((self.rd, self.g), d)
        return c[1]

    # ------------------------------------------------------------------ rows of a slot
    def lab_numbering(self, card, messages):
        """[(k, row id)] of the questions shown on the card, k = the number "### Question k" of the card text. The card text numbers
        questions in card-id order; questions dropped to fit a budget are not in the text. Matched in order on question text + gold
        lines (identical texts are common), so the k-th block maps to the next card id whose text it holds. None if the text has no
        consecutive Question 1..n blocks or a block matches no remaining card id (fail closed: the slot gets no lab)."""
        user = next((m["content"] for m in messages if m.get("role") == "user"), None)
        if user is None:
            return None
        t = _text(user)
        heads = list(QHEAD.finditer(t))
        if not heads or [int(m.group(1)) for m in heads] != list(range(1, len(heads) + 1)):
            return None
        out, j = [], 0
        for i, m in enumerate(heads):
            blk = t[m.end():heads[i + 1].start() if i + 1 < len(heads) else len(t)]
            while j < len(card):
                r = self.rows[card[j]]
                if "\n".join(["Question text:", str(r["prompt"]).strip()] + PR._gold_lines(r)) + "\n" in blk + "\n":
                    break
                j += 1
            if j == len(card):
                return None
            out.append((int(m.group(1)), card[j]))
            j += 1
        return out

    def lab_rows(self, p, messages):
        """(numbered shown [(k, rid)], other SHOW rows of the card's source not shown, sources) or (None, reason). Other rows need at
        least one graded tip sample (right-before needs it), as card rows do."""
        card = list(p.get("card") or [])
        if not card:
            return None, "no card"
        if any((self.rows.get(r) or {}).get("v4_role") != "show" for r in card):
            return None, "card has rows that are not SHOW rows"
        num = self.lab_numbering(card, messages)
        if not num:
            return None, "the card text's question numbering does not match the card ids"
        srcs = [p["card_source"]] if p.get("card_source") else sorted({self.rows[r]["bench"] for r in card})
        shown = {r for _, r in num}
        tip = self.state["tip"]["entries"]
        other = [r for r in self.role["show"] if self.rows[r]["bench"] in srcs and r not in shown
                 and PR._samples({r: self.samples(tip, r)}, r)]
        assert all(self.rows[r]["v4_role"] == "show" for r in list(shown) + other), "lab rows must be SHOW rows only"
        return (num, other, srcs), None

    # ------------------------------------------------------------------ entry of an answer
    def lab_entry(self, x, tv, card):
        """(entry, note, skip reason) of a proposer answer. A valid answer gives its entry and no note. A repairable-only failure
        (format limit or naming) still gives an entry when the condition is a non-empty string and the payload is not empty; the note
        is the validator message, shown to the proposer. Anything else: skip reason."""
        if x is None:
            return None, None, "no answer to the previous turn"
        cand, err, fix = self.check(x, tv, card)
        if cand is None:
            return None, None, f"answer does not parse: {err}"
        if err is not None and not fix:
            return None, None, err
        c = cand.get("condition")
        if not isinstance(c, str) or not c.strip() or not (cand.get("procedure") or cand.get("settings") or cand.get("ops")):
            return None, None, f"no entry can be made: {err}"
        e = dict(condition=c, cond="always" if c == "always" else _sha16(c), procedure=cand.get("procedure"), settings=cand.get("settings") or {},
                 ops=PR.canon_ops(cand.get("ops")))
        return e, (err[len("invalid: "):] if err and err.startswith("invalid: ") else err), None

    # ------------------------------------------------------------------ feedback
    def lab_keys(self, entry, r):
        """AM18: read keys of the draft on row r (tip entries + draft), samples 0 .. LAB_SAMPLES - 1."""
        tip = self.state["tip"]["entries"]
        return [self.read_req(tip + [entry], r, s, 1)[0]["key"] for s in range(LAB_SAMPLES)]

    def lab_row(self, entry, r):
        """Lab result of one row: routed, p, router answered, the tip's graded samples ps and the entry's graded samples xs (those in;
        error lines and missing reads are not samples), samples asked, tip accuracy and entry accuracy (share of samples right)."""
        tip = self.state["tip"]["entries"]
        always = entry["cond"] == "always"
        routed = self.fires(entry, r)
        ps = PR._samples({r: self.samples(tip, r)}, r)
        xs = PR._samples({r: [self.io.get(k) for k in self.lab_keys(entry, r)]}, r) if routed else []
        return dict(routed=routed, p=None if always else self.route_p.get((entry["cond"], r)), answered=always or (entry["cond"], r) in self.routes,
                    ps=ps, xs=xs, asked=LAB_SAMPLES if routed else 0, tip_acc=_acc(ps), acc=_acc(xs))

    @staticmethod
    def lab_counts(res):
        """Numbers over row results (AM18): rows, routed, read (routed with tip samples and >= 1 entry sample: the rows of the mean),
        missing (routed without any entry sample), partial (routed with some but not all entry samples), entry samples asked / in,
        right / total samples of tip and entry on the read rows, and mean, sd, se (sd / sqrt(n), n >= 2) of (entry accuracy - tip accuracy)
        over the read rows."""
        R = [z for z in res if z["routed"]]
        D = [z for z in R if z["xs"] and z["ps"]]
        d = [z["acc"] - z["tip_acc"] for z in D]
        n = len(d)
        mean = sum(d) / n if n else None
        sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n >= 2 else None
        return dict(n=len(res), routed=len(R), read=n, missing=sum(1 for z in R if not z["xs"]), partial=sum(1 for z in R if 0 < len(z["xs"]) < z["asked"]),
                    asked=sum(z["asked"] for z in R), samples_in=sum(len(z["xs"]) for z in R),
                    tip_right=sum(sum(bool(v["ok"]) for v in z["ps"]) for z in D), tip_samples=sum(len(z["ps"]) for z in D),
                    entry_right=sum(sum(bool(v["ok"]) for v in z["xs"]) for z in D), entry_samples=sum(len(z["xs"]) for z in D),
                    mean_diff=mean, sd_diff=sd, se_diff=(sd / math.sqrt(n)) if sd is not None else None)

    @staticmethod
    def _count_lines(c, always, what):
        L = [f"Your entry ran on all {c['n']} {what}." if always else f"The condition held on {c['routed']} of {c['n']} {what}."]
        if not c["routed"]:
            return L
        n = c["read"]
        if not n:
            L.append("Mean of (accuracy with your entry - accuracy of the tip): no question with samples of both the tip and your entry.")
        else:
            m = (f"Mean of (accuracy with your entry - accuracy of the tip) over the {n} question{'' if n == 1 else 's'} with samples of both the tip "
                 f"and your entry: {c['mean_diff']:+.3f}")
            m += (f", standard error {c['se_diff']:.3f} (sd / sqrt(n)), n = {n}." if c["se_diff"] is not None else
                  f"; standard error not computed (n = {n}; it needs n >= 2).")
            L.append(m)
        if c["samples_in"] < c["asked"]:
            L.append(f"Samples with your entry available in time: {c['samples_in']} of {c['asked']} ({LAB_SAMPLES} per question where your entry ran).")
        return L

    @staticmethod
    def _frac(vs):
        return f"right {sum(bool(v['ok']) for v in vs)} of {len(vs)} sample{'' if len(vs) == 1 else 's'}"

    def lab_feedback(self, entry, num, other, srcs, note, last):
        """(user message text, stats) of one lab turn for one slot. Data only (AM4, AM18): what was run, router decisions with p, per shown
        question the tip's and the entry's samples as fractions (never a verdict from one sample), the executor's text tails of one
        entry sample on shown questions (tails as on the card), and per group the mean difference of accuracies with se and n."""
        always = entry["cond"] == "always"
        src = "the same source" if len(srcs) == 1 else "the same sources as the shown questions"
        n = len(num)
        ns = f"{LAB_SAMPLES} times ({LAB_SAMPLES} samples)"
        L = ["## Trial of your entry"]
        s = f"Your entry was added to the tip and run on the {n} shown question{'' if n == 1 else 's'} above"
        s += f" and on {len(other)} other question{'' if len(other) == 1 else 's'} from {src} that {'is' if len(other) == 1 else 'are'} not shown to you." if other else "."
        if always:
            s += f' Your condition is "always", so it holds on every question, and the executor answered each question {ns} with the tip and your entry.'
        else:
            s += (" For each question the router answered your condition; p is the router's probability of yes, and the condition holds when"
                  f" p is at least 0.5. On each question where the condition held, the executor answered {ns} with the tip and your entry;"
                  " on the other questions the entry has no effect, and nothing was run.")
        s += (" Samples that were not available in time are left out, so a question can have fewer. The accuracy on a question is the share"
              " of its samples that are right. These trial answers are not part of the judgement.")
        L.append(s)
        if note:
            L.append(f"Your entry as written cannot be judged: {note}. It was run as written.")
        L += ["", "## Shown questions with your entry"]
        shown_res = []
        for k, r in num:
            z = self.lab_row(entry, r)
            shown_res.append(z)
            if always:
                head = "always holds"
            elif not z["answered"]:
                head = "the router gave no answer in time (taken as not holding)"
            else:
                head = ("the condition held" if z["routed"] else "the condition did not hold") + (f" (p = {z['p']:.2f})" if _num(z["p"]) else "")
            tipl = f"Tip {self._frac(z['ps'])}" if z["ps"] else "Tip: no samples"
            L += ["", f"### Question {k}: {head}"]
            if not z["routed"]:
                L.append(tipl + ". Your entry has no effect on this question.")
            elif not z["xs"]:
                L.append(tipl + "; with your entry: no sample was available in time.")
            else:
                xs = z["xs"]
                L.append(tipl + f"; with your entry {self._frac(xs)}." +
                         (f" ({z['asked'] - len(xs)} of the {z['asked']} samples not available in time.)" if len(xs) < z["asked"] else ""))
                x = xs[0]
                L += [f"- Sample {PR._sidx(x) + 1} with your entry ({'right' if x['ok'] else 'wrong'})"] + PR._sample_lines(1, dict(x, ok=bool(x["ok"])))[1:]
        other_res = [self.lab_row(entry, r) for r in other]
        cs, co = self.lab_counts(shown_res), self.lab_counts(other_res)
        L += ["", "Summary of the shown questions:"] + self._count_lines(cs, always, "shown questions")
        if other:
            L += ["", f"## Other questions from {src} ({len(other)}, not shown to you)"] + self._count_lines(co, always, "other questions")
        L += ["", "## Your next answer",
              "Keep your entry, revise it or replace it with a different one. " +
              ("The entry in your next answer is the one that is judged." if last else
               "The entry in your next answer is tried in the same way once more, and then you answer once again; the entry in that last answer is the one that is judged."),
              "", PR.ANSWER_FORMAT]
        toks = [(sum(v.get("tokens") or 0 for v in z["xs"]), len(z["xs"]), sum(v.get("tokens") or 0 for v in z["ps"]) / len(z["ps"]))
                for z in shown_res + other_res if z["xs"] and z["ps"]]
        st = dict(shown=cs, other=co, samples=LAB_SAMPLES, read_tokens=sum(sum(v.get("tokens") or 0 for v in z["xs"]) for z in shown_res + other_res),
                  tok_entry=round(sum(a for a, _, _ in toks) / sum(m for _, m, _ in toks), 1) if toks else None,
                  tok_tip=round(sum(b for _, _, b in toks) / len(toks), 1) if toks else None)
        return "\n".join(L), st

    # ------------------------------------------------------------------ the step
    def lab_build(self, t, slot_props, recs, tv):
        """New records of lab turn t for the slots in slot_props (none of them in recs yet): routes (one batch), reads (one batch),
        feedback, guarded and size-checked chat requests. Nothing is submitted to the chat engine here."""
        tip = self.state["tip"]["entries"]
        new, act = {}, []
        for p in slot_props:
            sk = (t, p["slot"])
            prev = p["key"] if t == 1 else (recs.get((t - 1, p["slot"])) or {}).get("key")
            base = dict(turn=t, slot=p["slot"], author=p["author"], prev=prev)
            if not prev:
                new[sk] = dict(base, skip=f"no turn {t} (the lab stopped at turn {t - 1})")
                continue
            q0 = self.lab_p0_requests().get(p["key"])
            q1 = q0 if t == 1 else recs[(t - 1, p["slot"])].get("req")
            if q0 is None or q1 is None:
                new[sk] = dict(base, skip="the previous request is not in the prompt files")
                continue
            x = self.io.get(prev)
            e, note, why = self.lab_entry(x, tv, p.get("card"))
            if why:
                new[sk] = dict(base, skip=why[:300])
                continue
            ek = json.dumps(PR._entry_key(e))
            if t > 1 and recs[(t - 1, p["slot"])].get("entry_key") == ek:
                new[sk] = dict(base, skip="entry kept unchanged: the previous trial already shows it")
                continue
            rows, why = self.lab_rows(p, q0["messages"])
            if why:
                new[sk] = dict(base, skip=why)
                continue
            act.append((p, sk, base, q1, x, e, ek, note) + rows)
        if act and time.time() >= self.deadline:
            for a in act:
                new[a[1]] = dict(a[2], skip="the generation box closed before the lab")
            act = []
        rq = [q for a in act for q in self.route_reqs(a[5], [r for _, r in a[8]] + a[9])]
        nr = 0
        if rq:
            nr = self.io.submit(f"g{self.g:03d}_labroute{t}", rq)
            _, miss = self.io.wait([q["key"] for q in rq], self.deadline, label=f"lab {t} routes")
            if miss:
                _log(f"  lab {t}: {len(miss)} routes unanswered at the deadline (taken as no)")
        for a in act:
            self.load_routes([a[5]], [r for _, r in a[8]] + a[9])
        rd = [self.read_req(tip + [a[5]], r, s, 1)[0] for a in act for r in [r for _, r in a[8]] + a[9] if self.fires(a[5], r)
              for s in range(LAB_SAMPLES)]                                # AM18: LAB_SAMPLES samples of the draft on every routed lab row
        nd, close = 0, None
        if rd:
            nd = self.io.submit(f"g{self.g:03d}_labread{t}", rd)
            _, miss, close = wait_close(self.io, [q["key"] for q in rd], self.deadline, every=self.lab_every, label=f"lab {t} reads")
            if miss:
                _log(f"  lab {t}: reads closed ({close['why']}) after {close['waited_s']:.0f} s with {close['k_in']} of {close['n']} in; "
                     f"{len(miss)} not in (left out of the feedback)")
        _log(f"  lab {t}: {len(act)} slots, routes {len(rq)} ({nr} new), reads {len(rd)} ({nd} new)")
        if act and time.time() >= self.deadline:                     # no chat turn is queued that could not come back in the box
            for a in act:
                new[a[1]] = dict(a[2], skip="the generation box closed during the lab reads")
            act = []
        for p, sk, base, q1, x, e, ek, note, num, other, srcs in act:
            last = t >= _turns(p)
            text, st = self.lab_feedback(e, num, other, srcs, note, last)
            msgs = q1["messages"] + [dict(role="assistant", content=(x or {}).get("content") or ""), dict(role="user", content=text)]
            key = f"{base['prev']}|lab{t}|{_sha16(json.dumps(msgs, ensure_ascii=False, sort_keys=True))[:8]}"
            req = dict({k: v for k, v in q1.items() if k not in ("key", "messages")}, key=key, messages=msgs)
            rec = dict(base, key=key, entry=e, entry_key=ek, note=note, shown_ids=[r for _, r in num], n_other=len(other), sources=srcs, chars=len(text),
                       close=close, **st)
            try:
                PR.guard_messages(msgs, self.rows)
            except (PR.LeakError, ValueError) as ex:
                new[sk] = dict(base, skip=f"guard: {ex}"[:300])
                continue
            pt = (x or {}).get("prompt_tokens")
            est = (pt + sum(_msg_tokens(m) for m in msgs[len(q1["messages"]):])) if _num(pt) else sum(_msg_tokens(m) for m in msgs)
            mt = req.get("max_tokens") or 12000
            if est + mt > self.LAB_CTX:
                new[sk] = dict(base, skip=f"context: ~{est:.0f} prompt tokens + max_tokens {mt} > {self.LAB_CTX}")
                continue
            new[sk] = dict(rec, req=req, est_prompt_tokens=round(est))
        return new

    def step_lab(self):
        """Lab turns for every slot with lab_turns >= 1 (between propose and validate). Sets p["key_final"] (last answered lab turn)
        and p["lab"] (one small record per turn) on G["props"]."""
        G = self.G()
        if "validate" in G.get("done", []):
            _log("  lab: validate already ran this generation; the lab is not run after it")
            return
        props = sorted((p for p in G.get("props", []) if p.get("key") and _turns(p) >= 1), key=lambda p: p["slot"])
        if not props:
            G["lab"] = dict(slots=0)
            return
        T = max(_turns(p) for p in props)
        tv = self.tip_view()
        recs = self.lab_load()
        for t in range(1, T + 1):
            todo = [p for p in props if _turns(p) >= t]
            miss = [p for p in todo if (t, p["slot"]) not in recs]
            if miss:
                recs.update(self.lab_build(t, miss, recs, tv))
                self.lab_write(recs)                                  # before any chat request of this turn is submitted
            reqs = [recs[(t, p["slot"])]["req"] for p in todo if recs[(t, p["slot"])].get("req")]
            n = self.io.submit(f"g{self.g:03d}_labchat{t}", reqs) if reqs else 0
            skips = [f"slot {p['slot']}: {recs[(t, p['slot'])]['skip'][:80]}" for p in todo if recs[(t, p["slot"])].get("skip")]
            _log(f"  lab {t}: {len(reqs)} chat turns ({n} new)" + (f"; skipped {len(skips)}: " + "; ".join(skips) if skips else "")
                 + ("" if miss else " (reused from the lab file)"))
            if reqs:
                _, m = self.io.wait([q["key"] for q in reqs], self.deadline, label=f"lab {t} chat")
                if m:
                    _log(f"  lab {t}: {len(m)} chat turns unanswered at the deadline (those slots keep their previous answer)")
        tot = dict(slots=len(props), turns=T, sent=0, answered=0, routed=0, read_tokens=0, final_from_lab=0)
        for p in props:
            out, kf = [], None
            for t in range(1, _turns(p) + 1):
                r = recs.get((t, p["slot"]))
                if r is None:
                    break
                if r.get("skip"):
                    out.append(dict(turn=t, skip=r["skip"]))
                    break
                x = self.io.get(r["key"])
                sm = dict(turn=t, key=r["key"], shown=r["shown"], other=r["other"], read_tokens=r["read_tokens"], tok_entry=r["tok_entry"],
                          tok_tip=r["tok_tip"], ran_invalid=bool(r.get("note")), answered=x is not None, answer_tokens=(x or {}).get("tokens"),
                          entry_key=r.get("entry_key"), samples=r.get("samples", 1))
                tot["sent"] += 1
                tot["routed"] += r["shown"]["routed"] + r["other"]["routed"]
                tot["read_tokens"] += r["read_tokens"] or 0
                if r.get("close"):
                    tot.setdefault("close", {}).setdefault(str(t), r["close"])
                if x is None:
                    out.append(sm)
                    break
                _, err, _ = self.check(x, tv, p.get("card"))
                sm["answer"] = "valid" if err is None else err[:160]
                out.append(sm)
                kf = r["key"]
                tot["answered"] += 1
            if kf:
                p["key_final"] = kf
                tot["final_from_lab"] += 1
            else:
                p.pop("key_final", None)
            p["lab"] = out
        G["lab"] = tot
        _log(f"  lab: {tot['slots']} slots, {tot['sent']} turns sent, {tot['answered']} answered, {tot['routed']} routed lab rows, "
             f"{tot['final_from_lab']} slots end on a lab turn")

    def lab_tested(self, p, cand, recs=None):
        """AM18 candidate field lab_tested of slot p with final parsed answer cand (after any repair): None when no lab turn ran an entry
        for the slot (lab_turns 0, or every turn skipped before its feedback was built); True when the final entry is one a lab turn ran
        (same entry key); False when the lab ran but the final entry is not one it ran ("final entry not lab-tested": replaced after the
        last lab turn, or no final entry). A replacement gets no further lab turn: lab_turns stays what the version says."""
        if _turns(p) < 1:
            return None
        recs = self.lab_load() if recs is None else recs
        ran = {r["entry_key"] for (t, s), r in recs.items() if s == p["slot"] and t <= _turns(p) and r.get("key") and r.get("entry_key")}
        if not ran:
            return None
        if cand is None:
            return False
        return json.dumps(PR._entry_key(cand)) in ran
