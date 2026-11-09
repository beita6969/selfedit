# selfedit 1109c: NEW adapter (§8.2 item 7, §5). SkillStore: skills for the PROPOSER only (never executor prompts). Model-written, stock only (D5: no seeds).
# Edits go through SkillOpt's apply_patch_with_report (verbatim, vendored) with its protected APPENDIX region holding the code-written evidence;
# a new version must keep >= 85% of its parent's lines and every section (Self-HPO parent_check rule, section-level); lifecycle per §5.4.
from __future__ import annotations

import hashlib
import json
import os
import re

import yaml

import harness  # noqa: F401  (puts vendored skillopt on sys.path)
from skillopt.optimizer.skill import APPENDIX_END, APPENDIX_START, apply_patch_with_report

PATTERNS = ("prior_recitation", "rule_inversion", "missing_detail", "commit_early", "dispersion")
MAX_WORDS = 300
COVER = 0.85
SECTIONS = ("## When it applies", "## What the output must contain", "## Known failure modes", "## Read first")

# leakage denylist for skill text (§5.5); the critic's regex precheck uses the same list
DENY = [
    (r"\b[a-z0-9_]+/[A-Za-z0-9_.:-]*\d[A-Za-z0-9_.:-]*\b", "row id"),
    (r"\b(vlmsbiased|babyvision|lego|blink|charxiv|visulogic)\b", "bench name in a rule"),
    (r"\b(option|answer|letter|choice)\s*\(?(?-i:[A-H])\)?\s*(is|=|:)", "option letter given as an answer"),
    (r"\b(answer|option|choice|letter)\s*(is|=|:)\s*\(?(?-i:[A-H])\)?(?![A-Za-z])", "option letter given as an answer"),
    (r"\b(answer|count|is|equals|total)\s*(is\s*)?[:=]?\s*\d+\b", "numeral next to answer/count/is"),
]


def precheck(text):
    t = str(text or "")
    return [why for pat, why in DENY if re.search(pat, t, re.I)]


def lines_cover(parent, child):
    """Self-HPO parent_check (line coverage): share of the parent's non-empty lines that appear unchanged in the child."""
    pl = [l.strip() for l in parent.splitlines() if l.strip()]
    cl = set(l.strip() for l in child.splitlines() if l.strip())
    return 1.0 if not pl else sum(l in cl for l in pl) / len(pl)


def parent_check(parent_body, child_body):
    cov = lines_cover(parent_body, child_body)
    lost = [s for s in SECTIONS if s in parent_body and s not in child_body]
    return cov >= COVER and not lost, dict(coverage=round(cov, 3), lost_sections=lost)


def render(front, body, evidence):
    return "---\n" + yaml.safe_dump(front, sort_keys=False, allow_unicode=True) + "---\n" + body.strip() + "\n\n" + APPENDIX_START + "\n" + json.dumps(evidence, sort_keys=True) + "\n" + APPENDIX_END + "\n"


def parse(text):
    m = re.match(r"---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        raise ValueError("no front matter")
    front = yaml.safe_load(m.group(1)) or {}
    rest = m.group(2)
    ev = {}
    if APPENDIX_START in rest:
        body, tail = rest.split(APPENDIX_START, 1)
        ev = json.loads(tail.split(APPENDIX_END, 1)[0].strip() or "{}")
    else:
        body = rest
    for k in ("id", "version", "status", "trigger", "procedure"):
        if k not in front:
            raise ValueError(f"front matter lacks {k}")
    if (front.get("trigger") or {}).get("pattern") not in PATTERNS:
        raise ValueError(f"trigger.pattern must be one of {PATTERNS}")
    return front, body.strip(), ev


class SkillStore:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    # ------------------------------------------------------------- storage
    def ids(self):
        return sorted(d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d)))

    def versions(self, sid):
        d = os.path.join(self.root, sid)
        return sorted((int(f[1:-3]) for f in os.listdir(d) if re.fullmatch(r"v\d+\.md", f)))

    def load(self, sid, v=None):
        v = v or self.versions(sid)[-1]
        return parse(open(os.path.join(self.root, sid, f"v{v}.md")).read())

    def latest(self):
        out = []
        for sid in self.ids():
            try:
                f, b, e = self.load(sid)
                out.append(dict(front=f, body=b, evidence=e))
            except (ValueError, OSError, IndexError):
                continue
        return out

    def _write(self, sid, v, text):
        d = os.path.join(self.root, sid)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"v{v}.md")
        assert not os.path.exists(p), f"skill versions are append-only: {p} exists"
        open(p, "w").write(text)
        return p

    # ------------------------------------------------------------- writing
    def write(self, edit, round_, code_sha=""):
        """One reflector skill edit: {"id", "new": true, "pattern", "scope", "detector", "procedure": [...], "body"} or {"id", "edits": [SkillOpt Edit dicts]}.
        Returns (status, info). New skills enter probation; any denylist hit quarantines; an amendment is a new version that must pass parent_check."""
        sid = re.sub(r"[^a-z0-9_]", "_", str(edit.get("id") or "").lower())[:40]
        if not sid:
            return "rejected", dict(why="no id")
        if edit.get("new") or sid not in self.ids():
            body = str(edit.get("body") or "")
            if len(body.split()) > MAX_WORDS:
                return "rejected", dict(why=f"body over {MAX_WORDS} words")
            front = dict(id=sid, version=1, parent=None, status="probation",
                         trigger=dict(pattern=edit.get("pattern"), scope=edit.get("scope"), detector=edit.get("detector")),
                         preconditions=edit.get("preconditions") or [], procedure=[str(x) for x in (edit.get("procedure") or [])][:8],
                         calls=edit.get("calls") or {}, budget=dict(max_engine_edits=2, max_calls=4), compat=dict(model="Qwen3.8-27B", max_tokens=32768, effort="medium"))
            hits = precheck(body + "\n" + "\n".join(front["procedure"]) + "\n" + str(front["trigger"]))
            if hits:
                front["status"] = "quarantined"
            ev = dict(rounds_used=[], proposals=0, passed_null=0, born=round_, provenance=dict(author="model", round=round_, seed=False, code_sha=code_sha, denylist_hits=hits))
            text = render(front, body, ev)
            parse(text)                 # schema
            self._write(sid, 1, text)
            return front["status"], dict(id=sid, version=1, denylist=hits)
        front, body, ev = self.load(sid)
        if front["status"] in ("retired", "quarantined"):
            return "rejected", dict(why=f"skill is {front['status']}")
        old = open(os.path.join(self.root, sid, f"v{front['version']}.md")).read()
        new, rep = apply_patch_with_report(old, dict(edits=edit.get("edits") or []))
        try:
            f2, b2, e2 = parse(new)
        except ValueError as e:
            return "rejected", dict(why=f"schema after edit: {e}", report=rep)
        ok, pc = parent_check(body, b2)
        if not ok or e2 != ev:
            return "rejected", dict(why="parent_check failed (rewrite)" if not ok else "evidence region changed", parent_check=pc, report=rep)
        hits = precheck(b2)
        f2.update(version=front["version"] + 1, parent=f"{sid}/v{front['version']}", status="quarantined" if hits else front["status"])
        self._write(sid, f2["version"], render(f2, b2, ev))
        return f2["status"], dict(id=sid, version=f2["version"], parent_check=pc, report=rep, denylist=hits)

    def set_meta(self, sid, status=None, evidence=None):
        """Code-only update (status, evidence) as a new version with the same body."""
        f, b, e = self.load(sid)
        f = dict(f, version=f["version"] + 1, parent=f"{sid}/v{f['version']}", status=status or f["status"])
        self._write(sid, f["version"], render(f, b, dict(e, **(evidence or {}))))

    # ------------------------------------------------------------- selection
    def match(self, pattern=None, scope=None, k=2, force_unused=False):
        """Candidates: probation/active skills whose trigger matches (pattern, or any when None); scored by Beta(1+passed, 1+proposals-passed) mean.
        At most k; with force_unused one never-used skill takes the last place (RRSI exploration slot). Code chooses, not the model."""
        c = [s for s in self.latest() if s["front"]["status"] in ("probation", "active")
             and (pattern is None or s["front"]["trigger"].get("pattern") == pattern) and (scope is None or s["front"]["trigger"].get("scope") in (None, scope))]
        sc = lambda s: (1 + s["evidence"].get("passed_null", 0)) / (2 + s["evidence"].get("proposals", 0))
        c = sorted(c, key=lambda s: (-sc(s), s["front"]["id"]))
        out = c[:k]
        if force_unused:
            un = [s for s in c if not s["evidence"].get("proposals") and s not in out]
            if un:
                out = out[: k - 1] + un[:1]
        return out

    @staticmethod
    def block(skills):
        """The text injected into a PROPOSER prompt (Run.prompts user text)."""
        if not skills:
            return ""
        L = ["\n\n## Skills written by you in earlier rounds (procedures for finding and building a change; use one if it fits, ignore otherwise)"]
        for s in skills:
            f = s["front"]
            L.append(f"### skill {f['id']} v{f['version']} ({f['status']}) - when: {f['trigger'].get('pattern')}, scope {f['trigger'].get('scope')}; detector: {f['trigger'].get('detector')}")
            L += [f"{i + 1}. {p}" for i, p in enumerate(f.get("procedure") or [])]
            L.append(s["body"][:1500])
        L.append('If you follow a skill, put its id in your JSON answer as "skill": "<id>".')
        return "\n".join(L)

    # ------------------------------------------------------------- lifecycle (§5.4)
    def lifecycle(self, r, used, passed, prune_ids=()):
        """used: {skill id: proposals this round}; passed: {skill id: members admitted in stage 2 this round}. Returns transitions."""
        out = []
        for s in self.latest():
            f, e = s["front"], dict(s["evidence"])
            sid = f["id"]
            if f["status"] not in ("probation", "active"):
                continue
            if used.get(sid):
                e["proposals"] = e.get("proposals", 0) + used[sid]
                e["rounds_used"] = sorted(set(e.get("rounds_used", [])) | {r})
            if passed.get(sid):
                e["passed_null"] = e.get("passed_null", 0) + passed[sid]
                e.setdefault("passed_rounds", []).append(r)
            born = e.get("born", r)
            st = f["status"]
            if st == "probation":
                if e.get("rounds_used") and e.get("passed_null", 0) >= 1:
                    st = "active"
                elif (r - born >= 2 and not e.get("rounds_used")) or (r - born >= 4 and not e.get("passed_null")):
                    st = "retired"
            elif st == "active" and sid in set(prune_ids):
                st = "retired"
            if st != f["status"] or e != s["evidence"]:
                self.set_meta(sid, status=st, evidence=e)
            if st != f["status"]:
                out.append(dict(id=sid, frm=f["status"], to=st, round=r))
        return out

    def sha(self):
        h = hashlib.sha256()
        for sid in self.ids():
            for v in self.versions(sid):
                h.update(open(os.path.join(self.root, sid, f"v{v}.md"), "rb").read())
        return h.hexdigest()
