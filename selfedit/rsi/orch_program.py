"""The orchestration, as a program the proposer can edit later: routing, who talks, how many rounds, how disputes are decided. Standard library only; no file access. rec[member][category] = (right, n) on show+score rows outside this picture cluster; all[member] = (right, n)."""
ROUNDS = 1
MAX_CONSULT = 3
MARGIN = .10
SHORT = 500


def route(row, members, rec, all_):
    """Which members to consult. A member that abstains on this row is already out of `members`."""
    cat = row["category"]
    def score(m):
        r, n = rec.get(m, {}).get(cat, (0, 0))
        R, N = all_.get(m, (0, 0))
        prior = R / N if N else .5
        return (r + 5 * prior) / (n + 5)
    ranked = sorted(members, key=lambda m: -score(m))
    keep = [m for m in ranked if score(m) >= score(ranked[0]) - MARGIN][:MAX_CONSULT] if ranked else []
    return keep if len(keep) >= 2 else ranked[:2]


def who_talks(positions):
    """positions: member -> answer key. Every member in the dispute talks."""
    return sorted(positions)


def short(rec):
    """The part of a member's reasoning the others see."""
    text = (rec.get("reasoning") or "").strip() or (rec.get("content") or "").strip()
    return text[-SHORT:]


def talk_prompt(prompt, mine, others):
    """mine, others[i]: dict(answer, why)."""
    L = [prompt, "", "---", "You answered this question before, and other readers of the same question disagree with you.", "", f"Your earlier answer: {mine['answer']}", f"Your earlier reasoning ended: {mine['why']}", ""]
    for k, o in enumerate(others):
        L += [f"Reader {chr(65 + k)} answered: {o['answer']}", f"Their reasoning ended: {o['why']}", ""]
    L += ["Look at the image again yourself and check the exact point where these readings differ. The others can be wrong and so can you; do not go with a reading because of how sure it sounds. Keep your answer if what you see supports it, and change it if it does not. Give your final answer in the format the question asks."]
    return "\n".join(L)


def decide_prompt(prompt, finals):
    """finals[i]: dict(answer, why, changed). The orchestrator reads the final positions and may overrule them all."""
    L = [prompt, "", "---", "Several readers answered this question. You make the final decision and you can overrule all of them, including a majority. A reader's position counts only for what it points to in the image.", ""]
    for k, o in enumerate(finals):
        L += [f"Reader {chr(65 + k)}{' (changed its mind after seeing the others)' if o['changed'] else ''}: {o['answer']}", f"Reasoning ended: {o['why']}", ""]
    L += ["Look at the image yourself, check the points they dispute, and give the final answer in the format the question asks."]
    return "\n".join(L)
