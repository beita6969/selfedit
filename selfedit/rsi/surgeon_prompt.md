You are operating on yourself. The weights you are running on right now ("generation {gen}") are the patient. Your job this round is to propose concrete edits to your OWN internal parameters that make you better at {target_family} without making you worse elsewhere. Nothing you write here becomes a prompt or setting for the future model; only weight edits you propose and that pass evaluation will be kept.

## What you can touch (edit menu)
{edit_menu}

Each edit is applied to the real weights, evaluated on held-out problems with fixed decoding, and kept only if it measurably helps. Failed edits are reverted exactly; they cost evaluation budget but do no harm.

## What we know about you
### Performance on the selection pool (generation {gen})
{pool_summary}

### Typical failures (excerpts of your own outputs)
{failure_excerpts}

### Edit history (all generations so far)
{edit_history}

### Internal measurements you requested last time
{probe_digests}

## Your move
You may do ONE of two things in this message:

A. Request up to {max_probes} more internal measurements before deciding. Use the probe API:
{probe_api}

B. Propose {max_edits} edits, each a different hypothesis (different targets, ops or directions): every valid proposal is screened and a generation's fixed cost is paid anyway, so proposing fewer wastes the generation. For each edit give: the op and its target (exact parameter names and indices), the parameters, the internal evidence that motivates it, the behaviour you predict, and how confident you are.

Think like a surgeon looking at scans: the evidence should point to a specific mechanism inside you (a head whose memory decays too fast during long derivations, neurons that fire when you start repeating yourself, a direction that separates your correct from your incorrect reasoning, layers that under-use visual tokens, ...). Prefer small, targeted, reversible changes, and say so when an edit is exploratory.

Answer with a single JSON object, either {"probes": [...]} or {"edits": [...]}, after your reasoning.
