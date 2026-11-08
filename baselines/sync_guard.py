"""sync_guard.py DIR...: stopgap for trainers started before the rank0_done fix. Rank 0 removes .sync_{k-1} when it writes .sync_k, which can be before rank 1 has seen it.
This puts the previous file back at once. Only touches empty .sync_* files in our own run directories."""
import os, re, sys, time
while True:
    for d in sys.argv[1:]:
        try:
            ks = [int(m.group(1)) for f in os.listdir(d) for m in [re.fullmatch(r"\.sync_(\d+)", f)] if m]
        except FileNotFoundError:
            continue
        if ks:
            k = max(ks)
            for j in (k - 1, k - 2):
                if j > 0 and not os.path.exists(f"{d}/.sync_{j}"):
                    open(f"{d}/.sync_{j}", "w").close()
    time.sleep(.05)
