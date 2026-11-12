"""Throughput sweep: steps the in-flight caps in engines.json (the dispatcher reloads it every 20 s). Writes V4/logs/sweep.jsonl with step times."""
import json, time
V4 = "/data/fd/mingdazhang/selfedit/v4"
STEP = 360
plan = [({8200: 16, 8201: 96}, STEP), ({8200: 32, 8201: 96}, STEP), ({8200: 48, 8201: 128}, STEP), ({8200: 64, 8201: 128}, STEP)]


def set_caps(caps):
    e = json.load(open(f"{V4}/engines.json"))
    have = {s["port"] for s in e["servers"]}
    if 8201 not in have:
        e["servers"].append(dict(name="e1", host="127.0.0.1", port=8201, card=1, role="E", state="up", cap=96, max_num_seqs=128, job="selfedit-v4e-s1", started=1794293926))
    for s in e["servers"]:
        if s["port"] in caps:
            s["cap"] = caps[s["port"]]
    e["updated"] = time.time()
    json.dump(e, open(f"{V4}/engines.json.tmp", "w"), indent=1)
    import os
    os.replace(f"{V4}/engines.json.tmp", f"{V4}/engines.json")


for caps, dur in plan:
    set_caps(caps)
    with open(f"{V4}/logs/sweep.jsonl", "a") as f:
        f.write(json.dumps(dict(t=time.time(), caps=caps)) + "\n")
    print(time.strftime("%H:%M:%S"), caps, flush=True)
    time.sleep(dur)
print("sweep done", flush=True)
