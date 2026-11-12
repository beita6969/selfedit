"""Checks 1a (prompt identity vs Surgeon.render + HF processor = what vllm.LLM.chat tokenizes) and 1b (per-request vision settings).
usage: python check1.py PORT"""
import json, sys, time, urllib.request
sys.path.insert(0, "/workspace/fudan/mingdazhang/selfedit-v4/engine_v4")
from common import *

PORT = int(sys.argv[1])
ROWS = "/data/fd/mingdazhang/selfedit/v4/rows/learn.jsonl"
rows = [json.loads(l) for l in open(ROWS)]
rows = [r for r in rows if r.get("image_paths")]


def post(body, path="/v1/chat/completions"):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=600).read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:400]}


from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained(MODEL)
tok = proc.tokenizer


def ref_ids(msgs, images, size=None):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort=EFFORT, enable_thinking=True)
    kw = {"images_kwargs": {"size": size}} if size else {}
    out = proc(text=[text], images=images or None, return_tensors=None, **kw)
    return list(out["input_ids"][0]), text


def server_ids(msgs, how="ctk", mmkw=None):
    body = dict(model="stock", messages=msgs, max_tokens=1, return_token_ids=True, temperature=0)
    if how == "ctk":
        body["chat_template_kwargs"] = {"reasoning_effort": EFFORT, "enable_thinking": True}
    else:
        body["reasoning_effort"] = EFFORT
    if mmkw:
        body["mm_processor_kwargs"] = mmkw
    r = post(body)
    if "error" in r:
        return None, r["error"]
    return r.get("prompt_token_ids"), r["usage"]["prompt_tokens"]


print("== 1a prompt identity (5 rows, stock view, effort medium)")
for row in rows[:5]:
    content, nt = content_for(row["prompt"], row["image_paths"])
    msgs = [{"role": "user", "content": content}]
    imgs = [load_rgb(p) for p in row["image_paths"]]
    ref, text = ref_ids(msgs, imgs)
    for how in ("ctk", "top"):
        ids, pt = server_ids(msgs, how)
        print(json.dumps(dict(row=row["id"], how=how, ref_len=len(ref), srv_len=(len(ids) if ids else pt), identical=(ids == ref) if ids else None,
                              first_diff=(next((k for k, (a, b) in enumerate(zip(ids, ref)) if a != b), None) if ids else None),
                              img_tok_ref=ref.count(IMAGE_TOKEN_ID), img_tok_srv=(ids.count(IMAGE_TOKEN_ID) if ids else None), img_tok_pred=sum(nt))))
print("rendered tail:", repr(text[-120:]))

print("== 1b per-request vision settings (20 images): image tokens")
seen, imgs = set(), []
for row in rows:
    for p in row["image_paths"]:
        if p not in seen and len(imgs) < 20:
            seen.add(p); imgs.append(p)
VIEW = dict(min_tokens=3600, max_tokens=None)
mn, mx = view_pixels(**VIEW)
tab = []
for p in imgs:
    msgs0 = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(p)}}, {"type": "text", "text": "Describe."}]}]
    im = load_rgb(p)
    ref_stock = ref_ids(msgs0, [im])[0].count(IMAGE_TOKEN_ID)
    ref_view = ref_ids(msgs0, [im], size={"shortest_edge": mn, "longest_edge": mx})[0].count(IMAGE_TOKEN_ID)
    s_stock = server_ids(msgs0)[0]
    s_size = server_ids(msgs0, mmkw={"size": {"shortest_edge": mn, "longest_edge": mx}})
    s_px = server_ids(msgs0, mmkw={"min_pixels": mn, "max_pixels": mx})
    c, nt = content_for("Describe.", [p], **VIEW)
    s_client = server_ids([{"role": "user", "content": c}])[0]
    f = lambda s: s[0].count(IMAGE_TOKEN_ID) if s[0] else ("ERR " + str(s[1])[:120])
    row = dict(img=p.split("/")[-1][:30], wh=im.size, ref_stock=ref_stock, srv_stock=s_stock.count(IMAGE_TOKEN_ID) if s_stock else None,
               ref_3600=ref_view, srv_mmkw_size=f(s_size), srv_mmkw_pixels=f(s_px), client_resize=s_client.count(IMAGE_TOKEN_ID) if s_client else None, client_pred=nt[0])
    tab.append(row)
    print(json.dumps(row))
ok = lambda k: sum(1 for r in tab if r[k] == r["ref_3600"])
print("match ref_3600: mmkw_size", ok("srv_mmkw_size"), "mmkw_pixels", ok("srv_mmkw_pixels"), "client_resize", ok("client_resize"), "of", len(tab),
      "; stock srv==ref", sum(1 for r in tab if r["srv_stock"] == r["ref_stock"]))
