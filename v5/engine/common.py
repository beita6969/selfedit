"""v4 engine common pieces: prompt content (identical to code-1109c engine/surgery.py:image_content), v3 sampling,
client-side image resize to a visual-token target (Qwen2-VL smart_resize, 32 px per merged token side), grading as rsi/mm_eval.py:_collect."""
import base64, io, json, math, mimetypes, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grade_mm  # copy of code-1109c/mm/grade_mm.py (unchanged)

MODEL = "/data/fd/mingdazhang/selfedit/models/Qwen3.8-27B"
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0)
EFFORT = "medium"
FACTOR = 32                       # patch 16 x merge 2
STOCK_MIN_PX, STOCK_MAX_PX = 65536, 16777216     # preprocessor_config size (shortest_edge / longest_edge are pixel areas)
IMAGE_TOKEN_ID = 248056


def data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def smart_resize(h, w, factor=FACTOR, min_pixels=STOCK_MIN_PX, max_pixels=STOCK_MAX_PX):
    """transformers qwen2_vl image_processing smart_resize."""
    if max(h, w) / min(h, w) > 200:
        raise ValueError(f"aspect ratio > 200: {h}x{w}")
    hb = round(h / factor) * factor
    wb = round(w / factor) * factor
    if hb * wb > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


def view_pixels(min_tokens=None, max_tokens=None):
    """vision_patch.py semantics: longest_edge = max_tokens * 32^2, shortest_edge = min_tokens * 32^2 (stock when None)."""
    mx = max_tokens * FACTOR ** 2 if max_tokens else STOCK_MAX_PX
    mn = min_tokens * FACTOR ** 2 if min_tokens else min(STOCK_MIN_PX, mx)
    return mn, mx


def load_rgb(path):
    """As vLLM ImageMediaIO(image_mode='RGB'): RGBA composited on white."""
    from PIL import Image
    im = Image.open(path)
    im.load()
    if im.mode == "RGB":
        return im
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


def image_url_for(path, min_tokens=None, max_tokens=None):
    """Stock view: the file's bytes unchanged (v3 path). Other views: resize client-side to the smart_resize size of that view
    (bicubic, as Qwen2VLImageProcessor), sent as PNG; the server's stock processor then keeps the size (it is a multiple of 32
    inside the stock bounds), so the token count equals the v3 checkpoint route's. Returns (url, image_tokens)."""
    from PIL import Image
    if max_tokens and max_tokens * FACTOR ** 2 >= STOCK_MAX_PX:
        max_tokens = None                    # the stock cap (16384): stock view
    if min_tokens and min_tokens * FACTOR ** 2 <= STOCK_MIN_PX:
        min_tokens = None                    # at or below the stock minimum (64): stock view
    if not min_tokens and not max_tokens:
        with Image.open(path) as im:
            w, h = im.size
        hb, wb = smart_resize(h, w)
        return data_url(path), (hb // FACTOR) * (wb // FACTOR)
    mn, mx = view_pixels(min_tokens, max_tokens)
    im = load_rgb(path)
    w, h = im.size
    hb, wb = smart_resize(h, w, min_pixels=mn, max_pixels=mx)
    hs, ws = smart_resize(hb, wb)            # what the stock processor does to the resized image (identity unless outside stock bounds)
    if (hb, wb) != (h, w):
        im = im.resize((wb, hb), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), (hs // FACTOR) * (ws // FACTOR)


def content_for(prompt, image_paths, min_tokens=None, max_tokens=None, prefix=None):
    """engine/surgery.py:image_content layout; prefix (routed procedures) goes as a text part right before the prompt text."""
    imgs, ntok = [], []
    for p in image_paths:
        u, n = image_url_for(p, min_tokens, max_tokens)
        imgs.append({"type": "image_url", "image_url": {"url": u}})
        ntok.append(n)
    parts = prompt.split("<image>")
    if len(parts) - 1 == len(imgs) and imgs:
        content = []
        if prefix:
            content.append({"type": "text", "text": prefix.rstrip() + "\n\n"})
        for k, txt in enumerate(parts):
            if txt.strip():
                content.append({"type": "text", "text": txt.strip()})
            if k < len(imgs):
                content.append(imgs[k])
        return content, ntok
    pre = [{"type": "text", "text": prefix.rstrip() + "\n\n"}] if prefix else []
    return imgs + pre + [{"type": "text", "text": prompt}], ntok


def grade(row, content, reasoning, finish, tokens, sample=0):
    """rsi/mm_eval.py:_collect for an image row."""
    r = {"id": row["id"], "kind": row["kind"], "answer": row["answer"], "sample": sample, "reasoning": reasoning or "",
         "content": content or "", "finish": finish, "tokens": tokens, "options": row.get("options") or [],
         "category": row.get("category"), "images": list(row.get("image_paths") or [])}
    if row.get("precision") is not None:
        r["precision"] = row["precision"]
    err, pred = None, None
    if finish != "stop" or not content:
        ok = False
    else:
        try:
            ok, pred = grade_mm.correct(r)
            ok = bool(ok)
        except Exception as e:
            ok, err = False, f"{type(e).__name__}: {str(e)[:160]}"
    try:
        loop = bool(grade_mm.looped(r))
    except Exception as e:
        loop, err = False, err or f"{type(e).__name__}: {str(e)[:160]}"
    if pred is not None and not isinstance(pred, (str, int, float)):
        pred = str(pred)
    return ok, loop, err, pred          # pred = grade_mm.correct(r)[1], the normalised extracted answer (None when unfinished / empty / grader error)


def chat_messages(messages):
    """Chat (proposer) messages: a content list may hold {"type": "text", "text"} and {"type": "image", "path", "min_image_tokens",
    "max_image_tokens"} parts; image parts go through the same client-side resize as reads. Returns (messages, image token counts)."""
    out, ntok = [], []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                if p.get("type") == "image":
                    u, n = image_url_for(p["path"], p.get("min_image_tokens"), p.get("max_image_tokens"))
                    parts.append({"type": "image_url", "image_url": {"url": u}})
                    ntok.append(n)
                elif p.get("type") == "text":
                    parts.append({"type": "text", "text": p["text"]})
                else:
                    parts.append(p)
            m = dict(m, content=parts)
        out.append(m)
    return out, ntok


# ------------------------------------------------------------------ harness ops (engine_api.md §7): zoom, tiles(n), check
ZOOM_Q = ("Do not answer the question. Name the image region that matters most for answering it: reply with only [i, x0, y0, x1, y1], "
          "where i is the image number (1 for a single image) and x0 y0 x1 y1 are normalised coordinates between 0 and 1.")
CHECK_Q = ("A proposed final answer to this question: {a}. Check it against the image(s) and the question. If it is wrong, correct it. "
           "Give your final answer in \\boxed{{Answer}} format.")
MIN_OP_TOKENS, MIN_SIDE = 256, 0.05


def parse_box(text, n_images):
    import re
    m = re.search(r"\[\s*(\d+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]", text or "")
    if not m:
        return None
    i = int(m.group(1))
    x0, y0, x1, y1 = (min(1.0, max(0.0, float(m.group(k)))) for k in range(2, 6))
    if not 1 <= i <= n_images:
        return None
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    for a, b in ((0, 1), (2, 3)):
        pass
    def widen(lo, hi):
        if hi - lo >= MIN_SIDE:
            return lo, hi
        c = (lo + hi) / 2
        lo, hi = c - MIN_SIDE / 2, c + MIN_SIDE / 2
        if lo < 0:
            lo, hi = 0.0, MIN_SIDE
        if hi > 1:
            lo, hi = 1 - MIN_SIDE, 1.0
        return lo, hi
    x0, x1 = widen(x0, x1)
    y0, y1 = widen(y0, y1)
    return [i, round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]


def upscaled_part(im, tokens):
    """PIL image -> image_url part resized (bicubic) to about `tokens` visual tokens (smart_resize with min = max = tokens * 32^2)."""
    from PIL import Image
    T = max(MIN_OP_TOKENS, int(tokens))
    w, h = im.size
    hb, wb = smart_resize(h, w, min_pixels=T * FACTOR ** 2, max_pixels=T * FACTOR ** 2)
    im = im.resize((wb, hb), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=1)
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}, (hb // FACTOR) * (wb // FACTOR)


def crop_part(path, box, tokens):
    im = load_rgb(path)
    w, h = im.size
    _, x0, y0, x1, y1 = box
    c = im.crop((int(x0 * w), int(y0 * h), max(int(x0 * w) + 1, int(round(x1 * w))), max(int(y0 * h) + 1, int(round(y1 * h)))))
    return upscaled_part(c, tokens)


def tile_parts(path, n, tokens):
    im = load_rgb(path)
    w, h = im.size
    out = []
    for r in range(n):
        for c in range(n):
            t = im.crop((c * w // n, r * h // n, (c + 1) * w // n, (r + 1) * h // n))
            out.append(upscaled_part(t, tokens))
    return out


def insert_after_last_image(content, parts):
    k = max((i for i, p in enumerate(content) if p.get("type") == "image_url"), default=-1)
    return content[:k + 1] + list(parts) + content[k + 1:]


def proposed_answer(content):
    b = grade_mm.last_boxed(content or "")
    return b if b is not None else (content or "")[-300:]


OPS_IMAGE_LIMIT, OPS_MAXLEN, OPS_GEN_ROOM = 40, 46000, 4096


def ops_need(row, ops, settings=None):
    """(images, estimated prompt tokens) a read with these ops would send; the box of a zoom is unknown before pass 1, so its crop counts at the
    larger of its source image tokens and 256 for the largest image."""
    st = settings or {}
    paths = row.get("image_paths") or []
    toks = [image_url_for_tokens(p, st.get("min_image_tokens"), st.get("max_image_tokens")) for p in paths]
    n_img, est = len(paths), sum(toks)
    for o in ops:
        if o["op"] == "zoom" and paths:
            n_img += 1
            est += max(MIN_OP_TOKENS, max(toks))
        elif o["op"] == "tiles":
            k = int(o.get("n", 2))
            n_img += k * k * len(paths)
            est += sum(k * k * max(MIN_OP_TOKENS, t) for t in toks)
    est += len(row.get("prompt") or "") // 3 + 64
    return n_img, est


def ops_skip_reason(row, ops, settings=None):
    n, est = ops_need(row, ops, settings)
    if n > OPS_IMAGE_LIMIT:
        return f"image_limit: {n} images > {OPS_IMAGE_LIMIT}"
    if est > OPS_MAXLEN - OPS_GEN_ROOM:
        return f"context: about {est} prompt tokens > {OPS_MAXLEN - OPS_GEN_ROOM}"
    return None


def image_url_for_tokens(path, min_tokens=None, max_tokens=None):
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    if max_tokens and max_tokens * FACTOR ** 2 >= STOCK_MAX_PX:
        max_tokens = None
    if min_tokens and min_tokens * FACTOR ** 2 <= STOCK_MIN_PX:
        min_tokens = None
    mn, mx = view_pixels(min_tokens, max_tokens) if (min_tokens or max_tokens) else (STOCK_MIN_PX, STOCK_MAX_PX)
    hb, wb = smart_resize(h, w, min_pixels=mn, max_pixels=mx)
    return (hb // FACTOR) * (wb // FACTOR)
