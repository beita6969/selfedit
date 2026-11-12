"""Declarative image-transform pipelines ("views") for the v4 harness (AM22 REVISED).

A view is DATA: a list of 1-4 stages drawn from the fixed vocabulary VOCAB.
Each stage is {"op": name, <params>, "on": "all" | 0-based image index,
"keep": "replace" | "append"}.  This module validates views, applies them to
PIL images with deterministic, trusted Pillow/numpy code, draws random
single-stage views for the blind arm, and generates neutral affordance text.

No code execution, no file or network access.

Public API: VOCAB, LIMITS, validate, apply, random_view, describe, ViewSkip.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

__all__ = ["VOCAB", "LIMITS", "validate", "apply", "random_view", "describe", "ViewSkip"]


class ViewSkip(Exception):
    """The view cannot be applied to these images within LIMITS (row is op_skipped)."""


LIMITS = {
    "max_stages": 4,
    "max_images_out": 40,
    "min_side": 28,
    "max_side": 8192,
    "max_total_px": 64_000_000,
}

# ---------------------------------------------------------------------------
# Vocabulary table.  Ranges are properties of the operations.
# ---------------------------------------------------------------------------

VOCAB = {
    "scale": {
        "what": "resize the image by factor f (Lanczos when shrinking, bicubic when enlarging)",
        "params": {"f": {"interval": [0.25, 4.0]}},
    },
    "crop": {
        "what": "cut out the region box = [x0, y0, x1, y1], given as fractions of width and height from the top-left corner",
        "params": {"box": {"box": True}},
    },
    "crop_border": {
        "what": "trim outer rows and columns whose pixels are all within tol of the background colour (median of the four corners), keeping a 2% margin; unchanged if nothing is trimmed",
        "params": {"tol": {"int": [0, 64]}},
    },
    "pad_square": {
        "what": "pad the image to a square with its background colour (median of the four corners), image centred",
        "params": {},
    },
    "split_grid": {
        "what": "cut the image into rows x cols equal tiles (not both 1), output in reading order",
        "params": {"rows": {"int": [1, 6]}, "cols": {"int": [1, 6]}},
    },
    "split_panels": {
        "what": "cut the image along background gutters at least min_gap px wide (measured with the longer side scaled to 1024); axis rows = horizontal cuts only, cols = vertical cuts only, auto = one axis then the other inside each part; at most 16 panels, reading order; unchanged if no gutter is found",
        "params": {"axis": {"choice": ["auto", "rows", "cols"]}, "min_gap": {"int": [2, 64]}},
    },
    "grid": {
        "what": "draw thin lines dividing the image into nx columns and ny rows; labels none, cells (A1-style tag in each cell's top-left corner: column letter, row number) or axes (column letters and row numbers in a margin above and left of the image)",
        "params": {"nx": {"int": [2, 16]}, "ny": {"int": [2, 16]}, "labels": {"choice": ["none", "cells", "axes"]}},
    },
    "dots": {
        "what": "draw small dots at the centres of an nx x ny lattice of cells, with a (col,row) tag next to each dot (1-based, from the top left) when labels is true",
        "params": {"nx": {"int": [2, 12]}, "ny": {"int": [2, 12]}, "labels": {"choice": [True, False]}},
    },
    "ruler": {
        "what": "add white margins on all four sides with tick marks dividing each edge into ticks equal steps, numbered 0 to 100 (percent of width on top and bottom, of height on left and right); the image itself is not drawn on",
        "params": {"ticks": {"int": [4, 20]}},
    },
    "sharpen": {
        "what": "unsharp mask with radius 2 and strength amount (percent = 100 x amount)",
        "params": {"amount": {"interval": [0.5, 3.0]}},
    },
    "contrast": {
        "what": "scale contrast around the mean grey by factor (1 = unchanged)",
        "params": {"factor": {"interval": [0.5, 2.5]}},
    },
    "autocontrast": {
        "what": "stretch levels to the full range, clipping 1% of pixels at each end",
        "params": {},
    },
    "equalize": {"what": "equalize the brightness histogram", "params": {}},
    "grayscale": {"what": "convert to grayscale (kept as an RGB image)", "params": {}},
    "invert": {"what": "invert colours (photographic negative)", "params": {}},
    "edges": {
        "what": "replace the image by its edge-strength map (Sobel), dark edges on white",
        "params": {},
    },
    "label_images": {
        "what": "add a band above each image reading \"Image k\", k = its position in the current list; on must be \"all\"",
        "params": {},
    },
    "contact_sheet": {
        "what": "place all current images on one white canvas in reading order with cols columns, each scaled to a common cell (aspect kept) under a band reading \"Image k\"; the result is one image; on must be \"all\"",
        "params": {"cols": {"int": [1, 8]}},
    },
}

_DEFAULT_KEEP = {"split_grid": "append", "split_panels": "append", "edges": "append"}
_ALL_ONLY = ("label_images", "contact_sheet")
# keeps that random_view draws from (appending a padded / border-trimmed /
# labelled duplicate carries no extra information)
_RANDOM_KEEPS = {"pad_square": ["replace"], "crop_border": ["replace"], "label_images": ["replace"]}
_LOG_PARAMS = {("scale", "f"), ("sharpen", "amount"), ("contrast", "factor")}
_STAGE_KEYS = ("op", "on", "keep")
_EPS = 1e-9


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _spec_text(spec):
    if "interval" in spec:
        lo, hi = spec["interval"]
        return f"a number in [{_fmt(lo)}, {_fmt(hi)}]"
    if "int" in spec:
        lo, hi = spec["int"]
        return f"an integer in [{lo}, {hi}]"
    if "choice" in spec:
        return "one of " + ", ".join(json.dumps(c) for c in spec["choice"])
    return "[x0, y0, x1, y1]: four numbers in [0, 1] with x1 - x0 >= 0.05 and y1 - y0 >= 0.05"


def _check_param(spec, v):
    """Return (canonical_value, None) or (None, reason)."""
    if "interval" in spec:
        lo, hi = spec["interval"]
        if not _is_num(v):
            return None, f"must be {_spec_text(spec)}, got {json.dumps(v) if _jsonable(v) else type(v).__name__}"
        if v < lo - _EPS or v > hi + _EPS:
            return None, f"must be {_spec_text(spec)}, got {v!r}"
        return round(float(v), 4), None
    if "int" in spec:
        lo, hi = spec["int"]
        if isinstance(v, float) and not isinstance(v, bool) and math.isfinite(v) and v.is_integer():
            v = int(v)
        if not isinstance(v, int) or isinstance(v, bool):
            return None, f"must be {_spec_text(spec)}, got {json.dumps(v) if _jsonable(v) else type(v).__name__}"
        if v < lo or v > hi:
            return None, f"must be {_spec_text(spec)}, got {v}"
        return int(v), None
    if "choice" in spec:
        for c in spec["choice"]:
            if type(c) is type(v) and c == v:
                return c, None
        return None, f"must be {_spec_text(spec)}, got {json.dumps(v) if _jsonable(v) else type(v).__name__}"
    # box
    if not isinstance(v, (list, tuple)) or len(v) != 4 or not all(_is_num(x) for x in v):
        return None, f"must be {_spec_text(spec)}, got {json.dumps(v) if _jsonable(v) else type(v).__name__}"
    b = [round(float(x), 4) for x in v]
    if any(x < -_EPS or x > 1 + _EPS for x in v):
        return None, f"values must each be in [0, 1], got {json.dumps(list(v))}"
    if b[2] - b[0] < 0.05 - _EPS:
        return None, f"needs x1 - x0 >= 0.05, got x0={_fmt(b[0])}, x1={_fmt(b[2])}"
    if b[3] - b[1] < 0.05 - _EPS:
        return None, f"needs y1 - y0 >= 0.05, got y0={_fmt(b[1])}, y1={_fmt(b[3])}"
    return b, None


def _jsonable(v):
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def validate(view) -> dict:
    """Check a view; return {"ok", "error", "canonical", "sha"}."""
    def fail(msg):
        return {"ok": False, "error": msg, "canonical": None, "sha": None}

    if isinstance(view, (str, bytes)):
        try:
            view = json.loads(view)
        except (ValueError, TypeError) as e:
            return fail(f"view is not valid JSON: {e}")
    if not isinstance(view, (list, tuple)):
        return fail(f"view must be a list of 1 to {LIMITS['max_stages']} stages, got {type(view).__name__}")
    if not 1 <= len(view) <= LIMITS["max_stages"]:
        return fail(f"view must have 1 to {LIMITS['max_stages']} stages, got {len(view)}")
    errors = []
    canonical = []
    ns = len(view)
    for k, st in enumerate(view, 1):
        where = f"stage {k} of {ns}"
        if not isinstance(st, dict):
            errors.append(f"{where}: must be an object with an \"op\" key, got {type(st).__name__}")
            continue
        op = st.get("op")
        if "op" not in st:
            errors.append(f"{where}: missing \"op\" (one of {', '.join(sorted(VOCAB))})")
            continue
        if not isinstance(op, str) or op not in VOCAB:
            shown = json.dumps(op) if _jsonable(op) else repr(op)
            errors.append(f"{where}: unknown op {shown} (allowed: {', '.join(sorted(VOCAB))})")
            continue
        where = f"stage {k} of {ns} ({op})"
        pspecs = VOCAB[op]["params"]
        allowed = set(_STAGE_KEYS) | set(pspecs)
        cst = {"op": op}
        for key in sorted(st, key=str):
            if key not in allowed:
                plist = ", ".join(sorted(pspecs)) if pspecs else "none"
                errors.append(f"{where}: unknown key {json.dumps(key) if _jsonable(key) else repr(key)} "
                              f"(parameters of {op}: {plist}; plus \"on\", \"keep\")")
        for pname in sorted(pspecs):
            spec = pspecs[pname]
            if pname not in st:
                errors.append(f"{where}: missing parameter '{pname}' ({_spec_text(spec)})")
                continue
            val, why = _check_param(spec, st[pname])
            if why:
                errors.append(f"{where}: parameter '{pname}' {why}")
            else:
                cst[pname] = val
        on = st.get("on", "all")
        if on == "all" and isinstance(on, str):
            cst["on"] = "all"
        elif isinstance(on, int) and not isinstance(on, bool) and 0 <= on < LIMITS["max_images_out"]:
            if op in _ALL_ONLY:
                errors.append(f"{where}: \"on\" must be \"all\" for {op}, got {on}")
            else:
                cst["on"] = on
        else:
            errors.append(f"{where}: \"on\" must be \"all\" or a 0-based image index "
                          f"(integer 0 to {LIMITS['max_images_out'] - 1}), got "
                          f"{json.dumps(on) if _jsonable(on) else type(on).__name__}")
        keep = st.get("keep", _DEFAULT_KEEP.get(op, "replace"))
        if keep in ("replace", "append") and isinstance(keep, str):
            cst["keep"] = keep
        else:
            errors.append(f"{where}: \"keep\" must be \"replace\" or \"append\", got "
                          f"{json.dumps(keep) if _jsonable(keep) else type(keep).__name__}")
        if op == "split_grid" and cst.get("rows") == 1 and cst.get("cols") == 1:
            errors.append(f"{where}: rows and cols cannot both be 1")
        canonical.append(dict(sorted(cst.items())))
    if errors:
        return fail("; ".join(errors))
    sha = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"ok": True, "error": None, "canonical": canonical, "sha": sha}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(x):
    """Number formatting for notes and messages: 2 -> '2.0', 0.3333 -> '0.3333'."""
    x = float(x)
    if x == int(x):
        return f"{x:.1f}"
    return f"{x:.4f}".rstrip("0").rstrip(".")


def _pct(x):
    s = f"{100 * float(x):.2f}".rstrip("0").rstrip(".")
    return s + "%"


@lru_cache(maxsize=256)
def _font(size):
    size = max(6, int(size))
    try:
        return ImageFont.load_default(size=size)
    except Exception:  # Pillow without FreeType or without the size argument
        return ImageFont.load_default()


def _tsize(font, text):
    l, t, r, b = font.getbbox(text)
    return r - l, b - t, l, t


def _text(draw, x, y, text, font, fill, halo=None):
    """Draw text so that its visible top-left lands at (x, y); optional 1-px halo."""
    _, _, l, t = _tsize(font, text)
    x, y = x - l, y - t
    if halo is not None:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((x + dx, y + dy), text, font=font, fill=halo)
    draw.text((x, y), text, font=font, fill=fill)


def _to_rgb(img):
    if not isinstance(img, Image.Image):
        raise TypeError("images must be PIL.Image objects")
    if img.mode == "RGB":
        return img.copy()
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA", "PA", "RGBa", "La"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    if img.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        a = np.asarray(img.convert("F"), dtype=np.float64)
        hi = float(a.max()) if a.size else 0.0
        if hi > 255:
            a = a / hi * 255.0
        a = np.clip(a, 0, 255)
        return Image.fromarray(np.rint(a).astype(np.uint8)).convert("RGB")
    return img.convert("RGB")


def _bg_colour(arr):
    c = np.stack([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]).astype(np.float64)
    return np.rint(np.median(c, axis=0)).astype(np.int16)


def _lum(img):
    a = np.asarray(img, dtype=np.float64)
    return float((a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114).mean())


def _ink(img):
    """(line, halo) colours: dark line + light halo on light images, and the reverse."""
    if _lum(img) >= 128.0:
        return (0, 0, 0), (255, 255, 255)
    return (255, 255, 255), (0, 0, 0)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _resize(img, size):
    w, h = img.size
    if (w, h) == tuple(size):
        return img.copy()
    rs = Image.LANCZOS if size[0] * size[1] < w * h else Image.BICUBIC
    return img.resize(size, rs)


def _bounds(n, length):
    return [round(i * length / n) for i in range(n + 1)]


def _letters(i):
    return chr(ord("A") + i)


# ---------------------------------------------------------------------------
# Operations: per-image ops take (img, params) and return (list_of_images, info)
# ---------------------------------------------------------------------------

def _op_scale(img, p):
    w, h = img.size
    f = p["f"]
    size = (max(1, round(w * f)), max(1, round(h * f)))
    return [_resize(img, size)], {}


def _op_crop(img, p):
    w, h = img.size
    x0, y0, x1, y1 = p["box"]
    a, b = round(x0 * w), round(x1 * w)
    c, d = round(y0 * h), round(y1 * h)
    b = max(b, a + 1)
    d = max(d, c + 1)
    a, b = min(a, w - 1), min(b, w)
    c, d = min(c, h - 1), min(d, h)
    return [img.crop((a, c, b, d))], {}


def _op_crop_border(img, p):
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    bg = _bg_colour(arr)
    far = (np.abs(arr.astype(np.int16) - bg).max(axis=2) > p["tol"])
    rows = np.flatnonzero(far.any(axis=1))
    cols = np.flatnonzero(far.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return [img.copy()], {"changed": False}
    mx, my = round(0.02 * w), round(0.02 * h)
    x0 = max(0, int(cols[0]) - mx)
    x1 = min(w, int(cols[-1]) + 1 + mx)
    y0 = max(0, int(rows[0]) - my)
    y1 = min(h, int(rows[-1]) + 1 + my)
    if (x0, y0, x1, y1) == (0, 0, w, h):
        return [img.copy()], {"changed": False}
    return [img.crop((x0, y0, x1, y1))], {"changed": True}


def _op_pad_square(img, p):
    w, h = img.size
    if w == h:
        return [img.copy()], {"changed": False}
    s = max(w, h)
    bg = tuple(int(v) for v in _bg_colour(np.asarray(img)))
    out = Image.new("RGB", (s, s), bg)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return [out], {"changed": True}


def _op_split_grid(img, p):
    w, h = img.size
    xs, ys = _bounds(p["cols"], w), _bounds(p["rows"], h)
    out = []
    for r in range(p["rows"]):
        for c in range(p["cols"]):
            out.append(img.crop((xs[c], ys[r], xs[c + 1], ys[r + 1])))
    return out, {"count": len(out)}


def _gutter_cuts(frac, min_gap):
    """Cut positions at the middle of interior gutter runs (frac >= 0.99) of length >= min_gap."""
    g = frac >= 0.99
    n = g.size
    cuts = []
    i = 0
    while i < n:
        if g[i]:
            j = i
            while j < n and g[j]:
                j += 1
            if i > 0 and j < n and (j - i) >= min_gap:
                cuts.append((i + j) // 2)
            i = j
        else:
            i += 1
    return cuts


def _segments(near, axis, min_gap, min_seg):
    """Split a boolean 'near background' region along one axis; returns [(start, end)]."""
    frac = near.mean(axis=1) if axis == "rows" else near.mean(axis=0)
    length = frac.size
    bnds = [0] + _gutter_cuts(frac, min_gap) + [length]
    segs = [[bnds[i], bnds[i + 1]] for i in range(len(bnds) - 1)]
    content = np.flatnonzero(frac < 0.99)

    def extent(a, b):  # length of the non-gutter content inside [a, b)
        idx = content[(content >= a) & (content < b)]
        return int(idx[-1] - idx[0] + 1) if idx.size else 0

    while len(segs) > 1:
        lens = [extent(a, b) for a, b in segs]
        small = [i for i, L in enumerate(lens) if L < min_seg]
        if not small:
            break
        i = min(small, key=lambda k: (lens[k], k))
        if i == 0:
            j = 1
        elif i == len(segs) - 1:
            j = i - 1
        else:
            j = i - 1 if lens[i - 1] <= lens[i + 1] else i + 1
        a, b = min(i, j), max(i, j)
        segs[a] = [segs[a][0], segs[b][1]]
        del segs[b]
    return [tuple(s) for s in segs]


def _op_split_panels(img, p):
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    bg = _bg_colour(arr)
    near = np.abs(arr.astype(np.int16) - bg).max(axis=2) <= 12
    gap = max(1, round(p["min_gap"] * max(w, h) / 1024))
    min_r, min_c = math.ceil(0.05 * h), math.ceil(0.05 * w)
    boxes = []  # (x0, y0, x1, y1)
    axis = p["axis"]
    if axis == "rows":
        boxes = [(0, a, w, b) for a, b in _segments(near, "rows", gap, min_r)]
    elif axis == "cols":
        boxes = [(a, 0, b, h) for a, b in _segments(near, "cols", gap, min_c)]
    else:
        rsegs = _segments(near, "rows", gap, min_r)
        if len(rsegs) > 1:
            for a, b in rsegs:
                for c, d in _segments(near[a:b, :], "cols", gap, min_c):
                    boxes.append((c, a, d, b))
        else:
            csegs = _segments(near, "cols", gap, min_c)
            for c, d in csegs:
                for a, b in _segments(near[:, c:d], "rows", gap, min_r):
                    boxes.append((c, a, d, b))
    if len(boxes) <= 1:
        return [img.copy()], {"count": 1}
    if len(boxes) > 16:
        order = sorted(range(len(boxes)),
                       key=lambda i: (-(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]), i))
        keep = sorted(order[:16])
        boxes = [boxes[i] for i in keep]
    return [img.crop(b) for b in boxes], {"count": len(boxes)}


def _vline(draw, x, width, h, fill):
    x0 = x - width // 2
    draw.rectangle([x0, 0, x0 + width - 1, h - 1], fill=fill)


def _hline(draw, y, width, w, fill):
    y0 = y - width // 2
    draw.rectangle([0, y0, w - 1, y0 + width - 1], fill=fill)


def _op_grid(img, p):
    w, h = img.size
    nx, ny, labels = p["nx"], p["ny"], p["labels"]
    lw = max(1, round(min(w, h) / 500))
    line, halo = _ink(img)
    out = img.copy()
    d = ImageDraw.Draw(out)
    xs, ys = _bounds(nx, w), _bounds(ny, h)
    for x in xs[1:-1]:
        _vline(d, x, lw + 2, h, halo)
    for y in ys[1:-1]:
        _hline(d, y, lw + 2, w, halo)
    for x in xs[1:-1]:
        _vline(d, x, lw, h, line)
    for y in ys[1:-1]:
        _hline(d, y, lw, w, line)
    cw, ch = w / nx, h / ny
    if labels == "cells":
        font = _font(_clamp(round(min(cw, ch) * 0.2), 7, 40))
        off = lw // 2 + 3
        for r in range(ny):
            for c in range(nx):
                _text(d, xs[c] + off, ys[r] + off, f"{_letters(c)}{r + 1}", font, line, halo)
        return [out], {}
    if labels == "axes":
        fs = _clamp(round(min(w, h) / 32), 9, 48)
        fs = min(fs, max(7, int(min(cw, ch) * 0.9)))
        font = _font(fs)
        pad = max(2, fs // 4)
        th = max(_tsize(font, s)[1] for s in ("A", "Q", "1"))
        top = th + 2 * pad + 2
        left = max(_tsize(font, str(r + 1))[0] for r in range(ny)) + 2 * pad + 2
        canvas = Image.new("RGB", (w + left, h + top), (255, 255, 255))
        canvas.paste(out, (left, top))
        dc = ImageDraw.Draw(canvas)
        for x in xs[1:-1]:  # boundary ticks in the margin
            dc.line([(left + x, top - pad), (left + x, top - 1)], fill=(0, 0, 0))
        for y in ys[1:-1]:
            dc.line([(left - pad, top + y), (left - 1, top + y)], fill=(0, 0, 0))
        for c in range(nx):
            s = _letters(c)
            tw, _, _, _ = _tsize(font, s)
            cx = left + (xs[c] + xs[c + 1]) / 2
            _text(dc, round(cx - tw / 2), pad, s, font, (0, 0, 0))
        for r in range(ny):
            s = str(r + 1)
            tw, tht, _, _ = _tsize(font, s)
            cy = top + (ys[r] + ys[r + 1]) / 2
            _text(dc, left - pad - 2 - tw, round(cy - tht / 2), s, font, (0, 0, 0))
        return [canvas], {"offset": (left, top)}
    return [out], {}


def _op_dots(img, p):
    w, h = img.size
    nx, ny = p["nx"], p["ny"]
    line, halo = _ink(img)
    out = img.copy()
    d = ImageDraw.Draw(out)
    r = max(2, round(min(w, h) / 200))
    cw, ch = w / nx, h / ny
    font = _font(_clamp(round(min(cw, ch) * 0.16), 7, 36))
    pts = [(round((i + 0.5) * cw), round((j + 0.5) * ch), i, j) for j in range(ny) for i in range(nx)]
    for x, y, _, _ in pts:
        d.ellipse([x - r - 1, y - r - 1, x + r + 1, y + r + 1], fill=halo)
        d.ellipse([x - r, y - r, x + r, y + r], fill=line)
    if p["labels"]:
        for x, y, i, j in pts:
            s = f"({i + 1},{j + 1})"
            _, th, _, _ = _tsize(font, s)
            _text(d, x + r + 3, round(y - th / 2), s, font, line, halo)
    return [out], {}


def _op_ruler(img, p):
    w, h = img.size
    ticks = p["ticks"]
    fs = _clamp(round(min(w, h) / 40), 9, 32)
    font = _font(fs)
    pad = max(2, fs // 4)
    tl = max(4, fs // 2)
    th = max(_tsize(font, s)[1] for s in ("0", "100"))
    w100 = _tsize(font, "100")[0]
    bh = tl + th + 2 * pad
    bw = tl + w100 + 2 * pad
    canvas = Image.new("RGB", (w + 2 * bw, h + 2 * bh), (255, 255, 255))
    canvas.paste(img, (bw, bh))
    d = ImageDraw.Draw(canvas)
    k_x = max(1, math.ceil((w100 + 4) / (w / ticks)))
    k_y = max(1, math.ceil((th + 3) / (h / ticks)))
    blk = (0, 0, 0)
    for j in range(ticks + 1):
        val = str(round(100 * j / ticks))
        x = bw + round(j / ticks * (w - 1))
        y = bh + round(j / ticks * (h - 1))
        d.line([(x, bh - tl), (x, bh - 1)], fill=blk)
        d.line([(x, bh + h), (x, bh + h + tl - 1)], fill=blk)
        d.line([(bw - tl, y), (bw - 1, y)], fill=blk)
        d.line([(bw + w, y), (bw + w + tl - 1, y)], fill=blk)
        tw, tht, _, _ = _tsize(font, val)
        if j % k_x == 0:
            tx = _clamp(round(x - tw / 2), 0, canvas.width - tw)
            _text(d, tx, bh - tl - pad - tht, val, font, blk)
            _text(d, tx, bh + h + tl + pad, val, font, blk)
        if j % k_y == 0:
            ty = _clamp(round(y - tht / 2), 0, canvas.height - tht)
            _text(d, bw - tl - pad - tw, ty, val, font, blk)
            _text(d, bw + w + tl + pad, ty, val, font, blk)
    return [canvas], {}


def _op_sharpen(img, p):
    return [img.filter(ImageFilter.UnsharpMask(radius=2, percent=int(round(100 * p["amount"])), threshold=0))], {}


def _op_contrast(img, p):
    return [ImageEnhance.Contrast(img).enhance(p["factor"])], {}


def _op_autocontrast(img, p):
    return [ImageOps.autocontrast(img, cutoff=1)], {}


def _op_equalize(img, p):
    return [ImageOps.equalize(img)], {}


def _op_grayscale(img, p):
    return [img.convert("L").convert("RGB")], {}


def _op_invert(img, p):
    return [ImageOps.invert(img)], {}


def _op_edges(img, p):
    g = np.asarray(img.convert("L"), dtype=np.float32)
    q = np.pad(g, 1, mode="edge")
    gx = (q[:-2, 2:] + 2 * q[1:-1, 2:] + q[2:, 2:]) - (q[:-2, :-2] + 2 * q[1:-1, :-2] + q[2:, :-2])
    gy = (q[2:, :-2] + 2 * q[2:, 1:-1] + q[2:, 2:]) - (q[:-2, :-2] + 2 * q[:-2, 1:-1] + q[:-2, 2:])
    mag = np.sqrt(gx * gx + gy * gy)
    ref = float(np.percentile(mag, 99))
    if ref < 1e-3:
        v = np.full(g.shape, 255, np.uint8)
    else:
        v = (255 - np.rint(np.clip(mag / ref, 0, 1) * 255)).astype(np.uint8)
    return [Image.fromarray(v, "L").convert("RGB")], {}


def _band_font(width, base):
    fs = base
    while fs > 6 and _tsize(_font(fs), "Image 40")[0] + 2 * max(2, fs // 4) > width:
        fs -= 1
    return _font(fs), fs


def _with_band(img, k, font, fs, band_h=None, width=None):
    w, h = img.size
    pad = max(2, fs // 4)
    th = _tsize(font, "Image 40")[1]
    bh = band_h if band_h is not None else th + 2 * pad + 1
    out = Image.new("RGB", (w, h + bh), (255, 255, 255))
    out.paste(img, (0, bh))
    d = ImageDraw.Draw(out)
    d.line([(0, bh - 1), (w - 1, bh - 1)], fill=(160, 160, 160))
    _text(d, pad, pad, f"Image {k}", font, (0, 0, 0))
    return out


def _op_label_images(images, p):
    out = []
    for k, img in enumerate(images, 1):
        font, fs = _band_font(img.width, _clamp(round(max(img.size) / 40), 10, 96))
        out.append(_with_band(img, k, font, fs))
    return out, {}


def _op_contact_sheet(images, p):
    n = len(images)
    c = min(p["cols"], n)
    rows = math.ceil(n / c)
    a = float(np.median([im.height / im.width for im in images]))
    a = _clamp(a, 0.25, 4.0)
    cell_w = max(64, min(768, 2048 // c, int(4096 / (rows * a))))
    cell_h = max(28, round(cell_w * a))
    fs = _clamp(round(cell_w / 12), 10, 40)
    font = _font(fs)
    pad = max(2, fs // 4)
    band = _tsize(font, "Image 40")[1] + 2 * pad
    gap = 8
    W = c * cell_w + (c + 1) * gap
    H = rows * (band + cell_h) + (rows + 1) * gap
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, im in enumerate(images):
        r, q = divmod(i, c)
        x0 = gap + q * (cell_w + gap)
        y0 = gap + r * (band + cell_h + gap)
        s = min(cell_w / im.width, cell_h / im.height)
        size = (max(1, round(im.width * s)), max(1, round(im.height * s)))
        t = _resize(im, size)
        px = x0 + (cell_w - size[0]) // 2
        py = y0 + band + (cell_h - size[1]) // 2
        sheet.paste(t, (px, py))
        d.rectangle([px - 1, py - 1, px + size[0], py + size[1]], outline=(170, 170, 170))
        _text(d, x0 + pad, y0 + pad, f"Image {i + 1}", font, (0, 0, 0))
    return [sheet], {"cols": c}


_OPS = {
    "scale": _op_scale, "crop": _op_crop, "crop_border": _op_crop_border, "pad_square": _op_pad_square,
    "split_grid": _op_split_grid, "split_panels": _op_split_panels, "grid": _op_grid, "dots": _op_dots,
    "ruler": _op_ruler, "sharpen": _op_sharpen, "contrast": _op_contrast, "autocontrast": _op_autocontrast,
    "equalize": _op_equalize, "grayscale": _op_grayscale, "invert": _op_invert, "edges": _op_edges,
    "label_images": _op_label_images, "contact_sheet": _op_contact_sheet,
}
assert set(_OPS) == set(VOCAB)


# ---------------------------------------------------------------------------
# Notes: one factual sentence (or two) per stage
# ---------------------------------------------------------------------------

def _phrase(op, p):
    """Past-participle phrase for single-output per-image ops."""
    if op == "scale":
        f = p["f"]
        if f > 1:
            return f"enlarged {_fmt(f)} times"
        if f < 1:
            return f"reduced to {_fmt(f)} times its size"
        return "kept at its size"
    if op == "crop":
        x0, y0, x1, y1 = p["box"]
        return (f"cropped to the region from {_pct(x0)} to {_pct(x1)} of its width "
                f"and {_pct(y0)} to {_pct(y1)} of its height")
    if op == "crop_border":
        return "trimmed of its uniform background border (2% margin kept)"
    if op == "pad_square":
        return "padded to a square with its background colour, centred"
    if op == "grid":
        nx, ny = p["nx"], p["ny"]
        base = f"overlaid with a grid of {nx} columns and {ny} rows"
        last = f"{_letters(nx - 1)}{ny}"
        if p["labels"] == "cells":
            return base + f", each cell tagged A1 to {last} (column letter, row number) in its top-left corner"
        if p["labels"] == "axes":
            return (base + f", with column letters A to {_letters(nx - 1)} along the top and row numbers 1 to {ny} "
                    "down the left in a white margin added outside the image")
        return base
    if op == "dots":
        nx, ny = p["nx"], p["ny"]
        s = f"marked with a lattice of {nx} by {ny} dots at the cell centres"
        if p["labels"]:
            s += f", each tagged (col,row) from (1,1) at the top left to ({nx},{ny}) at the bottom right"
        return s
    if op == "ruler":
        return (f"given white margins with {p['ticks']} tick steps marking 0 to 100 percent of its width "
                "(top and bottom) and height (left and right), the image itself not drawn on")
    if op == "sharpen":
        return f"sharpened (unsharp mask, radius 2, {int(round(100 * p['amount']))}%)"
    if op == "contrast":
        return f"contrast-adjusted by a factor of {_fmt(p['factor'])} (1 = unchanged)"
    if op == "autocontrast":
        return "auto-contrasted (levels stretched to the full range, 1% clipped at each end)"
    if op == "equalize":
        return "histogram-equalized"
    if op == "grayscale":
        return "converted to grayscale"
    if op == "invert":
        return "colour-inverted (negative)"
    if op == "edges":
        return "converted to an edge map showing edges as dark lines on white"
    raise KeyError(op)


_UNCHANGED_REASON = {"crop_border": "had no uniform border to trim", "pad_square": "was already square"}


def _span(a, b):
    return f"image {a}" if a == b else f"images {a} to {b}"


def _list_idx(idx):
    idx = [i + 1 for i in idx]
    if len(idx) == 1:
        return f"image {idx[0]}"
    return "images " + ", ".join(str(i) for i in idx[:-1]) + f" and {idx[-1]}"


def _note(op, p, ctx):
    n, targets, keep, n_after = ctx["n"], ctx["targets"], ctx["keep"], ctx["n_after"]
    single = len(targets) == 1
    t1 = targets[0] + 1
    if ctx["on"] == "all":
        subj = "The image" if n == 1 else "Each image"
    else:
        subj = f"Image {t1}"
    if op == "label_images":
        if keep == "replace":
            if n == 1:
                return 'A band reading "Image 1" was added above the image.'
            return f'A band reading "Image k" was added above each image, k being its position (1 to {n}).'
        return (f'{_span(n + 1, n_after).capitalize()} {"was" if n == 1 else "were"} added at the end: '
                f'{"a copy" if n == 1 else "copies"} of {_span(1, n)}, each with a band reading "Image k" above it, '
                'k being the position of the original.')
    if op == "contact_sheet":
        c = ctx["infos"][0]["cols"]
        if keep == "replace":
            return (f'All {n} images were placed on one contact sheet in {c} columns, each scaled to a common cell '
                    f'under a band reading "Image k" (k = its previous position); the sheet is now the only image.'
                    if n > 1 else
                    'The image was placed on a contact sheet under a band reading "Image 1"; the sheet replaces it.')
        return (f'Image {n_after} was added at the end: a contact sheet of {_span(1, n)} in {c} columns, '
                f'each under a band reading "Image k".')
    if op in ("split_grid", "split_panels"):
        counts = ctx["counts"]
        if op == "split_grid":
            how = f"{p['rows']} rows by {p['cols']} columns of equal tiles"
            word = "tiles"
        else:
            how = "panels along background gutters"
            word = "panels"
        if single:
            k = counts[0]
            if op == "split_panels" and k == 1:
                if keep == "replace":
                    return f"No panel gutters were found in image {t1}, so it was left unchanged."
                return (f"No panel gutters were found in image {t1}; an unchanged copy was added at the end "
                        f"as image {n_after}.")
            if keep == "replace":
                return (f"Image {t1} was cut into {k} {word}" + (f" ({how})" if op == "split_grid" else
                        " along background gutters") +
                        f"; in reading order they are now {_span(t1, t1 + k - 1)}.")
            return (f"The {k} {word} of image {t1}" + (f" ({how})" if op == "split_grid" else
                    ", cut along background gutters") +
                    f" follow after the originals in reading order as {_span(n + 1, n_after)}.")
        detail = "" if op == "split_grid" else (
            " (" + ", ".join(str(k) for k in counts) + f" panels for {_span(1, n)})")
        if keep == "replace":
            return (f"Each image was cut into {how}{detail}; the parts replace it in place in reading order, "
                    f"giving {n_after} images.")
        return (f"The {word} of every image ({how}{detail}) follow after the originals, image by image in "
                f"reading order, as {_span(n + 1, n_after)}.")
    # single-output per-image ops
    phrase = _phrase(op, p)
    extra = ""
    if op in _UNCHANGED_REASON:
        unchanged = [t for t, inf in zip(targets, ctx["infos"]) if not inf.get("changed", True)]
        if unchanged:
            one = len(unchanged) == 1
            who = "The image" if (n == 1 and ctx["on"] == "all") else _list_idx(unchanged).capitalize()
            reason = _UNCHANGED_REASON[op] if one else _UNCHANGED_REASON[op].replace("was ", "were ")
            if len(unchanged) == len(targets) and keep == "replace":
                return f"{who} {reason}, so nothing was changed."
            extra = f" ({who} {reason}, so {'its copy is' if one else 'their copies are'} unchanged.)" \
                if keep == "append" else f" ({who} {reason} and {'was' if one else 'were'} left unchanged.)"
    if keep == "replace":
        return f"{subj} was {phrase}.{extra}"
    if single:
        return f"Image {n_after} was added at the end: a copy of image {t1}, {phrase}.{extra}"
    return (f"Images {n + 1} to {n_after} were added at the end: copies of images 1 to {n} in order, "
            f"each {phrase}.{extra}")


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def _check_limits(images, new, stage_no, op):
    L = LIMITS
    if len(images) > L["max_images_out"]:
        raise ViewSkip(f"stage {stage_no} ({op}): {len(images)} images exceed the limit of {L['max_images_out']}")
    for im in new:
        w, h = im.size
        if min(w, h) < L["min_side"] or max(w, h) > L["max_side"]:
            raise ViewSkip(f"stage {stage_no} ({op}): output size {w}x{h} outside sides "
                           f"{L['min_side']}..{L['max_side']}")
    tot = sum(im.width * im.height for im in images)
    if tot > L["max_total_px"]:
        raise ViewSkip(f"stage {stage_no} ({op}): {tot} pixels in total exceed {L['max_total_px']}")


def _precheck(op, p, imgs, others_px, stage_no, keep):
    """Refuse size-increasing ops before allocating huge images."""
    L = LIMITS
    if op not in ("scale", "pad_square"):
        return
    tot = others_px
    for im in imgs:
        w, h = im.size
        if op == "scale":
            nw, nh = max(1, round(w * p["f"])), max(1, round(h * p["f"]))
        else:
            nw = nh = max(w, h)
        if min(nw, nh) < L["min_side"] or max(nw, nh) > L["max_side"]:
            raise ViewSkip(f"stage {stage_no} ({op}): output size {nw}x{nh} outside sides "
                           f"{L['min_side']}..{L['max_side']}")
        tot += nw * nh
    if tot > L["max_total_px"]:
        raise ViewSkip(f"stage {stage_no} ({op}): {tot} pixels in total exceed {L['max_total_px']}")


def apply(images, view):
    """Apply a view to a list of PIL images. Returns (list of RGB images, note)."""
    v = validate(view)
    if not v["ok"]:
        raise ValueError("invalid view: " + v["error"])
    if images is None or len(images) == 0:
        raise ViewSkip("no input images")
    cur = [_to_rgb(im) for im in images]
    n_in = len(cur)
    notes = []
    for k, st in enumerate(v["canonical"], 1):
        op, on, keep = st["op"], st["on"], st["keep"]
        p = {key: st[key] for key in VOCAB[op]["params"]}
        n = len(cur)
        if on == "all":
            targets = list(range(n))
        else:
            if on >= n:
                raise ViewSkip(f"stage {k} ({op}): image index {on} out of range for {n} images")
            targets = [on]
        fn = _OPS[op]
        if op in _ALL_ONLY:
            outs, info = fn(cur, p)
            infos, counts = [info], [len(outs)]
            new_imgs = outs
            nxt = outs if keep == "replace" else cur + outs
        else:
            if keep == "replace":
                others = sum(cur[i].width * cur[i].height for i in range(n) if i not in targets)
            else:
                others = sum(im.width * im.height for im in cur)
            _precheck(op, p, [cur[t] for t in targets], others, k, keep)
            per = {}
            infos, counts = [], []
            for t in targets:
                outs, info = fn(cur[t], p)
                per[t] = outs
                infos.append(info)
                counts.append(len(outs))
            new_imgs = [im for t in targets for im in per[t]]
            if keep == "replace":
                nxt = []
                for i, im in enumerate(cur):
                    nxt.extend(per[i] if i in per else [im])
            else:
                nxt = cur + new_imgs
        _check_limits(nxt, new_imgs, k, op)
        ctx = {"n": n, "targets": targets, "keep": keep, "on": on, "n_after": len(nxt),
               "infos": infos, "counts": counts}
        notes.append(_note(op, p, ctx))
        cur = nxt
    if len(cur) != n_in:
        notes.append(f"{len(cur)} images are shown in total.")
    return cur, " ".join(notes)


# ---------------------------------------------------------------------------
# random_view, describe
# ---------------------------------------------------------------------------

def _draw_param(rng, op, name, spec):
    if "interval" in spec:
        lo, hi = spec["interval"]
        if (op, name) in _LOG_PARAMS:
            x = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        else:
            x = rng.uniform(lo, hi)
        return round(_clamp(float(x), lo, hi), 4)
    if "int" in spec:
        lo, hi = spec["int"]
        return int(rng.integers(lo, hi + 1))
    if "choice" in spec:
        ch = spec["choice"]
        return ch[int(rng.integers(len(ch)))]
    out = []
    for _ in range(2):
        wdt = round(float(rng.uniform(0.05, 1.0)), 4)
        a = round(float(rng.uniform(0.0, 1.0 - wdt)), 4)
        b = round(min(1.0, a + wdt), 4)
        a = round(min(a, b - 0.05), 4)
        out.append((a, b))
    (x0, x1), (y0, y1) = out
    return [x0, y0, x1, y1]


def random_view(rng):
    """One random stage: op uniform over VOCAB, params uniform (log-uniform for factors)."""
    ops = sorted(VOCAB)
    op = ops[int(rng.integers(len(ops)))]
    st = {"op": op}
    while True:
        for name in sorted(VOCAB[op]["params"]):
            st[name] = _draw_param(rng, op, name, VOCAB[op]["params"][name])
        if not (op == "split_grid" and st["rows"] == 1 and st["cols"] == 1):
            break
    st["on"] = "all"
    keeps = _RANDOM_KEEPS.get(op, ["replace", "append"])
    st["keep"] = keeps[int(rng.integers(len(keeps)))]
    return [st]


def _param_text(name, spec):
    if "interval" in spec:
        lo, hi = spec["interval"]
        return f"{name}: number {_fmt(lo)} to {_fmt(hi)}"
    if "int" in spec:
        lo, hi = spec["int"]
        return f"{name}: integer {lo} to {hi}"
    if "choice" in spec:
        return f"{name}: one of " + " | ".join(json.dumps(c) for c in spec["choice"])
    return f"{name}: [x0, y0, x1, y1], fractions 0 to 1, x1 - x0 >= 0.05, y1 - y0 >= 0.05"


def describe() -> str:
    L = LIMITS
    app = ", ".join(sorted(_DEFAULT_KEEP))
    lines = [
        f"A view is a list of 1 to {L['max_stages']} stages, applied in order to the question's images before "
        "they are shown to the model; a factual note describing the stages is added after the question.",
        'Stage format: {"op": <name>, <parameters>, "on": "all" or a 0-based image index, '
        '"keep": "replace" or "append"}.',
        '"replace": the stage\'s outputs take the place of the image(s) it was applied to; '
        '"append": the current images stay and the outputs are added at the end.',
        f'All listed parameters are required. "on" defaults to "all"; "keep" defaults to "replace" '
        f'("append" for {app}). {" and ".join(_ALL_ONLY)} accept only "on": "all".',
        f"Limits: at most {L['max_images_out']} images, every side {L['min_side']} to {L['max_side']} px, "
        f"at most {L['max_total_px']} pixels in total; a view that breaks a limit is skipped for that question.",
        "Operations:",
    ]
    for op in sorted(VOCAB):
        ps = VOCAB[op]["params"]
        sig = "; ".join(_param_text(k, ps[k]) for k in sorted(ps))
        lines.append(f"- {op}" + (f" ({sig})" if sig else "") + f": {VOCAB[op]['what']}.")
    return "\n".join(lines)
