"""Tests for views.py.  Run: python t_views.py"""

import io
import json
import os
import sys
import time
import traceback

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import views  # noqa: E402
from views import ViewSkip, apply, describe, random_view, validate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = []

RECT_COLOURS = [(220, 40, 40), (40, 160, 60), (40, 70, 200), (230, 180, 30), (150, 50, 170), (30, 170, 170)]


def case(name):
    def deco(fn):
        try:
            fn()
            RESULTS.append((name, True, ""))
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            RESULTS.append((name, False, repr(e)))
            print(f"FAIL {name}: {e!r}")
            traceback.print_exc(limit=3)
        return fn
    return deco


# ---------------------------------------------------------------- inputs

def panel_canvas():
    """White 1200x800 with a 2x3 arrangement of rectangles, 40-px gutters and outer margins."""
    a = np.full((800, 1200, 3), 255, np.uint8)
    rw, rh = (1200 - 4 * 40) // 3, (800 - 3 * 40) // 2
    rects = []
    for r in range(2):
        for c in range(3):
            x0, y0 = 40 + c * (rw + 40), 40 + r * (rh + 40)
            a[y0:y0 + rh, x0:x0 + rw] = RECT_COLOURS[r * 3 + c]
            rects.append((x0, y0, x0 + rw, y0 + rh))
    return Image.fromarray(a), rects


def photo(w=1024, h=768, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([128 + 80 * np.sin(x / 57.0), 128 + 80 * np.cos(y / 43.0), 128 + 60 * np.sin((x + y) / 91.0)], -1)
    a = np.clip(base + rng.normal(0, 25, (h, w, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def tiny():
    return photo(64, 48, seed=3)


def five():
    sizes = [(640, 480), (300, 500), (1000, 250), (128, 128), (800, 600)]
    return [photo(w, h, seed=i + 10) for i, (w, h) in enumerate(sizes)]


def big():
    return photo(6000, 4000, seed=7)


def png(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def arr(img):
    return np.asarray(img).astype(np.int32)


def assert_rgb(imgs):
    for im in imgs:
        assert im.mode == "RGB", im.mode


PHOTO = photo()
CANVAS, RECTS = panel_canvas()
FIVE = five()


# ---------------------------------------------------------------- ops

@case("scale enlarge/shrink sizes")
def _():
    o, n = apply([PHOTO], [{"op": "scale", "f": 2}])
    assert len(o) == 1 and o[0].size == (2048, 1536), o[0].size
    assert "enlarged 2.0 times" in n, n
    o, n = apply([PHOTO], [{"op": "scale", "f": 0.25}])
    assert o[0].size == (256, 192) and "reduced" in n
    assert_rgb(o)


@case("crop box region preserved")
def _():
    o, _ = apply([PHOTO], [{"op": "crop", "box": [0.25, 0.5, 0.75, 1.0]}])
    assert o[0].size == (512, 384)
    assert np.array_equal(arr(o[0]), arr(PHOTO)[384:768, 256:768])


@case("crop_border trims white margin")
def _():
    a = np.full((800, 1000, 3), 255, np.uint8)
    a[200:600, 300:700] = (10, 100, 200)
    o, n = apply([Image.fromarray(a)], [{"op": "crop_border", "tol": 10}])
    w, h = o[0].size
    # content 400x400 plus 2 % margin each side (20 px horizontally, 16 px vertically)
    assert (w, h) == (440, 432), (w, h)
    assert "trimmed" in n, n
    o2, n2 = apply([PHOTO], [{"op": "crop_border", "tol": 5}])
    assert o2[0].size == PHOTO.size and "nothing was changed" in n2, n2


@case("crop_border on 2x3 canvas keeps all rectangles")
def _():
    o, _ = apply([CANVAS], [{"op": "crop_border", "tol": 12}])
    w, h = o[0].size
    assert w < 1200 and h < 800 and w >= 1120 and h >= 720, (w, h)


@case("pad_square centred with background colour")
def _():
    o, n = apply([CANVAS], [{"op": "pad_square"}])
    assert o[0].size == (1200, 1200)
    a = arr(o[0])
    assert np.array_equal(a[200:1000], arr(CANVAS)) and (a[:200] == 255).all() and (a[1000:] == 255).all()
    o, n = apply([Image.new("RGB", (100, 100), "red")], [{"op": "pad_square"}])
    assert "already square" in n, n


@case("split_grid 2x3 tiles in reading order")
def _():
    o, n = apply([CANVAS], [{"op": "split_grid", "rows": 2, "cols": 3, "keep": "replace"}])
    assert len(o) == 6 and all(im.size == (400, 400) for im in o)
    for k, im in enumerate(o):
        med = np.median(arr(im).reshape(-1, 3), axis=0)
        assert tuple(int(v) for v in med) == RECT_COLOURS[k], (k, med)


@case("split_panels auto on 2x3 canvas -> 6 panels with rectangle colours")
def _():
    o, n = apply([CANVAS], [{"op": "split_panels", "axis": "auto", "min_gap": 8, "keep": "replace"}])
    assert len(o) == 6, len(o)
    for k, im in enumerate(o):
        a = arr(im)
        h, w = a.shape[:2]
        centre = a[h // 4: 3 * h // 4, w // 4: 3 * w // 4].reshape(-1, 3).mean(0)
        assert np.abs(centre - np.array(RECT_COLOURS[k])).max() < 1, (k, centre)
        # each panel contains its whole rectangle
        assert (np.abs(a - np.array(RECT_COLOURS[k])).max(2) == 0).sum() == RECTS[k][2] * 0 + \
            (RECTS[k][2] - RECTS[k][0]) * (RECTS[k][3] - RECTS[k][1])
    assert "6 panels" in n, n


@case("split_panels rows/cols axes and no-gutter no-op")
def _():
    o, _ = apply([CANVAS], [{"op": "split_panels", "axis": "rows", "min_gap": 8, "keep": "replace"}])
    assert len(o) == 2 and o[0].width == 1200
    o, _ = apply([CANVAS], [{"op": "split_panels", "axis": "cols", "min_gap": 8, "keep": "replace"}])
    assert len(o) == 3 and o[0].height == 800
    o, n = apply([PHOTO], [{"op": "split_panels", "axis": "auto", "min_gap": 8, "keep": "replace"}])
    assert len(o) == 1 and np.array_equal(arr(o[0]), arr(PHOTO)) and "No panel gutters" in n, n
    # gutters narrower than min_gap are ignored (40 px gutter = 34 normalised px)
    o, _ = apply([CANVAS], [{"op": "split_panels", "axis": "auto", "min_gap": 40, "keep": "replace"}])
    assert len(o) == 1


@case("split_panels caps at 16 panels and drops slivers")
def _():
    a = np.full((1000, 1000, 3), 255, np.uint8)
    for r in range(5):
        for c in range(5):
            a[20 + r * 196: 20 + r * 196 + 160, 20 + c * 196: 20 + c * 196 + 160] = (0, 0, 0)
    o, _ = apply([Image.fromarray(a)], [{"op": "split_panels", "axis": "auto", "min_gap": 8, "keep": "replace"}])
    assert len(o) == 16, len(o)
    b = np.full((1000, 1000, 3), 255, np.uint8)
    b[100:900, 100:480] = 0
    b[100:900, 520:900] = 0
    b[100:900, 950:960] = 0  # 10-px sliver (< 5 % of 1000) after a 40-px gutter
    o, _ = apply([Image.fromarray(b)], [{"op": "split_panels", "axis": "cols", "min_gap": 8, "keep": "replace"}])
    assert len(o) == 2, len(o)


@case("grid none/cells sizes and unchanged cell interiors")
def _():
    for lab in ("none", "cells"):
        o, n = apply([PHOTO], [{"op": "grid", "nx": 8, "ny": 6, "labels": lab}])
        assert o[0].size == PHOTO.size
        a, b = arr(o[0]), arr(PHOTO)
        # interior of the bottom-right cell away from lines and the label corner is unchanged
        assert np.array_equal(a[700:760, 950:1015], b[700:760, 950:1015])
        assert not np.array_equal(a[:, 128], b[:, 128])  # line at x=128
    assert "A1 to H6" in n, n


@case("grid axes: original pixels unshifted inside margin band")
def _():
    o, n = apply([PHOTO], [{"op": "grid", "nx": 8, "ny": 6, "labels": "axes"}])
    W, H = o[0].size
    left, top = W - 1024, H - 768
    assert left > 0 and top > 0
    a, b = arr(o[0]), arr(PHOTO)
    inner = a[top:, left:]
    mask = np.ones((768, 1024), bool)
    for x in [128 * i for i in range(1, 8)]:
        mask[:, x - 3:x + 3] = False
    for y in [128 * i for i in range(1, 6)]:
        mask[y - 3:y + 3, :] = False
    assert np.array_equal(inner[mask], b[mask])
    band = a[:top, left:]
    assert (band < 128).any(), "no letters in top band"
    assert "column letters A to H" in n


@case("dots with and without labels")
def _():
    o1, n1 = apply([PHOTO], [{"op": "dots", "nx": 6, "ny": 4, "labels": True}])
    o2, n2 = apply([PHOTO], [{"op": "dots", "nx": 6, "ny": 4, "labels": False}])
    assert o1[0].size == PHOTO.size == o2[0].size
    a1, a2, b = arr(o1[0]), arr(o2[0]), arr(PHOTO)
    cx, cy = round(0.5 * 1024 / 6), round(0.5 * 768 / 4)
    assert (a2[cy, cx] != b[cy, cx]).any()
    assert (a1 != a2).sum() > (a2 != b).sum() / 3  # labels add marks
    assert "(6,4)" in n1 and "(6,4)" not in n2


@case("ruler margins on four sides, image unchanged")
def _():
    o, n = apply([PHOTO], [{"op": "ruler", "ticks": 10}])
    W, H = o[0].size
    bw, bh = (W - 1024) // 2, (H - 768) // 2
    assert bw > 0 and bh > 0 and W - 1024 == 2 * bw and H - 768 == 2 * bh
    assert np.array_equal(arr(o[0])[bh:bh + 768, bw:bw + 1024], arr(PHOTO))
    assert "0 to 100 percent" in n


@case("pixel ops keep size and mode")
def _():
    for st in ({"op": "sharpen", "amount": 1.5}, {"op": "contrast", "factor": 2.0}, {"op": "autocontrast"},
               {"op": "equalize"}, {"op": "grayscale"}, {"op": "invert"}, {"op": "edges", "keep": "replace"}):
        o, n = apply([PHOTO], [st])
        assert len(o) == 1 and o[0].size == PHOTO.size and o[0].mode == "RGB", st
        assert not np.array_equal(arr(o[0]), arr(PHOTO)), st
    o, _ = apply([PHOTO], [{"op": "grayscale"}])
    a = arr(o[0])
    assert (a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all()
    o, _ = apply([PHOTO], [{"op": "invert"}])
    assert np.array_equal(arr(o[0]), 255 - arr(PHOTO))
    o, _ = apply([CANVAS], [{"op": "edges", "keep": "replace"}])
    a = arr(o[0])
    assert a[400, 20].min() == 255 and a[RECTS[0][1] + 50, RECTS[0][0]].max() < 128  # edge dark, flat white


@case("edges default keep is append")
def _():
    o, n = apply([PHOTO], [{"op": "edges"}])
    assert len(o) == 2 and np.array_equal(arr(o[0]), arr(PHOTO)) and "added at the end" in n


@case("label_images adds band above, content intact")
def _():
    o, n = apply(FIVE, [{"op": "label_images"}])
    assert len(o) == 5
    for im, src in zip(o, FIVE):
        bh = im.height - src.height
        assert im.width == src.width and bh > 0
        assert np.array_equal(arr(im)[bh:], arr(src))
        assert (arr(im)[:bh] < 100).any()
    assert '"Image k"' in n


@case("contact_sheet returns one image")
def _():
    o, n = apply(FIVE, [{"op": "contact_sheet", "cols": 3}])
    assert len(o) == 1 and o[0].mode == "RGB"
    o2, n2 = apply(FIVE, [{"op": "contact_sheet", "cols": 3, "keep": "append"}])
    assert len(o2) == 6 and all(np.array_equal(arr(a), arr(b)) for a, b in zip(o2[:5], FIVE))
    assert np.array_equal(arr(o2[5]), arr(o[0])) and "Image 6" in n2


@case("tiny 64x48 image through every op (ok or ViewSkip)")
def _():
    t = tiny()
    for op in sorted(views.VOCAB):
        st = {"op": op}
        for k, spec in views.VOCAB[op]["params"].items():
            st[k] = {"interval": lambda s: s["interval"][0] if op != "scale" else 2.0,
                     "int": lambda s: max(s["int"][0], 2), "choice": lambda s: s["choice"][0],
                     "box": lambda s: [0.0, 0.0, 1.0, 1.0]}[next(iter(spec))](spec)
        try:
            o, n = apply([t], [st])
            assert_rgb(o)
            assert n
        except ViewSkip:
            assert op in ("split_grid",), op  # 2x2 tiles of 64x48 -> 32x24 < 28
    o, _ = apply([t], [{"op": "label_images"}])
    assert o[0].width == 64


@case("RGBA / L / P inputs become RGB")
def _():
    rgba = Image.new("RGBA", (100, 80), (255, 0, 0, 0))
    o, _ = apply([rgba, Image.new("L", (90, 90), 7), Image.new("P", (50, 60))], [{"op": "invert", "on": 1}])
    assert_rgb(o)
    assert arr(o[0])[0, 0].tolist() == [255, 255, 255]  # transparent composited on white
    assert arr(o[1])[0, 0].tolist() == [248, 248, 248]


# ---------------------------------------------------------------- keep / on

@case("keep/on: split_grid 2x2 on index 2, replace -> 8 in order")
def _():
    o, n = apply(FIVE, [{"op": "split_grid", "rows": 2, "cols": 2, "on": 2, "keep": "replace"}])
    assert len(o) == 8
    for i in (0, 1):
        assert o[i] is not FIVE[i] and np.array_equal(arr(o[i]), arr(FIVE[i]))
    src = arr(FIVE[2])
    tiles = [src[:125, :500], src[:125, 500:], src[125:, :500], src[125:, 500:]]
    for j in range(4):
        assert np.array_equal(arr(o[2 + j]), tiles[j]), j
    for i in (3, 4):
        assert np.array_equal(arr(o[i + 3]), arr(FIVE[i]))
    assert "images 3 to 6" in n, n


@case("keep/on: append -> 9, originals first")
def _():
    o, n = apply(FIVE, [{"op": "split_grid", "rows": 2, "cols": 2, "on": 2, "keep": "append"}])
    assert len(o) == 9
    assert all(np.array_equal(arr(o[i]), arr(FIVE[i])) for i in range(5))
    assert np.array_equal(arr(o[5]), arr(FIVE[2])[:125, :500])
    assert "images 6 to 9" in n, n


@case("keep/on: replace on all and per-index single-output op")
def _():
    o, _ = apply(FIVE, [{"op": "invert", "on": 3}])
    assert len(o) == 5 and np.array_equal(arr(o[3]), 255 - arr(FIVE[3]))
    assert np.array_equal(arr(o[4]), arr(FIVE[4]))
    o, n = apply(FIVE, [{"op": "invert", "keep": "append"}])
    assert len(o) == 10 and np.array_equal(arr(o[7]), 255 - arr(FIVE[2])) and "Images 6 to 10" in n


@case("keep/on: out-of-range index and empty list -> ViewSkip")
def _():
    for imgs, view in ((FIVE, [{"op": "invert", "on": 5}]), ([], [{"op": "invert"}])):
        try:
            apply(imgs, view)
            raise AssertionError("no ViewSkip")
        except ViewSkip:
            pass


@case("multi-stage note mentions every stage, inputs not mutated")
def _():
    before = [png(im) for im in FIVE]
    o, n = apply(FIVE, [{"op": "crop", "box": [0, 0, 0.5, 0.5], "on": 0},
                        {"op": "grid", "nx": 4, "ny": 4, "labels": "cells"},
                        {"op": "label_images"}, {"op": "contact_sheet", "cols": 2, "keep": "append"}])
    assert len(o) == 6
    assert n.count(".") >= 4 and "cropped" in n and "grid" in n and "contact sheet" in n, n
    assert [png(im) for im in FIVE] == before
    for w in ("should", "helps", "improve", "please", "look at"):
        assert w not in n.lower(), w


# ---------------------------------------------------------------- validate

INVALID = [
    ([], "1 to 4 stages"),
    ({"op": "scale", "f": 2}, "must be a list"),
    ([{"op": "invert"}] * 5, "got 5"),
    (["scale"], "must be an object"),
    ([{"f": 2}], "missing \"op\""),
    ([{"op": "rotate"}], "unknown op \"rotate\""),
    ([{"op": 3}], "unknown op 3"),
    ([{"op": "scale", "f": 2, "mode": "x"}], "unknown key \"mode\""),
    ([{"op": "invert", "amount": 2}], "unknown key \"amount\""),
    ([{"op": "scale"}], "missing parameter 'f' (a number in [0.25, 4.0])"),
    ([{"op": "scale", "f": 5}], "stage 1 of 1 (scale): parameter 'f' must be a number in [0.25, 4.0], got 5"),
    ([{"op": "scale", "f": 0.1}], "got 0.1"),
    ([{"op": "scale", "f": "2"}], "parameter 'f' must be a number"),
    ([{"op": "scale", "f": True}], "got true"),
    ([{"op": "scale", "f": float("nan")}], "parameter 'f'"),
    ([{"op": "grid", "nx": 1, "ny": 4, "labels": "none"}], "parameter 'nx' must be an integer in [2, 16], got 1"),
    ([{"op": "grid", "nx": 4.5, "ny": 4, "labels": "none"}], "parameter 'nx' must be an integer"),
    ([{"op": "grid", "nx": 4, "ny": 4, "labels": "rows"}], "parameter 'labels' must be one of \"none\", \"cells\", \"axes\""),
    ([{"op": "dots", "nx": 4, "ny": 4, "labels": 1}], "parameter 'labels' must be one of true, false, got 1"),
    ([{"op": "split_grid", "rows": 1, "cols": 1}], "cannot both be 1"),
    ([{"op": "split_grid", "rows": 7, "cols": 1}], "parameter 'rows' must be an integer in [1, 6], got 7"),
    ([{"op": "split_panels", "axis": "diag", "min_gap": 8}], "parameter 'axis'"),
    ([{"op": "split_panels", "axis": "auto", "min_gap": 1}], "parameter 'min_gap' must be an integer in [2, 64]"),
    ([{"op": "crop", "box": [0, 0, 1]}], "parameter 'box' must be [x0, y0, x1, y1]"),
    ([{"op": "crop", "box": [0.5, 0, 0.52, 1]}], "x1 - x0 >= 0.05"),
    ([{"op": "crop", "box": [0, 0.9, 1, 0.5]}], "y1 - y0 >= 0.05"),
    ([{"op": "crop", "box": [0, 0, 1.2, 1]}], "each be in [0, 1]"),
    ([{"op": "crop", "box": "0,0,1,1"}], "parameter 'box' must be"),
    ([{"op": "contact_sheet", "cols": 2, "on": 0}], "\"on\" must be \"all\" for contact_sheet"),
    ([{"op": "label_images", "on": 1}], "\"on\" must be \"all\" for label_images"),
    ([{"op": "invert", "on": -1}], "\"on\" must be \"all\" or a 0-based image index"),
    ([{"op": "invert", "on": "first"}], "\"on\" must be \"all\""),
    ([{"op": "invert", "on": True}], "got true"),
    ([{"op": "invert", "keep": "both"}], "\"keep\" must be \"replace\" or \"append\""),
    ([{"op": "ruler", "ticks": 3}], "parameter 'ticks' must be an integer in [4, 20], got 3"),
    ([{"op": "sharpen", "amount": 3.5}], "parameter 'amount' must be a number in [0.5, 3.0]"),
    ([{"op": "contrast", "factor": 0.2}], "parameter 'factor'"),
    ([{"op": "crop_border", "tol": 65}], "parameter 'tol' must be an integer in [0, 64]"),
    ([{"op": "invert"}, {"op": "scale", "f": 9}], "stage 2 of 2 (scale)"),
    ("not json", "not valid JSON"),
]


@case(f"validate rejects {len(INVALID)} invalid views with precise messages")
def _():
    bad = []
    for view, frag in INVALID:
        r = validate(view)
        if r["ok"] or frag not in (r["error"] or "") or r["canonical"] is not None or r["sha"] is not None:
            bad.append((view, frag, r["error"]))
    assert not bad, bad


@case("validate reports all errors of a view at once")
def _():
    r = validate([{"op": "grid", "nx": 1, "ny": 40, "labels": "x", "extra": 1}])
    for frag in ("'nx'", "'ny'", "'labels'", "\"extra\""):
        assert frag in r["error"], (frag, r["error"])


@case("canonical / sha stable under key order and float formatting")
def _():
    a = validate([{"op": "scale", "f": 2}, {"labels": "cells", "ny": 6, "nx": 8, "op": "grid"}])
    b = validate(json.loads('[{"f": 2.00000, "op": "scale", "on": "all", "keep": "replace"},'
                            ' {"op": "grid", "nx": 8.0, "ny": 6, "labels": "cells"}]'))
    c = validate('[{"op":"scale","f":2.000001},{"op":"grid","labels":"cells","nx":8,"ny":6}]')
    assert a["ok"] and b["ok"] and c["ok"]
    assert a["sha"] == b["sha"] == c["sha"] and a["canonical"] == b["canonical"]
    assert a["canonical"][0] == {"f": 2.0, "keep": "replace", "on": "all", "op": "scale"}
    assert list(a["canonical"][1]) == sorted(a["canonical"][1])
    assert validate([{"op": "edges"}])["canonical"][0]["keep"] == "append"
    assert validate([{"op": "scale", "f": 2.5}])["sha"] != a["sha"]
    d = validate([{"op": "crop", "box": [0.123456, 0, 1, 1]}])
    assert d["canonical"][0]["box"][0] == 0.1235
    import hashlib
    assert a["sha"] == hashlib.sha256(json.dumps(a["canonical"], sort_keys=True,
                                                 separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------- limits

@case("limits: split_grid 6x6 append on 5 images -> ViewSkip")
def _():
    try:
        apply(FIVE, [{"op": "split_grid", "rows": 6, "cols": 6, "keep": "append"}])
        raise AssertionError("no ViewSkip")
    except ViewSkip as e:
        assert "exceed the limit of 40" in str(e), e


BIG = big()


@case("limits: scale 4 on 6000x4000 -> ViewSkip")
def _():
    try:
        apply([BIG], [{"op": "scale", "f": 4}])
        raise AssertionError("no ViewSkip")
    except ViewSkip as e:
        assert "outside sides" in str(e) or "pixels" in str(e), e


@case("limits: 6000x4000 grid axes works; split_grid on tiny -> ViewSkip (min side)")
def _():
    o, _ = apply([BIG], [{"op": "grid", "nx": 8, "ny": 8, "labels": "axes"}])
    assert o[0].width > 6000
    try:
        apply([tiny()], [{"op": "split_grid", "rows": 3, "cols": 3}])
        raise AssertionError("no ViewSkip")
    except ViewSkip as e:
        assert "outside sides" in str(e)


# ---------------------------------------------------------------- random_view

@case("random_view: 2000 draws validate, every op appears, apply ok or ViewSkip")
def _():
    rng = np.random.default_rng(1234)
    seen, keeps = set(), set()
    skips = 0
    for i in range(2000):
        v = random_view(rng)
        r = validate(v)
        assert r["ok"], (v, r["error"])
        assert len(v) == 1 and v[0]["on"] == "all"
        seen.add(v[0]["op"])
        keeps.add((v[0]["op"], v[0]["keep"]))
        if True:  # apply every draw on the 5-image list
            try:
                o, n = apply(FIVE, v)
                assert_rgb(o)
                assert isinstance(n, str) and n
            except ViewSkip:
                skips += 1
    assert seen == set(views.VOCAB), set(views.VOCAB) - seen
    assert ("label_images", "append") not in keeps and ("scale", "append") in keeps
    print(f"     ({skips} of 2000 applied random views raised ViewSkip)")


@case("random_view: log-uniform factor, deterministic for a seed")
def _():
    rng = np.random.default_rng(5)
    fs = []
    while len(fs) < 3000:
        v = random_view(rng)[0]
        if v["op"] == "scale":
            fs.append(v["f"])
    fs = np.array(fs)
    frac_below1 = (fs < 1).mean()
    assert 0.45 < frac_below1 < 0.55, frac_below1  # log-uniform on [0.25, 4] is symmetric around 1
    a = [random_view(np.random.default_rng(9)) for _ in range(3)]
    assert a[0] == a[1] == a[2]


# ---------------------------------------------------------------- determinism

@case("determinism: 300 random views applied twice give identical PNG bytes")
def _():
    rng = np.random.default_rng(77)
    inputs = [FIVE, [PHOTO], [CANVAS]]
    for i in range(300):
        v = random_view(rng)
        imgs = inputs[i % 3]
        try:
            o1, n1 = apply(imgs, v)
        except ViewSkip:
            continue
        o2, n2 = apply([im.copy() for im in imgs], json.loads(json.dumps(v)))
        assert n1 == n2 and len(o1) == len(o2)
        assert all(png(a) == png(b) for a, b in zip(o1, o2)), v


# ---------------------------------------------------------------- describe

@case("describe: generated from VOCAB, neutral")
def _():
    d = describe()
    for op, ent in views.VOCAB.items():
        assert f"- {op}" in d and ent["what"] in d
    for w in ("helps", "improve", "better", "should", "useful", "recommend", "best"):
        assert w not in d.lower(), w


# ---------------------------------------------------------------- timing + samples

@case("timing")
def _():
    im = photo(1024, 768, seed=1)
    v1 = [{"op": "grid", "nx": 8, "ny": 8, "labels": "cells"}]
    v2 = [{"op": "split_panels", "axis": "auto", "min_gap": 8}]
    for name, imgs, v in (("grid(8,8,cells) 1024x768", [im], v1), ("split_panels auto 2x3 canvas", [CANVAS], v2)):
        apply(imgs, v)
        t = time.perf_counter()
        for _ in range(200):
            apply(imgs, v)
        print(f"     {name}: {(time.perf_counter() - t) / 200 * 1000:.2f} ms per apply")


@case("write sample PNGs")
def _():
    out = os.path.join(HERE, "views_samples")
    os.makedirs(out, exist_ok=True)
    o, _ = apply([PHOTO], [{"op": "grid", "nx": 8, "ny": 6, "labels": "axes"}])
    o[0].save(os.path.join(out, "grid_axes.png"))
    o, _ = apply([PHOTO], [{"op": "dots", "nx": 8, "ny": 6, "labels": True}])
    o[0].save(os.path.join(out, "dots_labels.png"))
    o, n = apply([CANVAS], [{"op": "split_panels", "axis": "auto", "min_gap": 8, "keep": "replace"},
                            {"op": "contact_sheet", "cols": 3}])
    o[0].save(os.path.join(out, "split_panels_sheet.png"))
    print("     note:", n)


if __name__ == "__main__":
    p = sum(1 for r in RESULTS if r[1])
    f = len(RESULTS) - p
    print(f"{p} passed, {f} failed")
    sys.exit(1 if f else 0)
