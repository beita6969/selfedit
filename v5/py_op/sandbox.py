"""PY operator sandbox (selfedit v5 pilot, 11-11): run ONE model-written Python block on a row's images in a separate process.

run(code, images) -> dict(status, stdout, error, images, wall, cpu)
    status   "ok" | "error" (the block raised) | "refused" (static check; nothing was run) | "timeout" (wall clock) | "killed" (CPU / signal)
    stdout   what the block printed, cut at OUT_CAP characters
    error    short text for the model (None when ok)
    images   up to MAX_SHOW PNG byte strings the block passed to show()
images: list of the row's original image files as bytes (question order).

Layers (it runs on a SHARED server where nothing may ever be written or deleted):
 1. static check in the parent (ast): imports only from the allowlist, no file / exec / reflection builtins, no name or attribute starting with
    "_", no attribute or name from the deny lists. A refused block is never started.
 2. the child gets a restricted builtins dict (no open / exec / eval / compile / getattr / ...; __import__ returns only the preloaded modules).
 3. the child process (python -B, environment = thread-count variables only, cwd = an EMPTY read-only directory that is never written and never
    removed) imports its modules, decodes the images, and then locks itself down BEFORE the block runs:
       root: chroot into the empty directory, drop to uid / gid nobody;  everyone: close every descriptor except 0-2 and RLIMIT_NOFILE = 3
       (no file and no socket can be opened), RLIMIT_FSIZE = 0 (no byte can be written to a file), RLIMIT_NPROC = 0 (no fork, no thread),
       RLIMIT_CORE = 0, RLIMIT_AS ~6 GB, RLIMIT_CPU = 10 s for the block (soft -> an exception in the block, hard +2 s -> kill).
    The parent kills the child at 20 s of wall clock. Images go in through stdin, results come back through stdout as JSON (never pickle).
Nothing touches the disk: no temp files, no .pyc, no core dumps.
"""
import ast
import base64
import io
import json
import os
import signal
import subprocess
import sys
import time

OUT_CAP = 2000            # characters of printed output returned to the model
MAX_SHOW = 2              # images per run
SHOW_MAX_SIDE = 4096      # a shown image is shrunk to this side at most
CPU_S = 10                # CPU seconds for the block
WALL_S = 20               # wall-clock seconds for the whole child
AS_BYTES = 6 * 1024 ** 3  # address space
MAX_CODE_CHARS = 20000
MAX_RESULT_BYTES = 96 * 1024 ** 2

# ---------------------------------------------------------------------------------------------------- what exists in this interpreter
def _have(mod):
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


HAVE_CV2 = _have("cv2")
HAVE_NDIMAGE = _have("scipy") and _have("scipy.ndimage")
PIL_SUB = ("Image", "ImageDraw", "ImageFilter", "ImageOps")
NP_SUB = ("numpy.linalg", "numpy.fft", "numpy.random", "numpy.ma")


def allowed_modules():
    m = {"numpy", "PIL", "math", "collections", "itertools"} | set(NP_SUB) | {"PIL." + s for s in PIL_SUB}
    if HAVE_CV2:
        m.add("cv2")
    if HAVE_NDIMAGE:
        m |= {"scipy", "scipy.ndimage"}
    return m


def available_text():
    """the module list of the interface note (depends only on what is installed, never on the row)."""
    parts = ["`np` (numpy)", "`PIL` (`Image`, `ImageDraw`, `ImageFilter`, `ImageOps`)"]
    if HAVE_CV2:
        parts.append("`cv2`")
    if HAVE_NDIMAGE:
        parts.append("`scipy.ndimage`")
    parts += ["`math`", "`collections`", "`itertools`"]
    return ", ".join(parts)


# ---------------------------------------------------------------------------------------------------- static check
BAD_NAMES = {            # bare names (the builtins below do not exist in the block's builtins either)
    "open", "exec", "eval", "compile", "getattr", "setattr", "delattr", "globals", "locals", "vars", "input", "breakpoint", "help", "exit",
    "quit", "os", "sys", "subprocess", "shutil", "pathlib", "socket", "ctypes", "importlib", "builtins",
}
BAD_ATTRS = (BAD_NAMES - {"open"}) | {  # attribute names (a module reached through another module's namespace, file / process methods)
    # `.open` itself is allowed since smoke 2: the only reachable one is PIL's Image.open, which the child replaces by a shim that returns
    # the already-loaded image (the model writes Image.open(...) by habit); the bare builtin `open` stays refused.
    "codecs", "tokenize", "dbm", "wave", "fileinput", "sqlite3", "fdopen", "urlopen", "openpty", "open_code", "startfile",
    "io", "pickle", "attrgetter", "methodcaller", "operator", "inspect", "types", "gc", "signal", "resource", "threading",
    "multiprocessing", "tempfile", "logging", "atexit", "memoryview",
    "system", "popen", "remove", "unlink", "rmdir", "rmtree", "removedirs", "rename", "renames", "save", "imwrite", "imread", "tofile",
    "fromfile", "savez", "dump", "load", "memmap", "write", "writelines", "truncate", "chmod", "chown", "mkdir", "makedirs", "mkdtemp",
    "mkstemp", "kill", "killpg", "fork", "spawn", "environ", "getenv", "putenv", "chdir", "chroot", "listdir", "scandir", "walk", "glob",
    "rglob", "iterdir", "expanduser", "touch", "symlink", "link", "Path", "PurePath", "write_text", "write_bytes", "read_text", "read_bytes",
    "ctypeslib", "f2py", "distutils", "testing", "test", "bench", "npyio", "core", "fp", "truetype", "load_path", "safe_eval",
    "marshal", "shelve", "zipfile", "tarfile", "gzip", "bz2", "lzma", "mmap", "asyncio", "concurrent", "webbrowser", "urllib", "http",
    "ftplib", "smtplib", "pty", "fcntl", "code", "codeop", "pdb", "runpy", "pydoc", "linecache", "FileIO", "VideoWriter", "VideoCapture",
    "FileStorage", "imwritemulti", "imreadmulti", "imshow", "waitKey", "namedWindow", "DataSource", "func_globals", "gi_frame", "cr_frame",
    "f_globals", "f_locals", "f_builtins", "tb_frame",
}
BAD_SUBSTR = ("save", "dump", "pickle", "memmap", "unlink", "rmtree", "popen", "subprocess", "ctypes", "socket", "importlib", "builtins",
              "fromfile", "tofile", "imwrite", "imread", "loadtxt", "genfromtxt", "fromregex", "datasource", "__")


OK_UNDERSCORE = ("_", "__name__")     # the throwaway variable `_` and the `if __name__ == "__main__":` idiom: plain variables, no capability


class Refused(Exception):
    pass


def _bad_attr(name):
    if name.startswith("_"):
        return "names starting with an underscore are not allowed"
    if name in BAD_ATTRS:
        return "not allowed (no file, process, network or reflection access)"
    low = name.lower()
    for s in BAD_SUBSTR:
        if s in low:
            return "not allowed (no file, process, network or reflection access)"
    if name.startswith("Image") and name not in ("Image",) + PIL_SUB:
        return "not available (PIL: " + ", ".join(PIL_SUB) + ")"
    if name.endswith("Plugin"):
        return "not available"
    return None


class _Prune(ast.NodeTransformer):
    """Remove statically dead branches (`a if False else b`, `if False: ...`). check() returns the pruned tree and the child compiles
    exactly that tree, so what is checked is what runs (smoke 11-11: the model writes `Image.open("x.png") if False else None` by habit)."""

    @staticmethod
    def _const(t):
        return isinstance(t, ast.Constant) and (isinstance(t.value, (bool, int)) or t.value is None) and not isinstance(t.value, float)

    def visit_IfExp(self, node):
        self.generic_visit(node)
        if self._const(node.test):
            return node.body if node.test.value else node.orelse
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if self._const(node.test):
            keep = node.body if node.test.value else node.orelse
            return keep if keep else ast.copy_location(ast.Pass(), node)
        return node


class _DropUnusedImports(ast.NodeTransformer):
    """Remove an import of a module that is not available when the name it would bind is used nowhere in the block (smoke 2: the model
    writes `import matplotlib.pyplot as plt  # no matplotlib; use show()` by habit). A used one is still refused by the check."""

    def __init__(self, used, mods):
        self.used, self.mods = used, mods

    def visit_Import(self, node):
        keep = [a for a in node.names if a.name in self.mods or (a.asname or a.name.split(".")[0]) in self.used]
        if len(keep) == len(node.names):
            return node
        if not keep:
            return ast.copy_location(ast.Pass(), node)
        node.names = keep
        return node

    def visit_ImportFrom(self, node):
        if node.level or not node.module or node.module not in self.mods:
            if not any((a.asname or a.name) in self.used or a.name == "*" for a in node.names):
                return ast.copy_location(ast.Pass(), node)
        return node


def check(code):
    """-> ast tree (dead branches pruned) of an accepted block; raises Refused(text for the model) otherwise. Nothing is executed."""
    if not isinstance(code, str) or not code.strip():
        raise Refused("empty code block")
    if len(code) > MAX_CODE_CHARS:
        raise Refused(f"code block longer than {MAX_CODE_CHARS} characters")
    if "\x00" in code:
        raise Refused("null byte in code")
    try:
        tree = ast.parse(code, filename="<code>", mode="exec")
    except SyntaxError as e:
        raise Refused(f"SyntaxError: {e.msg} (line {e.lineno})")
    except (ValueError, RecursionError, MemoryError) as e:
        raise Refused(f"code could not be parsed: {type(e).__name__}")
    mods = allowed_modules()
    try:
        tree = _Prune().visit(tree)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        tree = ast.fix_missing_locations(_DropUnusedImports(used, mods).visit(tree))
    except RecursionError:
        raise Refused("code could not be parsed: RecursionError")
    avail = "available modules: " + ", ".join(sorted(m for m in mods if "." not in m or m == "scipy.ndimage"))
    for node in ast.walk(tree):
        ln = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in mods:
                    raise Refused(f"line {ln}: `import {a.name}` is not allowed ({avail})")
                if a.asname and a.asname.startswith("_"):
                    raise Refused(f"line {ln}: names starting with an underscore are not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module or node.module not in mods:
                raise Refused(f"line {ln}: `from {node.module or '.'} import ...` is not allowed ({avail})")
            for a in node.names:
                if a.name == "*":
                    raise Refused(f"line {ln}: `import *` is not allowed")
                if node.module == "PIL" and a.name not in PIL_SUB:
                    raise Refused(f"line {ln}: `from PIL import {a.name}` is not available (PIL: {', '.join(PIL_SUB)})")
                if node.module == "scipy" and a.name != "ndimage":
                    raise Refused(f"line {ln}: `from scipy import {a.name}` is not available (scipy: ndimage)")
                why = _bad_attr(a.name) or (a.asname and _bad_attr(a.asname))
                if why:
                    raise Refused(f"line {ln}: `{a.name}` is {why}")
        elif isinstance(node, ast.Name):
            if node.id in OK_UNDERSCORE:
                continue
            if node.id.startswith("_"):
                raise Refused(f"line {ln}: `{node.id}`: names starting with an underscore are not allowed")
            if node.id in BAD_NAMES:
                raise Refused(f"line {ln}: `{node.id}` is not allowed (no file, process, network or reflection access)")
        elif isinstance(node, ast.Attribute):
            why = _bad_attr(node.attr)
            if why:
                raise Refused(f"line {ln}: `.{node.attr}` is {why}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                raise Refused(f"line {ln}: names starting with an underscore are not allowed")
        elif isinstance(node, ast.arg):
            if node.arg.startswith("_") and node.arg != "_":
                raise Refused(f"line {ln}: names starting with an underscore are not allowed")
        elif isinstance(node, ast.keyword):
            if node.arg and node.arg.startswith("_"):
                raise Refused(f"line {ln}: names starting with an underscore are not allowed")
        elif isinstance(node, ast.alias):
            pass
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            if any(n.startswith("_") or n in BAD_NAMES for n in node.names):
                raise Refused(f"line {ln}: name not allowed")
        elif isinstance(node, ast.ExceptHandler):
            if node.name and node.name.startswith("_"):
                raise Refused(f"line {ln}: names starting with an underscore are not allowed")
        elif isinstance(node, (ast.Await, ast.AsyncFor, ast.AsyncWith)):
            raise Refused(f"line {ln}: async code is not available")
        elif isinstance(node, ast.MatchAs) and node.name and node.name.startswith("_") and node.name != "_":
            raise Refused(f"line {ln}: names starting with an underscore are not allowed")
    return tree


# ---------------------------------------------------------------------------------------------------- parent side
JAIL_ENV = "PY_OP_JAIL"


def jail_dir():
    """The one EMPTY directory the children use as cwd (and chroot target). Created once, mode r-x, never written, never removed."""
    d = os.environ.get(JAIL_ENV)
    if not d:
        base = "/data/fd/mingdazhang/selfedit/v5/py_op" if os.path.isdir("/data/fd/mingdazhang/selfedit") else os.path.join(
            os.path.expanduser("~"), ".cache", "selfedit_py_op")
        d = os.path.join(base, "jail_empty")
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
        os.chmod(d, 0o555)
    if os.listdir(d):
        raise RuntimeError(f"sandbox directory {d} is not empty: refusing to run")
    return d


CHILD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
             "VECLIB_MAXIMUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0"}


def run(code, images, wall_s=WALL_S, cpu_s=CPU_S, prelude=(), _unrestricted=False):
    """Run one block. `prelude`: the blocks this read ran before (notebook state: the child re-runs them silently in the same
    namespace, then runs `code`; each is checked again). `_unrestricted` (tests only) skips layers 1 and 2 so that the OS layer can be
    checked by itself."""
    t0 = time.time()
    out = dict(status="refused", stdout="", error=None, images=[], wall=0.0, cpu=None)
    if not _unrestricted:
        try:
            check(code)
        except Refused as e:
            out.update(error=str(e), wall=time.time() - t0)
            return out
    payload = json.dumps(dict(code=code, images=[base64.b64encode(b).decode() for b in images], cpu_s=cpu_s, unrestricted=bool(_unrestricted),
                              prelude=[c for c in prelude if isinstance(c, str)][:8])).encode()
    jail = jail_dir()
    try:
        p = subprocess.Popen([sys.executable, "-B", os.path.abspath(__file__), "--child"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, env=dict(CHILD_ENV), cwd=jail, close_fds=True, start_new_session=True)
    except OSError as e:
        out.update(status="error", error=f"sandbox could not start: {type(e).__name__}", wall=time.time() - t0)
        return out
    try:
        so, se = p.communicate(payload, timeout=wall_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            p.kill()
        so, se = p.communicate()
        out.update(status="timeout", error=f"stopped: the run exceeded the time limit ({wall_s} s wall clock)", wall=time.time() - t0)
        return out
    out["wall"] = time.time() - t0
    res = None
    if so and len(so) <= MAX_RESULT_BYTES:
        try:
            res = json.loads(so.decode("utf-8", "replace"))
        except ValueError:
            res = None
    if not isinstance(res, dict):
        rc = p.returncode
        if rc is not None and rc < 0 and -rc in (signal.SIGXCPU, signal.SIGKILL):
            out.update(status="killed", error=f"stopped: the run exceeded the CPU time limit ({cpu_s} s)")
        elif rc is not None and rc < 0:
            out.update(status="killed", error=f"the run crashed (signal {-rc})")
        else:
            tail = (se or b"").decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
            out.update(status="error", error=("the run failed: " + tail[0][:300]) if tail[0] else "the run failed with no output")
        return out
    out["stdout"] = str(res.get("stdout") or "")[:OUT_CAP + 80]
    out["cpu"] = res.get("cpu")
    out["lock"] = res.get("lock")
    err = res.get("error")
    out["error"] = str(err)[:1200] if err else None
    out["status"] = "ok" if not err else ("timeout" if res.get("cpu_hit") else "error")
    imgs = []
    for b in (res.get("images") or [])[:MAX_SHOW]:
        try:
            raw = base64.b64decode(b)
            if raw[:8] == b"\x89PNG\r\n\x1a\n":
                imgs.append(raw)
        except Exception:
            pass
    out["images"] = imgs
    return out


# ---------------------------------------------------------------------------------------------------- child side
def _child():
    import resource
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass
    real_out = os.fdopen(1, "wb", closefd=False)
    job = json.loads(sys.stdin.buffer.read().decode())
    code, cpu_s, unrestricted = job["code"], float(job.get("cpu_s") or CPU_S), bool(job.get("unrestricted"))

    import builtins as _b
    import collections
    import itertools
    import linecache
    import math
    import traceback
    import numpy as np
    import numpy.linalg, numpy.fft, numpy.random, numpy.ma            # noqa: E401,F401
    import numpy.lib.stride_tricks                                     # noqa: F401
    import PIL
    from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageFont, ImageChops, ImageStat, ImageColor   # noqa: F401
    mods = {"numpy": np, "math": math, "collections": collections, "itertools": itertools, "PIL.Image": Image, "PIL.ImageDraw": ImageDraw,
            "PIL.ImageFilter": ImageFilter, "PIL.ImageOps": ImageOps, "numpy.linalg": np.linalg, "numpy.fft": np.fft,
            "numpy.random": np.random, "numpy.ma": np.ma}

    class _NS:
        """a module-like holder that exposes only the listed names"""
        def __init__(self, name, **kw):
            self.__dict__.update(kw)
            self.__dict__["__name__"] = name

        def __repr__(self):
            return f"<module {self.__dict__['__name__']}>"

    pil = _NS("PIL", Image=Image, ImageDraw=ImageDraw, ImageFilter=ImageFilter, ImageOps=ImageOps)
    mods["PIL"] = pil
    ns = dict(np=np, numpy=np, PIL=pil, Image=Image, ImageDraw=ImageDraw, ImageFilter=ImageFilter, ImageOps=ImageOps, math=math,
              collections=collections, itertools=itertools)
    cv2 = None
    if HAVE_CV2:
        try:
            import cv2
            cv2.setNumThreads(0)
            try:
                cv2.ocl.setUseOpenCL(False)
            except Exception:
                pass
            mods["cv2"] = cv2
            ns["cv2"] = cv2
        except Exception:
            cv2 = None
    ndimage = None
    if HAVE_NDIMAGE:
        try:
            import scipy
            import scipy.ndimage as ndimage
            sp = _NS("scipy", ndimage=ndimage)
            mods["scipy"] = sp
            mods["scipy.ndimage"] = ndimage
            ns["scipy"] = sp
            ns["ndimage"] = ndimage
        except Exception:
            ndimage = None

    # ---- images (decoded before the lock-down; RGBA composited on white, as the model's view)
    def load_rgb(raw):
        im = Image.open(io.BytesIO(raw))
        im.load()
        if im.mode == "RGB":
            out = im.copy()
        elif im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            out = Image.new("RGB", im.size, (255, 255, 255))
            out.paste(im, mask=im.split()[3])
        else:
            out = im.convert("RGB")
        return out

    imgs = [load_rgb(base64.b64decode(b)) for b in job["images"]]
    ns["imgs"] = imgs
    ns["img"] = imgs[0] if imgs else None

    # ---- warm-up: everything that imports lazily must do so before no file can be opened
    try:
        t = Image.new("RGB", (16, 16), (200, 30, 30))
        d = ImageDraw.Draw(t)
        d.text((1, 1), "a", fill=(0, 0, 0))
        d.line((0, 0, 5, 5), fill=(0, 0, 255), width=2)
        d.rectangle((2, 2, 8, 8), outline=(0, 255, 0))
        d.ellipse((2, 2, 8, 8), outline=(0, 255, 0))
        t.filter(ImageFilter.GaussianBlur(1)); t.filter(ImageFilter.FIND_EDGES); t.filter(ImageFilter.MedianFilter(3))
        ImageOps.autocontrast(t); ImageOps.grayscale(t); ImageOps.invert(t); ImageOps.expand(t, 2)
        t.resize((8, 8)); t.resize((32, 32), Image.BICUBIC); t.rotate(30, expand=True); t.convert("HSV"); t.convert("L").point(lambda v: 255 * (v > 9))
        t.quantize(4); t.getcolors(); t.crop((0, 0, 4, 4)); t.getpixel((1, 1)); t.transpose(Image.FLIP_LEFT_RIGHT); t.histogram(); t.getbbox()
        a = np.asarray(t).astype(float)
        np.linalg.lstsq(np.eye(3), np.ones(3), rcond=None); np.fft.fft2(a[..., 0]); np.random.default_rng(0).random(2)
        np.polyfit([0, 1, 2], [0, 1, 2], 1); np.unique(a, return_counts=True); np.percentile(a, 50); np.histogram(a, 8); np.median(a)
        np.lib.stride_tricks.sliding_window_view(a[..., 0], (3, 3)); np.argwhere(a > 0); np.gradient(a[..., 0]); np.convolve([1, 2], [1, 1])
        np.cov(a[..., 0]); np.corrcoef(a[0, :, 0] + np.arange(16), np.arange(16)); np.linalg.norm(a); np.linalg.eigh(np.eye(2))
        repr(a); str(a[:2, :2]); np.array2string(a[:2, :2, 0]); np.set_printoptions(precision=4, suppress=True, linewidth=160, threshold=400)
        Image.fromarray((a[..., 0] > 0).astype(np.uint8) * 255); Image.fromarray(a.astype(np.uint8))
        if ndimage is not None:
            m = a[..., 0] > 100
            ndimage.label(m); ndimage.find_objects(ndimage.label(m)[0]); ndimage.center_of_mass(m); ndimage.gaussian_filter(a[..., 0], 1)
            ndimage.binary_opening(m); ndimage.binary_closing(m); ndimage.binary_dilation(m); ndimage.binary_erosion(m); ndimage.binary_fill_holes(m)
            ndimage.distance_transform_edt(m); ndimage.zoom(a[..., 0], 0.5); ndimage.rotate(a[..., 0], 10); ndimage.median_filter(a[..., 0], 3)
            ndimage.sobel(a[..., 0]); ndimage.uniform_filter(a[..., 0], 3); ndimage.maximum_filter(a[..., 0], 3); ndimage.minimum_filter(a[..., 0], 3)
            ndimage.sum(a[..., 0], ndimage.label(m)[0], [1]); ndimage.map_coordinates(a[..., 0], [[1.5], [1.5]]); ndimage.convolve(a[..., 0], np.ones((3, 3)))
            ndimage.generate_binary_structure(2, 2); ndimage.laplace(a[..., 0]); ndimage.shift(a[..., 0], 1)
        if cv2 is not None:
            g = cv2.cvtColor(np.asarray(t), cv2.COLOR_RGB2GRAY)
            cv2.Canny(g, 50, 150); cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); cv2.connectedComponentsWithStats(g)
            cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.GaussianBlur(g, (3, 3), 0); cv2.resize(g, (8, 8))
            cv2.HoughLinesP(g, 1, math.pi / 180, 5); cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1, 5); cv2.cvtColor(np.asarray(t), cv2.COLOR_RGB2HSV)
            cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)); cv2.distanceTransform(g, cv2.DIST_L2, 3); cv2.inRange(g, 0, 9)
    except Exception:
        pass

    # ---- capture
    buf = io.StringIO()
    shown = []
    state = dict(extra_show=0, quiet=False)

    def _print(*a, sep=" ", end="\n", file=None, flush=False):
        if state["quiet"] or buf.tell() > 4 * OUT_CAP:
            return
        try:
            buf.write(str(sep).join(str(x) for x in a) + str(end))
        except Exception as e:
            buf.write(f"<unprintable: {type(e).__name__}>\n")

    def _to_pil(x):
        if isinstance(x, Image.Image):
            im = x
        else:
            a = np.asarray(x)
            if a.dtype == bool:
                a = a.astype(np.uint8) * 255
            elif a.dtype != np.uint8:
                a = a.astype(np.float64)
                lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
                if lo >= 0.0 and hi <= 1.0:
                    a = a * 255.0
                elif lo < 0.0 or hi > 255.0:
                    a = (a - lo) / max(hi - lo, 1e-12) * 255.0
                a = np.nan_to_num(a).clip(0, 255).astype(np.uint8)
            if a.ndim == 3 and a.shape[2] == 1:
                a = a[..., 0]
            if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] not in (3, 4)):
                raise ValueError(f"show(): cannot display an array of shape {a.shape}")
            im = Image.fromarray(np.ascontiguousarray(a))
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        if w < 1 or h < 1:
            raise ValueError("show(): empty image")
        if max(w, h) > SHOW_MAX_SIDE:
            s = SHOW_MAX_SIDE / max(w, h)
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
        return im

    def show(x=None, *a, **k):
        if x is None or state["quiet"]:
            return
        if len(shown) >= MAX_SHOW:
            state["extra_show"] += 1
            return
        im = _to_pil(x)
        b = io.BytesIO()
        im.save(b, format="PNG", compress_level=1)
        shown.append(b.getvalue())

    Image.Image.show = lambda self, *a, **k: show(self)

    def _open_shim(x=None, *a, **k):
        """Image.open never touches a file here: it hands back an image that is already loaded."""
        if isinstance(x, Image.Image):
            return x
        if isinstance(x, int) and not isinstance(x, bool) and -len(imgs) <= x < len(imgs):
            return imgs[x]
        if len(imgs) == 1:
            return imgs[0]
        raise FileNotFoundError(f"there are no image files: the {len(imgs)} images are already loaded in the list `imgs`")

    Image.open = _open_shim

    class _W:
        def __init__(self, keep):
            self.keep = keep

        def write(self, t):
            if self.keep and not state["quiet"] and buf.tell() <= 4 * OUT_CAP:
                buf.write(str(t))
            return len(str(t))

        def flush(self):
            pass

        def isatty(self):
            return False

    import warnings
    warnings.simplefilter("ignore")
    sys.stdout, sys.stderr = _W(True), _W(False)

    def _echo(v):
        if v is not None:
            _print(repr(v))
        return None

    def _imp(name, globals=None, locals=None, fromlist=(), level=0):
        if level:
            raise ImportError("relative imports are not available")
        if name not in mods:
            raise ImportError(f"module {name!r} is not available")
        if fromlist:
            return mods[name]
        return mods[name.split(".")[0]]

    SAFE = ("abs all any ascii bin bool bytearray bytes callable chr classmethod complex dict dir divmod enumerate filter float format frozenset "
            "hasattr hash hex id int isinstance issubclass iter len list map max min next object oct ord pow property range repr reversed round "
            "set slice sorted staticmethod str sum super tuple type zip True False None NotImplemented Ellipsis __build_class__ "
            "ArithmeticError AssertionError AttributeError BaseException Exception FloatingPointError ImportError IndexError KeyError "
            "KeyboardInterrupt LookupError MemoryError NameError NotImplementedError OSError OverflowError RecursionError RuntimeError "
            "StopIteration GeneratorExit SyntaxError TypeError UnboundLocalError ValueError ZeroDivisionError TimeoutError Warning DeprecationWarning "
            "RuntimeWarning UserWarning").split()
    if unrestricted:
        bi = dict(vars(_b))
    else:
        bi = {k: getattr(_b, k) for k in SAFE if hasattr(_b, k)}
        bi["__import__"] = _imp
    bi["print"] = _print
    ns.update(__builtins__=bi, __name__="__main__", show=show, print=_print)

    # ---- compile (the parent has checked the block; check again here so that the child never trusts its input)
    err = None
    try:
        tree = ast.parse(code, filename="<code>", mode="exec") if unrestricted else check(code)
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = tree.body[-1]
            v = last.value
            is_print = isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in ("print", "show")
            if not is_print:
                call = ast.Call(func=ast.Name(id="_echo_", ctx=ast.Load()), args=[v], keywords=[])
                tree.body[-1] = ast.copy_location(ast.Expr(value=ast.copy_location(call, v)), last)
                ast.fix_missing_locations(tree)
                ns["_echo_"] = _echo
        comp = compile(tree, "<code>", "exec")
        lines = code.splitlines(True)
        linecache.cache["<code>"] = (len(code), None, lines, "<code>")
    except Refused as e:
        err, comp = "not run: " + str(e), None
    except SyntaxError as e:
        err, comp = f"SyntaxError: {e.msg} (line {e.lineno})", None
    earlier = []                       # the read's earlier blocks: checked again, re-run silently before `code` (notebook state)
    for c in job.get("prelude") or []:
        try:
            earlier.append(compile(ast.parse(c, filename="<earlier>", mode="exec") if unrestricted else check(c), "<earlier>", "exec"))
        except Exception:
            pass

    # ---- lock-down
    lock = dict(root=(os.geteuid() == 0))
    try:
        os.closerange(3, 4096)           # descriptors 0-2 stay open and taken, so RLIMIT_NOFILE = 3 leaves no free slot
    except Exception:
        pass
    if lock["root"]:
        try:
            os.chroot(".")
            os.chdir("/")
            lock["chroot"] = True
        except Exception as e:
            lock["chroot"] = f"{type(e).__name__}"
        try:
            os.setgroups([])
            os.setgid(65534)
            os.setuid(65534)
            lock["uid"] = os.getuid()
        except Exception as e:
            lock["uid"] = f"{type(e).__name__}"
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)

    class CpuLimit(BaseException):
        pass

    def _xcpu(sig, frame):
        raise CpuLimit()

    signal.signal(signal.SIGXCPU, _xcpu)
    used = time.process_time()
    for name, val in (("RLIMIT_FSIZE", (0, 0)), ("RLIMIT_NOFILE", (3, 3)), ("RLIMIT_NPROC", (0, 0)), ("RLIMIT_AS", (AS_BYTES, AS_BYTES)),
                      ("RLIMIT_CPU", (int(used + cpu_s) + 1, int(used + cpu_s) + 3))):
        try:
            resource.setrlimit(getattr(resource, name), val)
            lock[name] = True
        except Exception as e:
            lock[name] = f"{type(e).__name__}"

    # ---- run
    t0 = time.process_time()
    cpu_hit = False
    state["quiet"] = True
    for e_comp in earlier:
        try:
            exec(e_comp, ns)
        except CpuLimit:
            cpu_hit, comp = True, None
            err = f"stopped: the run exceeded the CPU time limit ({int(cpu_s)} s)"
            break
        except BaseException:
            pass
    state["quiet"] = False
    if comp is not None:
        try:
            exec(comp, ns)
        except CpuLimit:
            cpu_hit = True
            err = f"stopped: the run exceeded the CPU time limit ({int(cpu_s)} s)"
        except SystemExit:
            pass
        except MemoryError:
            err = "MemoryError: the run exceeded the memory limit"
        except BaseException as e:
            try:
                tb = traceback.extract_tb(e.__traceback__)
                where = [f for f in tb if f.filename == "<code>"]
                loc = ""
                if where:
                    f = where[-1]
                    src = (f.line or "").strip()
                    loc = f" (line {f.lineno}" + (f": {src[:120]}" if src else "") + ")"
                msg = str(e)
                err = f"{type(e).__name__}: {msg[:600]}{loc}"
            except BaseException:
                err = f"{type(e).__name__}"
    cpu = time.process_time() - t0
    try:
        signal.signal(signal.SIGXCPU, signal.SIG_IGN)
    except Exception:
        pass
    text = buf.getvalue()
    if len(text) > OUT_CAP:
        text = text[:OUT_CAP] + "\n... [output cut at 2,000 characters]"
    if state["extra_show"]:
        text += f"\n[show(): only the first {MAX_SHOW} images are returned; {state['extra_show']} more ignored]"
    res = dict(stdout=text, error=err, images=[base64.b64encode(b).decode() for b in shown], cpu=round(cpu, 3), lock=lock, cpu_hit=cpu_hit)
    real_out.write(json.dumps(res).encode())
    real_out.flush()
    os._exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child()
    else:
        print("available:", available_text())
        print("jail:", jail_dir())
