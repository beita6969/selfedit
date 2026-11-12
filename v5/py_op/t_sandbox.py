"""Tests of the PY operator sandbox (plain checks: ordinary image code works, every class of refusal is refused, the OS limits hold).
usage: python -B t_sandbox.py            (Mac or the server; on the server as root the chroot / uid drop is checked too)"""
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sandbox
from PIL import Image


def png(im):
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def picture():
    """white 200 x 100 with a red 50 x 40 block at (50, 20) and a blue 10 x 10 block at (150, 70)"""
    im = Image.new("RGB", (200, 100), (255, 255, 255))
    for x in range(50, 100):
        for y in range(20, 60):
            im.putpixel((x, y), (255, 0, 0))
    for x in range(150, 160):
        for y in range(70, 80):
            im.putpixel((x, y), (0, 0, 255))
    return im


IMG = [png(picture())]
RGBA = Image.new("RGBA", (20, 10), (0, 0, 0, 0))


class Ordinary(unittest.TestCase):
    def ok(self, code, images=IMG, **k):
        r = sandbox.run(code, images, **k)
        self.assertEqual(r["status"], "ok", r)
        return r

    def test_size_crop_stats(self):
        r = self.ok("w, h = img.size\nprint(w, h)\nc = img.crop((50, 20, 100, 60))\na = np.asarray(c)\nprint(a.shape, a[..., 0].mean(), a[..., 1].mean())")
        self.assertEqual(r["stdout"].split("\n")[0], "200 100")
        self.assertIn("(40, 50, 3) 255.0 0.0", r["stdout"])

    def test_components(self):
        if not sandbox.HAVE_NDIMAGE:
            self.skipTest("no scipy.ndimage")
        r = self.ok("a = np.asarray(img).astype(int)\nm = a.sum(axis=2) < 700\nlab, n = scipy.ndimage.label(m)\nprint(n)\n"
                    "print(sorted(int(v) for v in ndimage.sum(m, lab, range(1, n + 1))))")
        self.assertEqual(r["stdout"].strip().split("\n"), ["2", "[100, 2000]"])

    def test_imports_allowed(self):
        code = ("import numpy as np\nimport math, itertools\nfrom collections import Counter\nfrom PIL import Image, ImageDraw, ImageFilter, ImageOps\n"
                "import PIL.Image\nimport numpy.linalg\nd = ImageDraw.Draw(img)\nd.line((0, 0, 10, 10), fill=(0, 255, 0), width=3)\n"
                "print(Counter(img.getdata()).most_common(1)[0][1], round(math.pi, 2), PIL.Image.new('L', (2, 2)).size)")
        if sandbox.HAVE_NDIMAGE:
            code += "\nfrom scipy import ndimage\nimport scipy.ndimage as ndi\nprint(ndi.label(np.ones((2, 2)))[1])"
        r = self.ok(code)
        self.assertIn("3.14 (2, 2)", r["stdout"])

    def test_show_and_echo(self):
        r = self.ok("show(img.crop((40, 10, 110, 70)))\nshow(np.asarray(img)[..., 0] > 200)\nimg.size")
        self.assertEqual(len(r["images"]), 2)
        self.assertEqual(Image.open(io.BytesIO(r["images"][0])).size, (70, 60))
        self.assertEqual(Image.open(io.BytesIO(r["images"][1])).size, (200, 100))
        self.assertEqual(r["stdout"].strip(), "(200, 100)")

    def test_show_cap_and_method(self):
        r = self.ok("for k in range(4):\n    img.show()\nprint('x')")
        self.assertEqual(len(r["images"]), sandbox.MAX_SHOW)
        self.assertIn("only the first 2 images", r["stdout"])

    def test_show_float_array(self):
        r = self.ok("show(np.linspace(0, 1, 64 * 64).reshape(64, 64))\nshow(np.arange(12).reshape(3, 4) * 100.0)")
        self.assertEqual(len(r["images"]), 2)

    def test_several_images_and_alpha(self):
        r = self.ok("print(len(imgs), [im.size for im in imgs], [im.mode for im in imgs], imgs[1].getpixel((0, 0)))", IMG + [png(RGBA)])
        self.assertEqual(r["stdout"].strip(), "2 [(200, 100), (20, 10)] ['RGB', 'RGB'] (255, 255, 255)")

    def test_defs_classes_fstrings(self):
        r = self.ok("class P:\n    def area(self, w, h):\n        return w * h\ndef f(x):\n    return [v * 2 for v in x]\n"
                    "print(f'{P().area(*img.size)} {f([1, 2])} {sorted({3, 1})} {max(range(4))}')")
        self.assertEqual(r["stdout"].strip(), "20000 [2, 4] [1, 3] 3")

    def test_common_idioms(self):
        r = self.ok("h, w, _ = np.asarray(img).shape\nfor _ in range(2):\n    pass\ndef main():\n    print(h, w)\nif __name__ == '__main__':\n    main()")
        self.assertEqual(r["stdout"].strip(), "100 200")

    def test_dead_branches_are_pruned_not_run(self):
        r = self.ok("im = Image.open('image.png') if False else None\nif False:\n    img.save('x.png')\nelse:\n    print(im, img.size)\nif 0:\n    np.save('a', 1)\nprint('end')")
        self.assertEqual(r["stdout"].strip(), "None (200, 100)\nend")
        tree = sandbox.check("x = Image.open('a') if False else 5")
        import ast as _ast
        self.assertNotIn("open", _ast.dump(tree))
        for c in ("x = img.save('a') if True else None", "x = None if False else img.save('a')", "if True:\n    img.save('x.png')",
                  "if False:\n    pass\nelse:\n    img.save('x.png')", "k = 0\nx = img.save('a') if k else None", "x = [np.load(p) for p in []]"):
            with self.assertRaises(sandbox.Refused, msg=c):
                sandbox.check(c)

    def test_image_open_is_a_shim(self):
        r = self.ok("from PIL import Image\nim = Image.open('image.png')\nprint(im.size, Image.open(imgs[0]) is imgs[0], Image.open('/etc/hostname').size)")
        self.assertEqual(r["stdout"].strip(), "(200, 100) True (200, 100)")
        r = sandbox.run("im = Image.open('a.png')", IMG + IMG)
        self.assertEqual(r["status"], "error")
        self.assertIn("already loaded", r["error"])
        r = self.ok("print(Image.open(imgs[1]).size, PIL.Image.open(1).size)", IMG + [png(RGBA)])
        self.assertEqual(r["stdout"].strip(), "(20, 10) (20, 10)")

    def test_unused_unavailable_imports_are_dropped(self):
        r = self.ok("import matplotlib.pyplot as plt  # habit\nimport matplotlib, os as unused_os\nfrom scipy.signal import find_peaks\nimport numpy as np\nprint(np.asarray(img).shape)")
        self.assertEqual(r["stdout"].strip(), "(100, 200, 3)")
        for c in ("import matplotlib.pyplot as plt\nplt.imshow(img)", "import os\nprint(os.getcwd())", "from scipy.signal import find_peaks\nfind_peaks([1])",
                  "from os import *", "import subprocess as sp\nsp.run(['ls'])"):
            with self.assertRaises(sandbox.Refused, msg=c):
                sandbox.check(c)

    def test_error_text(self):
        r = sandbox.run("x = 1\ny = x / 0\nprint('no')", IMG)
        self.assertEqual(r["status"], "error")
        self.assertIn("ZeroDivisionError", r["error"])
        self.assertIn("line 2", r["error"])
        r = sandbox.run("print(1)\nprint(undefined_thing)", IMG)
        self.assertEqual(r["status"], "error")
        self.assertIn("NameError", r["error"])
        self.assertEqual(r["stdout"].strip(), "1")
        self.assertLess(len(r["error"]), 1300)

    def test_notebook_state(self):
        a = "keep_me = 5\nprint('first')\nshow(img)\ndef f(v):\n    return v + 1"
        b = "keep_me = f(keep_me)\nprint('second')\nz = 1 / 0\nlost = 1"
        self.ok(a)
        r = sandbox.run("print(keep_me)", IMG)                       # without the earlier block: unknown name
        self.assertEqual(r["status"], "error")
        r = self.ok("print(keep_me, f(1))", prelude=[a])               # earlier block re-run silently: no 'first', no image
        self.assertEqual((r["stdout"].strip(), len(r["images"])), ("5 2", 0))
        r = self.ok("print(keep_me, 'lost' in dir())", prelude=[a, b])  # a block that failed midway keeps what it did before the error
        self.assertEqual(r["stdout"].strip(), "6 False")
        r = self.ok("print(keep_me)", prelude=[a, "import os\nkeep_me = 99\nprint(os.getcwd())"])   # an earlier block that does not pass the check is skipped
        self.assertEqual(r["stdout"].strip(), "5")

    def test_fresh_state_and_no_images(self):
        self.ok("keep_me = 5")
        r = sandbox.run("print(keep_me)", IMG)
        self.assertEqual(r["status"], "error")
        r = self.ok("print(imgs, img)", [])
        self.assertEqual(r["stdout"].strip(), "[] None")


class Refusals(unittest.TestCase):
    def refused(self, code):
        with self.assertRaises(sandbox.Refused, msg=code):
            sandbox.check(code)
        r = sandbox.run(code, IMG)
        self.assertEqual(r["status"], "refused", code)
        self.assertTrue(r["error"])
        self.assertEqual(r["stdout"], "")

    def test_imports(self):
        for m in ("os", "sys", "subprocess", "shutil", "pathlib", "socket", "ctypes", "importlib", "builtins", "io", "pickle", "matplotlib",
                  "matplotlib.pyplot", "scipy.io", "scipy.signal", "PIL.ImageFont", "PIL.ImageGrab", "numpy.ctypeslib", "numpy.lib.npyio",
                  "time", "requests", "tempfile", "glob", "random", "skimage"):
            self.refused(f"import {m}\nprint({m})")
            self.refused(f"import {m} as z\nprint(z)")
            self.refused(f"from {m} import x\nprint(x)")
        self.refused("from numpy import *")
        self.refused("from . import x\nprint(x)")
        self.refused("from PIL import ImageFont")
        self.refused("from PIL import ImageShow")
        self.refused("from numpy import load")
        self.refused("from numpy import save as s")
        self.refused("from numpy import memmap")
        self.refused("from numpy import ctypeslib")
        if sandbox.HAVE_NDIMAGE:
            self.refused("from scipy import io")
            self.refused("from scipy import signal")
        if not sandbox.HAVE_CV2:
            self.refused("import cv2\nprint(cv2)")

    def test_builtins(self):
        for n in ("open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr", "globals", "locals", "vars", "input",
                  "breakpoint"):
            self.refused(f"{n}('x')")
            self.refused(f"f = {n}")
            self.refused(f"print([{n}])")

    def test_underscore(self):
        for c in ("x = img.__class__", "_a = 1", "print(__builtins__)", "print(__file__)", "x = __doc__", "x = ().__class__.__base__", "def _f():\n    pass",
                  "class _C:\n    pass", "f = lambda _x: 1", "x = img._size", "import numpy as _np", "print(np.__dict__)",
"x = _private_name", "print(f'{img.__class__}')", "x = np.__version__", "try:\n    pass\nexcept Exception as _e:\n    pass"):
            self.refused(c)

    def test_attribute_names(self):
        for a in ("os", "sys", "subprocess", "shutil", "pathlib", "socket", "ctypes", "importlib", "builtins", "system", "popen", "remove",
                  "unlink", "rmdir", "rmtree", "rename", "save", "imwrite", "tofile", "savez", "dump", "load", "memmap",
                  "savez_compressed", "savetxt", "loadtxt", "genfromtxt", "fromfile", "dumps", "fdopen", "write", "ctypeslib", "testing", "codecs",
                  "io", "pickle", "operator", "ImageFont", "ImageShow", "ImageGrab", "truetype", "eval", "exec", "core", "fp", "Path"):
            self.refused(f"np.{a}")
            self.refused(f"x = img.{a}('a')")
            self.refused(f"y = np.lib.{a}.z")

    def test_typical_file_calls(self):
        for c in ("img.save('out.png')", "np.save('a.npy', np.zeros(3))", "np.asarray(img).tofile('a.bin')", "a = np.load('a.npy')",
                  "np.memmap('a', mode='w+', shape=(3,))", "np.savetxt('a.txt', np.zeros(3))",
                  "x = np.os.listdir('.')", "Image.os.remove('x')", "ImageOps.operator.attrgetter('a')", "np.asarray(img).dump('a')",
                  "open('a.txt', 'w').write('x')"):
            self.refused(c)

    def test_syntax_and_size(self):
        self.refused("def f(:\n  pass")
        self.refused("")
        self.refused("x = 1\n" * 30000)
        self.refused("async def f():\n    await g()")


class OsLayer(unittest.TestCase):
    """The limits of the child hold even with the static check and the restricted builtins switched off (_unrestricted: tests only)."""

    def test_no_file_can_be_written(self):
        jail = sandbox.jail_dir()
        for code in ("f = open('t_sandbox_probe.txt', 'w')\nf.write('x')\nf.close()\nprint('WROTE')",
                     "import os\nfd = os.open('t_sandbox_probe.txt', os.O_CREAT | os.O_WRONLY)\nos.write(fd, b'x')\nprint('WROTE')",
                     "img.save('t_sandbox_probe.png')\nprint('WROTE')"):
            r = sandbox.run(code, IMG, _unrestricted=True)
            self.assertNotIn("WROTE", r["stdout"], r)
            self.assertEqual(r["status"], "error", r)
            self.assertEqual(os.listdir(jail), [])
        self.assertFalse(os.access(jail, os.W_OK) and os.geteuid() != 0)

    def test_no_file_can_be_read_or_descriptor_opened(self):
        r = sandbox.run("f = open('/etc/hostname')\nprint('OPENED')", IMG, _unrestricted=True)
        self.assertNotIn("OPENED", r["stdout"], r)
        r = sandbox.run("import socket\ns = socket.socket()\nprint('OPENED')", IMG, _unrestricted=True)
        self.assertNotIn("OPENED", r["stdout"], r)

    def test_no_new_process(self):
        r = sandbox.run("import os\npid = os.fork()\nif pid == 0:\n    os._exit(0)\nprint('FORKED')", IMG, _unrestricted=True)
        self.assertNotIn("FORKED", r["stdout"], r)

    def test_lock_report(self):
        r = sandbox.run("print(1)", IMG)
        lock = r["lock"]
        for k in ("RLIMIT_FSIZE", "RLIMIT_NOFILE", "RLIMIT_NPROC", "RLIMIT_CPU"):
            self.assertIs(lock[k], True, lock)
        if sys.platform.startswith("linux"):
            self.assertIs(lock["RLIMIT_AS"], True, lock)
            if os.geteuid() == 0:
                self.assertIs(lock["chroot"], True, lock)
                self.assertEqual(lock["uid"], 65534, lock)

    def test_root_sees_empty_root(self):
        if not (sys.platform.startswith("linux") and os.geteuid() == 0):
            self.skipTest("needs root on Linux")
        r = sandbox.run("import os\nprint(os.getuid(), os.getgid(), os.getcwd(), [os.path.exists(p) for p in ('/etc', '/data', '/workspace', '/proc', '/tmp', '/usr')])",
                        IMG, _unrestricted=True)
        self.assertEqual(r["stdout"].strip(), "65534 65534 / [False, False, False, False, False, False]", r)

    def test_cpu_loop_is_stopped(self):
        r = sandbox.run("k = 0\nwhile True:\n    k += 1", IMG, cpu_s=2)
        self.assertEqual(r["status"], "timeout", r)
        self.assertIn("time limit", r["error"])
        self.assertLess(r["wall"], 12)

    def test_cpu_loop_that_swallows_is_killed(self):
        r = sandbox.run("while True:\n    try:\n        while True:\n            pass\n    except BaseException:\n        pass", IMG, cpu_s=1, wall_s=6)
        self.assertIn(r["status"], ("killed", "timeout"), r)       # Linux: the hard CPU limit kills; elsewhere the parent's wall clock does
        self.assertLess(r["wall"], 10)

    def test_wall_clock_kill(self):
        r = sandbox.run("k = 0\nwhile True:\n    k += 1", IMG, cpu_s=60, wall_s=3)
        self.assertEqual(r["status"], "timeout", r)
        self.assertIn("wall clock", r["error"])
        self.assertLess(r["wall"], 8)

    def test_output_cap(self):
        r = sandbox.run("for k in range(100000):\n    print('abcdefghij' * 5)", IMG)
        self.assertEqual(r["status"], "ok", r)
        self.assertLessEqual(len(r["stdout"]), sandbox.OUT_CAP + 80)
        self.assertIn("output cut", r["stdout"])
        r = sandbox.run("print(np.arange(10 ** 6))\nnp.ones((300, 300))", IMG)
        self.assertLessEqual(len(r["stdout"]), sandbox.OUT_CAP + 80)

    def test_memory_limit(self):
        if not sys.platform.startswith("linux"):
            self.skipTest("RLIMIT_AS is Linux only")
        r = sandbox.run("a = np.ones((1 << 33,), dtype=np.float64)\nprint('BIG', a[-1])", IMG)
        self.assertNotIn("BIG", r["stdout"], r)
        self.assertIn(r["status"], ("error", "killed"), r)

    def test_environment_is_empty(self):
        r = sandbox.run("import os\nprint(sorted(os.environ))", IMG, _unrestricted=True)
        self.assertEqual(r["status"], "ok", r)
        extra = set(eval(r["stdout"].strip())) - set(sandbox.CHILD_ENV)
        # names the interpreter itself adds inside the child (locale coercion, macOS text encoding, cv2's loader): nothing is inherited
        self.assertLessEqual(extra, {"LC_CTYPE", "__CF_USER_TEXT_ENCODING", "LD_LIBRARY_PATH"})


if __name__ == "__main__":
    print("platform", sys.platform, "euid", os.geteuid(), "| available:", sandbox.available_text(), "| jail:", sandbox.jail_dir())
    unittest.main(verbosity=1)
