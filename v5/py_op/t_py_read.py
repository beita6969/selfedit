"""End-to-end test of py_read.py against a scripted fake chat server (no model): turn handling, the sandbox round trip, the round limit,
refusals, role guard, records and token sums.   usage: python -B t_py_read.py [SCRATCH_DIR]"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SEEN = []


def text_of(content):
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content if p.get("type") == "text")


def n_images(content):
    return 0 if isinstance(content, str) else sum(p.get("type") == "image_url" for p in content)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body)
        msgs = body["messages"]
        q = text_of(msgs[1]["content"])
        last = text_of(msgs[-1]["content"])
        turns = sum(m["role"] == "assistant" for m in msgs)
        reasoning, finish = f"thinking {turns}", "stop"
        if "DIRECT" in q:
            content = "It is A. \\boxed{A}"
        elif "LOOP" in q:
            content = "again\n```python\nprint(len(imgs))\n```"
        elif "LATE" in q:
            content = "It is B: \\boxed{B}" if "no runs left. The last block was not run" in last else "more\n```python\nkeep = 1 if turns == 0 else keep + 1\nprint(keep)\n```".replace("turns == 0", "'keep' not in dir()")
        elif "BAD" in q:
            content = "```python\nimport os\nprint(os.listdir('.'))\n```" if turns == 0 else ("\\boxed{B}" if "was not run" in last and "`import os` is not allowed" in last else "\\boxed{A}")
        elif "CUT" in q:
            content, finish = "", "length"
        elif "CODE" in q:
            if turns == 0:
                content = "Let me measure.\n```python\nprint(img.size)\nshow(img.crop((0, 0, 50, 50)))\n```\n"
            elif turns == 1:
                good = "(200, 100)" in last and "[python run 1 of 2]" in last and n_images(msgs[-1]["content"]) == 1 and msgs[-2].get("reasoning") == "thinking 0"
                content = "```python\nprint(1 / 0)\n```" if good else "\\boxed{A}"
            else:
                good = "ZeroDivisionError" in last and "[no runs left]" in last and "[python run 2 of 2]" in last
                content = "Final: \\boxed{B}" if good else "\\boxed{A}"
        else:
            content = "\\boxed{A}"
        out = dict(choices=[dict(message=dict(content=content, reasoning_content=reasoning), finish_reason=finish)],
                   usage=dict(prompt_tokens=1000 + 100 * turns, completion_tokens=50))
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main():
    from PIL import Image
    scratch = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
    os.makedirs(scratch, exist_ok=True)
    img = os.path.join(scratch, "t_img.png")
    Image.new("RGB", (200, 100), (255, 255, 255)).save(img)
    rows = []
    for k, (tag, role) in enumerate([("DIRECT", "show"), ("CODE", "show"), ("LOOP", "monitor"), ("BAD", "monitor"), ("CUT", "show"), ("DIRECT", "judge"), ("LATE", "show")]):
        rows.append(dict(id=f"t/{tag}{k}", bench="t", kind="mc", prompt=f"{tag} question?\n(A) a\n(B) b\nPut the letter in \\boxed{{}}.", image_paths=[img],
                         answer="B" if tag in ("CODE", "BAD", "LATE") else "A", options=["a", "b"], category="c", pic=f"p{k}", v4_role=role))
    rf = os.path.join(scratch, "t_rows.jsonl")
    open(rf, "w").write("".join(json.dumps(r) + "\n" for r in rows))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    out = os.path.join(scratch, "t_out")
    env = dict(os.environ, PY_OP_JAIL=os.environ.get("PY_OP_JAIL", ""))
    if not env["PY_OP_JAIL"]:
        env.pop("PY_OP_JAIL")
    name = f"t{os.getpid()}"
    p = subprocess.run([sys.executable, "-B", os.path.join(HERE, "py_read.py"), "--name", name, "--rows", rf, "--no-sha", "--servers", f"127.0.0.1:{port}",
                        "--out", out, "--samples", "0,1", "--per-server", "3"], capture_output=True, text=True, env=env)
    print(p.stdout[-1500:], p.stderr[-3000:])
    assert p.returncode == 0
    R = {(x["row"], x["sample"]): x for x in map(json.loads, open(os.path.join(out, name, "reads.jsonl")))}
    C = {(x["row"], x["sample"]): x for x in map(json.loads, open(os.path.join(out, name, "compact.jsonl")))}
    assert len(R) == 12 and len(C) == 12, len(R)
    assert not any(k[0].startswith("t/DIRECT5") for k in R), "judge row was read"
    for s in (0, 1):
        d = R[("t/DIRECT0", s)]
        assert d["ok"] and d["n_calls"] == 1 and d["n_runs"] == 0 and d["tokens"] == 50 and d["prompt_tokens"] == 1000, d
        c = R[("t/CODE1", s)]
        assert c["ok"] and c["pred"] == "B" and c["n_calls"] == 3 and c["n_runs"] == 2, (c["ok"], c["pred"], c["n_calls"], c["runs"])
        assert c["runs"][0]["status"] == "ok" and c["runs"][0]["n_sent"] == 1 and c["runs"][0]["image_tokens"] == [4096], c["runs"][0]
        assert c["runs"][1]["status"] == "error" and "ZeroDivisionError" in c["runs"][1]["error"]
        assert c["tokens"] == 150 and c["prompt_tokens"] == 3300, (c["tokens"], c["prompt_tokens"])
        l = R[("t/LOOP2", s)]      # 2 runs, then one "no runs left" turn, then the reply is graded as it stands
        assert not l["ok"] and l["n_runs"] == 2 and l["n_calls"] == 4 and l["flags"].get("code_after_limit") == 2, (l["n_calls"], l["flags"])
        b = R[("t/BAD3", s)]
        assert b["ok"] and b["runs"][0]["status"] == "refused" and b["n_calls"] == 2, b["runs"]
        la = R[("t/LATE6", s)]     # code after the limit -> one more turn -> the answer; state kept between runs (keep = 1, then 2)
        assert la["ok"] and la["n_runs"] == 2 and la["n_calls"] == 4 and la["flags"].get("code_after_limit") == 1, (la["ok"], la["n_calls"], la["flags"])
        assert [r["stdout"].strip() for r in la["runs"]] == ["1", "2"], la["runs"]
        u = R[("t/CUT4", s)]
        assert not u["ok"] and u["finish"] == "length" and u["n_runs"] == 0
    b0 = next(b for b in SEEN if "CODE" in text_of(b["messages"][1]["content"]) and len(b["messages"]) == 2)
    import py_read
    assert b0["messages"][0] == {"role": "system", "content": py_read.NOTE}
    assert b0["model"] == "stock" and b0["max_tokens"] == 32768 and b0["chat_template_kwargs"] == {"reasoning_effort": "medium", "enable_thinking": True}
    assert b0["temperature"] == 1.0 and b0["top_p"] == 0.95 and b0["top_k"] == 20 and b0["seed"] in (py_read.crn_seed("t/CODE1", 0), py_read.crn_seed("t/CODE1", 1))
    assert n_images(b0["messages"][1]["content"]) == 1 and b0["messages"][1]["content"][-1]["text"].startswith("CODE question?")
    notes = {b["messages"][0]["content"] for b in SEEN}
    assert len(notes) == 1, "the note must be identical on every request"
    # resume: nothing left to do
    p2 = subprocess.run([sys.executable, "-B", os.path.join(HERE, "py_read.py"), "--name", name, "--rows", rf, "--no-sha", "--servers", f"127.0.0.1:{port}",
                         "--out", out, "--samples", "0,1"], capture_output=True, text=True, env=env)
    assert "todo 0" in p2.stdout, p2.stdout + p2.stderr
    # final_block unit checks
    fb = py_read.final_block
    assert fb("x\n```python\nprint(1)\n```")[::2] == ("print(1)\n", "fenced")
    assert fb("x\n```python\nprint(1)\n```\nI will run this.")[0] == "print(1)\n"
    assert fb("```python\nprint(1)\n```\nSo the answer is \\boxed{3}") is None
    assert fb("```\nplain\n```") is None and fb("no code \\boxed{A}") is None
    assert fb("```python\nprint(1)\n```\nthen\n```py\nprint(2)")[0] == "print(2)\n"
    assert fb("```Python \nprint(1)\n```") is not None
    # the tool-call forms seen in the smoke
    assert fb("<tool_call>\n<python>\nfrom PIL import Image\nprint(img.size)\n</python>\n</parameter>\n\n<tool_call>\n<python>\nw, h = img.size\nshow(img)\n</parameter>\n</function>\n</tool_call>") == ("w, h = img.size\nshow(img)\n", "", "tool_call")
    assert fb("<tool_call>\n<function from __future__ import print_function\nimport numpy as np\nprint(1)\n</function>\n</tool_call>")[0] == "import numpy as np\nprint(1)\n"
    assert fb("<tool_call>\n<function/python>\nprint(2)\n</function>\n</tool_call>")[0] == "print(2)\n"
    assert fb("<tool_call>\npython\nimport numpy as np\nprint(3)\n\n</parameter>")[0] == "import numpy as np\nprint(3)\n"
    assert fb('<tool_call>\n{"name": "python", "arguments": {"code": "print(4)"}}\n</tool_call>')[0] == "print(4)\n"
    assert fb("<tool_call>\n<function=python>\n<parameter=code>\n```python\nprint(5)\n```\n</parameter>\n</function>\n</tool_call>")[0] == "print(5)\n"
    assert fb("<tool_call>\n<python>\nprint(1)\n</python>\n</tool_call>\nThe answer is \\boxed{B}") is None
    # empty stubs after a complete block (seen in full1) do not hide the block
    assert fb("```python\nprint(7)\n```\n\n<tool_call>\n</function>")[0] == "print(7)\n"
    assert fb("text\n<tool_call>\n\n<tool_call>") is None
    assert fb("```python\nprint(8)\n```\n\n```python\n\n```")[0] == "print(8)\n"
    print("t_py_read: all ok")
    print("NOTE:\n" + py_read.NOTE)


if __name__ == "__main__":
    main()
