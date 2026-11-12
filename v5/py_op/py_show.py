"""Print PY traces exported by py_export.py as plain text.   usage: py_show.py TRACES.jsonl.gz [--klass K] [--row SUBSTR] [--short]"""
import gzip
import json
import sys


def arg(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def main():
    kl, rw, short = arg("--klass"), arg("--row"), "--short" in sys.argv
    for l in gzip.open(sys.argv[1], "rt"):
        x = json.loads(l)
        if x.get("meta"):
            print("META", x)
            continue
        if (kl and x["klass"] != kl) or (rw and rw not in x["row"]):
            continue
        print("=" * 150)
        print(f"{x['row']} s{x['sample']} [{x['klass']}] bench {x['bench']} / {x.get('category')} | gold {x['answer']!r} | PY pred {x.get('pred')!r} ok {x['ok']} | "
              f"base8 {x['base8']:.2f} preds {x['base_preds']} | gen {x['tokens']} prompt {x['prompt_tokens']} | images {x['n_images']} {x.get('image_tokens')} | flags {x.get('flags')}")
        print("Q:", x["prompt"][-700:] if short else x["prompt"])
        for k, c in enumerate(x["calls"]):
            print(f"--- call {k + 1}: gen {c['tokens']} prompt {c['prompt_tokens']} finish {c['finish']} reasoning chars {c['reasoning_chars']}")
            if not short:
                print("  [reasoning tail]", c["reasoning_tail"])
            print("  [reply]", c["content"])
            if k < len(x["runs"]):
                u = x["runs"][k]
                print(f"  >>> run {k + 1}: {u['status']} cpu {u.get('cpu')} wall {u.get('wall')} images {u['n_images']} sent {u['n_sent']} {u.get('image_tokens')} {u.get('lock') or ''}")
                if u["error"]:
                    print("  >>> error:", u["error"])
                print("  >>> stdout:", u["stdout"])


if __name__ == "__main__":
    main()
