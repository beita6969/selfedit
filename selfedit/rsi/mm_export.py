"""Write engine edits to a new checkpoint on disk, unchanged shards linked. The checkpoint runs on stock vLLM.

Usage: python -u rsi/mm_export.py --edits JSON --out DST [loop.py args, with --mm]
"""
import json, os, sys


def pop(argv, flag):
    i = argv.index(flag)
    val = argv[i + 1]
    del argv[i:i + 2]
    return val


def main():                                                     # vLLM TP>1 workers re-import this file: nothing at top level
    argv = sys.argv[1:]
    edits = json.load(open(pop(argv, "--edits")))
    out = pop(argv, "--out")
    sys.argv = ["loop.py", "--run-dir", os.path.dirname(out)] + argv
    CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
    import rsi.loop as L
    from surgery import Surgeon

    a = L.args
    S = Surgeon(a.model, tp=a.tp, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens,
                gpu_memory_utilization=a.gpu_util, kv_cache_dtype=a.kv_cache_dtype, multimodal=True)
    S.export(edits, out, link="sym")
    L.say(f"mm_export: wrote {out}")


if __name__ == "__main__":
    main()
