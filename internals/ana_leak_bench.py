"""ana_leak_bench (fable_int): what a cell does to reads it is not meant for, by benchmark (b_fn5 records on unrouted rows: reference = no cell). Per cell and benchmark:
KL(no cell || cell) per thinking position; change of log p(read token); and the STOP side: at paragraph ends (positions whose next token in the read is a line break) the change of
log p("\\n") - log p("\\n\\n"), i.e. of closing the reasoning against opening another paragraph (negative = the cell pushes the read to go on).
python ana_leak_bench.py EV_DIR [NLID NNID are stored per record as lp_nl / lp_nn]"""
import glob, os, sys, collections
import numpy as np
R = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz")))]
names = [n for n in R[0]["names"].tolist() if n not in ("m", "x")]; B = sorted({str(r["bench"]) for r in R}) + ["ALL"]
print(f"{len(R)} reads: " + ", ".join(f"{b} {sum(str(r['bench']) == b for r in R)}" for b in B[:-1]))
print("cell | benchmark: KL(no cell || cell) per position, change of log p(read token), stop-side change at paragraph ends (n positions)")
for n in names:
    line = []
    for b in B:
        S = [r for r in R if b == "ALL" or str(r["bench"]) == b]
        if not S:
            continue
        kl = np.mean([r[n + "/kl_mc"].mean() for r in S]); dl = np.mean([(r[n + "/lpy"] - r["m/lpy"]).mean() for r in S]); st = []
        for r in S:
            pe = np.zeros(len(r["y"]), bool); pe[:-1] = r["pstart"][1:]              # position predicting a paragraph break
            if pe.sum():
                st.append((((r[n + "/lp_nl"] - r[n + "/lp_nn"]) - (r["m/lp_nl"] - r["m/lp_nn"]))[pe]).mean())
        line.append(f"{b} {kl:.4f}, {dl:+.4f}, {np.mean(st) if st else float('nan'):+.3f}")
    print(f"{n} | " + " | ".join(line))
