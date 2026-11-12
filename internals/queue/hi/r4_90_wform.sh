# weight-form check of the W4 candidate (benefit-gated round-4 cell): the spec file applied to the weights on one held-out vlmsbiased read of the round-3 student, against the functional units
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
grep '^vlmsbiased' $W/proc_test_loop4_items.txt | head -1 > $W/wform4_items.txt
FI_TRACES=$T/loop_r4.json FI_MAX_REASON=3000 $R b_xb.py --mode proc --bf16 --rows wform4_items.txt --limit 1 --out ev_w4 --specs "tE=cells_r4/edits/PR4a64K2BE_g002c26_+1.5.json" > $L/ev_w4.log 2>&1
. /workspace/fudan/mingdazhang/selfedit/venv/bin/activate; cd $W; python - > $W/tables_w4.txt 2>&1 <<'PY'
import numpy as np, glob, os
for f in sorted(glob.glob('ev_w4/*.npz')):
    w = np.load(f, allow_pickle=True); e = np.load(os.path.join('ev_r4_k2b_own', os.path.basename(f)), allow_pickle=True)
    kw = w['tE/kl_xc']; ke = e['tE_x1.5/kl_xc']; n = min(len(kw), len(ke))
    print(os.path.basename(f), 'benefit-gated round-4 cell: KL(text||no text) %.4f | R weight-form %+.3f | functional %+.3f | KL(m||c) %.4f vs %.4f | corr of per-position KL(x||c) %.4f' % (w['m/kl_xc'].mean(), 1 - kw.sum() / w['m/kl_xc'].sum(), 1 - ke.sum() / e['m/kl_xc'].sum(), w['tE/kl_mc'].mean(), e['tE_x1.5/kl_mc'].mean(), np.corrcoef(kw[:n], ke[:n])[0, 1]))
PY
