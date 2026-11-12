# weight-form check of the route-keyed round-3 cell (new gate type): the spec file applied to the weights on one held-out read of the round-2 student, against the functional units
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
FI_TRACES=$T/loop_r3.json FI_MAX_REASON=3000 $R b_xb.py --mode proc --bf16 --rows proc_test_loop3_items.txt --limit 1 --out ev_w3 --specs "tE=cells_r3/edits/PR3a64tE_g002c26_+1.8.json" > $L/ev_w3.log 2>&1
. /workspace/fudan/mingdazhang/selfedit/venv/bin/activate; cd $W; python - > $W/tables_w3.txt 2>&1 <<'PY'
import numpy as np, glob, os
for f in sorted(glob.glob('ev_w3/*.npz')):
    w = np.load(f, allow_pickle=True); e = np.load(os.path.join('ev_r3_gate2_own', os.path.basename(f)), allow_pickle=True)
    kw = w['tE/kl_xc']; ke = e['tE_x1.8/kl_xc']; n = min(len(kw), len(ke))
    print(os.path.basename(f), 'route-keyed cell: R weight-form %+.3f | functional %+.3f | KL(m||c) %.4f vs %.4f | corr of per-position KL(x||c) %.4f' % (1 - kw.sum() / w['m/kl_xc'].sum(), 1 - ke.sum() / e['m/kl_xc'].sum(), w['tE/kl_mc'].mean(), e['tE_x1.8/kl_mc'].mean(), np.corrcoef(kw[:n], ke[:n])[0, 1]))
PY
