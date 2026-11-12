# route key, second fit: states on further unrouted rows
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
FI_MAX_REASON=1500 $R kcap.py --rows key2_unr_items.txt --out kcap_um > $L/key2_cap_u.log 2>&1
