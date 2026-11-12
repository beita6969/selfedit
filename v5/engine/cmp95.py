import json,sys,collections
t0=float(sys.argv[1]); t1=float(sys.argv[2])
g=collections.defaultdict(list); pre={}; run=collections.defaultdict(list)
for l in open('/data/fd/mingdazhang/selfedit/v4/stats.jsonl'):
    try: d=json.loads(l)
    except: continue
    if not (t0<=d['t']<=t1) or d.get('state')!='up' or 'gen_tok_s' not in d: continue
    c=d['card']; g[c].append(d['gen_tok_s']); run[c].append(d['running'] or 0)
    p=d.get('preemptions') or 0; pre.setdefault(c,[p,p]); pre[c][1]=p
for c in sorted(g):
    print(c, 'n=%d gen_tok_s=%.0f running=%.1f preempt_delta=%d'%(len(g[c]),sum(g[c])/len(g[c]),sum(run[c])/len(run[c]),pre[c][1]-pre[c][0]))
