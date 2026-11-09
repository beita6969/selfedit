import sys; import os; _r=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0]=[_r, _r+'/rsi']
from rsi import hboost as HB
pool=[dict(cat="a|x",note="Count the bars; the answer is (B) when two match. Paris is the label.",name="p1",round=1),
      dict(cat="b|y",note="Short note.",name="p2",round=1),
      dict(cat="c|z",note="Look at the left object first then compare its size to the right one carefully.",name="p3",round=1)]
g={"a|x":{"Paris","B"}}
print(HB.strip_gold(pool[0]["note"],g["a|x"]))
own="Check the left object and compare its height to the right object before you answer it."
n,src=HB.control_note(own,"c|z",pool,g,"FALLBACK","t1"); print(n,src); assert src["cat"]!="c|z"
n2,_=HB.control_note(own,"c|z",pool,g,"FALLBACK","t1"); assert n2==n
assert HB.control_note(own,"a|x",[pool[0]],g,"FB","t")[0]=="FB"
s=HB.strip_gold(pool[0]["note"],g["a|x"]); assert "Paris" not in s and "(B)" not in s
print("ALL GREEN")
