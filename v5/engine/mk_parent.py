"""Launch parent cache: stock, medium, 4 samples per learn row, as ordinary read batches p000_parent_<s>.jsonl (keys p000|parent|<row>|<s>), priority 2."""
import json, os
V4 = "/data/fd/mingdazhang/selfedit/v4"
rows = [json.loads(l) for l in open(f"{V4}/rows/learn.jsonl")]
os.makedirs(f"{V4}/q/in", exist_ok=True)
for s in range(4):
    name = f"{V4}/q/in/p000_parent_{s}.jsonl"
    if os.path.exists(name):
        print("exists", name); continue
    with open(name + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(dict(key=f"p000|parent|{r['id']}|{s}", mode="read", engine="E", priority=2, row=r["id"], sample=s,
                                    prefix=None, settings={"min_image_tokens": None, "max_image_tokens": None, "temperature": None})) + "\n")
    os.replace(name + ".tmp", name)
    print("wrote", name, len(rows))
