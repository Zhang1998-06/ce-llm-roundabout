"""Single-step decision accuracy of a (V)LM on the offline set.
    python scripts/offline_eval.py results/offline_dev.pkl <model> <version 1|2> <vision 0|1> [n_follow]"""
import os, pickle, random, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from collections import Counter
import coc_supervisor as C

items = pickle.load(open(sys.argv[1], "rb"))
model, ver, vis = sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
nf = int(sys.argv[5]) if len(sys.argv) > 5 else 100
random.seed(0)
hz = [i for i in items if i["label"] != "FOLLOW"]
fo = random.sample([i for i in items if i["label"] == "FOLLOW"], nf)
sup = C.CoCSupervisor("http://127.0.0.1:11434", model, vision=vis, version=ver)
OK = {"FOLLOW": {"FOLLOW_POLICY"}, "CHANGE_INNER": {"CHANGE_TO_INNER_LANE"}, "WAIT": {"SLOW_DOWN", "KEEP_SPEED"}, "BYPASS": {"BYPASS_OBSTACLE"}}
if ver == 4:
    for i in items: i["label"] = i["label4"]
res, lat, bad = Counter(), [], []
for it in hz + fo:
    r = sup.ask(it[f"v{ver}"], it["png"] if vis else None)
    lat.append(r["latency_s"])
    good = r["meta_action"] in OK[it["label"]]
    res[(it["label"], "ok" if good else "wrong")] += 1
    res[("invalid",)] += not r["valid"]
    if it["label"] == "FOLLOW" and not good:
        res[("fp", it["scenario"], "ring" if "ring" in it["v2"][:25] else ("exit" if "exit road" in it["v2"][:30] else "access"), r["meta_action"])] += 1
    if not good and len(bad) < 6:
        bad.append((it["label"], r["meta_action"], it[f"v{ver}"], r["causal_reasoning"]))
n = lambda l: res[(l, "ok")] + res[(l, "wrong")]
print(f"model={model} v{ver} vision={vis}")
for l in [l for l in OK if any(i["label"] == l for i in items)]:
    print(f"  {l:13s} {res[(l, 'ok')]}/{n(l)} correct")
fp = res[("FOLLOW", "wrong")] / max(1, n("FOLLOW"))
det = sum(res[(l, k)] for l in ("CHANGE_INNER", "WAIT") for k in ("ok", "wrong")) and \
      (n("CHANGE_INNER") + n("WAIT"))
print(f"  false-override rate {100 * fp:.1f}%  invalid {res[('invalid',)]}  mean latency {sum(lat) / len(lat):.2f}s")
for k, v in sorted((k, v) for k, v in res.items() if k[0] == "fp"):
    print("  false override", k[1:], v)
for b in bad[:0]:
    print("  WRONG", b[0], "->", b[1], "|", b[3]); print("   ", b[2].replace("\n", "\n    "))
