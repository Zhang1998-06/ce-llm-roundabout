"""Phase-4 tables: results/hazard_test_v5/final_tables.md (+ .csv)
    python scripts/make_final_tables_v5.py"""
import os
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
T = "results/hazard_test_v5"
NAMES = {"dt": "DT (baseline)", "acc": "DT + same longitudinal safety, no LLM (control)",
         "oracle": "Perfect recognition + same maneuvers (reference)", "llm": "**DT + LLM-CoC v5 (Qwen3-4B, text)**",
         "llm_v4": "DT + LLM-CoC v4 (previous version, executor v1)"}
ORDER = ["nominal", "stall_entry", "stall_mid", "stall_exit", "slow_entry", "slow_mid", "slow_exit"]


def table(e, methods):
    g = e.groupby(["scenario", "method"]).agg(n=("seed", "size"), success=("success", "mean"), collision=("collision", "mean"),
                                             at_fault=("collision_at_fault", "mean"), timeout=("timeout", "mean"),
                                             min_dist=("min_dist", "mean"), speed=("avg_speed", "mean"),
                                             time=("completion_time", "mean"), calls=("n_calls", "mean"), latency=("mean_latency", "mean"))
    for k in ("success", "collision", "at_fault", "timeout"):
        g[k] *= 100
    rows = []
    for sc in ORDER + ["hazard mean"]:
        for m in methods:
            if sc == "hazard mean":
                x = g.xs(m, level=1).drop("nominal")
                r = x.mean(numeric_only=True); r["n"] = int(x.n.sum())
            elif (sc, m) in g.index:
                r = g.loc[(sc, m)]
            else:
                continue
            rows.append(dict(scenario=sc, method=m, **r.to_dict()))
    return pd.DataFrame(rows)


def fmt(v, nd=1):
    return "–" if pd.isna(v) else f"{v:.{nd}f}"


e = pd.concat([pd.read_csv(f"{T}/episodes_dt_acc_oracle.csv"), pd.read_csv(f"{T}/episodes_llm.csv")])
old = pd.read_csv("results/hazard_test/episodes_llm.csv"); old["method"] = "llm_v4"; old["collision_at_fault"] = float("nan")
e = pd.concat([e, old])
D = table(e, ["dt", "acc", "oracle", "llm", "llm_v4"])
D.to_csv(f"{T}/final_tables.csv", index=False)
L = ["### Test seeds 500–549 (50 episodes per scenario and method), horizon 11 s", "",
     "| Scenario | Method | Success % | Collision % | At-fault collision % | Timeout % | Min dist (m) | Avg speed (m/s) | Completion time (s) | LLM calls/ep | Latency (s/call) |",
     "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for _, r in D.iterrows():
    L.append(f"| {r.scenario} | {NAMES[r.method]} | {fmt(r.success)} | {fmt(r.collision)} | {fmt(r.at_fault)} | {fmt(r.timeout)} | "
             f"{fmt(r.min_dist, 2)} | {fmt(r.speed, 2)} | {fmt(r.time, 2)} | {fmt(r.calls) if r.method.startswith('llm') else '–'} | "
             f"{fmt(r.latency, 2) if r.method.startswith('llm') else '–'} |")
L += ["", "At-fault: the ego is not at fault only when hit from behind while keeping its lane outside the merge zone "
      "(nuPlan-style); not computed for the v4 run. Completion time: successful episodes only.", ""]
h = pd.read_csv("results/hazard_test_v5_h20/episodes_dt_oracle.csv")
H = table(h, ["dt", "oracle"])
L += ["### Sensitivity: 20 s horizon (protocol is 11 s), DT and the perfect-recognition reference, same seeds", "",
      "| Scenario | Method | Success % | Collision % | At-fault % | Timeout % |", "|---|---|---:|---:|---:|---:|"]
for _, r in H.iterrows():
    L.append(f"| {r.scenario} | {NAMES[r.method]} | {fmt(r.success)} | {fmt(r.collision)} | {fmt(r.at_fault)} | {fmt(r.timeout)} |")
open(f"{T}/final_tables.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
