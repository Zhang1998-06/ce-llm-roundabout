"""Final tables (success / safety / efficiency) on the test seeds, references restricted to the seeds each
(V)LM run covered.  Uses the finished episode files if present, else the *_partial.csv files.
    python scripts/make_final_tables.py   -> results/hazard_test/final_tables.md (+ .csv)"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "hazard_test")
ORDER = ["nominal", "stall_entry", "stall_mid", "stall_exit", "slow_entry", "slow_mid", "slow_exit"]
NAMES = {"dt": "DT (baseline)", "rule": "Stationary-vehicle rule (ref.)", "skill": "Oracle trigger + executor (ref.)",
         "llm": "DT + LLM-CoC (Qwen3-4B, text)", "vlm": "DT + VLM-CoC (Qwen3-VL-2B, text+BEV)"}


def load(m):
    f = os.path.join(D, f"episodes_{m}.csv")
    if os.path.exists(f):
        return pd.read_csv(f), "complete"
    f = os.path.join(D, f"episodes_{m}_partial.csv")
    return (pd.read_csv(f), "partial") if os.path.exists(f) else (None, "missing")


ref = pd.read_csv(os.path.join(D, "episodes_dt_rule_skill.csv"))
out, rows = [], []
for m in ("llm", "vlm"):
    df, status = load(m)
    if df is None:
        continue
    out.append(f"\n### {NAMES[m]} — {status} run, {len(df)} episodes\n")
    out.append("| Scenario | n | Method | Success % | Collision % | Timeout % | Min dist (m) | Avg speed (m/s) | Completion time (s) | (V)LM calls/ep | Latency (s/call) |")
    out.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sc in ORDER:
        g = df[df.scenario == sc]
        if g.empty:
            continue
        seeds = set(g.seed)
        for meth, x in [(k, ref[(ref.scenario == sc) & (ref.method == k) & ref.seed.isin(seeds)]) for k in ("dt", "rule", "skill")] + [(m, g)]:
            r = dict(run=m, scenario=sc, n=len(x), method=meth, success=100 * x.success.mean(),
                     collision=100 * x.collision.mean(), timeout=100 * x.timeout.mean(), min_dist=x.min_dist.mean(),
                     speed=x.avg_speed.mean(), time=x.completion_time.mean(),
                     calls=x.n_calls.mean(), latency=x.mean_latency.mean())
            rows.append(r)
            b = "**" if meth == m else ""
            t = "–" if pd.isna(r["time"]) else f"{r['time']:.2f}"
            c = "–" if meth != m else f"{r['calls']:.1f}"
            lat = "–" if meth != m or pd.isna(r["latency"]) else f"{r['latency']:.2f}"
            out.append(f"| {sc} | {r['n']} | {b}{NAMES[meth]}{b} | {b}{r['success']:.1f}{b} | {r['collision']:.1f} | {r['timeout']:.1f} | "
                       f"{r['min_dist']:.2f} | {r['speed']:.2f} | {t} | {c} | {lat} |")
pd.DataFrame(rows).round(3).to_csv(os.path.join(D, "final_tables.csv"), index=False)
open(os.path.join(D, "final_tables.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
