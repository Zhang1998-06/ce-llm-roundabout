"""Phase 5 (v6) tables: closed-loop outcomes, decision quality, paired comparisons.
    python scripts/make_tables_v6.py results/hazard_test_v6 [out.md]
Reads every episodes_*.csv / decisions_*.csv in the folder.
"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest

D = sys.argv[1] if len(sys.argv) > 1 else "results/hazard_test_v6"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, "tables_v6.md")
NEED = ["stall_mid", "stall_exit", "slow_exit"]                 # intervention needed
NONEED = ["stall_past_exit", "stall_upstream"]                  # same object, not on the ego's route
SCEN = ["nominal"] + NEED + NONEED
ORDER = ["dt", "naive", "llm5", "llm", "designed"]
NAME = {"dt": "DT", "naive": "DT + naive trigger rule + executor", "llm5": "DT + LLM v5 (labels + rule prompt, executor picks target)",
        "llm": "DT + LLM v6 (facts only, model picks target)", "designed": "DT + designed rule + executor (answer key)"}

E = pd.concat([pd.read_csv(f) for f in glob.glob(f"{D}/episodes_*.csv") if "partial" not in f])
K = pd.concat([pd.read_csv(f) for f in glob.glob(f"{D}/decisions_*.csv") if os.path.getsize(f) > 2])
E = E.drop_duplicates(["method", "scenario", "seed"], keep="last")
methods = [m for m in ORDER if m in set(E.method)]
L = [f"# v6 results ({D})", ""]


def pct(x):
    return f"{100 * x:.0f}"


# 1. closed loop
L += ["## 1. Closed loop: success / collision / at-fault collision / timeout (%)", "",
      "| Method | " + " | ".join(SCEN) + " | mean (need) | mean (no need) |", "|---|" + "---|" * (len(SCEN) + 2)]
for m in methods:
    row, need, non = [], [], []
    for s in SCEN:
        g = E[(E.method == m) & (E.scenario == s)]
        if g.empty:
            row.append("–"); continue
        v = [g.success.mean(), g.collision.mean(), g.collision_at_fault.mean(), g.timeout.mean()]
        row.append(" / ".join(pct(x) for x in v))
        (need if s in NEED else non if s in NONEED else []).append(v)
    agg = lambda a: " / ".join(pct(x) for x in np.mean(a, 0)) if a else "–"
    L.append(f"| {NAME[m]} | " + " | ".join(row) + f" | {agg(need)} | {agg(non)} |")
n = E.groupby(["method", "scenario"]).size()
L += ["", f"Episodes per cell: {n.min()}–{n.max()}. Seeds identical across methods (paired).", ""]

# 2. efficiency + interventions
L += ["## 2. Efficiency and interventions (mean per episode)", "",
      "| Method | scenario | avg speed m/s | completion time s (successes) | maneuver starts | supervisor calls | latency s |",
      "|---|---|---:|---:|---:|---:|---:|"]
for m in methods:
    for s in SCEN:
        g = E[(E.method == m) & (E.scenario == s)]
        if g.empty:
            continue
        L.append(f"| {m} | {s} | {g.avg_speed.mean():.1f} | {g.completion_time.mean():.2f} | {g.n_starts.mean():.2f} | "
                 f"{g.n_calls.mean():.1f} | {g.mean_latency.mean():.2f} |")
L.append("")

# 3. decision quality (per call against the privileged reference; reference = intervene on the injected vehicle only
#    while it is on the ego's route ahead before the exit)
L += ["## 3. Decision quality per supervisor call (privileged reference, never shown to any supervisor)", "",
      "| Method | scenario | calls | ref says intervene | correct decision + target when ref intervenes | "
      "intervenes when ref says no | of which on the injected vehicle | invalid target refused |", "|---|---|---:|---:|---:|---:|---:|---:|"]
for m in [x for x in methods if x != "dt"]:
    for s in SCEN:
        k = K[(K.method == m) & (K.scenario == s)]
        if k.empty:
            continue
        pos = k[k.ref_decision != "FOLLOW_POLICY"]
        neg = k[k.ref_decision == "FOLLOW_POLICY"]
        ok = ((pos.decision == pos.ref_decision) & pos.target_is_hazard.astype(bool)).mean() if len(pos) else np.nan
        fp = (neg.decision != "FOLLOW_POLICY")
        fp_h = (fp & neg.target_is_hazard.astype(bool)).sum()
        rej = (k.status == "rejected_invalid_target").sum() if "status" in k else 0
        L.append(f"| {m} | {s} | {len(k)} | {len(pos)} | {'–' if np.isnan(ok) else pct(ok)} % | "
                 f"{pct(fp.mean()) if len(neg) else '–'} % ({fp.sum()}) | {fp_h} | {rej} |")
L.append("")

L += ["Note: llm5 calls in stall_mid count as 'intervenes when ref says no' because v5's path perception reaches ~70 m along the",
      "route at t = 0, while the v6 facts (and so the reference) stop at 50 m: these are early detections of the injected car,",
      "not judgment errors. In stall_past_exit / stall_upstream every intervention is unnecessary.", ""]

# 4. episode-level: did the first maneuver on the injected vehicle use the right maneuver?
L += ["## 4. Episode level: first maneuver started on the injected vehicle", "",
      "| Method | scenario | episodes with a maneuver on it | BYPASS | FOLLOW_VEHICLE | episodes with any maneuver |", "|---|---|---:|---:|---:|---:|"]
for m in [x for x in methods if x != "dt"]:
    for s in SCEN[1:]:
        k = K[(K.method == m) & (K.scenario == s)]
        if k.empty:
            continue
        st = k[(k.decision != "FOLLOW_POLICY") & ((k.status == "executed") if "status" in k and m != "llm5" else k.skill_started.astype(bool) if "skill_started" in k else True)]
        on = st[st.target_is_hazard.astype(bool)].sort_values("t").groupby("seed").first()
        neps = E[(E.method == m) & (E.scenario == s)].shape[0]
        L.append(f"| {m} | {s} | {len(on)}/{neps} | {(on.decision == 'BYPASS').sum()} | {(on.decision == 'FOLLOW_VEHICLE').sum()} | "
                 f"{st.seed.nunique()}/{neps} |")
L.append("")

# 5. paired comparisons (same seeds): success and collision, exact McNemar (binomial on discordant pairs)
L += ["## 5. Paired comparisons (same scenario and seed), exact McNemar test", "",
      "| Comparison | scenarios | outcome | A better | B better | p |", "|---|---|---|---:|---:|---:|"]
P = E.set_index(["method", "scenario", "seed"])
for a, b in [("llm", "dt"), ("llm", "naive"), ("llm", "llm5"), ("llm", "designed")]:
    if a not in methods or b not in methods:
        continue
    for grp, ss in [("need", NEED), ("no need", NONEED), ("nominal", ["nominal"])]:
        for col, good in [("success", 1), ("collision", 0)]:
            A = P.loc[a].loc[ss][col]; B = P.loc[b].loc[ss][col]
            j = A.index.intersection(B.index)
            A, B = A[j], B[j]
            ab = int(((A == good) & (B != good)).sum()); ba = int(((B == good) & (A != good)).sum())
            p = binomtest(ab, ab + ba).pvalue if ab + ba else 1.0
            L.append(f"| {a} vs {b} | {grp} | {'no collision' if col == 'collision' else col} | {ab} | {ba} | {p:.3g} |")
L.append("")
open(OUT, "w").write("\n".join(L))
print("\n".join(L))
