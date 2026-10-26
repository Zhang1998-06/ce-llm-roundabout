"""Offline single-step check of prompt v6 on DEV states only (seeds 200-219, never the test seeds).
States = observations logged along the designed-rule trajectories (results/hazard_v6probe_rules), with the privileged
reference label of each state. The LLM sees exactly the text the closed loop would send.
    python scripts/offline_v6.py [n_per_cell] [out.csv]
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
import pandas as pd
import coc_supervisor as C

n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
out = sys.argv[2] if len(sys.argv) > 2 else "results/offline_v6_dev.csv"
D = pd.read_csv("results/hazard_v6probe_rules/decisions_designed_naive.csv")
D = D[(D.method == "designed") & D.seed.between(200, 219)].drop_duplicates("user_message")
# balance: per scenario up to n states where the reference intervenes and n where it does not
S = pd.concat([g.sample(min(n, len(g)), random_state=0) for _, g in D.groupby(["scenario", D.ref_decision != "FOLLOW_POLICY"])])
sup = C.CoCSupervisor("http://127.0.0.1:11434", os.environ.get("V6_MODEL", "qwen3:4b-instruct-2507-q4_K_M"))
rows = []
for _, r in S.iterrows():
    q = C.ask_v6(sup, r.user_message)
    ok_dec = q["decision"] == r.ref_decision
    ok_tgt = r.ref_decision == "FOLLOW_POLICY" or (q["target_id"] is not None and q["target_id"] == int(r.ref_target))
    rows.append(dict(scenario=r.scenario, seed=r.seed, t=r.t, ref=r.ref_decision, ref_target=r.ref_target,
                     decision=q["decision"], target_id=q["target_id"], correct=ok_dec and ok_tgt, valid=q["valid"],
                     consequence=q["consequence"], critical_objects=q["critical_objects"], thinking=q["thinking"], latency=q["latency_s"], user_message=r.user_message))
    print(f"{r.scenario:16s} ref={r.ref_decision:14s} llm={q['decision']:14s} tgt={q['target_id']} ref_t={r.ref_target} | {q['critical_objects']} || {q['consequence']}", flush=True)
R = pd.DataFrame(rows)
R.to_csv(out, index=False)
print(R.groupby(["scenario", "ref"]).correct.agg(["mean", "size"]).round(2).to_string())
print("overall", round(R.correct.mean(), 3), "valid", R.valid.mean())
