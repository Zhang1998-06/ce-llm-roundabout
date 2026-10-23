"""B0 (DT) vs B1 (DT + VLM) on the frozen paired seed list; writes raw CSVs, comparison table, figures.
    python scripts/run_comparison.py            # runs both and analyses
"""
import hashlib, json, os, subprocess, sys, threading, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
import numpy as np, pandas as pd
import gymnasium as gym, highway_env  # noqa
import bev_renderer as bev
from evaluation import *
from vlm_supervisor import VLMSupervisor

cfg = load_cfg()
TAG = os.environ.get("TAG", "")                       # "" = v1 ; e.g. "v2_s100"
if os.environ.get("PROMPT"): cfg["vlm"]["prompt"] = os.environ["PROMPT"]
if os.environ.get("VLM_MODEL"): cfg["vlm"]["model"] = os.environ["VLM_MODEL"]
seeds = [int(x) for x in os.environ["SEEDS"].split(",")] if os.environ.get("SEEDS") else cfg["seeds"]["eval"]
RAW, FIG = "results/raw", "results/figures"
OUT = f"results/{TAG}" if TAG else "results"
os.makedirs(OUT + "/raw", exist_ok=True); RAW = OUT + "/raw"
peak = [0]
def poll():
    while True:
        try:
            m = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
            peak[0] = max(peak[0], m)
        except Exception: pass
        time.sleep(0.5)
threading.Thread(target=poll, daemon=True).start()

mod, model, c, _ = load_dt(cfg)
env = gym.make("roundabout-v0")
if os.environ.get("LLM_TEXT"):
    from llm_supervisor import LLMSupervisor
    vlm = LLMSupervisor(cfg["vlm"])
else:
    vlm = VLMSupervisor(cfg["vlm"])
os.makedirs(f"{OUT}/bev", exist_ok=True)
rows = {"B0": [], "B1": []}; steps = {"B0": [], "B1": []}; decs = []
for name in ("B0", "B1"):
    for s in seeds:
        if name == "B0":
            ep, st, _ = run_episode(mod, model, c, env, s)
        else:
            ep, st, d = run_episode(mod, model, c, env, s, vlm, cfg["vlm"]["query_every_n_steps"], bev,
                                    f"{OUT}/bev" if s == seeds[0] else None)
            decs += d
        rows[name].append(ep); steps[name] += st
        print(name, s, {k: ep[k] for k in ("success", "collision", "timeout", "avg_speed", "n_vlm_calls", "n_overrides")}, flush=True)

pd.DataFrame(rows["B0"]).to_csv(f"{RAW}/dt_episodes.csv", index=False)
pd.DataFrame(rows["B1"]).to_csv(f"{RAW}/dt_vlm_episodes.csv", index=False)
pd.DataFrame(decs).to_csv(f"{RAW}/vlm_decisions.csv", index=False)
json.dump({k: v for k, v in steps.items()}, open(f"{RAW}/steps.json", "w"))

d0, d1, dd = pd.DataFrame(rows["B0"]), pd.DataFrame(rows["B1"]), pd.DataFrame(decs)
def agg(df, name):
    return {"Method": name, "Episodes": len(df), "Success Rate (%)": 100 * df.success.mean(),
            "Collision Rate (%)": 100 * df.collision.mean(), "Average Speed (m/s)": df.avg_speed.mean(),
            "Completion Time (s)": df.completion_time.mean(), "Average Reward": df.reward.mean(),
            "Successes": int(df.success.sum()), "Collisions": int(df.collision.sum()),
            "Timeouts": int(df.timeout.sum())}
comp = pd.DataFrame([agg(d0, "Original DT"), agg(d1, "DT + VLM")])
comp["VLM calls"] = [0, int(d1.n_vlm_calls.sum())]
comp["Overrides"] = [0, int(d1.n_overrides.sum())]
comp["Override rate (% of calls)"] = [0, 100 * d1.n_overrides.sum() / max(1, d1.n_vlm_calls.sum())]
comp["VLM JSON validity (%)"] = [np.nan, 100 * dd.valid.mean() if len(dd) else np.nan]
comp["VLM mean latency (s)"] = [np.nan, dd.latency_s.mean() if len(dd) else np.nan]
comp["VLM YIELD share (%)"] = [np.nan, 100 * (dd.mode_returned == "YIELD").mean() if len(dd) else np.nan]
comp["Peak GPU mem (MiB, whole GPU)"] = [np.nan, peak[0]]
comp.to_csv(f"{OUT}/comparison.csv", index=False)
m = d0.merge(d1, on="seed", suffixes=("_dt", "_vlm"))
pair = dict(fail_to_success=int(((m.success_dt == 0) & (m.success_vlm == 1)).sum()),
            success_to_fail=int(((m.success_dt == 1) & (m.success_vlm == 0)).sum()),
            both_success=int(((m.success_dt == 1) & (m.success_vlm == 1)).sum()),
            both_fail=int(((m.success_dt == 0) & (m.success_vlm == 0)).sum()))
json.dump(pair, open(f"{OUT}/paired_outcomes.json", "w"))
md = ["| Method | Success Rate (%) | Collision Rate (%) | Average Speed (m/s) | Completion Time (s) | Average Reward |",
      "|---|---:|---:|---:|---:|---:|"]
for _, r in comp.iterrows():
    md.append(f"| {r['Method']} | {r['Success Rate (%)']:.1f} | {r['Collision Rate (%)']:.1f} | "
              f"{r['Average Speed (m/s)']:.2f} | {r['Completion Time (s)']:.2f} | {r['Average Reward']:.2f} |")
open(f"{OUT}/comparison.md", "w").write("\n".join(md) + "\n")
print("\n".join(md)); print(comp.T.to_string()); print(pair)
