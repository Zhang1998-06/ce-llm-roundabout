"""B0 only over a seed range: list the seeds where the original DT fails (no VLM/LLM)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/src"); os.chdir(ROOT)
import gymnasium as gym, highway_env  # noqa
import pandas as pd
from evaluation import *
a, b = int(sys.argv[1]), int(sys.argv[2])
cfg = load_cfg(); mod, model, c, _ = load_dt(cfg); env = gym.make("roundabout-v0")
rows = [run_episode(mod, model, c, env, s)[0] for s in range(a, b)]
df = pd.DataFrame(rows); df.to_csv(f"results/phase1_nominal/b0_scan_{a}_{b}.csv", index=False)
print("n", len(df), "success", df.success.mean(), "collision", df.collision.mean(), "timeout", df.timeout.mean())
print("fail seeds:", df[df.success == 0].seed.tolist())
print(df[df.success == 0][["seed", "n_other_vehicles", "collision", "avg_speed"]].to_string())
