import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
import gymnasium as gym, highway_env  # noqa
import bev_renderer as bev
from evaluation import *
from vlm_supervisor import VLMSupervisor
cfg = load_cfg(); mod, model, c, miss = load_dt(cfg)
print("load_state_dict:", len(miss.missing_keys), "missing,", len(miss.unexpected_keys), "unexpected")
env = gym.make("roundabout-v0")
print("obs shape", env.reset(seed=0)[0].shape, "policy_freq", env.unwrapped.config["policy_frequency"])
vlm = VLMSupervisor(cfg["vlm"]); os.makedirs("results/bev_samples/smoke", exist_ok=True)
for s in cfg["seeds"]["smoke"]:
    a, *_ = run_episode(mod, model, c, env, s)
    a2, *_ = run_episode(mod, model, c, env, s)          # determinism check
    b, st, dec = run_episode(mod, model, c, env, s, vlm, cfg["vlm"]["query_every_n_steps"], bev, "results/bev_samples/smoke")
    print(s, "B0 succ/coll", a["success"], a["collision"], "repeat-same", a["reward"] == a2["reward"],
          "| B1 succ/coll", b["success"], b["collision"], "calls", b["n_vlm_calls"], "overrides", b["n_overrides"],
          "region_steps", b["steps_in_region"], "speed", round(a["avg_speed"],1), round(b["avg_speed"],1))
    for d in dec: print("   ", d["t"], d["dt_action"], d["mode_returned"], d["valid"], f"{d['latency_s']:.1f}s", d["reason"][:90])
