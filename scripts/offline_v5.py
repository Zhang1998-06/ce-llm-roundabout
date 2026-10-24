"""Offline single-step check of prompt v5 (feasibility before closed loop).
States: DT-only rollouts, seeds 200-209, t = 0, 2, ..., 10. Label: nominal -> FOLLOW_POLICY;
hazard in the ego's path: FOLLOW_VEHICLE if it is moving and in the exit zone / on the exit road, else BYPASS_OBSTACLE.
    python scripts/offline_v5.py [model] [vision 0|1]"""
import os, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
import numpy as np, torch, pandas as pd
import evaluation as E, hazards as H, coc_supervisor as C, bev_renderer as B, highway_env  # noqa
ACTIONS = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3:4b-instruct-2507-q4_K_M"
vision = len(sys.argv) > 2 and sys.argv[2] == "1"
mod, model, c = E.load_dt(E.load_cfg())[:3]
sup = C.CoCSupervisor("http://127.0.0.1:11434", model_name, vision, version=5)
rows = []
for scen in ["nominal", "stall_entry", "stall_mid", "stall_exit", "slow_entry", "slow_mid", "slow_exit"]:
    for seed in range(200, 210):
        env = H.make_env(scen); u = env.unwrapped
        np.random.seed(seed); obs, _ = env.reset(seed=seed)
        haz = u.road.vehicles[-1] if scen != "nominal" else None
        states = torch.tensor(obs, dtype=torch.float32)[None, None]; actions = torch.zeros((1, 0, c.act_dim))
        rtg = torch.tensor([[[mod.MAX_EP_LEN / mod.SCALE]]]); ts = torch.zeros((1, 1), dtype=torch.long)
        for t in range(11):
            actions = torch.cat([actions, torch.zeros((1, 1, c.act_dim))], dim=1)
            with torch.no_grad():
                a = int(mod.get_action(model, states, actions, rtg, ts)[1])
            if t % 2 == 0:
                in_path = haz is not None and any(v is haz for _, v in C.path_ahead_x(env))
                dx = C.to_exit(env, haz) if haz is not None else None             # label = the rule the prompt states
                follow = haz is not None and haz.speed >= 0.5 and dx is not None and dx < C.EXIT_ZONE
                label = ("FOLLOW_POLICY" if scen == "nominal" else
                         ("FOLLOW_VEHICLE" if follow else "BYPASS_OBSTACLE") if in_path else None)
                if label is not None:
                    r = sup.ask(C.describe_v5(env, ACTIONS[a]), B.render_bev(env) if vision else None)
                    rows.append(dict(scen=scen, seed=seed, t=t, label=label, pred=r["meta_action"], valid=r["valid"],
                                     lat=r["latency_s"], cause=r["abnormal_cause"], why=r["causal_reasoning"]))
            actions[0, -1] = torch.nn.functional.one_hot(torch.tensor(a), c.act_dim).float()
            obs, rew, done, trunc, info = env.step(a)
            if done or trunc or u.vehicle.crashed:
                break
            states = torch.cat([states, torch.tensor(obs, dtype=torch.float32)[None, None]], dim=1)
            rtg = torch.cat([rtg, (rtg[0, -1, 0] - rew / mod.SCALE).view(1, 1, 1)], dim=1)
            ts = torch.cat([ts, torch.full((1, 1), t + 1, dtype=torch.long)], dim=1)
df = pd.DataFrame(rows)
df["ok"] = df.label == df.pred
tag = "vlm" if vision else "llm"
df.to_csv(f"results/offline_v5_{tag}.csv", index=False)
print(df.groupby("scen").agg(n=("ok", "size"), acc=("ok", "mean")).round(2).to_string())
print("overall", round(df.ok.mean(), 3), "valid", round(df.valid.mean(), 3), "latency", round(df.lat.mean(), 2))
print(pd.crosstab(df.label, df.pred))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 110)
print(df[~df.ok][["scen", "seed", "t", "pred", "cause", "why"]].head(12).to_string())
