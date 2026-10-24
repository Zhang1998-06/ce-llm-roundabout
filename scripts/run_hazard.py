"""DT vs DT + CoC (V)LM high-level driver on nominal and long-tail (stalled-vehicle) roundabout scenarios.
    METHODS=dt,rule,llm,vlm SCEN=nominal,stall_entry,stall_mid,stall_exit SEEDS=200-224 TAG=dev python scripts/run_hazard.py
"""
import json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
import multiprocessing as mp
import numpy as np, pandas as pd, torch

ACTIONS = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
URL = "http://127.0.0.1:11434"
MODELS = {"llm": ("qwen3:4b-instruct-2507-q4_K_M", False), "vlm": ("qwen3-vl:2b-instruct-q4_K_M", True)}
G = None
G_HAZ = {}


def episode(method, scen, seed):
    import evaluation as E, hazards as H, coc_supervisor as C, bev_renderer as B, highway_env  # noqa
    global G
    if G is None:
        G = E.load_dt(E.load_cfg())[:3]
    mod, model, c = G
    env = H.make_env(scen); u = env.unwrapped
    VER = int(os.environ.get("PROMPT_VER", "3"))
    sup = C.CoCSupervisor(URL, *MODELS[method], version=VER) if method in MODELS else None
    desc = {1: C.describe, 2: C.describe_v2, 3: C.describe_v3, 4: C.describe_v4, 5: C.describe_v5}[VER]
    SKV = int(os.environ.get("SKILL_V", "3" if os.environ.get("PROMPT_VER") == "5" else "1"))
    skill = {1: lambda: H.BypassSkill(17.0, 6.0, slow_on_access=False), 2: H.SafeBypassSkill, 3: H.SmoothBypassSkill}[SKV]()
    SCALE, MAX = mod.SCALE, mod.MAX_EP_LEN
    HOR = float(os.environ.get("HORIZON_S", "11"))           # sensitivity analysis only; 11 s is the protocol
    if HOR != 11:
        u.config["duration"] = HOR
        MAX = int(HOR * u.config["policy_frequency"])
    np.random.seed(seed); obs, _ = env.reset(seed=seed)
    G_HAZ[id(env)] = u.road.vehicles[-1] if scen != "nominal" else None      # the hazard (appended by the Stall wrapper)
    states = torch.tensor(obs, dtype=torch.float32)[None, None]; actions = torch.zeros((1, 0, c.act_dim))
    rtg = torch.tensor([[[mod.MAX_EP_LEN / SCALE]]]); ts = torch.zeros((1, 1), dtype=torch.long)   # target RTG as trained (2.2)
    dt_s = 1 / float(u.config["policy_frequency"])
    t, done, trunc, crashed, ok, tte = 0, False, False, False, False, np.nan
    speeds, mind, decs, nov = [], 99.0, [], 0
    fault, lane_hist = None, []                                   # at-fault classification of the first ego collision
    while not (done or trunc) and t < MAX:
        actions = torch.cat([actions, torch.zeros((1, 1, c.act_dim))], dim=1)
        with torch.no_grad():
            a = int(mod.get_action(model, states, actions, rtg, ts)[1])
        ae = a
        if method == "acc":
            ae = H.acc_action(env, a) if SKV == 2 else H.smooth_acc_action(env, a)
        elif SKV >= 2 and skill.active:                       # v2: the maneuver runs to its end (also on the exit road)
            ae = skill.act(env, a)
        elif method == "oracle" and SKV >= 3 and not C._where(u.vehicle).startswith("north exit"):
            # reference: perfect recognition (the scenario's hazard) + the maneuver rule stated in prompt v5
            haz = G_HAZ.get(id(env))
            pa = C.path_ahead_x(env)
            if haz is not None and any(v is haz for _, v in pa[:3]):
                dx = C.to_exit(env, haz)
                follow = haz.speed >= 0.5 and dx is not None and dx < C.EXIT_ZONE
                skill = H.FollowSkill() if follow else H.SmoothBypassSkill()
                skill.start(haz); ae = skill.act(env, a)
        elif method == "follow" and SKV >= 3 and not C._where(u.vehicle).startswith("north exit"):   # oracle follow
            ob = next((v for g, v in C.path_ahead(env)[:3] if v.speed < 5), None)
            if ob is not None:
                skill = H.FollowSkill(); skill.start(ob); ae = skill.act(env, a)
        elif method == "skill" and SKV >= 2 and not C._where(u.vehicle).startswith("north exit"):
            ob = H.stationary_ahead_on_path(env, float(os.environ.get("SKILL_DM", "40")))
            if ob is None:                                    # oracle for the slow hazard too: the scenario's hazard
                ob = next((v for g, v in C.path_ahead(env)[:3] if v.speed < 5), None)
            if ob is not None:
                skill.start(ob); ae = skill.act(env, a)
        elif len(u.vehicle.route) > 0 and not C._where(u.vehicle).startswith("north exit"):   # active until the ego leaves the ring
            if method == "rule" and H.blocked_ahead(env):
                ae = 0
            elif method == "skill" or skill.active:          # oracle-triggered skill / maneuver in progress
                if not skill.active and method == "skill":
                    ob = H.stationary_ahead_on_path(env, float(os.environ.get("SKILL_DM", "40")))
                    if ob is not None:
                        skill.start(ob)
                if skill.active:
                    ae = skill.act(env, a)
            elif sup is not None:
                png = B.render_bev(env) if sup.vision else None
                state_text = desc(env, ACTIONS[a])
                r = sup.ask(state_text, png)
                if VER == 5:
                    pa = C.path_ahead_x(env)
                    if r["meta_action"] in ("BYPASS_OBSTACLE", "FOLLOW_VEHICLE") and pa:
                        skill = H.SmoothBypassSkill() if r["meta_action"] == "BYPASS_OBSTACLE" else H.FollowSkill()
                        skill.start(min(pa[:3], key=lambda x: x[1].speed)[1]); ae = skill.act(env, a)   # the abnormal one
                elif VER == 4:
                    ts_ = float(getattr(u.vehicle, "target_speed", u.vehicle.speed))
                    if r["meta_action"] == "BYPASS_OBSTACLE":
                        pa = C.path_ahead(env)
                        if pa:
                            skill.start(min(pa[:3], key=lambda x: x[1].speed)[1]); ae = skill.act(env, a)   # the abnormal one
                    elif r["meta_action"] == "SLOW_DOWN":
                        ae = 4 if ts_ > 8 else 1                       # one level, never to a stop
                    elif r["meta_action"] == "KEEP_SPEED":
                        ae = 1
                elif r["action"] is not None:
                    ae = r["action"]
                decs.append(dict(method=method, scenario=scen, seed=seed, t=t, dt_action=ACTIONS[a],
                                 meta_action=r["meta_action"], valid=r["valid"], latency_s=r["latency_s"],
                                 scene=r["scene"], critical_objects=r["critical_objects"],
                                 causal_reasoning=r["causal_reasoning"], abnormal_cause=r.get("abnormal_cause", ""),
                                 executed=ACTIONS[ae], skill_started=bool(skill.active), user_message=state_text, raw=r.get("raw", "")))
        nov += ae != a
        actions[0, -1] = torch.nn.functional.one_hot(torch.tensor(ae), c.act_dim).float()   # history = executed action
        obs, rew, done, trunc, info = env.step(ae); t += 1
        ego = u.vehicle; speeds.append(float(ego.speed))
        mind = min([mind] + [float(np.linalg.norm(v.position - ego.position)) for v in u.road.vehicles if v is not ego])
        lane_hist.append((C._ring_lane(ego), ae in (0, 2)))
        if ego.crashed and fault is None:
            fault = at_fault(env, C, lane_hist)
        crashed |= bool(info.get("crashed", False))
        if not ok and len(ego.route) == 0 and not ego.crashed:
            ok, tte = True, t * dt_s
        states = torch.cat([states, torch.tensor(obs, dtype=torch.float32)[None, None]], dim=1)
        rtg = torch.cat([rtg, (rtg[0, -1, 0] - rew / SCALE).view(1, 1, 1)], dim=1)
        ts = torch.cat([ts, torch.full((1, 1), t, dtype=torch.long)], dim=1)
    success = bool(ok and not crashed)
    ep = dict(method=method, scenario=scen, seed=seed, success=int(success), collision=int(crashed),
              collision_at_fault=int(crashed and fault is not False),
              timeout=int(not success and not crashed), avg_speed=float(np.mean(speeds)),
              completion_time=tte if success else np.nan, min_dist=mind, n_steps=t, n_overrides=nov,
              n_calls=len(decs), mean_latency=float(np.mean([d["latency_s"] for d in decs])) if decs else np.nan,
              invalid=sum(not d["valid"] for d in decs))
    return ep, decs


def at_fault(env, C, lane_hist):
    """nuPlan-style simplification: the ego is NOT at fault only if it is hit from behind (partner > 2 m behind along the
    ego heading) while keeping its lane (no lane change command in the last 3 steps, same ring lane) outside the merge zone.
    Everything else (hitting a vehicle ahead, side impacts, lane-change and entry conflicts) counts as at fault."""
    u = env.unwrapped
    e = u.vehicle
    others = [v for v in u.road.vehicles if v is not e and v.crashed]
    if not others:
        return True
    p = min(others, key=lambda v: np.linalg.norm(v.position - e.position))
    fwd = np.array([np.cos(e.heading), np.sin(e.heading)])
    rel = float(np.dot(p.position - e.position, fwd))
    where = C._where(e)
    th_en, _ = C._entry_angle(env)
    in_merge = where.startswith("south access") or (where.startswith("ring outer") and
                                                    24.0 * ((th_en - C._ring_angle(e.position)) % (2 * np.pi)) < 10)
    recent = lane_hist[-3:]
    kept_lane = not any(lc for _, lc in recent) and len({l for l, _ in recent}) == 1
    return not (rel < -2 and kept_lane and not in_merge)


def work(x):
    return episode(*x)


if __name__ == "__main__":
    methods = os.environ.get("METHODS", "dt,rule,llm").split(",")
    scens = os.environ.get("SCEN", "nominal,stall_entry,stall_mid,stall_exit").split(",")
    lo, hi = map(int, os.environ.get("SEEDS", "200-224").split("-"))
    tag = os.environ.get("TAG", "dev")
    out = f"results/hazard_{tag}"; os.makedirs(out, exist_ok=True)
    jobs = [(m, s, k) for m in methods for s in scens for k in range(lo, hi + 1)]
    eps, decs = [], []
    with mp.Pool(int(os.environ.get("NP", "4"))) as p:
        for i, (ep, d) in enumerate(p.imap_unordered(work, jobs)):
            eps.append(ep); decs += d
            if i % 20 == 0:
                print(f"{i + 1}/{len(jobs)}", ep, flush=True)
                pd.DataFrame(eps).to_csv(f"{out}/episodes_{'_'.join(methods)}_partial.csv", index=False)
    E = pd.DataFrame(eps).sort_values(["method", "scenario", "seed"])
    E.to_csv(f"{out}/episodes_{'_'.join(methods)}.csv", index=False)
    pd.DataFrame(decs).to_csv(f"{out}/decisions_{'_'.join(methods)}.csv", index=False)
    S = E.groupby(["scenario", "method"]).agg(n=("seed", "size"), success=("success", "mean"), collision=("collision", "mean"),
                                              at_fault=("collision_at_fault", "mean"),
                                              timeout=("timeout", "mean"), speed=("avg_speed", "mean"),
                                              time=("completion_time", "mean"), min_dist=("min_dist", "mean"),
                                              overrides=("n_overrides", "mean"), calls=("n_calls", "mean"),
                                              latency=("mean_latency", "mean"), invalid=("invalid", "sum"))
    for k in ("success", "collision", "at_fault", "timeout"):
        S[k] *= 100
    print(S.round(2).to_string())
    S.round(3).to_csv(f"{out}/summary_{'_'.join(methods)}.csv")
