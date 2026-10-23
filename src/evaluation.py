"""Closed-loop episode runner shared by B0 (DT) and B1 (DT + VLM)."""
import importlib.util
import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTIONS = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
FASTER, SLOWER = 3, 4
HIST_EXECUTED = os.environ.get('HIST_EXECUTED', '1') == '1'


def load_cfg():
    with open(os.path.join(ROOT, "configs", "experiment.yaml")) as f:
        return yaml.safe_load(f)


def load_dt(cfg):
    """Import the ORIGINAL DT code (TrainableDT / get_action) and load the frozen checkpoint."""
    from safetensors.torch import load_file
    from transformers import DecisionTransformerConfig
    spec = importlib.util.spec_from_file_location("orig_dt", cfg["dt"]["source_script"])
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    d = cfg["dt"]["checkpoint_dir"]
    c = DecisionTransformerConfig.from_pretrained(d)
    model = mod.TrainableDT(c).eval()
    missing = model.load_state_dict(load_file(os.path.join(d, "model.safetensors"), device="cpu"), strict=False)
    return mod, model, c, missing


REGION_R = float(os.environ.get('REGION_R', '70'))
YIELD_ON = os.environ.get('YIELD_ON', 'FASTER')       # FASTER = only FASTER is overridden; ANY = every action not already YIELD_TO
YIELD_TO = int(os.environ.get('YIELD_TO', str(4)))   # 4 = SLOWER (v1), 1 = IDLE (v2)


def in_entry_region(env):
    e = env.unwrapped.vehicle
    node = e.lane_index[0] if e.lane_index else None
    return node in ("ser", "ses", "se") and float(np.hypot(*e.position)) <= REGION_R


def run_episode(mod, model, c, env, seed, vlm=None, qevery=4, bev=None, save_bev=None):
    """vlm=None -> B0.  Returns (episode dict, step list, vlm decision list)."""
    SCALE, MAX_EP_LEN = mod.SCALE, mod.MAX_EP_LEN
    np.random.seed(seed)                       # env draws traffic density from the global RNG
    obs, _ = env.reset(seed=seed)
    states = torch.tensor(obs, dtype=torch.float32)[None, None]
    actions = torch.zeros((1, 0, c.act_dim))
    rtg = torch.tensor([[[MAX_EP_LEN / SCALE]]])
    tsteps = torch.zeros((1, 1), dtype=torch.long)
    n_veh = len(env.unwrapped.road.vehicles) - 1
    ep_ret, t, done, trunc = 0.0, 0, False, False
    speeds, steps, decisions = [], [], []
    exit_ok, crashed, tte = False, False, np.nan
    mode, qcount = "NORMAL", 0               # latest valid VLM mode, steps spent in region
    dt_s = 1.0 / float(env.unwrapped.config["policy_frequency"])
    while not (done or trunc) and t < MAX_EP_LEN:
        actions = torch.cat([actions, torch.zeros((1, 1, c.act_dim))], dim=1)
        with torch.no_grad():
            got = mod.get_action(model, states, actions, rtg, tsteps)
        onehot, a_dt = got[0], int(got[1])
        actions[0, -1] = onehot              # provisional; replaced by the EXECUTED action below
        a_exec, overridden, region = a_dt, False, False
        if vlm is not None:
            region = in_entry_region(env)
            if region:
                if qcount % qevery == 0:
                    path = os.path.join(save_bev, f"s{seed}_t{t}.png") if save_bev else None
                    if getattr(vlm, "text_only", False):
                        import llm_supervisor
                        path, png = None, None
                        res = vlm.ask(None, llm_supervisor.describe_kin(env, ACTIONS[a_dt]))
                    else:
                        png = bev.render_bev(env, path)
                        res = vlm.ask(png, bev.describe(env, ACTIONS[a_dt]))
                    if res["valid"]:
                        mode = res["mode"]
                    decisions.append(dict(seed=seed, t=t, dt_action=ACTIONS[a_dt], mode_returned=res["mode"],
                                          valid=res["valid"], latency_s=res["latency_s"], reason=res["reason"],
                                          raw=res["raw"], bev=path or ""))
                    last = decisions[-1]
                qcount += 1
                if mode == "YIELD" and (a_dt == FASTER or (YIELD_ON == 'ANY' and a_dt != YIELD_TO)):
                    a_exec, overridden = YIELD_TO, True
                    if decisions: decisions[-1]["overrode_any"] = True
            else:
                mode = "NORMAL"
        ego = env.unwrapped.vehicle
        steps.append(dict(seed=seed, t=t, x=float(ego.position[0]), y=float(ego.position[1]),
                          speed=float(ego.speed), dt_action=a_dt, exec_action=a_exec,
                          in_region=region, vlm_mode=mode if vlm else "", overridden=overridden,
                          others=[[float(v.position[0]), float(v.position[1])]
                                  for v in env.unwrapped.road.vehicles if v is not ego]))
        if HIST_EXECUTED:                    # DT history = the action actually executed (identical to B0, where exec == DT)
            actions[0, -1] = torch.nn.functional.one_hot(torch.tensor(a_exec), c.act_dim).float()
        obs, r, done, trunc, info = env.step(a_exec)
        ep_ret += r; t += 1
        ego = env.unwrapped.vehicle
        speeds.append(float(ego.speed))
        crashed |= bool(info.get("crashed", False))
        if not exit_ok and len(ego.route) == 0 and not ego.crashed:   # original exit evaluator
            exit_ok, tte = True, t * dt_s
        states = torch.cat([states, torch.tensor(obs, dtype=torch.float32)[None, None]], dim=1)
        rtg = torch.cat([rtg, (rtg[0, -1, 0] - r / SCALE).view(1, 1, 1)], dim=1)
        tsteps = torch.cat([tsteps, torch.full((1, 1), t, dtype=torch.long)], dim=1)
    sp = np.asarray(speeds)
    success = bool(exit_ok and not crashed)
    ep = dict(seed=seed, n_other_vehicles=n_veh, success=int(success), collision=int(crashed),
              timeout=int(not success and not crashed), avg_speed=float(sp.mean()),
              completion_time=tte if success else np.nan, sim_time=t * dt_s, reward=ep_ret,
              n_steps=t, n_vlm_calls=len(decisions), n_overrides=sum(s["overridden"] for s in steps),
              steps_in_region=sum(s["in_region"] for s in steps))
    return ep, steps, decisions
