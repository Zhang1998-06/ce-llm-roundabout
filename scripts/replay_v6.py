"""Replay one logged v6 episode step by step (the logged LLM decisions are fed back, no new LLM call), to show what the
executor did after each decision. Env and DT are deterministic given the seed, so the replay reproduces the logged
episode; the script checks this against the logged outcome.
    python scripts/replay_v6.py stall_mid 602 [results/hazard_test_v6]
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ["PROMPT_VER"] = "6"
import numpy as np, pandas as pd
import run_hazard as R, coc_supervisor as C, hazards as H

scen, seed = sys.argv[1], int(sys.argv[2])
D = sys.argv[3] if len(sys.argv) > 3 else "results/hazard_test_v6"
K = pd.read_csv(f"{D}/decisions_llm_llm5.csv")
K = K[(K.method == "llm") & (K.scenario == scen) & (K.seed == seed)].set_index("t")
E = pd.read_csv(f"{D}/episodes_llm_llm5.csv")
E = E[(E.method == "llm") & (E.scenario == scen) & (E.seed == seed)].iloc[0]
st = {"t": 0}


def ask_v6(sup, text):
    r = K.loc[st["t"]]
    assert text == r.user_message, f"observation differs from the log at t={st['t']}"
    tid = None if pd.isna(r.target_id) else int(r.target_id)
    return dict(decision=r.decision, target_id=tid, valid=True, latency_s=r.latency_s, raw=r.raw,
                consequence=r.consequence, critical_objects="")


C.ask_v6 = ask_v6
_make = H.make_env


def make(sc):
    env = _make(sc); u = env.unwrapped; _reset, _step = env.reset, env.step

    def reset(**kw):
        o = _reset(**kw); st["haz"] = u.road.vehicles[-1]; return o

    def step(a):
        out = _step(a)
        e = u.vehicle
        hz = st["haz"]
        g = C._arc(float(np.hypot(*e.position)), C._ring_angle(e.position), C._ring_angle(hz.position)) \
            if C.road_of(e)[0] == "ring" else np.nan
        road, lane = C.road_of(e)
        print(f"t={st['t']:2d} ({st['t'] * 0.5:4.1f} s) executed={R.ACTIONS[int(a)]:10s} ego: {road}{', ' + lane if lane else ''} "
              f"v={e.speed:4.1f} target_v={e.target_speed:4.1f} | stalled car: {'%+.0f m along the ring' % g if g == g else '-'}"
              f"{' | CRASH' if e.crashed else ''}")
        st["t"] += 1
        return out
    env.reset, env.step = reset, step
    return env


H.make_env = make
ep, decs = R.episode("llm", scen, seed)
print({k: ep[k] for k in ("success", "collision", "timeout", "completion_time", "n_starts")})
print("matches the logged episode:", all(ep[k] == E[k] for k in ("success", "collision", "timeout", "n_steps")))
