"""Offline decision test set: states visited by DT+reference-rule in every scenario, with the description texts,
the BEV image and the expected meta-action class (FOLLOW / CHANGE_INNER / WAIT)."""
import os, pickle, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
import numpy as np, torch
import evaluation as E, hazards as H, coc_supervisor as C, bev_renderer as B, highway_env  # noqa

mod, model, c = E.load_dt(E.load_cfg())[:3]
ACTIONS = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]


def inner_free(env):
    u = env.unwrapped; e = u.vehicle
    th = C._ring_angle(e.position); r = float(np.hypot(*e.position))
    for v in u.road.vehicles:
        if v is e or C._ring_lane(v) != "inner" or not C._where(v).startswith("ring"):
            continue
        g = r * ((th - C._ring_angle(v.position) + np.pi) % (2 * np.pi) - np.pi)
        if -10 <= g <= 10:
            return False
    return True


items = []
for scen in H.SCENARIOS:
    for seed in range(int(sys.argv[1]), int(sys.argv[2])):
        env = H.make_env(scen); u = env.unwrapped
        np.random.seed(seed); obs, _ = env.reset(seed=seed)
        st = torch.tensor(obs, dtype=torch.float32)[None, None]; ac = torch.zeros((1, 0, c.act_dim))
        rtg = torch.tensor([[[mod.MAX_EP_LEN / mod.SCALE]]]); ts = torch.zeros((1, 1), dtype=torch.long)
        t, done, trunc = 0, False, False
        while not (done or trunc) and t < mod.MAX_EP_LEN and len(u.vehicle.route) > 0 and not C._where(u.vehicle).startswith("north exit"):
            ac = torch.cat([ac, torch.zeros((1, 1, c.act_dim))], dim=1)
            with torch.no_grad():
                a = int(mod.get_action(model, st, ac, rtg, ts)[1])
            hz = H.blocked_ahead(env)
            on_ring = C._where(u.vehicle).startswith("ring")
            label = "FOLLOW" if not hz else ("CHANGE_INNER" if on_ring and inner_free(env) else "WAIT")
            items.append(dict(scenario=scen, seed=seed, t=t, label=label, dt_action=ACTIONS[a],
                              v1=C.describe(env, ACTIONS[a]), v2=C.describe_v2(env, ACTIONS[a]), v3=C.describe_v3(env, ACTIONS[a]), v4=C.describe_v4(env, ACTIONS[a]),
                              label4="BYPASS" if any(v.speed < 0.5 for _, v in C.path_ahead(env)[:1]) else "FOLLOW", png=B.render_bev(env)))
            ae = 0 if hz else a
            ac[0, -1] = torch.nn.functional.one_hot(torch.tensor(ae), c.act_dim).float()
            obs, r, done, trunc, _ = env.step(ae); t += 1
            st = torch.cat([st, torch.tensor(obs, dtype=torch.float32)[None, None]], dim=1)
            rtg = torch.cat([rtg, (rtg[0, -1, 0] - r / mod.SCALE).view(1, 1, 1)], dim=1)
            ts = torch.cat([ts, torch.full((1, 1), t, dtype=torch.long)], dim=1)
pickle.dump(items, open(sys.argv[3], "wb"))
from collections import Counter
print(len(items), Counter((i["scenario"], i["label"]) for i in items))
ex = [i for i in items if i["label"] == "CHANGE_INNER"][0]; print(ex["v2"])
ex = [i for i in items if i["label"] == "FOLLOW" and "ring" in i["v2"][:20]][0]; print(ex["v2"])
