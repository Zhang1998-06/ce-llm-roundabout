import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread
ACT = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
comp = pd.read_csv("results/comparison.csv"); st = json.load(open("results/raw/steps.json"))
dd = pd.read_csv("results/raw/vlm_decisions.csv")
# Fig 1
fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
for a, (k, u) in zip(ax, [("Success Rate (%)", "%"), ("Collision Rate (%)", "%"), ("Average Speed (m/s)", "m/s")]):
    a.bar(["DT", "DT + VLM"], comp[k], color=["#1f77b4", "#e8a33d"]); a.set_title(k)
    for i, v in enumerate(comp[k]): a.text(i, v, f"{v:.1f}", ha="center", va="bottom")
plt.tight_layout(); plt.savefig("results/figures/performance_comparison.png", dpi=150); plt.close()
# Fig 2: trajectories of one seed
seed = 0
fig, ax = plt.subplots(figsize=(6, 6))
for name, col in (("B0", "#1f77b4"), ("B1", "#e8a33d")):
    s = [x for x in st[name] if x["seed"] == seed]
    ax.plot([x["x"] for x in s], [x["y"] for x in s], "-o", ms=3, color=col, label={"B0": "DT", "B1": "DT + VLM"}[name])
    if name == "B1":
        ov = [x for x in s if x["overridden"]]
        ax.scatter([x["x"] for x in ov], [x["y"] for x in ov], marker="X", s=120, c="red", zorder=5, label="VLM override (FASTER→SLOWER)")
t0 = [x for x in st["B0"] if x["seed"] == seed][5]
ax.scatter(*zip(*t0["others"]), c="gray", s=25, label="other vehicles (DT run, t=5)")
th = np.linspace(0, 2 * np.pi, 200)
for r in (20, 24): ax.plot(r * np.cos(th), r * np.sin(th), "k-", lw=0.6)
ax.set_xlim(-45, 45); ax.set_ylim(60, -45); ax.set_aspect("equal"); ax.legend(fontsize=7); ax.set_title(f"Seed {seed}: ego trajectories (south→north entry)")
plt.tight_layout(); plt.savefig("results/figures/trajectory_comparison.png", dpi=150); plt.close()
# Fig 3: VLM example
ex = dd[(dd.seed == 0) & (dd.bev.notna())].iloc[0]
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
ax[0].imshow(imread(ex.bev)); ax[0].axis("off")
final = "SLOWER" if (ex.dt_action == "FASTER" and ex.mode_returned == "YIELD") else ex.dt_action
txt = f"seed {ex.seed}, step {ex.t}\nDT proposed action: {ex.dt_action}\nVLM mode: {ex.mode_returned}\nFinal executed action: {final}\n\nVLM reason:\n{ex.reason}\n\nlatency {ex.latency_s:.2f} s"
ax[1].text(0, 1, txt, va="top", wrap=True, fontsize=10); ax[1].axis("off")
plt.tight_layout(); plt.savefig("results/figures/vlm_decision_example.png", dpi=150); plt.close()
# diagnostics
b1 = pd.DataFrame(st["B1"]); b0 = pd.DataFrame(st["B0"])
stall = b1[(b1.speed < 1)].groupby("seed").t.min()
print("B1 episodes with stall (speed<1):", len(stall), "first stall step median", stall.median())
sp0 = b1[b1.speed < 1]; print("stall distance to centre (m): median", np.hypot(sp0.x, sp0.y).median())
first = b1[b1.overridden].groupby("seed").t.min(); print("first override step: median", first.median(), "min", first.min(), "max", first.max())
print("VLM replies containing 'no conflict'/'no conflicting' yet YIELD:", dd.reason.str.contains("no conflict", case=False).sum(), "of", len(dd))
print("B0 speed at t=0:", b0[b0.t==0].speed.mean())
print("B1 YIELD by dt_action:\n", dd.groupby("dt_action").mode_returned.value_counts())
