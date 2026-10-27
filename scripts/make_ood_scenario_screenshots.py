"""Figure: the six roundabout OOD scenarios as rendered by highway-env's own pygame renderer (EnvViewer), no redrawing.
Same seeding as scripts/run_hazard.py; the state is the one after reset (t = 0). Only the render window is set (fixed, centred
on the roundabout, instead of following the ego) - this changes the picture, not the simulation (checked below).
Overlay: a circle around the injected vehicle, placed with the renderer's own world -> pixel transform (pos2pix).
    python scripts/make_ood_scenario_screenshots.py [seed=500] [out_dir=reports/figures]
Writes the unmodified screenshots too: <out_dir>/highway_env_raw/<scenario>_seed<seed>_t0.png (north up, as rendered).
The figure rotates them 90 deg clockwise (north to the right), the same orientation as roundabout_ood_scenarios.pdf.
"""
import os, sys
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, gymnasium as gym
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from PIL import Image
import highway_env  # noqa: F401  (registers roundabout-v0)
from highway_env.envs.common.graphics import EnvViewer
from highway_env.vehicle.graphics import VehicleGraphics as VG
import hazards as H
from make_ood_scenario_figure import where_text          # same distance labels as the vector figure

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 500
OUT = sys.argv[2] if len(sys.argv) > 2 else "reports/figures"
COLS = ["entry", "mid", "exit"]
ROWS = [("stall", "Stationary vehicle\n(0 m/s)"), ("slow", "Slow-moving vehicle\n(3 m/s)")]
X0, X1, Y0, Y1 = -30.0, 30.0, -49.0, 50.0          # world window [m] (x east, y south), as in the vector figure
PX = 16                                            # renderer scaling [px/m]
HL = "#FF2D8A"                                     # overlay colour, distinct from the renderer's green / blue / yellow
rgb = lambda c: tuple(x / 255 for x in c[:3])

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                     "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5})


def snapshot(scen):
    base = gym.make("roundabout-v0", render_mode="rgb_array")
    base.unwrapped.config.update({"offscreen_rendering": True, "scaling": PX, "centering_position": [0.5, 0.5],
                                  "screen_width": int((X1 - X0) * PX), "screen_height": int((Y1 - Y0) * PX)})
    env = H.Stall(base, H.SCENARIOS[scen]); u = env.unwrapped
    np.random.seed(SEED); env.reset(seed=SEED)                       # identical to run_hazard.episode
    ref = H.make_env(scen); np.random.seed(SEED); ref.reset(seed=SEED)  # the evaluation env, no render settings
    a = [(tuple(v.position), v.heading, v.speed) for v in u.road.vehicles]
    b = [(tuple(v.position), v.heading, v.speed) for v in ref.unwrapped.road.vehicles]
    assert a == b, "render settings changed the scenario state"
    u.viewer = EnvViewer(u); u.viewer.enabled = True
    u.viewer.window_position = lambda: np.array([(X0 + X1) / 2, (Y0 + Y1) / 2])
    img = env.render()
    hz = u.road.vehicles[-1]
    px, py = u.viewer.sim_surface.pos2pix(*hz.position)
    if hz.speed == 0:                                                # sanity: the injected car is where the transform says
        assert np.allclose(img[py, px], VG.DEFAULT_COLOR, atol=40), img[py, px]
    return u, hz, img


def main():
    os.makedirs(f"{OUT}/highway_env_raw", exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.6), gridspec_kw=dict(wspace=0.04, hspace=0.20))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.10)
    for r, (kind, rlab) in enumerate(ROWS):
        for c, pos in enumerate(COLS):
            ax, scen = axes[r, c], f"{kind}_{pos}"
            u, hz, img = snapshot(scen)
            Image.fromarray(img).save(f"{OUT}/highway_env_raw/{scen}_seed{SEED}_t0.png")
            # rotate 90 deg clockwise; extent in display metres (x' = -y, y' = -x) so overlays use simulator coordinates
            ax.imshow(np.rot90(img, k=-1), extent=[-Y1, -Y0, -X1, -X0], origin="upper", interpolation="lanczos")
            ax.add_patch(Circle((-hz.position[1], -hz.position[0]), 4.8, fc="none", ec=HL, lw=1.1, zorder=5))
            assert abs(hz.speed - (0.0 if kind == "stall" else 3.0)) < 1e-9 and hz.lane_index == H.SCENARIOS[scen][0]
            ax.text(-Y1 + 1.2, -X0 - 1.2, f"({'abcdef'[3 * r + c]}) {scen}", ha="left", va="top", fontsize=7, color="white")
            ax.set_xlim(-Y1, -Y0); ax.set_ylim(-X1, -X0)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title({"entry": "Entry", "mid": "Middle", "exit": "Exit"}[pos], fontsize=9, pad=3)
            if c == 0:
                ax.set_ylabel(rlab, fontsize=8, labelpad=4)
            ax.set_xlabel(where_text(u.road.network, H.SCENARIOS[scen]), fontsize=7, labelpad=1.5)
    sq = dict(marker="s", ms=6, mec="#3C3C3C", mew=0.5, ls="none")
    h = [Line2D([], [], mfc=rgb(VG.EGO_COLOR), **sq), Line2D([], [], mfc=rgb(VG.BLUE), **sq),
         Line2D([], [], mfc=rgb(VG.DEFAULT_COLOR), **sq), Line2D([], [], marker="o", ms=9, mfc="none", mec=HL, mew=1.1, ls="none")]
    labels = ["Ego vehicle (DT)", "IDM traffic (incl. the 3 m/s vehicle)", "Stationary vehicle", "Injected vehicle (overlay)"]
    fig.legend(h, labels, loc="lower center", ncol=4, frameon=False, fontsize=7.5, handletextpad=0.4, columnspacing=1.6,
               bbox_to_anchor=(0.535, 0.0))
    fig.suptitle(f"Roundabout Out-of-Distribution Driving Scenarios (highway-env render, seed {SEED}, t = 0 s)", fontsize=10, y=0.985)
    for ext, kw in (("pdf", {"dpi": 600}), ("jpg", {"dpi": 600, "pil_kwargs": {"quality": 95}})):
        fig.savefig(f"{OUT}/roundabout_ood_scenarios_highway_env.{ext}", **kw)
    print("written", OUT, "seed", SEED)


if __name__ == "__main__":
    main()
