"""Figure: the six roundabout OOD scenarios (stationary / slow vehicle at the entry / middle / exit of the ego's ring route).
Every element is drawn from the simulator state after reset (same seeding as scripts/run_hazard.py): lanes are sampled
from the road network (lane.position, lane.width_at, lane.line_types), vehicles from position / heading / LENGTH / WIDTH,
the route from the ego's planned route, landmarks and distances from the lane geometry. Nothing is placed by hand.
    python scripts/make_ood_scenario_figure.py [seed=500] [out_dir=reports/figures]
The map is rotated 90 deg clockwise (north to the right) so the route reads left to right; rotation keeps handedness.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import hazards as H
from highway_env.road.lane import LineType
from highway_env.road.graphics import LaneGraphics

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 500
OUT = sys.argv[2] if len(sys.argv) > 2 else "reports/figures"
COLS = ["entry", "mid", "exit"]
ROWS = [("stall", "Stationary vehicle\n(0 m/s)"), ("slow", "Slow-moving vehicle\n(3 m/s)")]
XLIM, YLIM = (-50.0, 49.0), (-30.0, 30.0)          # display frame [m]: x = -y_sim (north right), y = -x_sim
EGO, HAZ, OTHER, INK, ROAD, EDGE = "#0072B2", "#D55E00", "#9A9A9A", "#222222", "#E3E3E3", "#7F7F7F"

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                     "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5})


def D(p):
    """Simulator (x, y; y = south) -> display frame, north to the right."""
    p = np.asarray(p, float)
    return np.stack([-p[..., 1], -p[..., 0]], -1)


def state(scen):
    env = H.make_env(scen); u = env.unwrapped
    np.random.seed(SEED); env.reset(seed=SEED)                  # identical to run_hazard.episode
    return u, (u.road.vehicles[-1] if scen != "nominal" else None)


def draw_road(ax, net):
    lanes = [l for f in net.graph for t in net.graph[f] for l in net.graph[f][t]]
    for l in lanes:                                             # surface
        s = np.linspace(0, l.length, max(2, int(l.length / 0.5)))
        w = np.array([l.width_at(x) for x in s])
        a = np.array([l.position(x, -wi / 2) for x, wi in zip(s, w)]); b = np.array([l.position(x, wi / 2) for x, wi in zip(s, w)])
        ax.add_patch(Polygon(D(np.vstack([a, b[::-1]])), closed=True, fc=ROAD, ec=ROAD, lw=0.3, zorder=1))
    for l in lanes:                                             # markings, as in highway_env.road.graphics.LaneGraphics
        for side in (0, 1):
            lat = (side - 0.5) * l.width_at(0)
            if l.line_types[side] in (LineType.CONTINUOUS, LineType.CONTINUOUS_LINE):
                s = np.linspace(0, l.length, max(2, int(l.length / 0.5)))
                ax.plot(*D(np.array([l.position(x, lat) for x in s])).T, color=EDGE, lw=0.6, zorder=2, solid_capstyle="butt")
            elif l.line_types[side] == LineType.STRIPED:
                for s0 in np.arange(0, l.length, LaneGraphics.STRIPE_SPACING):
                    s = np.linspace(s0, min(s0 + LaneGraphics.STRIPE_LENGTH, l.length), 6)
                    ax.plot(*D(np.array([l.position(x, lat) for x in s])).T, color="white", lw=0.7, zorder=2,
                            solid_capstyle="butt")


def box(v):
    c, s = np.cos(v.heading), np.sin(v.heading)
    f, r = np.array([c, s]), np.array([-s, c])
    L, W = v.LENGTH / 2, v.WIDTH / 2
    return D(np.array([v.position + f * L + r * W, v.position + f * L - r * W, v.position - f * L - r * W, v.position - f * L + r * W]))


def draw_vehicle(ax, v, fc, ec=INK, z=5):
    ax.add_patch(Polygon(box(v), closed=True, fc=fc, ec=ec, lw=0.5, zorder=z))
    nose = box(v)[:2].mean(0)                                   # short heading tick ending at the front bumper
    c, s = np.cos(v.heading), np.sin(v.heading)
    p = D(np.array([v.position + np.array([c, s]) * (v.LENGTH / 2 - 1.4)]))[0]
    ax.plot(*np.stack([p, nose]).T, color="white" if fc != "white" else INK, lw=0.6, zorder=z + 0.1, solid_capstyle="butt")


def route_xy(u):
    """Centreline of the ego's planned route (lane id None on the ring = outer lane, the lane the access road merges into)."""
    net, pts = u.road.network, []
    for f, t, i in u.vehicle.route:
        lanes = net.graph[f][t]
        l = lanes[i if i is not None else len(lanes) - 1]
        s = np.linspace(0, l.length, max(2, int(l.length / 0.5)))
        pts.append(np.array([l.position(x, 0) for x in s]))
    p = np.vstack(pts)
    p = p[p[:, 1] < u.vehicle.position[1] - u.vehicle.LENGTH / 2]        # start at the ego's front bumper
    return p[p[:, 1] > -(XLIM[1] - 2.5)]                                  # stop inside the panel


def landmarks(net):
    lane = lambda f, t: net.get_lane((f, t, 1))
    merge = lane("se", "ex").position(0, 0)                                # outer lane where the access road joins
    exit_ = lane("ee", "nx").position(lane("ee", "nx").length, 0)          # outer lane where the north exit leaves
    return merge, exit_


def where_text(net, spec):
    """Position of the injected vehicle along the ego's ring route, from the lane lengths."""
    (f, t, i), s = spec[0], spec[1]
    seq = [("se", "ex"), ("ex", "ee"), ("ee", "nx")]
    k = seq.index((f, t))
    past = sum(net.get_lane(seq[j] + (i,)).length for j in range(k)) + s
    total = sum(net.get_lane(x + (i,)).length for x in seq)
    lane = "outer" if i == 1 else "inner"
    return f"{lane} lane, {past:.1f} m past merge" if past <= total / 2 else f"{lane} lane, {total - past:.1f} m before exit"


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.6), gridspec_kw=dict(wspace=0.04, hspace=0.20))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.10)
    for r, (kind, rlab) in enumerate(ROWS):
        for c, pos in enumerate(COLS):
            ax, scen = axes[r, c], f"{kind}_{pos}"
            u, hz = state(scen)
            net = u.road.network
            draw_road(ax, net)
            rt = D(route_xy(u))
            ax.plot(*rt[:-3].T, color=EGO, lw=1.0, ls=(0, (3.0, 1.6)), zorder=3)
            ax.add_patch(FancyArrowPatch(rt[-6], rt[-1], arrowstyle="-|>,head_length=3.2,head_width=1.8", color=EGO,
                                         lw=1.0, zorder=3, shrinkA=0, shrinkB=0))
            m, x = landmarks(net)
            for p in (m, x):
                ax.plot(*D(p), marker="o", ms=2.6, mfc="white", mec=INK, mew=0.6, zorder=4, ls="none")
            for v in u.road.vehicles:
                if v is u.vehicle or v is hz:
                    continue
                draw_vehicle(ax, v, OTHER)
            draw_vehicle(ax, u.vehicle, EGO, z=6)
            draw_vehicle(ax, hz, HAZ, z=7)
            ax.add_patch(Circle(D(hz.position), 4.6, fc="none", ec=HAZ, lw=0.9, zorder=7))
            assert abs(hz.speed - (0.0 if kind == "stall" else 3.0)) < 1e-9 and hz.lane_index == H.SCENARIOS[scen][0]
            ax.text(XLIM[0] + 1.2, YLIM[1] - 1.2, f"({'abcdef'[3 * r + c]}) {scen}", ha="left", va="top", fontsize=7,
                    color=INK, zorder=9)
            ax.set_xlabel(where_text(net, H.SCENARIOS[scen]), fontsize=7, labelpad=1.5, color=INK)
            if r == 0 and c == 0:                                  # orientation labels, once (geometry identical in all panels)
                ax.annotate("merge", D(m), xytext=(-3, -3), textcoords="offset points", fontsize=7, ha="right", va="top", color=INK)
                ax.annotate("exit", D(x), xytext=(3, -3), textcoords="offset points", fontsize=7, ha="left", va="top", color=INK)
                ax.text(XLIM[0] + 1.2, D(u.vehicle.position)[1] - 3.3, "south\nentry", fontsize=7, va="top", color=INK)
                ax.text(XLIM[1] - 1.0, D(u.road.network.get_lane(("nxs", "nxr", 0)).position(0, 0))[1] - 3.3, "north\nexit", fontsize=7,
                        va="top", ha="right", color=INK)
            ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#BBBBBB")
            if r == 0:
                ax.set_title({"entry": "Entry", "mid": "Middle", "exit": "Exit"}[pos], fontsize=9, pad=3)
            if c == 0:
                ax.set_ylabel(rlab, fontsize=8, labelpad=4)
    h = [Line2D([], [], marker="s", ms=6, mfc=EGO, mec=INK, mew=0.5, ls="none", label="Ego vehicle (DT)"),
         Line2D([], [], marker="o", ms=8, mfc=HAZ, mec=HAZ, mew=0.9, ls="none", label="Injected abnormal vehicle"),
         Line2D([], [], marker="s", ms=6, mfc=OTHER, mec=INK, mew=0.5, ls="none", label="Other traffic (IDM)"),
         Line2D([], [], color=EGO, lw=1.0, ls=(0, (3.0, 1.6)), label="Ego route (S entry → N exit)"),
         Line2D([], [], marker="o", ms=3.5, mfc="white", mec=INK, mew=0.6, ls="none", label="Merge / exit point")]
    h[1] = (Line2D([], [], marker="o", ms=9, mfc="none", mec=HAZ, mew=0.9, ls="none"),
            Line2D([], [], marker="s", ms=4, mfc=HAZ, mec=INK, mew=0.5, ls="none"))
    labels = ["Ego vehicle (DT)", "Injected abnormal vehicle", "Other traffic (IDM)", "Ego route (S entry → N exit)",
              "Merge / exit point"]
    fig.legend(h, labels, loc="lower center", ncol=5, frameon=False, fontsize=7.5, handletextpad=0.4, columnspacing=1.1,
               bbox_to_anchor=(0.535, 0.0), handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=1)})
    fig.suptitle("Roundabout Out-of-Distribution Driving Scenarios", fontsize=10, y=0.985)
    for ext, kw in (("pdf", {}), ("jpg", {"dpi": 600, "pil_kwargs": {"quality": 95}})):
        fig.savefig(f"{OUT}/roundabout_ood_scenarios.{ext}", **kw)
    print("written", OUT, "seed", SEED)


if __name__ == "__main__":
    main()
