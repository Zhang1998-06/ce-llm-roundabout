"""Current-state BEV image + text description restricted to the DT's observation window.

The window is the OccupancyGrid extent of the DT observation (+-50 m / +-41 m around
the ego, current time step only).  Only vehicles inside it are drawn / listed.
"""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

HX, HY = 50.0, 41.0            # half extents of the observation grid [m]
CENTER = np.zeros(2)


def _box(v):
    c, s = np.cos(v.heading), np.sin(v.heading)
    L, W = v.LENGTH / 2, v.WIDTH / 2
    pts = np.array([[L, W], [L, -W], [-L, -W], [-L, W]])
    return pts @ np.array([[c, s], [-s, c]]) + v.position


def visible(env):
    u = env.unwrapped
    e = u.vehicle
    out = []
    for v in u.road.vehicles:
        if v is e:
            continue
        d = v.position - e.position
        if abs(d[0]) <= HX and abs(d[1]) <= HY:
            out.append(v)
    return out


def render_bev(env, path=None):
    u = env.unwrapped
    e = u.vehicle
    fig, ax = plt.subplots(figsize=(5, 4.1), dpi=100)
    ax.set_facecolor("#e8e8e8")
    for lane in u.road.network.lanes_list():
        s = np.linspace(0, lane.length, 40)
        l = np.array([lane.position(x, -lane.width_at(x) / 2) for x in s])
        r = np.array([lane.position(x, lane.width_at(x) / 2) for x in s])
        ax.add_patch(Polygon(np.vstack([l, r[::-1]]), fc="#555555", ec="none", zorder=1))
    for v in [e] + visible(env):
        ego = v is e
        ax.add_patch(Polygon(_box(v), fc="#1f77b4" if ego else "#e8a33d", ec="white", lw=0.8, zorder=3))
        ax.annotate(f"{'EGO ' if ego else ''}{v.speed:.0f}", v.position, xytext=(4, 2),
                    textcoords="offset points", fontsize=7, zorder=4,
                    color="white" if ego else "black",
                    bbox=dict(fc="#1f77b4" if ego else "white", ec="none", alpha=0.85, pad=0.8))
    ax.set_xlim(e.position[0] - HX, e.position[0] + HX)
    ax.set_ylim(e.position[1] + HY, e.position[1] - HY)      # y grows downward; ego drives "up"
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("BEV (blue = ego, orange = others, numbers = speed m/s; ego drives upward)", fontsize=6)
    fig.tight_layout(pad=0.3)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    png = buf.getvalue()
    if path:
        with open(path, "wb") as f:
            f.write(png)
    return png


def describe(env, proposed_name):
    u = env.unwrapped
    e = u.vehicle
    lines = [f"Ego speed: {e.speed:.1f} m/s.",
             f"Ego distance to roundabout centre: {np.hypot(*e.position):.0f} m "
             f"(ring radius about 20-24 m).",
             f"Baseline DT proposed action: {proposed_name}.",
             "Nearby vehicles (relative to ego; ahead = along ego heading, right = ego's right):"]
    fwd = np.array([np.cos(e.heading), np.sin(e.heading)])
    rgt = np.array([-fwd[1], fwd[0]])
    near = sorted(visible(env), key=lambda v: np.linalg.norm(v.position - e.position))[:6]
    if not near:
        lines.append("  none in sensing range.")
    for v in near:
        d = v.position - e.position
        vr = v.velocity - e.velocity
        on_ring = 18 <= np.hypot(*v.position) <= 30
        lines.append(f"  - {np.dot(d, fwd):+.0f} m ahead, {np.dot(d, rgt):+.0f} m right, "
                     f"speed {v.speed:.0f} m/s, rel. speed along ego heading {np.dot(vr, fwd):+.0f} m/s, "
                     f"{'on roundabout ring' if on_ring else 'not on ring'}")
    return "\n".join(lines)
