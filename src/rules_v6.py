"""v6 mechanism controls and the reference label. All of them read the same facts as the LLM (coc_supervisor.perceive_v6).

- naive_rule: context-free trigger. Any slow (< 5 m/s) vehicle within 40 m in front of the ego (straight-line geometry,
  no road, lane or route reasoning) -> BYPASS it. "A detector plus a hand-written avoidance maneuver, no understanding."
- designed_rule: the experiment designer's own route-aware logic (the "answer key"): a slow vehicle on the ego's own
  route ahead and before its exit -> BYPASS, or FOLLOW_VEHICLE if it is moving and about to take the single-lane exit.
  It encodes the human understanding of these scenarios, so it is an upper reference, not a competitor.
- reference(): the same route-aware logic applied only to the scenario's injected vehicle (privileged identity), used to
  score each supervisor decision offline. It is never shown to any supervisor.
"""
import numpy as np

SLOW = 5.0          # [m/s] well below the 12-16 m/s of ordinary ring traffic
EXIT_ZONE = 20.0    # [m] a moving vehicle this close to the ego's exit will take the single exit lane before the ego


def _on_route_ahead(ego, f):
    """Is vehicle f on the ego's own route ahead (own lane / merge lane, before the ego's exit, or on the exit road)?"""
    if f["road"] == "ring":                                # any sensed vehicle (<= 50 m) in that lane before the ego's exit
        lane_ok = f["lane"] == (ego["lane"] if ego["road"] == "ring" else "outer")
        return lane_ok and 0 < f["along"] < ego.get("exit_ahead", 1e9)
    if "along" in f:                                       # own straight road (access road)
        return 0 < f["along"] <= 50
    if f["road"] == "north exit road" and ego["road"] == "ring":
        return ego.get("exit_ahead", 1e9) < 50
    return False


def _maneuver(ego, f):
    if f["road"] == "north exit road":
        return "FOLLOW_VEHICLE"
    to_exit = ego.get("exit_ahead", 1e9) - f.get("along", 0.0)
    return "FOLLOW_VEHICLE" if (f["speed"] >= 0.5 and to_exit < EXIT_ZONE) else "BYPASS"


def designed_rule(ego, facts):
    c = [f for f in facts if f["speed"] < SLOW and _on_route_ahead(ego, f)]
    if not c:
        return "FOLLOW_POLICY", None
    f = min(c, key=lambda f: f["dist"])
    return _maneuver(ego, f), f["id"]


def naive_rule(env, ego, facts, d_max=40.0):
    e = env.unwrapped.vehicle
    fwd = np.array([np.cos(e.heading), np.sin(e.heading)])
    c = [f for f in facts if f["speed"] < SLOW and f["dist"] <= d_max and np.dot(f["veh"].position - e.position, fwd) > 0]
    if not c:
        return "FOLLOW_POLICY", None
    return "BYPASS", min(c, key=lambda f: f["dist"])["id"]


def reference(ego, facts, hazard):
    """Correct decision w.r.t. the injected vehicle only (privileged): (decision, hazard id or None)."""
    if hazard is None or ego["road"] not in ("ring", "south access road"):
        return "FOLLOW_POLICY", None
    f = next((f for f in facts if f["veh"] is hazard), None)
    if f is None or not _on_route_ahead(ego, f):
        return "FOLLOW_POLICY", None
    return _maneuver(ego, f), f["id"]
