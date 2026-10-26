"""Chain-of-Causation (CoC) high-level driver: scene -> critical objects -> causal reasoning -> meta-action.

The (V)LM sits above the frozen DT. It returns one meta-action per query; FOLLOW_POLICY keeps the DT
action, every other meta-action overrides it for that step. Text mode (LLM) gets the kinematic
description only; vision mode (VLM) additionally gets the BEV image.
"""
import base64
import json
import time

import numpy as np
import requests

META = {"FOLLOW_POLICY": None, "CHANGE_TO_INNER_LANE": 0, "CHANGE_TO_OUTER_LANE": 2,
        "SLOW_DOWN": 4, "KEEP_SPEED": 1}

SYSTEM = """You are the high-level reasoning module of a self-driving car (the EGO) driving through a two-lane roundabout.
A learned low-level driving policy controls the car. It was trained only on normal traffic and drives well there,
but it has never seen unusual situations. Each step you may let it drive or override it with one meta-action.

ROAD
- The ego comes from the south access road, enters the ring on the OUTER lane, drives counter-clockwise and leaves at the north exit.
- The ring has an INNER lane (radius about 20 m) and an OUTER lane (radius about 24 m). Both ring lanes lead to the ego's exit.
- Normal traffic on the ring moves at about 12-16 m/s. Vehicles on the ring have priority over entering vehicles.

META-ACTIONS
- FOLLOW_POLICY: let the learned policy drive (correct choice in normal traffic; use it whenever nothing unusual is happening).
- CHANGE_TO_INNER_LANE / CHANGE_TO_OUTER_LANE: lane change on the ring (only possible when the ego is on the ring).
- SLOW_DOWN: brake. KEEP_SPEED: hold the current speed.

THINK IN THIS ORDER (chain of causation), then decide:
1. scene: where is the ego (access road / ring outer lane / ring inner lane / exit) and how fast is it going?
2. critical_objects: which vehicles can affect the ego in the next 3 seconds? Note their lane relative to the ego, the gap, and speed. Pay attention to anything abnormal, for example a vehicle that is stopped or much slower than traffic.
3. causal_reasoning: what will happen if the policy just keeps driving? Is there a cause for a collision or blockage? Which meta-action removes that cause, and is the target lane free (no vehicle within about 10 m beside or just behind the ego in that lane)?
4. meta_action: one of FOLLOW_POLICY, CHANGE_TO_INNER_LANE, CHANGE_TO_OUTER_LANE, SLOW_DOWN, KEEP_SPEED.

Base every statement only on the given data. Keep each field under 30 words.
Return only JSON: {"scene": "...", "critical_objects": "...", "causal_reasoning": "...", "meta_action": "..."}"""

VISION_NOTE = ("\nYou also get a bird's-eye-view image of the current scene: blue box = ego, orange boxes = other vehicles, "
               "labels = speed in m/s; the ego drives upward in the image. Use the image together with the numbers.")


def _ring_lane(v):
    """'inner' / 'outer' / None from the distance to the roundabout centre."""
    r = float(np.hypot(*v.position))
    if 17 <= r <= 22:
        return "inner"
    if 22 < r <= 27:
        return "outer"
    return None


def _where(v):
    li = v.lane_index
    ring = _ring_lane(v)
    if ring and li and li[0] not in ("ser", "ses", "nxs", "nxr"):
        return f"ring {ring} lane"
    if li and li[0] in ("ser", "ses"):
        return "south access road (before the ring)"
    if li and li[0] in ("nx", "nxs", "nxr"):
        return "north exit road"
    return "other road"


def describe(env, proposed_name, max_veh=8):
    u = env.unwrapped
    e = u.vehicle
    fwd = np.array([np.cos(e.heading), np.sin(e.heading)])
    rgt = np.array([-fwd[1], fwd[0]])            # y axis points down on screen, so this is the ego's right
    ego_where = _where(e)
    L = [f"EGO: {ego_where}, speed {e.speed:.1f} m/s.",
         f"Learned policy wants: {proposed_name}.",
         "OTHER VEHICLES (nearest first; 'ahead' is along the ego's heading, negative = behind):"]
    rows = []
    for v in u.road.vehicles:
        if v is e:
            continue
        d = v.position - e.position
        if abs(d[0]) > 50 or abs(d[1]) > 41:
            continue
        rows.append((float(np.linalg.norm(d)), v, d))
    rows.sort(key=lambda x: x[0])
    for dist, v, d in rows[:max_veh]:
        w = _where(v)
        rel = ""
        if w.startswith("ring") and ego_where.startswith("ring"):
            rel = " (SAME lane as ego)" if w == ego_where else " (other ring lane)"
        L.append(f"- {np.dot(d, fwd):+.0f} m ahead, {np.dot(d, rgt):+.0f} m to the right, distance {dist:.0f} m, "
                 f"speed {v.speed:.1f} m/s, on {w}{rel}")
    if not rows:
        L.append("- none within sensing range.")
    return "\n".join(L)


class CoCSupervisor:
    def __init__(self, url, model, vision=False, num_predict=320, num_ctx=4096, version=1):
        self.url = url.rstrip("/") + "/api/chat"
        self.version = version
        self.model, self.vision = model, vision
        self.num_predict, self.num_ctx = num_predict, num_ctx

    def ask(self, state_text, png=None):
        user = {"role": "user", "content": state_text + "\n\nReason step by step in the JSON fields, then give meta_action."}
        if self.vision and png is not None:
            user["images"] = [base64.b64encode(png).decode()]
        body = {"model": self.model, "stream": False, "format": "json", "keep_alive": "60m",
                "options": {"temperature": 0, "num_predict": self.num_predict, "num_ctx": self.num_ctx},
                "messages": [{"role": "system", "content": {1: SYSTEM, 2: SYSTEM_V2, 3: SYSTEM_V3, 4: SYSTEM_V4, 5: SYSTEM_V5}[self.version] + (VISION_NOTE if self.vision else "")}, user]}
        t0 = time.time()
        raw, meta, valid, out = "", "FOLLOW_POLICY", False, {}
        try:
            r = requests.post(self.url, json=body, timeout=180)
            r.raise_for_status()
            raw = r.json()["message"]["content"]
            out = json.loads(raw)
            m = str(out.get("meta_action", "")).strip().upper()
            if m in {4: META_V4, 5: META_V5}.get(self.version, META):
                meta, valid = m, True
                # v3+ grounding: an override must name a cause; "none" -> the policy keeps driving
                if self.version >= 3 and str(out.get("abnormal_cause", "none")).strip().lower().startswith("none"):
                    meta = "FOLLOW_POLICY"
        except Exception as ex:                       # failure -> keep the DT action
            out = {"error": str(ex)}
        return dict(meta_action=meta, action=META.get(meta), valid=valid, latency_s=time.time() - t0,
                    raw=raw, scene=out.get("scene", ""), critical_objects=out.get("critical_objects", ""),
                    causal_reasoning=out.get("causal_reasoning", ""), abnormal_cause=out.get("abnormal_cause", ""))


# ---------------------------------------------------------------- v2: lane-structured perception + few-shot CoC
SYSTEM_V2 = """You are the high-level reasoning module of a self-driving car (the EGO) in a two-lane roundabout.
A learned low-level policy drives the car. It handles normal traffic well (including entering the ring and keeping distance),
but it was never trained on abnormal situations. You override it ONLY when you can name a concrete abnormal cause.

ROAD: the ego enters the ring on the OUTER lane and leaves at the north exit. The INNER lane is on the ego's left.
Both ring lanes lead to the exit. Normal ring traffic moves at 12-16 m/s.

The perception system gives you, per lane, the nearest vehicles with their gap along the road and speed.

META-ACTIONS
- FOLLOW_POLICY: let the learned policy drive. This is the default. Use it whenever traffic is normal, even if
  vehicles are close: the policy already handles normal close traffic.
- CHANGE_TO_INNER_LANE / CHANGE_TO_OUTER_LANE: lane change, only on the ring, and only if the target lane is FREE
  beside the ego (no vehicle between 10 m behind and 10 m ahead in that lane).
- SLOW_DOWN: brake. KEEP_SPEED: hold speed.

ABNORMAL causes are, for example: a vehicle in the ego's lane ahead that is stationary or far slower than traffic,
a vehicle driving the wrong way, a blocked lane. Normal moving traffic is NOT abnormal.

CHAIN OF CAUSATION — fill the JSON fields in order:
1. scene: ego location and speed.
2. critical_objects: vehicles that matter in the next 3 s (lane, gap, speed); say whether any is abnormal.
3. causal_reasoning: if nothing abnormal -> FOLLOW_POLICY. If abnormal: what happens if the policy keeps driving,
   which action removes the cause, and is the target lane free beside the ego?
   If the cause needs a lane change but the target lane is not free -> SLOW_DOWN and wait for a gap.
4. meta_action.

EXAMPLES
- Ego on access road, ring vehicles moving at 16 m/s, nothing stationary -> FOLLOW_POLICY.
- Ego on ring outer lane; inner lane vehicle 4 m beside moving 16 m/s; own lane ahead clear -> FOLLOW_POLICY (normal traffic).
- Ego on ring outer lane; own lane ahead: stationary vehicle 15 m ahead; inner lane free beside -> CHANGE_TO_INNER_LANE.
- Ego on ring outer lane; own lane ahead: stationary vehicle 15 m ahead; inner lane vehicle 3 m beside -> SLOW_DOWN.

Keep each field under 30 words. Return only JSON:
{"scene": "...", "critical_objects": "...", "causal_reasoning": "...", "meta_action": "..."}"""


def _ring_angle(p):
    return float(np.arctan2(p[1], p[0]))


def describe_v2(env, proposed_name, sense=50.0):
    """Lane-structured perception summary (object list as a perception stack would output it)."""
    u = env.unwrapped
    e = u.vehicle
    ego_where = _where(e)
    L = [f"EGO: {ego_where}, speed {e.speed:.1f} m/s. Learned policy wants: {proposed_name}."]
    others = [v for v in u.road.vehicles if v is not e and np.linalg.norm(v.position - e.position) <= sense]

    def tag(v):
        return f"speed {v.speed:.1f} m/s" + (" (STATIONARY)" if v.speed < 0.5 else "")

    if ego_where.startswith("ring"):
        own = _ring_lane(e)
        r_e, th_e = float(np.hypot(*e.position)), _ring_angle(e.position)
        lanes = {"outer": [], "inner": []}
        for v in others:
            if not _where(v).startswith("ring"):
                continue
            d = (th_e - _ring_angle(v.position) + np.pi) % (2 * np.pi) - np.pi   # circulation = decreasing angle
            lanes[_ring_lane(v)].append((r_e * d, v))
        for name in (own, "inner" if own == "outer" else "outer"):
            side = "YOUR LANE" if name == own else ("INNER LANE (on your left)" if name == "inner" else "OUTER LANE (on your right)")
            vs = sorted(lanes[name], key=lambda x: x[0])
            ahead = [(g, v) for g, v in vs if g > 0]
            behind = [(g, v) for g, v in vs if g <= 0]
            s = f"{side} ({name}): "
            s += (f"ahead nearest {ahead[0][0]:.0f} m, {tag(ahead[0][1])}" if ahead else "ahead clear") + "; "
            s += (f"behind nearest {-behind[-1][0]:.0f} m, {tag(behind[-1][1])}" if behind else "behind clear")
            if name != own:
                near = [g for g, _ in vs if -10 <= g <= 10]
                s += "; beside the ego (-10..+10 m): " + (f"OCCUPIED (vehicle at {near[0]:+.0f} m)" if near else "free")
            L.append(s)
    else:
        L.append("Ego is not on the ring. Vehicles in range:")
        for v in sorted(others, key=lambda v: np.linalg.norm(v.position - e.position))[:6]:
            L.append(f"- {np.linalg.norm(v.position - e.position):.0f} m away on {_where(v)}, {tag(v)}")
        stat = [v for v in others if v.speed < 0.5]
        if not stat:
            L.append("No stationary vehicle in range.")
    return "\n".join(L)


# ---------------------------------------------------------------- v3: explicit abnormal_cause + lane-change options
SYSTEM_V3 = """You are the high-level reasoning module of a self-driving car (the EGO) in a two-lane roundabout.
A learned low-level policy drives the car. It handles all normal traffic well: entering the ring, following,
vehicles close beside it in the other lane. It was never trained on abnormal situations.
You override it ONLY when there is a concrete ABNORMAL CAUSE.

ROAD: the ego enters the ring on the OUTER lane and leaves at the north exit. The INNER lane is on the ego's left.
Both ring lanes lead to the exit. Normal ring traffic moves at 12-16 m/s.

WHAT COUNTS AS AN ABNORMAL CAUSE: something IN THE EGO'S PATH AHEAD (its own lane, ahead) that will block it or
that it will hit: e.g. a stationary or broken-down vehicle, an obstacle, a vehicle coming the wrong way.
NOT a cause: normal moving traffic (even close), anything behind the ego, vehicles beside the ego in the other lane,
stationary vehicles that are not ahead in the ego's own lane.

META-ACTIONS
- FOLLOW_POLICY: let the learned policy drive (default; required when abnormal_cause is "none").
- CHANGE_TO_INNER_LANE / CHANGE_TO_OUTER_LANE: only if that lane change is listed as POSSIBLE.
- SLOW_DOWN: brake (use when a cause exists but the needed lane change is NOT possible yet).
- KEEP_SPEED: hold speed.

CHAIN OF CAUSATION — fill the JSON fields in order:
1. scene: ego location and speed.
2. critical_objects: the vehicle(s) in the ego's own lane ahead (gap, speed); copy numbers exactly from the input.
3. abnormal_cause: "none", or the concrete cause in the ego's path ahead (which vehicle, gap, why abnormal).
4. causal_reasoning: if abnormal_cause is "none": "normal traffic, the policy keeps driving". Otherwise: what happens
   if the policy keeps driving, and which action removes the cause (check that the lane change is POSSIBLE).
5. meta_action.

EXAMPLES
- Own lane ahead: vehicle 20 m ahead, 15.8 m/s. Inner lane change NOT possible (vehicle beside). -> abnormal_cause "none" -> FOLLOW_POLICY.
- Own lane ahead clear; a stationary vehicle 15 m behind in the inner lane -> "none" (behind, other lane) -> FOLLOW_POLICY.
- Own lane ahead: STATIONARY vehicle 14 m ahead; inner lane change POSSIBLE -> cause: stationary vehicle blocks my lane -> CHANGE_TO_INNER_LANE.
- Own lane ahead: STATIONARY vehicle 14 m ahead; inner lane change NOT possible -> cause as above -> SLOW_DOWN.

Keep each field under 25 words. Return only JSON:
{"scene": "...", "critical_objects": "...", "abnormal_cause": "...", "causal_reasoning": "...", "meta_action": "..."}"""


def describe_v3(env, proposed_name, sense=50.0):
    u = env.unwrapped
    e = u.vehicle
    ego_where = _where(e)
    L = [f"EGO: {ego_where}, speed {e.speed:.1f} m/s. Learned policy wants: {proposed_name}."]
    others = [v for v in u.road.vehicles if v is not e and np.linalg.norm(v.position - e.position) <= sense]

    def tag(v):
        return f"speed {v.speed:.1f} m/s" + (" (STATIONARY)" if v.speed < 0.5 else "")

    if ego_where.startswith("ring"):
        own = _ring_lane(e)
        other = "inner" if own == "outer" else "outer"
        r_e, th_e = float(np.hypot(*e.position)), _ring_angle(e.position)
        lanes = {"outer": [], "inner": []}
        for v in others:
            if _where(v).startswith("ring"):
                d = (th_e - _ring_angle(v.position) + np.pi) % (2 * np.pi) - np.pi   # circulation = decreasing angle
                lanes[_ring_lane(v)].append((r_e * d, v))
        own_ahead = sorted([(g, v) for g, v in lanes[own] if g > 0], key=lambda x: x[0])[:2]
        L.append(f"YOUR LANE ({own} ring lane), ahead: " +
                 ("; ".join(f"vehicle {g:.0f} m ahead, {tag(v)}" for g, v in own_ahead) if own_ahead else "clear for 50 m"))
        beh = sorted([(g, v) for g, v in lanes[own] if g <= 0], key=lambda x: -x[0])[:1]
        L.append(f"YOUR LANE, behind: " + (f"vehicle {-beh[0][0]:.0f} m behind, {tag(beh[0][1])}" if beh else "clear"))
        near = sorted([(g, v) for g, v in lanes[other] if -10 <= g <= 10], key=lambda x: abs(x[0]))
        oth_ahead = sorted([(g, v) for g, v in lanes[other] if g > 10], key=lambda x: x[0])[:1]
        name = "CHANGE_TO_INNER_LANE" if other == "inner" else "CHANGE_TO_OUTER_LANE"
        L.append(f"LANE CHANGE OPTION {name}: " +
                 (f"NOT possible (vehicle beside the ego at {near[0][0]:+.0f} m in the {other} lane)" if near else "POSSIBLE (gap free)") +
                 (f"; {other} lane further ahead: vehicle {oth_ahead[0][0]:.0f} m ahead, {tag(oth_ahead[0][1])}" if oth_ahead else ""))
    else:
        L.append("Ego is not on the ring (no lane change possible). Vehicles in range:")
        for v in sorted(others, key=lambda v: np.linalg.norm(v.position - e.position))[:6]:
            L.append(f"- {np.linalg.norm(v.position - e.position):.0f} m away on {_where(v)}, {tag(v)}")
    return "\n".join(L)


# ---------------------------------------------------------------- v4: maneuver-level interface (BYPASS_OBSTACLE option)
META_V4 = ("FOLLOW_POLICY", "BYPASS_OBSTACLE", "SLOW_DOWN", "KEEP_SPEED")

SYSTEM_V4 = """You are the high-level reasoning module of a self-driving car (the EGO) in a two-lane roundabout.
A learned low-level policy drives the car. It handles all normal traffic well: entering the ring, following,
vehicles close beside it in the other lane. It was never trained on abnormal situations.
You override it ONLY when there is a concrete ABNORMAL CAUSE.

ROAD: the ego enters the ring on the OUTER lane and leaves at the north exit. The INNER lane is on the ego's left.
Both ring lanes lead to the exit. Normal ring traffic moves at 12-16 m/s.

VEHICLE DYNAMICS (important): speed is set in three levels: 0, 8 and 16 m/s. Ring traffic drives 16 m/s, so a car
that slows down or stops on the ring is likely to be hit from behind. A lane change needs the car to be moving.

WHAT COUNTS AS AN ABNORMAL CAUSE: something IN THE EGO'S PATH AHEAD (its own lane ahead, or the ring lane it is about to
enter) that will block it or that it will hit: e.g. a stationary or broken-down vehicle, an obstacle, a vehicle far
slower than traffic, a vehicle coming the wrong way.
NOT a cause: normal moving traffic (even close), anything behind the ego, vehicles beside the ego in the other lane,
stationary vehicles that are not in the ego's path.

MANEUVERS (meta_action)
- FOLLOW_POLICY: let the learned policy drive (default; required when abnormal_cause is "none").
- BYPASS_OBSTACLE: start the bypass maneuver around the obstacle in the ego's path. It is executed automatically:
  it keeps the car moving, changes to the free ring lane as soon as the gap is free (also right after entering the
  ring), and returns control to the policy once the obstacle is behind. Choose it as soon as an obstacle is in the path,
  even if the lane change is not possible yet.
- SLOW_DOWN: reduce speed one level (never to a stop). Only for a cause that a lane change cannot solve.
- KEEP_SPEED: hold the current speed.

CHAIN OF CAUSATION — fill the JSON fields in order:
1. scene: ego location and speed.
2. critical_objects: the vehicle(s) in the ego's path ahead (gap, speed); copy numbers exactly from the input.
3. abnormal_cause: "none", or the concrete cause in the ego's path (which vehicle, gap, why abnormal).
4. causal_reasoning: if abnormal_cause is "none": "normal traffic, the policy keeps driving". Otherwise: what happens
   if the policy keeps driving, and which maneuver removes the cause.
5. meta_action: one of FOLLOW_POLICY, BYPASS_OBSTACLE, SLOW_DOWN, KEEP_SPEED.

EXAMPLES
- Path ahead: vehicle 20 m ahead, 15.8 m/s; a vehicle beside in the inner lane -> "none" -> FOLLOW_POLICY.
- Path ahead clear; a stationary vehicle 15 m behind in the other lane -> "none" (not in my path) -> FOLLOW_POLICY.
- Path ahead: STATIONARY vehicle 30 m ahead in my lane -> cause: it blocks my lane -> BYPASS_OBSTACLE.
- Ego on the access road; the ring lane I enter has a STATIONARY vehicle 10 m after the entry -> cause: it blocks the
  lane I will enter -> BYPASS_OBSTACLE.

Keep each field under 25 words. Return only JSON:
{"scene": "...", "critical_objects": "...", "abnormal_cause": "...", "causal_reasoning": "...", "meta_action": "..."}"""


def _entry_angle(env):
    lane = env.unwrapped.road.network.get_lane(("se", "ex", 1))
    return _ring_angle(lane.position(0, 0)), lane.position(0, 0)


def path_ahead(env, sense=50.0):
    """Vehicles in the ego's path ahead as (gap m, vehicle), nearest first.
    On the ring: own ring lane ahead. Before the ring: same access lane ahead + outer ring lane just after the entry."""
    u = env.unwrapped
    e = u.vehicle
    out = []
    where = _where(e)
    if where.startswith("ring"):
        own = _ring_lane(e)
        r_e, th_e = float(np.hypot(*e.position)), _ring_angle(e.position)
        for v in u.road.vehicles:
            if v is e or not _where(v).startswith("ring") or _ring_lane(v) != own:
                continue
            g = r_e * ((th_e - _ring_angle(v.position) + np.pi) % (2 * np.pi) - np.pi)
            if 0 < g <= sense:
                out.append((g, v))
    elif where.startswith("south access"):
        th_en, p_en = _entry_angle(env)
        d_en = float(np.linalg.norm(p_en - e.position))
        fwd = np.array([np.cos(e.heading), np.sin(e.heading)])
        for v in u.road.vehicles:
            if v is e:
                continue
            if _where(v).startswith("south access") and v.lane_index == e.lane_index:
                g = float(np.dot(v.position - e.position, fwd))
                if 0 < g <= sense:
                    out.append((g, v))
            elif _where(v).startswith("ring") and _ring_lane(v) == "outer":
                a = 24.0 * ((th_en - _ring_angle(v.position)) % (2 * np.pi))   # arc after the entry
                if a <= 40 and d_en + a <= sense + 20:
                    out.append((d_en + a, v))
    return sorted(out, key=lambda x: x[0])


def describe_v4(env, proposed_name):
    u = env.unwrapped
    e = u.vehicle
    ego_where = _where(e)

    def tag(v):
        return f"speed {v.speed:.1f} m/s" + (" (STATIONARY)" if v.speed < 0.5 else "")

    L = [f"EGO: {ego_where}, speed {e.speed:.1f} m/s. Learned policy wants: {proposed_name}."]
    pa = path_ahead(env)[:2]
    if ego_where.startswith("ring"):
        L.append("YOUR PATH AHEAD (your ring lane): " + ("; ".join(f"vehicle {g:.0f} m ahead, {tag(v)}" for g, v in pa) if pa else "clear for 50 m"))
    else:
        th_en, p_en = _entry_angle(env)
        L.append(f"You will enter the ring OUTER lane in {np.linalg.norm(p_en - e.position):.0f} m.")
        L.append("YOUR PATH AHEAD (access lane and the outer ring lane up to 40 m after the entry): " +
                 ("; ".join(f"vehicle {g:.0f} m ahead along your path, {tag(v)}" for g, v in pa) if pa else "clear"))
    # other traffic (context only)
    rest = [v for v in u.road.vehicles if v is not e and all(v is not w for _, w in pa)
            and np.linalg.norm(v.position - e.position) <= 40]
    rest.sort(key=lambda v: np.linalg.norm(v.position - e.position))
    if rest:
        L.append("OTHER TRAFFIC (not in your path): " + "; ".join(
            f"{np.linalg.norm(v.position - e.position):.0f} m away on {_where(v)}, {tag(v)}" for v in rest[:4]))
    return "\n".join(L)


# ---------------------------------------------------------------- v5: maneuver choice bypass vs follow (+ exit geometry)
META_V5 = ("FOLLOW_POLICY", "BYPASS_OBSTACLE", "FOLLOW_VEHICLE")

SYSTEM_V5 = """You are the high-level reasoning module of a self-driving car (the EGO) in a two-lane roundabout.
A learned low-level policy drives the car. It handles all normal traffic well: entering the ring, following,
vehicles close beside it in the other lane. It was never trained on abnormal situations.
You override it ONLY when there is a concrete ABNORMAL CAUSE.

ROAD: the ego enters the ring on the OUTER lane and must leave at the north exit. The INNER lane is on the ego's left.
Normal ring traffic moves at 12-16 m/s. The NORTH EXIT ROAD IS A SINGLE LANE: every car that leaves there, including
the ego, must pass through it one after another.

WHAT COUNTS AS AN ABNORMAL CAUSE: something IN THE EGO'S PATH AHEAD (its own lane ahead, the ring lane it is about to
enter, or the exit road it will take) that will block it or that it will hit: e.g. a stationary or broken-down vehicle,
an obstacle, a vehicle far slower than traffic, a vehicle coming the wrong way.
NOT a cause: normal moving traffic (even close), anything behind the ego, vehicles beside the ego in the other lane,
stationary vehicles that are not in the ego's path.

MANEUVERS (meta_action). Both maneuvers are executed automatically and brake smoothly when needed.
- FOLLOW_POLICY: let the learned policy drive (default; required when abnormal_cause is "none").
- BYPASS_OBSTACLE: THE DEFAULT MANEUVER for an abnormal vehicle in your path. Passes it in the other ring lane: the car
  keeps moving, changes lane as soon as the gap is free, and returns control to the policy once the obstacle is behind.
  Also for a vehicle "stopped in the exit zone": it does not move, so you pass it and still take the exit.
- FOLLOW_VEHICLE: stay behind the abnormal vehicle at a safe distance (slow down, stop if needed), no lane change.
  ONLY when the abnormal vehicle is "entering the single-lane exit ahead of you" or "on the single-lane exit road":
  it will be in the single exit lane before you, so you can never get in front of it; passing it is impossible.
A vehicle entering the exit at normal speed (12-16 m/s) is NORMAL traffic -> "none".

CHAIN OF CAUSATION — fill the JSON fields in order:
1. scene: ego location, speed, distance to the exit.
2. critical_objects: the vehicle(s) in the ego's path ahead (gap, speed, label); copy numbers exactly.
3. abnormal_cause: "none", or the concrete cause in the ego's path (which vehicle, gap, why abnormal).
4. causal_reasoning: if abnormal_cause is "none": "normal traffic, the policy keeps driving". Otherwise: what happens
   if the policy keeps driving; can the vehicle be passed before the exit; which maneuver removes the cause.
5. meta_action: one of FOLLOW_POLICY, BYPASS_OBSTACLE, FOLLOW_VEHICLE.

EXAMPLES
- Path ahead: vehicle 20 m ahead, 15.8 m/s; a vehicle beside in the inner lane -> "none" -> FOLLOW_POLICY.
- Path ahead clear; a stationary vehicle 15 m behind in the other lane -> "none" (not in my path) -> FOLLOW_POLICY.
- Path ahead: STATIONARY vehicle 30 m ahead in my lane -> cause: it blocks my lane -> BYPASS_OBSTACLE.
- Ego on the access road; the ring lane I enter has a STATIONARY vehicle 10 m after the entry -> cause: it blocks the
  lane I will enter -> BYPASS_OBSTACLE.
- Path ahead: STATIONARY vehicle 25 m ahead, stopped in the exit zone -> cause: it blocks my lane; it does not move, so
  I can pass it and still exit -> BYPASS_OBSTACLE.
- Path ahead: vehicle 30 m ahead at 3.0 m/s (no exit label) -> cause: crawling vehicle in my lane, still on the ring;
  I can pass it -> BYPASS_OBSTACLE.
- Path ahead: vehicle 25 m ahead at 15.2 m/s, entering the single-lane exit ahead of you -> "none" (normal speed) ->
  FOLLOW_POLICY.
- Path ahead: vehicle 18 m ahead at 3.0 m/s, entering the single-lane exit ahead of you -> cause: crawling vehicle
  entering the single exit lane before me; I can never get in front of it -> FOLLOW_VEHICLE.
- Path ahead: vehicle 37 m ahead at 12.5 m/s, entering the single-lane exit ahead of you; vehicle 46 m ahead at 3.0 m/s,
  entering the single-lane exit ahead of you -> cause: the 3.0 m/s vehicle is crawling into the single exit lane (the
  12.5 m/s one is normal); I can never get in front of it -> FOLLOW_VEHICLE.

Keep each field under 25 words. Return only JSON:
{"scene": "...", "critical_objects": "...", "abnormal_cause": "...", "causal_reasoning": "...", "meta_action": "..."}
"""

EXIT_ZONE = 20.0                                             # [m] last part of the ring before the north exit (map zone)
_TH_NX = np.deg2rad(-66.0)                                   # ring angle of the north exit node (outer lane)


def to_exit(env, v):
    """Distance along the route from vehicle v to the start of the north exit road (0 if already on it), or None."""
    w = _where(v)
    if w.startswith("north exit"):
        return 0.0
    if w.startswith("ring"):
        return float(np.hypot(*v.position)) * ((_ring_angle(v.position) - _TH_NX) % (2 * np.pi))
    if w.startswith("south access"):
        th_en, p_en = _entry_angle(env)
        return float(np.linalg.norm(p_en - v.position)) + 24.0 * ((th_en - _TH_NX) % (2 * np.pi))
    return None


def path_ahead_x(env, sense=50.0):
    """path_ahead + vehicles already on the north exit road when the ego is near its exit."""
    u = env.unwrapped
    e = u.vehicle
    out = list(path_ahead(env, sense))
    de = to_exit(env, e)
    if de is not None and de <= sense and not _where(e).startswith("north exit"):
        for v in u.road.vehicles:
            if v is not e and _where(v).startswith("north exit") and all(v is not w for _, w in out):
                d = de + float(np.linalg.norm(v.position - u.road.network.get_lane(("nx", "nxs", 0)).position(0, 0)))
                if d <= sense:
                    out.append((d, v))
    return sorted(out, key=lambda x: x[0])


def describe_v5(env, proposed_name):
    u = env.unwrapped
    e = u.vehicle
    ego_where = _where(e)

    def tag(v):                                                  # motion state + map zone / predicted manoeuvre
        moving = v.speed >= 0.5
        s = f"speed {v.speed:.1f} m/s" + (" (STATIONARY)" if not moving else "")
        dx = to_exit(env, v)
        if dx is not None and dx == 0:
            s += ", on the single-lane exit road"
        elif dx is not None and dx < EXIT_ZONE and _where(v).startswith("ring"):
            s += ", entering the single-lane exit ahead of you" if moving else ", stopped in the exit zone"
        return s

    de = to_exit(env, e)
    L = [f"EGO: {ego_where}, speed {e.speed:.1f} m/s. Learned policy wants: {proposed_name}."]
    if de is not None:
        L.append(f"Your exit (north exit road, single lane) starts {de:.0f} m ahead along your path.")
    pa = path_ahead_x(env)[:2]
    if ego_where.startswith("ring"):
        L.append("YOUR PATH AHEAD (your ring lane, then the exit road): " + ("; ".join(f"vehicle {g:.0f} m ahead, {tag(v)}" for g, v in pa) if pa else "clear for 50 m"))
    else:
        th_en, p_en = _entry_angle(env)
        L.append(f"You will enter the ring OUTER lane in {np.linalg.norm(p_en - e.position):.0f} m.")
        L.append("YOUR PATH AHEAD (access lane and the outer ring lane up to 40 m after the entry): " +
                 ("; ".join(f"vehicle {g:.0f} m ahead along your path, {tag(v)}" for g, v in pa) if pa else "clear"))
    rest = [v for v in u.road.vehicles if v is not e and all(v is not w for _, w in pa)
            and np.linalg.norm(v.position - e.position) <= 40]
    rest.sort(key=lambda v: np.linalg.norm(v.position - e.position))
    if rest:
        L.append("OTHER TRAFFIC (not in your path): " + "; ".join(
            f"{np.linalg.norm(v.position - e.position):.0f} m away on {_where(v)}, speed {v.speed:.1f} m/s"
            + (" (STATIONARY)" if v.speed < 0.5 else "") for v in rest[:4]))
    return "\n".join(L)


# ---------------------------------------------------------------- v6: facts-only observation, task-only prompt, model picks the target
# What changed against v5 (and why): v5 told the model which vehicles are "in your path", labelled them STATIONARY /
# "entering the single-lane exit", and its prompt mapped each test condition to a maneuver with matching examples; the
# executor then picked the slowest path vehicle as the target. v6 removes all of that. The observation lists every vehicle
# near the ego with road, lane, position along the road and speed; deciding whether a vehicle matters for the ego's
# route, what happens if the policy keeps driving, and which vehicle a maneuver refers to is left to the model.
DECISIONS_V6 = ("FOLLOW_POLICY", "BYPASS", "FOLLOW_VEHICLE")
_RING_NODES = {"se", "ex", "ee", "nx", "ne", "wx", "we", "sx"}
_DIRS = {"s": "south", "e": "east", "n": "north", "w": "west"}
_ARC_MERGE_TO_EXIT = 24.0 * np.deg2rad(132.0)                    # outer lane, merge point (66 deg) -> north exit (-66 deg)

SYSTEM_V6 = """You supervise a learned driving policy that controls a car (the EGO) in a two-lane roundabout.
The policy was trained by imitation only on ordinary flowing traffic in this roundabout. It drives well in such traffic
(merging, keeping gaps, following other cars), but it may act wrongly in situations it never saw during training.
At every step you decide whether the policy keeps driving or one of two maneuvers takes over.

ROAD
- The ring has two lanes (inner and outer). Traffic on the ring drives counter-clockwise. Both ring lanes connect to every exit.
- Four access roads lead into the ring and four exit roads lead out of it (south, east, north, west). Each of these roads
  has a single lane; lane changes are only possible on the ring.
- Speed limit 16 m/s.
- The ego's route: south access road -> merge into the ring's outer lane -> counter-clockwise along the ring -> leave on
  the north exit road. Once it has left the ring it does not come back.

INPUT (every 0.5 s): the ego's state, the rest of its route, the policy's proposed action, and every vehicle within
50 m detected by perception, grouped by road and lane (id, position along the road, speed).

DECISIONS
- FOLLOW_POLICY: the policy's proposed action is executed.
- BYPASS with target_id: pass the target vehicle using the other ring lane. The executor changes lane when the gap in that
  lane is safe, drives past the target and hands control back to the policy once the target is behind the ego.
  While the ego is still on the access road, the policy merges first and the lane change happens on the ring.
- FOLLOW_VEHICLE with target_id: stay behind the target vehicle at a safe distance without changing lane, slowing down
  or stopping as needed. Control returns to the policy when the target is no longer ahead of the ego.
The maneuvers handle braking and gap checks. They do not judge the traffic situation; that is your job.
Every intervention has a cost: an unnecessary lane change or slowdown delays the ego and can create conflicts with
other traffic.

YOUR TASK: judge the situation for the ego's own route, in this order:
1. critical_objects: which vehicles, if any, can affect the ego on the part of the road it will actually drive in the
   next few seconds, and why.
2. consequence: what will happen if the policy's proposed action is executed? Will something block the ego or will it
   collide with something?
3. decision and target_id: the decision that lets the ego reach its exit safely and without unnecessary delay.

Return only JSON (each text field at most 25 words):
{"critical_objects": "...", "consequence": "...", "decision": "FOLLOW_POLICY | BYPASS | FOLLOW_VEHICLE", "target_id": <vehicle id or null>}"""


def road_of(v):
    """('ring', 'inner'|'outer') or (road name, None) from the lane index."""
    a, b, k = v.lane_index
    if a in _RING_NODES and b in _RING_NODES:
        return "ring", ("inner" if k == 0 else "outer")
    for n in (a, b):
        if len(n) == 3 and n[0] in _DIRS:
            kind = "access" if n[1] == "e" else "exit"
            return f"{_DIRS[n[0]]} {kind} road", None
    if a in _RING_NODES:                                        # ring node -> exit road (e.g. nx -> nxs)
        return f"{_DIRS[a[0]]} exit road", None
    return "unknown road", None


def _arc(r, th_from, th_to):
    """Signed arc length along the driving direction (counter-clockwise = decreasing angle here) from th_from to th_to."""
    return r * ((th_from - th_to + np.pi) % (2 * np.pi) - np.pi)


def perceive_v6(env, ids, sense=50.0, max_veh=8):
    """Ego facts + per-vehicle facts. `ids` (dict, kept per episode) maps vehicles to stable integer ids by first sighting."""
    u = env.unwrapped
    e = u.vehicle
    er, el = road_of(e)
    th_en, p_en = _entry_angle(env)
    on_ring = er == "ring"
    ref_r, ref_th = (float(np.hypot(*e.position)), _ring_angle(e.position)) if on_ring else (24.0, th_en)
    fwd = np.array([np.cos(e.heading), np.sin(e.heading)])
    ego = dict(road=er, lane=el, speed=float(e.speed), target_speed=float(getattr(e, "target_speed", e.speed)))
    if on_ring:
        ego["exit_ahead"] = float(np.hypot(*e.position)) * ((_ring_angle(e.position) - _TH_NX) % (2 * np.pi))
    elif er == "south access road":
        ego["to_merge"] = float(np.linalg.norm(p_en - e.position))
        ego["exit_ahead"] = _ARC_MERGE_TO_EXIT
    near = sorted([v for v in u.road.vehicles if v is not e and np.linalg.norm(v.position - e.position) <= sense],
                  key=lambda v: np.linalg.norm(v.position - e.position))[:max_veh]
    out = []
    for v in near:
        if id(v) not in ids:
            ids[id(v)] = len(ids) + 1
        r, ln = road_of(v)
        f = dict(id=ids[id(v)], veh=v, road=r, lane=ln, speed=float(v.speed), dist=float(np.linalg.norm(v.position - e.position)))
        if r == "ring":
            f["along"] = _arc(ref_r, ref_th, _ring_angle(v.position))        # > 0: ahead of the ego / past the merge point
        elif r == er:
            f["along"] = float(np.dot(v.position - e.position, fwd))         # same straight road as the ego
        elif r == "north exit road":
            f["from_ring"] = float(np.linalg.norm(v.position - u.road.network.get_lane(("nx", "nxs", 0)).position(0, 0)))
        out.append(f)
    return ego, out


V6_KIN = __import__("os").environ.get("V6_KIN", "1") == "1"            # dev switches; the frozen setting is in the report
V6_LANDMARK = __import__("os").environ.get("V6_LANDMARK", "1") == "1"


def describe_v6(env, proposed_name, ids):
    """Facts grouped by lane (lane association is a perception output); no relevance, motion-state or route labels."""
    ego, vs = perceive_v6(env, ids)
    on_ring = ego["road"] == "ring"
    L = []
    if on_ring:
        L.append(f"EGO: on the ring, {ego['lane']} lane, speed {ego['speed']:.1f} m/s.")
        L.append(f"ROUTE: your exit, the north exit road, leaves the ring {ego['exit_ahead']:.0f} m ahead of you along the ring.")
    elif "to_merge" not in ego:                                   # off the ring and not before it (exit road): not queried
        L.append(f"EGO: on the {ego['road']}, speed {ego['speed']:.1f} m/s.")
        return "\n".join(L), ego, vs
    else:
        L.append(f"EGO: on the south access road, {ego['to_merge']:.0f} m before the point where it merges into the ring's "
                 f"outer lane, speed {ego['speed']:.1f} m/s.")
        L.append(f"ROUTE: merge into the ring's outer lane; your exit, the north exit road, leaves the ring "
                 f"{ego['exit_ahead']:.0f} m past your merge point.")
    L.append(f"POLICY PROPOSES: {proposed_name} (current target speed {ego['target_speed']:.0f} m/s; "
             "target speed levels 0 / 8 / 16 m/s; SLOWER and FASTER change the level, LANE_LEFT = towards the inner lane).")
    ref = "you" if on_ring else "your merge point"
    L.append(f"PERCEPTION, vehicles within 50 m. Ring positions are along the ring in the driving direction, relative to {ref} "
             "(+ = ahead/downstream, - = behind/upstream):")
    groups = []
    if not on_ring:
        groups.append(("Your road (south access road), + = ahead of you", [f for f in vs if "along" in f and f["road"] != "ring"]))
    for ln in ("outer", "inner"):
        name = f"Ring, {ln} lane" + (" (your lane)" if on_ring and ego["lane"] == ln else
                                     " (the lane you merge into)" if not on_ring and ln == "outer" else "")
        groups.append((name, [f for f in vs if f["road"] == "ring" and f["lane"] == ln]))
    groups.append(("Other roads", [f for f in vs if f["road"] != "ring" and "along" not in f]))
    for name, fs in groups:
        if "along" in (fs[0] if fs else {}):
            fs = sorted([f for f in fs if f["along"] >= 0], key=lambda f: f["along"]) + \
                 sorted([f for f in fs if f["along"] < 0], key=lambda f: -f["along"])          # ahead near->far, then behind
        items = []
        for f in fs:
            if "along" in f:
                kin = ""
                if V6_KIN and on_ring:                              # kinematics at current speeds, same for every lane
                    dv = ego["speed"] - f["speed"]
                    if f["along"] > 0 and dv > 0.5:
                        kin = f" (at current speeds you reach its position in {f['along'] / dv:.1f} s)"
                    elif f["along"] < 0 and -dv > 0.5:
                        kin = f" (it reaches your position in {-f['along'] / -dv:.1f} s)"
                items.append(f"id {f['id']} at {f['along']:+.0f} m, {f['speed']:.1f} m/s{kin}")
            elif "from_ring" in f:
                items.append(f"id {f['id']} on the north exit road {f['from_ring']:.0f} m from where it leaves the ring, {f['speed']:.1f} m/s")
            else:
                items.append(f"id {f['id']} on the {f['road']}, {f['dist']:.0f} m from you, {f['speed']:.1f} m/s")
        if V6_LANDMARK and name.startswith("Ring"):                 # 1-D map: upstream -> downstream, with map landmarks
            marks = [(ego["exit_ahead"], f"[YOUR EXIT (north exit road) at {ego['exit_ahead']:+.0f} m]")]
            if not on_ring and "outer" in name:
                marks.append((0.0, "[YOUR MERGE POINT at 0 m]"))
            if on_ring and ego["lane"] in name:
                marks.append((0.0, "[YOU at 0 m]"))
            items = [it for _, it in sorted([(f["along"], it) for f, it in zip(fs, items)] + marks, key=lambda x: x[0])]
        L.append(f"- {name}: " + ("; ".join(items) if items else "none"))
    return "\n".join(L), ego, vs


SYSTEM_V6_NOCO = (SYSTEM_V6.replace("""1. critical_objects: which vehicles, if any, can affect the ego on the part of the road it will actually drive in the
   next few seconds, and why.
2. consequence""", "1. consequence").replace("3. decision and target_id", "2. decision and target_id")
                  .replace('{"critical_objects": "...", "consequence"', '{"consequence"'))


def ask_v6(sup, state_text):
    """One v6 query. Returns decision, target id, consequence, validity, latency and the raw response."""
    body = {"model": sup.model, "stream": False, "format": "json", "keep_alive": "60m",
            "options": {"temperature": 0, "num_predict": 220, "num_ctx": sup.num_ctx},
            "messages": [{"role": "system", "content": SYSTEM_V6_NOCO if __import__("os").environ.get("V6_CO", "0") == "0" else SYSTEM_V6},
                         {"role": "user", "content": state_text}]}
    if "thinking" in sup.model:                                 # reasoning model: thinks first (separate field), then the JSON
        body["think"] = True
        body["options"]["num_predict"] = 4096
    elif "instruct" not in sup.model:
        body["think"] = False                                   # hybrid models (qwen3:8b): answer directly, as the 4B instruct does
    t0 = time.time()
    raw, out, dec, tid, valid, thinking = "", {}, "FOLLOW_POLICY", None, False, ""
    try:
        r = requests.post(sup.url, json=body, timeout=180)
        r.raise_for_status()
        raw = r.json()["message"]["content"]
        thinking = r.json()["message"].get("thinking", "")
        out = json.loads(raw)
        d = str(out.get("decision", "")).strip().upper()
        if d in DECISIONS_V6:
            dec, valid = d, True
            t = out.get("target_id")
            tid = int(t) if t not in (None, "", "null") and str(t).lstrip("-").isdigit() else None
    except Exception as ex:                                     # failure -> keep the DT action
        out = {"error": str(ex)}
    return dict(decision=dec, target_id=tid, valid=valid, latency_s=time.time() - t0, raw=raw,
                consequence=str(out.get("consequence", "")), critical_objects=str(out.get("critical_objects", "")),
                thinking=thinking)
