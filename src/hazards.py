import os
"""Long-tail hazard scenarios on roundabout-v0: a broken-down (stationary) vehicle on the ego's ring route."""
import gymnasium as gym
import numpy as np

SCENARIOS = {
    "nominal": None,
    "stall_entry": (("se", "ex", 1), 12.0),   # just after the ego enters the ring (outer lane)
    "stall_mid": (("ex", "ee", 1), 10.0),     # mid-ring, outer lane
    "stall_exit": (("ee", "nx", 1), 8.0),     # outer lane shortly before the north exit
    # second hazard type (generalisation test, not used for prompt design): a crawling vehicle in the ego's lane
    "slow_entry": (("se", "ex", 1), 12.0, 3.0),
    "slow_mid": (("ex", "ee", 1), 10.0, 3.0),
    "slow_exit": (("ee", "nx", 1), 8.0, 3.0),
    # v6 context-contrast scenarios: the same stalled vehicle, but NOT on the ego's route (no intervention needed)
    "stall_mid_inner": (("ex", "ee", 0), 10.0),     # same ring position as stall_mid, inner lane (the ego stays outer)
    "stall_past_exit": (("nx", "ne", 1), 9.6),      # outer lane, 9.6 m after the ego's exit (stall_exit mirrored)
    "stall_upstream": (("sx", "se", 1), 8.0),       # outer lane, before the ego's entry point (behind the ego)
}


class Stall(gym.Wrapper):
    """After reset, park a stationary vehicle at (lane, s) and drop traffic overlapping it (8 m)."""

    def __init__(self, env, spec):
        super().__init__(env)
        self.spec_ = spec

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        if self.spec_ is not None:
            from highway_env.vehicle.kinematics import Vehicle
            from highway_env.vehicle.behavior import IDMVehicle
            lane_id, s, v_slow = (tuple(self.spec_) + (0.0,))[:3]
            u = self.env.unwrapped
            lane = u.road.network.get_lane(lane_id)
            p = lane.position(s, 0)
            u.road.vehicles = [v for v in u.road.vehicles if v is u.vehicle or np.linalg.norm(v.position - p) > 8]
            if v_slow > 0:                      # crawling vehicle that follows its lane, never changes lane
                veh = IDMVehicle(u.road, p, lane.heading_at(s), v_slow, target_lane_index=lane_id,
                                 target_speed=v_slow, enable_lane_change=False)
                veh.plan_route_to("nxs")
            else:
                veh = Vehicle(u.road, p, lane.heading_at(s), 0)
            u.road.vehicles.append(veh)
            obs = u.observation_type.observe()
        return obs, info


def make_env(scenario):
    return Stall(gym.make("roundabout-v0"), SCENARIOS[scenario])


def blocked_ahead(env, d_max=25.0):
    """Reference rule (hazard-specific, hand-written): stationary vehicle ahead in the ego's lane."""
    u = env.unwrapped
    e = u.vehicle
    li = e.lane_index
    fwd = np.array([np.cos(e.heading), np.sin(e.heading)])
    for v in u.road.vehicles:
        if v is e or v.speed > 0.5:
            continue
        if v.lane_index == li or (v.lane_index[2] == li[2] and v.lane_index[0] == li[1]):
            d = v.position - e.position
            if np.linalg.norm(d) < d_max and np.dot(d, fwd) > 0:
                return True
    return False


def stationary_ahead_on_path(env, d_max=40.0):
    """Stationary vehicle on the outer ring lane ahead of the ego within d_max (oracle trigger for tests)."""
    u = env.unwrapped
    e = u.vehicle
    fwd = np.array([np.cos(e.heading), np.sin(e.heading)])
    for v in u.road.vehicles:
        if v is e or v.speed > 0.5:
            continue
        r = float(np.hypot(*v.position))
        d = v.position - e.position
        if 22 < r <= 27 and np.linalg.norm(d) < d_max and np.dot(d, fwd) > 0:
            return v
    return None


class BypassSkill:
    """Generic lane-bypass maneuver executor (temporally extended option).
    While active: approach at a moderate speed, change to the inner ring lane as soon as the gap beside is free,
    never brake to a standstill (a lane change needs speed), finish once the obstacle is behind the ego."""

    def __init__(self, v_target=9.0, gap=10.0, slow_on_access=True):
        self.v_target, self.gap, self.slow_on_access = v_target, gap, slow_on_access
        self.active, self.obstacle, self.done_change = False, None, False

    def start(self, obstacle):
        self.active, self.obstacle, self.done_change = True, obstacle, False

    def stop(self):
        self.active, self.obstacle, self.done_change = False, None, False

    def act(self, env, a_dt):
        """Only takes over what the maneuver needs; longitudinal control stays with the policy otherwise."""
        import coc_supervisor as C
        u = env.unwrapped
        e = u.vehicle
        if self.obstacle is not None:
            fwd = np.array([np.cos(e.heading), np.sin(e.heading)])
            if np.dot(self.obstacle.position - e.position, fwd) < -3:      # obstacle passed
                self.stop()
                return a_dt
        ts = float(getattr(e, "target_speed", e.speed))                     # target speed levels 0 / 8 / 16 m/s

        def keep(a):                      # policy action, minus lane changes and minus stopping
            if a in (0, 2):
                return 1
            if a == 4 and ts <= 8:
                return 1
            return 3 if ts < 1 else a

        lane = C._ring_lane(e) if C._where(e).startswith("ring") else None
        if lane is None or self.done_change:
            return keep(a_dt)                                                # access road / already in the free lane
        other = "inner" if lane == "outer" else "outer"
        th, r = C._ring_angle(e.position), float(np.hypot(*e.position))
        for v in u.road.vehicles:
            if v is e or C._ring_lane(v) != other or not C._where(v).startswith("ring"):
                continue
            g = r * ((th - C._ring_angle(v.position) + np.pi) % (2 * np.pi) - np.pi)
            if -self.gap <= g <= self.gap * 0.6:
                return keep(a_dt)                                            # wait for a gap, keep rolling
        self.done_change = True
        return 0 if other == "inner" else 2                                  # LANE_LEFT / LANE_RIGHT


# ---------------------------------------------------------------- v2 executor: same maneuver + longitudinal safety
def _lane_leader(env, lane=None, ignore=()):
    """Nearest vehicle ahead in the ego's lane (or in ring lane `lane`), as (gap m, vehicle) or None.
    Ring/access: arc gap along the path; exit road: straight-line projection."""
    import coc_supervisor as C
    u = env.unwrapped
    e = u.vehicle
    where = C._where(e)
    best = None
    if lane is not None and where.startswith("ring"):
        r, th = float(np.hypot(*e.position)), C._ring_angle(e.position)
        for v in u.road.vehicles:
            if v is e or v in ignore or C._ring_lane(v) != lane or not C._where(v).startswith("ring"):
                continue
            g = r * ((th - C._ring_angle(v.position) + np.pi) % (2 * np.pi) - np.pi)
            if 0 < g <= 60 and (best is None or g < best[0]):
                best = (g, v)
        return best
    if where.startswith("ring") or where.startswith("south access"):
        pa = [x for x in C.path_ahead(env, 60.0) if x[1] not in ignore]
        if where.startswith("ring") and CORRIDOR:                         # also vehicles cutting into the ego's corridor
            r, th = float(np.hypot(*e.position)), C._ring_angle(e.position)
            for v in u.road.vehicles:
                if v is e or v in ignore or any(v is w for _, w in pa) or not C._where(v).startswith("ring"):
                    continue
                if abs(float(np.hypot(*v.position)) - r) < 2.6:
                    g = r * ((th - C._ring_angle(v.position) + np.pi) % (2 * np.pi) - np.pi)
                    if 0 < g <= 60:
                        pa.append((g, v))
            pa.sort(key=lambda x: x[0])
        if where.startswith("ring outer") and C._ring_angle(e.position) < np.deg2rad(-20):   # last segment before the north exit
            for v in u.road.vehicles:                                       # vehicles already on the exit road are ahead too
                if v is not e and v not in ignore and C._where(v).startswith("north exit"):
                    d = float(np.linalg.norm(v.position - e.position))
                    if d <= 60:
                        pa.append((d, v))
            pa.sort(key=lambda x: x[0])
        return pa[0] if pa else None
    fwd = np.array([np.cos(e.heading), np.sin(e.heading)])                  # exit road (straight)
    for v in u.road.vehicles:
        if v is e or v in ignore:
            continue
        d = v.position - e.position
        lon, lat = float(np.dot(d, fwd)), abs(float(np.cross(fwd, d)))
        if 0 < lon <= 60 and lat < 2.5 and (best is None or lon < best[0]):
            best = (lon, v)
    return best


def safe_level(gap, v_lead, s0=7.0, T=1.5, b=5.0):
    """Highest target-speed level (16/8/0) that can still stop behind the leader: gap >= s0 + T*dv + dv^2/(2b)."""
    for L in (16.0, 8.0):
        dv = max(0.0, L - v_lead)
        if gap >= s0 + T * dv + dv * dv / (2 * b):
            return L
    return 0.0


def longitudinal(a, ts, cap):
    """Combine a policy action with a speed cap: never above the cap, otherwise keep the policy's choice."""
    if ts > cap + 1:
        return 4                                    # SLOWER (one level per step, down to a stop if needed)
    if a == 3 and ts + 8 > cap + 1:
        return 1                                    # FASTER would exceed the cap
    return a


class SafeBypassSkill:
    """Bypass maneuver v2. Same trigger and lane-change idea as BypassSkill, plus
    (1) car-following safety while active: brake (also to a stop) behind the leader / obstacle,
    (2) lane change only with a speed-dependent gap in the target lane (front and rear),
    (3) ends only when the obstacle is > end_gap behind; stays active on the exit road."""

    def __init__(self, end_gap=8.0):
        self.end_gap = end_gap
        self.active, self.obstacle, self.done_change, self.target = False, None, False, None

    def start(self, obstacle):
        self.active, self.obstacle, self.done_change, self.target = True, obstacle, False, None

    def stop(self):
        self.active, self.obstacle, self.done_change, self.target = False, None, False, None

    def _gap_free(self, env, other):
        import coc_supervisor as C
        u = env.unwrapped
        e = u.vehicle
        r, th = float(np.hypot(*e.position)), C._ring_angle(e.position)
        for v in u.road.vehicles:
            if v is e or C._ring_lane(v) != other or not C._where(v).startswith("ring"):
                continue
            g = r * ((th - C._ring_angle(v.position) + np.pi) % (2 * np.pi) - np.pi)
            if g >= 0 and g < 6.0 + 1.0 * max(0.0, e.speed - v.speed):          # vehicle ahead in target lane
                return False
            if g < 0 and -g < 8.0 + 2.0 * max(0.0, v.speed - e.speed):         # vehicle behind, closing in
                return False
        return True

    def act(self, env, a_dt):
        import coc_supervisor as C
        u = env.unwrapped
        e = u.vehicle
        fwd = np.array([np.cos(e.heading), np.sin(e.heading)])
        if self.obstacle is not None and np.dot(self.obstacle.position - e.position, fwd) < -self.end_gap:
            self.stop()
            return a_dt
        ts = float(getattr(e, "target_speed", e.speed))
        a = 1 if a_dt in (0, 2) else a_dt                                        # lane changes only by the maneuver
        on_ring = C._where(e).startswith("ring")
        if self.done_change:                                                     # in the target lane: follow its leader
            ld = _lane_leader(env, self.target, ignore=(self.obstacle,)) if on_ring else _lane_leader(env, ignore=(self.obstacle,))
            cap = safe_level(ld[0], ld[1].speed) if ld else 16.0
            act = longitudinal(a, ts, cap)
            return 3 if (act != 4 and ts < 1 and cap > 0) else act               # restart from a stop
        ld = _lane_leader(env)
        cap = safe_level(ld[0], ld[1].speed) if ld else 16.0
        if on_ring:
            lane = C._ring_lane(e)
            other = "inner" if lane == "outer" else "outer"
            if self._gap_free(env, other):
                self.done_change, self.target = True, other
                return 0 if other == "inner" else 2                              # LANE_LEFT / LANE_RIGHT
        return longitudinal(a, ts, cap)


def acc_action(env, a_dt):
    """Non-LLM control baseline: the same car-following cap applied to the policy at every step."""
    e = env.unwrapped.vehicle
    ts = float(getattr(e, "target_speed", e.speed))
    ld = _lane_leader(env)
    cap = safe_level(ld[0], ld[1].speed) if ld else 16.0
    return longitudinal(a_dt, ts, cap)


# ---------------------------------------------------------------- v3 executor: smooth longitudinal control
# Two facts of this simulator drive the v3 design (Phase 4 diagnosis):
#  - SLOWER picks the level below the *current* speed: at 12 m/s it sets the target to 0 (a full stop, ~13 m/s^2 at first);
#  - the IDM traffic brakes gently (comfortable deceleration 1 m/s^2), so a hard-braking ego is rear-ended.
# IDLE keeps the target speed, so the executor sets the target speed itself and sends IDLE: rate-limited braking.
DEC_STEP, ACC_STEP = 1.0, 2.0                             # target-speed change per 0.5 s policy step [m/s]
CORRIDOR = os.environ.get("CORRIDOR", "1") == "1"         # leader = any vehicle within 2.6 m laterally (cut-ins), not only same lane
ENTRY_HOLD = os.environ.get("ENTRY_HOLD", "1") == "1"    # obstacle within 25 m after the merge: wait before entering
MERGE_GATE = False                                        # v3a gated the ring entry; it made the ego creep into ring traffic


KP_A = 1 / 0.6                                              # ControlledVehicle speed-tracking gain
LAG_COMP = float(os.environ.get("LAG_COMP", "1.0"))   # 1.6 tried: harder braking -> more rear-end and at-fault crashes


S0, T_GAP = float(os.environ.get("S0", "10")), float(os.environ.get("T_GAP", "1.0"))   # stop margin (centre gap) and time gap


def brake_target(e, leaders, s0=None, a_comf=1.5, a_max=6.0):
    """Target speed that brings the ego to v_lead (s0 + T*v) behind the most critical leader:
    decel = (v^2 - vl^2) / 2(gap - s0 - T*v). The time-gap term covers the speed-tracking lag and leaders that stop
    abruptly (e.g. traffic crashing into the hazard ahead, which IDM traffic often does in this simulator)."""
    s0 = S0 if s0 is None else s0
    a_req = 0.0
    for g, v in leaders:
        if e.speed > v.speed or g < s0:
            a_req = max(a_req, (e.speed ** 2 - v.speed ** 2) / (2 * max(0.5, g - s0 - T_GAP * e.speed)), 9.9 if g < s0 else 0)
    if a_req < 0.5:
        return None
    a_cmd = min(a_max, max(a_comf, 1.3 * a_req))
    return max(0.0, e.speed - LAG_COMP * a_cmd / KP_A)                     # LAG_COMP: the speed loop only reaches ~2/3 of it


def safe_speed(gap, v_lead, s0=None, b=2.0):
    """Highest speed that can still stop (s0 + T*v) behind the leader with deceleration b (leader keeps v_lead)."""
    s0 = S0 if s0 is None else s0
    d = max(0.0, gap - s0)
    return v_lead + max(0.0, -b * T_GAP + np.sqrt((b * T_GAP) ** 2 + 2 * b * d))      # solves v^2 / 2b + T v = d


def merge_blocked(env, horizon=3.0):
    """Ego on the access road: does outer-ring traffic reach the merge point before the ego is safely in (t_ego + horizon)?"""
    import coc_supervisor as C
    u = env.unwrapped
    e = u.vehicle
    th_en, p_en = C._entry_angle(env)
    d = float(np.linalg.norm(p_en - e.position))
    v, a = max(float(e.speed), 0.0), 2.0
    t_e = (-v + np.sqrt(v * v + 2 * a * d)) / a                         # time to the merge if accelerating gently
    for w in u.road.vehicles:
        if w is e or C._ring_lane(w) != "outer" or not C._where(w).startswith("ring"):
            continue
        up = 24.0 * ((C._ring_angle(w.position) - th_en) % (2 * np.pi))  # arc distance upstream of the merge
        if up > 80:
            continue
        if up < 8 or up / max(w.speed, 0.5) < t_e + horizon:
            return True, d
    return False, d


def obstacle_gap(env, ob):
    """(gap, obstacle) if the designated obstacle is in the ego's path ahead (ring / access / exit road), else None."""
    import coc_supervisor as C
    if ob is None:
        return None
    for g, v in C.path_ahead_x(env, 60.0):
        if v is ob:
            return (g, v)
    return None


class SmoothBypassSkill(SafeBypassSkill):
    """Bypass maneuver v3: v2 logic, but the executor owns the target speed (rate-limited, IDM-like safe speed),
    gates the ring entry on a gap in the outer lane, and keeps the obstacle as a leader until the lane change is done."""

    def _cmd(self, env, a_lat, v_des, leaders=()):
        e = env.unwrapped.vehicle
        ts = float(getattr(e, "target_speed", e.speed))
        bt = brake_target(e, leaders)
        if bt is not None:                                                    # braking needed: decel from the gap
            e.target_speed = float(min(ts, bt))
        else:
            e.target_speed = float(np.clip(v_des, min(ts, e.speed) - DEC_STEP, min(16.0, ts + ACC_STEP)))
        e.target_speed = max(0.0, e.target_speed)
        return a_lat                                                          # IDLE or a lane change; target speed kept

    def _obstacle_near_entry(self, env, d_max=25.0):
        import coc_supervisor as C
        if self.obstacle is None or C._ring_lane(self.obstacle) != "outer" or self.obstacle.speed >= 0.5:
            return False                                                      # only a stationary obstacle blocks the entry
        th_en, _ = C._entry_angle(env)
        return 24.0 * ((th_en - C._ring_angle(self.obstacle.position)) % (2 * np.pi)) < d_max

    def _entry_hold(self, env, a_dt):
        """Hold before the merge until the outer lane upstream AND the inner lane at the entry are free, then go in
        slowly (the ring logic then changes to the inner lane at once)."""
        import coc_supervisor as C
        u = env.unwrapped
        e = u.vehicle
        th_en, p_en = C._entry_angle(env)
        blocked, d = merge_blocked(env, horizon=4.0)
        for w in u.road.vehicles:                                             # inner lane near the entry
            if w is e or C._ring_lane(w) != "inner" or not C._where(w).startswith("ring"):
                continue
            g = 20.0 * ((C._ring_angle(w.position) - th_en + np.pi) % (2 * np.pi) - np.pi)   # > 0: upstream
            if -8 < g < 8 + 2.5 * w.speed:
                blocked = True
        stop = type("Stop", (), {"speed": 0.0})()
        og = obstacle_gap(env, self.obstacle)
        leaders = [og] if og else []
        if blocked:
            leaders.append((max(0.0, d - 4.0) + S0, stop))                    # stop ~4 m before the merge point
        ld = _lane_leader(env)
        if ld:
            leaders.append(ld)
        v_des = min([16.0] + [safe_speed(g, v.speed) for g, v in leaders])
        return self._cmd(env, 1, v_des, leaders)

    def _changed(self, env):
        import coc_supervisor as C
        r = float(np.hypot(*env.unwrapped.vehicle.position))
        return C._ring_lane(env.unwrapped.vehicle) == self.target and abs(r - (20.0 if self.target == "inner" else 24.0)) < 1.0

    def act(self, env, a_dt):
        import coc_supervisor as C
        u = env.unwrapped
        e = u.vehicle
        fwd = np.array([np.cos(e.heading), np.sin(e.heading)])
        if self.obstacle is not None and np.dot(self.obstacle.position - e.position, fwd) < -self.end_gap:
            self.stop()
            return a_dt
        where = C._where(e)
        on_ring = where.startswith("ring")
        if where.startswith("south access") and ENTRY_HOLD and self._obstacle_near_entry(env):
            return self._entry_hold(env, a_dt)                                # obstacle right after the merge: wait first
        if where.startswith("south access") and not MERGE_GATE:
            return 1 if a_dt in (0, 2) else a_dt                              # the policy's own (trained) ring entry
        leaders = []
        if self.done_change and on_ring:
            ld = _lane_leader(env, self.target, ignore=(self.obstacle,))
            if ld:
                leaders.append(ld)
            if not self._changed(env):                                        # still beside / behind the obstacle
                ld0 = _lane_leader(env)
                if ld0:
                    leaders.append(ld0)
        else:
            ld = _lane_leader(env)
            if ld:
                leaders.append(ld)
            og = obstacle_gap(env, self.obstacle)                             # the designated obstacle is always a leader
            if og:
                leaders.append(og)
        v_des = min([16.0] + [safe_speed(g, v.speed) for g, v in leaders])
        if where.startswith("south access"):
            blocked, d = merge_blocked(env)
            if blocked:                                                       # stop before the merge point
                v_des = min(v_des, safe_speed(d + 4.0, 0.0))
                leaders.append((d + 4.0, type("Stop", (), {"speed": 0.0})()))
        if on_ring and not self.done_change:
            lane = C._ring_lane(e)
            other = "inner" if lane == "outer" else "outer"
            if e.speed > 1.0 and self._gap_free(env, other):
                self.done_change, self.target = True, other
                return self._cmd(env, 0 if other == "inner" else 2, v_des, leaders)
            if e.speed <= 1.0 and self._gap_free(env, other):
                v_des = max(v_des, 3.0)                                       # creep to be able to steer
                leaders = []
        return self._cmd(env, 1, v_des, leaders)

    def _gap_free(self, env, other):
        import coc_supervisor as C
        u = env.unwrapped
        e = u.vehicle
        r, th = float(np.hypot(*e.position)), C._ring_angle(e.position)
        for v in u.road.vehicles:
            if v is e or C._ring_lane(v) != other or not C._where(v).startswith("ring"):
                continue
            g = r * ((th - C._ring_angle(v.position) + np.pi) % (2 * np.pi) - np.pi)
            dv = v.speed - e.speed
            if g >= 0 and g < 7.0 + 1.5 * max(0.0, -dv):                    # ahead in the target lane
                return False
            if g < 0 and -g < 9.0 + 1.5 * max(0.0, dv) + max(0.0, dv) ** 2 / 2.0:   # behind, closing (follower brakes ~1 m/s^2)
                return False
        return True


def smooth_acc_action(env, a_dt):
    """Non-LLM control baseline for v3: the same rate-limited safe-speed cap on the policy at every step (no lane change)."""
    e = env.unwrapped.vehicle
    ts = float(getattr(e, "target_speed", e.speed))
    ld = _lane_leader(env)
    bt = brake_target(e, [ld]) if ld else None
    if bt is None:
        v_cap = safe_speed(ld[0], ld[1].speed) if ld else 16.0
        return 1 if (a_dt == 3 and ts + 8 > v_cap + 1) else a_dt
    e.target_speed = float(min(ts, bt))
    return 1 if a_dt in (3, 4) else a_dt


class FollowSkill(SmoothBypassSkill):
    """Follow maneuver: stay behind the abnormal vehicle at a safe distance (smooth braking, also to a stop),
    no lane change. For causes a lane change cannot solve (e.g. a slow vehicle at the single-lane exit).
    Ends when the obstacle is no longer ahead within 60 m."""

    def act(self, env, a_dt):
        import coc_supervisor as C
        u = env.unwrapped
        e = u.vehicle
        fwd = np.array([np.cos(e.heading), np.sin(e.heading)])
        d = self.obstacle.position - e.position if self.obstacle is not None else None
        if d is None or np.dot(d, fwd) < -self.end_gap or np.linalg.norm(d) > 60:
            self.stop()
            return a_dt
        if C._where(e).startswith("south access"):
            return 1 if a_dt in (0, 2) else a_dt
        ld = _lane_leader(env)
        leaders = [ld] if ld else []
        og = obstacle_gap(env, self.obstacle)                                 # the designated obstacle is always a leader
        if og:
            leaders.append(og)
        v_des = min([16.0] + [safe_speed(g, v.speed) for g, v in leaders])
        return self._cmd(env, 1, v_des, leaders)
