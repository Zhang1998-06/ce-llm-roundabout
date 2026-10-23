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
