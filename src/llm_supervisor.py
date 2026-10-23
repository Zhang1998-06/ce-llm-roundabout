"""Text-only LLM yield supervisor: kinematic state description -> NORMAL / YIELD (local Ollama)."""
import json
import time

import numpy as np
import requests

SYSTEM = """You supervise a self-driving car (the EGO) that is approaching a roundabout entry. A learned baseline policy drives the car; you only decide whether the ego should hold back before entering.

ROAD GEOMETRY
- The roundabout ring has two lanes (radius about 20 m and 24 m) and circulates in one direction. Vehicles already on the ring have priority over vehicles entering.
- The ego drives on the access road toward the ENTRY POINT, where its lane merges into the ring.
- For every other vehicle you get its position relative to the ego, its speed, and, if it is on the ring, its arc distance along the ring to the entry point (ahead of the entry in the circulation direction) and the time it needs to get there at its current speed. Vehicles with "already past the entry" are downstream and cannot conflict with the ego's entry. Vehicles not on the ring (access roads, exits) are not circulating traffic.
- The ego also gets its own distance and time to the entry point at its current speed (a nominal time: the ego may speed up or slow down).

WHAT YIELD MEANS
YIELD makes the ego hold its current speed instead of accelerating. NORMAL lets the baseline policy act freely. The baseline policy is trained and usually drives well; unnecessary YIELD makes the ego slow and late, so use it only for a real entry conflict.

DECISION RULE
Answer YIELD only if at least one circulating vehicle on the ring would reach the entry point at nearly the same time as the ego, that is, its time to the entry point is within about 2.5 seconds of the ego's time to the entry point (before or after), and that vehicle is not already past the entry.
Answer NORMAL if:
- no ring vehicle is upstream of the entry within 35 m of arc distance, or
- every upstream ring vehicle arrives at the entry more than 2.5 s before or after the ego, or
- the ego is already past the entry point, or
- you are unsure. When unsure, answer NORMAL.
A ring full of vehicles that are far from the entry, or already past it, is NOT a reason to yield.
Base the decision only on the numbers given. Do not invent vehicle motion that is not in the data.

OUTPUT
Return only a JSON object: {"mode": "YIELD" or "NORMAL", "reason": "<one short sentence citing the key numbers>"}"""


def entry_point(env):
    """Ring-entry node: start of the ring lane the ego merges into (se -> ex)."""
    lane = env.unwrapped.road.network.get_lane(("se", "ex", 0))
    p = np.asarray(lane.position(0, 0), dtype=float)
    return p, float(np.arctan2(p[1], p[0]))


def describe_kin(env, proposed_name, max_veh=8):
    u = env.unwrapped
    e = u.vehicle
    ep, th_e = entry_point(env)
    d_e = float(np.linalg.norm(ep - e.position))
    t_e = d_e / max(float(e.speed), 1.0)
    fwd = np.array([np.cos(e.heading), np.sin(e.heading)])
    rgt = np.array([-fwd[1], fwd[0]])
    L = [f"EGO: speed {e.speed:.1f} m/s, distance to entry point {d_e:.0f} m, "
         f"time to entry point at current speed {t_e:.1f} s (speed floor 1 m/s).",
         f"Baseline policy proposes: {proposed_name}.",
         "OTHER VEHICLES (nearest first):"]
    rows = []
    for v in u.road.vehicles:
        if v is e:
            continue
        d = v.position - e.position
        r = float(np.hypot(*v.position))
        if abs(d[0]) > 50 or abs(d[1]) > 41:
            continue
        rows.append((float(np.linalg.norm(d)), v, d, r))
    rows.sort(key=lambda x: x[0])
    for _, v, d, r in rows[:max_veh]:
        s = (f"- {np.dot(d, fwd):+.0f} m ahead, {np.dot(d, rgt):+.0f} m to ego's right, "
             f"speed {v.speed:.1f} m/s, ")
        if 17 <= r <= 27:
            ang = float(np.arctan2(v.position[1], v.position[0]))
            arc = r * ((ang - th_e) % (2 * np.pi))          # circulation: decreasing angle
            if arc > 1.5 * np.pi * r:                        # just past the entry (downstream)
                s += "on ring, already past the entry"
            else:
                s += (f"on ring, {arc:.0f} m of arc before the entry point, "
                      f"reaches it in {arc / max(float(v.speed), 1.0):.1f} s")
        else:
            s += "not on the ring"
        L.append(s)
    if not rows:
        L.append("- none within sensing range.")
    return "\n".join(L)


class LLMSupervisor:
    text_only = True

    def __init__(self, cfg):
        self.url = cfg["ollama_url"].rstrip("/") + "/api/chat"
        self.cfg = cfg

    def ask(self, _png, state_text):
        body = {"model": self.cfg["model"], "stream": False, "format": "json", "keep_alive": "30m",
                "options": {"temperature": 0, "num_predict": self.cfg["num_predict"],
                            "num_ctx": self.cfg["num_ctx"]},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": state_text + "\n\nReturn the JSON object only."}]}
        t0 = time.time(); raw, mode, reason, valid = "", "NORMAL", "", False
        try:
            r = requests.post(self.url, json=body, timeout=120); r.raise_for_status()
            raw = r.json()["message"]["content"]; out = json.loads(raw)
            m = str(out["mode"]).strip().upper()
            if m in ("YIELD", "NORMAL"):
                mode, reason, valid = m, str(out.get("reason", "")), True
        except Exception as ex:
            reason = f"LLM_ERROR: {ex}"
        return dict(mode=mode, reason=reason, valid=valid, latency_s=time.time() - t0, raw=raw)
