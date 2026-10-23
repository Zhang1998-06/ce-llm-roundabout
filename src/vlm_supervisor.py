"""Frozen Qwen3-VL (Ollama, local) yield supervisor: BEV image + state text -> NORMAL / YIELD."""
import base64
import json
import time

import requests

PROMPT = """You are a driving decision supervisor for a vehicle approaching a roundabout.

Analyze the current bird's-eye-view traffic scene and the vehicle-state description.
Determine whether the ego vehicle should yield to potentially conflicting circulating traffic.

Return:
YIELD - if the ego is approaching the roundabout and circulating traffic presents a plausible entry conflict.
NORMAL - if there is no clear entry conflict or the ego is no longer approaching the entry.

Prioritize safe task completion and avoid unnecessary stopping.
Use only the provided current observations.
Do not assume future vehicle motions are known.

Return valid JSON containing:
- mode: "YIELD" or "NORMAL"
- reason: a short explanation (under 15 words)

Do not return additional text."""

PROMPT_V2 = """You are a driving decision supervisor for a vehicle approaching a roundabout.
You see a current bird's-eye-view image (blue = ego, orange = other vehicles, numbers = speed in m/s) and a state description.
Vehicles on the roundabout ring circulate and have priority over the ego at the entry.

Answer YIELD only if BOTH hold:
1. a vehicle on the ring is close to the entry point in front of the ego (within about 25 m, upstream of it) and would reach it about when the ego does;
2. the ego is still before the entry.
Otherwise answer NORMAL. Many vehicles on the ring far from the entry is NOT a reason to yield.
Default to NORMAL when unsure; unnecessary stopping is costly.

Examples:
- Ring vehicles are all more than 30 m from the entry or already past it -> {"mode": "NORMAL", "reason": "Ring traffic is far from the entry."}
- A ring vehicle is 10 m before the entry at 16 m/s while the ego is 15 m from it -> {"mode": "YIELD", "reason": "Ring vehicle will reach the entry with the ego."}

Use only the provided current observations. Do not assume future motion.
Return only valid JSON: {"mode": "YIELD" or "NORMAL", "reason": "<under 15 words>"}"""


class VLMSupervisor:
    def __init__(self, cfg):
        self.url = cfg["ollama_url"].rstrip("/") + "/api/chat"
        self.cfg = cfg

    def ask(self, png: bytes, state_text: str):
        """Returns dict(mode, reason, valid, latency_s, raw)."""
        body = {
            "model": self.cfg["model"], "stream": False, "format": "json", "keep_alive": "30m",
            "options": {"temperature": self.cfg["temperature"], "num_predict": self.cfg["num_predict"],
                        "num_ctx": self.cfg["num_ctx"]},
            "messages": [{"role": "user", "content": (PROMPT_V2 if self.cfg.get("prompt") == "v2" else PROMPT) + "\n\n" + state_text,
                          "images": [base64.b64encode(png).decode()]}],
        }
        t0 = time.time()
        raw, mode, reason, valid = "", "NORMAL", "", False
        try:
            r = requests.post(self.url, json=body, timeout=120)
            r.raise_for_status()
            raw = r.json()["message"]["content"]
            out = json.loads(raw)
            m = str(out["mode"]).strip().upper()
            if m in ("YIELD", "NORMAL"):
                mode, reason, valid = m, str(out.get("reason", "")), True
        except Exception as ex:               # failure -> fall back to the DT action
            reason = f"VLM_ERROR: {ex}"
        return dict(mode=mode, reason=reason, valid=valid, latency_s=time.time() - t0, raw=raw)
