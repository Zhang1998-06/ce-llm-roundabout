"""Replay run_hazard.episode('llm','stall_mid',500) with PROMPT_VER=4 and record the exact Ollama traffic + executed actions."""
import os, sys, json
os.environ["PROMPT_VER"] = "4"
ROOT = "/media/zhang/990/roudabout/CE-llms-roundabout/ce_vlm_experiment"
sys.path.insert(0, ROOT + "/scripts"); sys.path.insert(0, ROOT + "/src"); sys.path.insert(0, ROOT); os.chdir(ROOT)
import run_hazard as R, coc_supervisor as C, hazards as H
calls, steps, starts = [], [], []
_post = C.requests.post
def post(url, json=None, **kw):
    r = _post(url, json=json, **kw); calls.append(dict(request=json, response=r.json())); return r
C.requests.post = post
_make = H.make_env
def make(scen):
    env = _make(scen); _step = env.step
    def step(a):
        steps.append(int(a)); return _step(a)
    env.step = step; return env
H.make_env = make
_start = H.BypassSkill.start
def start(self, ob):
    starts.append(len(steps)); return _start(self, ob)
H.BypassSkill.start = start
ep, decs = R.episode("llm", "stall_mid", 500)
json.dump(dict(episode=ep, decisions=decs, calls=calls, executed=steps, skill_start_t=starts),
          open(sys.argv[1], "w"), indent=1, default=str)
print(ep); print("skill start at t =", starts); print("executed:", [R.ACTIONS[a] for a in steps])
