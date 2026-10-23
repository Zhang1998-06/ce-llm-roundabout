# Iteration log — LLM/VLM + DT on roundabout-v0

Purpose: record every attempt (idea → feasibility analysis → result → diagnosis) so work can resume from here.
Goal (CE requirement): propose an architecture with LLMs/VLMs involved, implement it on the roundabout scenario,
evaluate success / safety / efficiency, and beat the plain baseline. The LLM/VLM itself must be the source of the gain
(a non-LLM safety layer that does the work does not count).

Code: `ce_vlm_experiment/` (DT imported from `uncertaintyDT/CNNDT_baseline`, frozen checkpoint `output2`).
Python: `/home/zhang/anaconda3/envs/HF-DT/bin/python`. Ollama models: `qwen3:4b-instruct-2507-q4_K_M` (text),
`qwen3-vl:2b-instruct-q4_K_M` (vision). Seeds: dev 200–499 (tuning), test 500–799 (final, untouched).

---

## Phase 1 — supervisor on the NOMINAL scenario (YIELD/NORMAL) — FAILED, abandoned

| Attempt | Result |
|---|---|
| VLM 2B YIELD/NORMAL, query every 4 steps, YIELD: FASTER→SLOWER (variants A/B/C) | never better than DT (seeds 0–29) |
| Text LLM 4B kinematic prompt, same intervention | 0 overrides on DT-failure seeds → no change |
| Rule version of the prompt's decision rule, every step, FASTER→IDLE | fixes 4/11 DT failures, breaks 166/289 successes |
| 27 timing-based override variants (dev) | only "IDLE at step 3" net +1; on test seeds 500–799 net −1 |

Baseline DT: dev 289/300 (96.3 %), test 290/300 (96.7 %).

**Diagnosis:** DT failures in nominal traffic are timing accidents. At the decision step the state of failing seeds is
nearly identical to many successful seeds (e.g. 247/248/255 vs 220/287/375/464), so no state-based supervisor
(LLM or rule) can separate them. Nominal scenario leaves no room for an LLM.

## Phase 2 — predictive safety shield (simulator-copy look-ahead) — works, but gain is not from the LLM

Dev 200–499: DT 96.3 % / 3.7 % coll. Shield H=2: 98.7 % / 1.3 %; H=3 (continuation FASTER-first): 99.0 % / 0.3 %;
H=4: 88.3 % (timeouts). Uses a perfect world model (simulator copy). Kept only as a supplementary result; the user
pointed out that with a shield the proper baseline is DT+shield, so the LLM adds nothing here.
Script: scratchpad `shield.py` (not in repo).

## Phase 3 — long-tail hazard: stalled (stationary) vehicle on the ego's ring route

Idea: an LLM/VLM's value is common-sense handling of situations the DT never saw in training.
Scenarios (`src/hazards.py`): `nominal`, `stall_entry` (se→ex outer lane, s=12), `stall_mid` (ex→ee, s=10),
`stall_exit` (ee→nx, s=8). Ego always drives the OUTER ring lane; LANE_LEFT (action 0) moves to the inner lane.

Feasibility (seeds 200–299): plain DT 0 % success / 100 % collision in all three stall positions (95 % nominal).
Hand-written reference rule "stationary vehicle ahead in my lane within 25 m → LANE_LEFT": 54 / 78 / 82 %.
LANE_RIGHT: 0 %. → large headroom, gain must come from hazard recognition + choosing the manoeuvre.

### Iteration 3.1 — CoC prompt, raw relative-coordinate description (`src/coc_supervisor.py` v1)
- Prompt: scene → critical_objects → causal_reasoning → meta_action
  (FOLLOW_POLICY / CHANGE_TO_INNER_LANE / CHANGE_TO_OUTER_LANE / SLOW_DOWN / KEEP_SPEED), queried every step until exit.
- Runner: `scripts/run_hazard.py` (METHODS=dt,rule,llm,vlm).
- Dev 200–224 (25 seeds × 4 scenarios): DT nominal 92 %, stalls 0 %; rule 48 / 80 / 76 %.
- Smoke test: 4B LLM reasons correctly on a clean stall scene (0.6 s/call warm). 2B VLM misplaces the stalled car
  (says inner lane), inconsistent across calls.
- Closed loop (stopped early): LLM breaks nominal episodes (seed 200, 222 crash). VLM 15 s/call, 6/7 invalid JSON.
- **Diagnosis (seed 200 trace):** ego just entered the ring, inner-lane car 6 m beside it; LLM concluded
  "collision risk, change to inner lane" and drove into that car.
  1. raw relative x/y coordinates are mis-read by the 4B model (lane/side/gap not understood);
  2. the model does not verify that the target lane is free;
  3. prompt invites overriding; no explicit "no abnormal cause → FOLLOW_POLICY".
  VLM: output too verbose → truncated at num_predict=320 → invalid JSON; latency too high for every-step querying.

### Iteration 3.2 — feasibility-first: offline single-step decision test (`scripts/build_offline_set.py`, `scripts/offline_eval.py`)
Rationale: closed-loop runs are slow (≈1 s/LLM call, ~15 calls/episode), so every prompt change is first checked
offline. Set = states visited by DT+reference rule, seeds 200–209, all 4 scenarios; label FOLLOW / CHANGE_INNER
(stationary vehicle ≤25 m ahead in own lane, inner lane free ±10 m) / WAIT (same, inner lane occupied).
Targets before going closed-loop: nominal false-override ≈0 %, hazard accuracy ≥90 %.

| Version | Description / prompt | FOLLOW ok | CHANGE_INNER ok | WAIT ok | notes |
|---|---|---|---|---|---|
| v1 | raw relative x/y list, CoC 4 fields | 28/100 | 11/15 | 2/20 | 4B misreads geometry |
| v2 | lane-structured perception (own lane ahead/behind, other lane, beside free/OCCUPIED) + few-shot | 40/100 | 14/15 | 19/20 | nominal false override 30/143: "OCCUPIED" read as a hazard; reasoning says "no abnormal" but action overrides; hallucinated stationary vehicle |
| v3 | + `abnormal_cause` field (must be named; "none" ⇒ FOLLOW_POLICY), lane-change shown as option POSSIBLE/NOT possible, common-sense definition of a cause (in my path ahead; behind/beside/moving traffic is not), no queries on the exit road | 390/447 | 15/15 | 20/20 | **nominal false override 0/125**; remaining "false" overrides only in stall scenarios = early reactions (lane change / slow down while the stalled car is >25 m ahead, slow down on the access road when the entry is blocked) |

Decision: v3 passes the offline gate → closed loop on dev 200–224.

### Iteration 3.2 closed loop (dev 200–224, 25 seeds/scenario)
| Scenario | DT | reference rule | LLM-CoC v3 |
|---|---|---|---|
| nominal | 92 % | 92 % | 92 % (0 overrides) |
| stall_entry | 0 % | 48 % | 0 % |
| stall_mid | 0 % | 80 % | 32 % |
| stall_exit | 0 % | 76 % | 40 % |
Beats DT, nominal untouched, but far below the reference rule although offline single-step accuracy was 100 %.
**Diagnosis (decision traces, `results/hazard_dev_v3/decisions_llm.csv`):** failures are about multi-step execution,
not recognition: (1) SLOW_DOWN repeated while "waiting for a gap" brings the ego to a standstill, after which
CHANGE_TO_INNER_LANE has no effect (stall_mid 203: 9× SLOW_DOWN then 8× CHANGE, timeout); (2) detection flickers
(cause → "none" next step, control returns to DT, collision: stall_entry 200/202); (3) braking on the access road
before a blocked entry disturbs DT's entry and lane change is impossible there.
Env facts found: speed is commanded via 3 target levels (0/8/16 m/s) — one SLOWER from 16 goes to 8, two to 0;
slowing to 8 m/s on the ring gets the ego rear-ended by 16 m/s ring traffic.

Reference-rule failure analysis: stall_entry 10/13 collisions with the stalled car (ego enters at ~13 m/s and is
1–2 steps from it — too late to change lanes), others split between stalled car and inner-lane traffic.

### Iteration 3.3 — feasibility of a maneuver-level interface (`hazards.BypassSkill`, oracle trigger)
Idea: LLM picks a temporally extended maneuver (option) instead of per-step actions; a generic executor holds speed,
changes lane when the adjacent gap is free, ends when the obstacle is behind. Ceiling with oracle trigger (dev 25):
| variant | entry | mid | exit |
|---|---|---|---|
| slow to 8 on access + ring, gap 10, trigger 40 m | 44 % | 0 % | 64 % (rear-ended at low speed) |
| no slowing (VT 17), gap 6 | 44 % | 60 % | 80 % |
| no slowing on access, gap 6, trigger 40 m | 44 % | 60 % | 72 % |
Conclusion: the scenario's physical ceiling for simple bypass strategies is ≈ 45 / 60–80 / 75–80 %; the skill does not
raise the ceiling above the reference rule, but with the LLM as trigger it removes the LLM's execution failures
(flicker, standstill). Expected LLM+skill ≈ skill ceiling with a 40 m trigger, i.e. ~2× the v3 LLM.
→ go: iteration 3.4 = v4 prompt with BYPASS_OBSTACLE maneuver.

### Iteration 3.4 — v4: maneuver-level interface (LLM chooses BYPASS_OBSTACLE, executor carries it out)
- `SYSTEM_V4` / `describe_v4` / `path_ahead` in `src/coc_supervisor.py`: meta-actions FOLLOW_POLICY, BYPASS_OBSTACLE,
  SLOW_DOWN (one level, never to 0), KEEP_SPEED; prompt adds vehicle-dynamics facts (3 speed levels, braking on the ring
  gets you rear-ended, lane change needs speed); perception on the access road reports vehicles on the outer ring lane
  up to 40 m after the entry as "in your path". Obstacle for the executor = slowest of the first 3 path vehicles.
- Offline (v4 labels): nominal false override 0; BYPASS 52/52; remaining "false" overrides = BYPASS when the stalled
  car is in the path behind a moving car (label only checks the first path vehicle) → acceptable.
- Closed loop dev 200–224: nominal 92 % (0 overrides) | stall_entry 56 % | stall_mid 16 % (44 % timeouts) | stall_exit 76 %.
  Oracle-triggered skill (same executor): 92 / 56 / 56 / 76.
- **Diagnosis stall_mid:** the LLM recognised the stalled car at t0 (52 m along the path, earlier than the 40 m oracle
  trigger); the executor then held IDLE on the access road so the ego never accelerated beyond 8 m/s → timeout (11 s).
  Executor design flaw, not an LLM error.

### Iteration 3.5 — executor defers longitudinal control to the DT
`BypassSkill.act`: on the access road and after the lane change it returns the DT action, only removing lane changes
and stopping; while waiting for a gap on the ring it also keeps the DT action (no braking). Oracle-triggered, dev 25:
trigger 40 m: 92 / 48 / 72 / 76; trigger 60 m: 92 / 48 / 72 / 72 → robust to early triggering, no timeouts.

### Generalisation hazard prepared (not used for prompt or executor design)
`slow_entry/mid/exit`: an IDM vehicle crawling at 3 m/s in the ego's outer lane (no lane change). Dev 200–224:
DT 0 / 0 / 0 % (100 % collision). Stationary-vehicle reference rule and oracle skill: 0 / 48 / 0 % — they only detect
speed < 0.5 m/s. The v4 prompt already defines "a vehicle far slower than traffic" as an abnormal cause.

### Iteration 3.5 closed loop — LLM v4 + deferring executor (dev 200–224)
| Scenario | DT | stationary rule | oracle skill | LLM v4 |
|---|---|---|---|---|
| nominal | 92 | 92 | 92 | 92 (0 overrides) |
| stall_entry | 0 | 48 | 48 | 48 |
| stall_mid | 0 | 80 | 72 | 72 |
| stall_exit | 0 | 76 | 76 | 76 |
| slow_entry | 0 | 0 | 0 | **68** |
| slow_mid | 0 | 48 | 48 | **72** |
| slow_exit | 0 | 0 | 0 | 0 |
LLM = oracle-triggered executor on the stalled-car hazard and clearly better than the stationary rule on the unseen
slow-vehicle hazard (no prompt/code change). slow_exit diagnosis: the crawling car is at the north-exit split when the
ego arrives (collisions at node nx with that car); bypassing on the inner lane then exiting crosses its position →
physically not bypassable by any tested strategy. Not tuned (would be test-fitting); reported as a limitation.

VLM 2B offline (v4 prompt + BEV image, 95 states): nominal false override 0, BYPASS 27/27, extra BYPASS calls in stall
scenarios after/around the obstacle, 4 invalid JSON, 1.28 s/call → admitted to the final evaluation without tuning.

## Final evaluation protocol (frozen here, nothing tuned after this point)
Seeds 500–599 for references; LLM on 500–549 and VLM on 500–529 (reduced for run time; comparisons use the same seeds), 7 scenarios, methods dt, rule, skill (references), llm (qwen3 4B, text),
vlm (qwen3-vl 2B, text + BEV). Outputs: `results/hazard_test/`.

## Final results (test seeds; see `reports/final_report.md`, `results/hazard_test/final_tables.md`)
LLM (500–549, 50/scenario, complete): nominal 100 = DT (0 overrides) | stall 56 / 70 / 94 (DT 0; oracle 56 / 70 / 96) |
slow 68 / 86 / 0 (DT 0; stationary rule 0 / 36 / 0). 1911 calls, 100 % valid JSON.
VLM (500–529, 30/scenario, complete): nominal 93.3 (2 broken, 15 overrides) | stall 56.7 / 76.7 / 70.0 | slow 33.3 / 30.0 / 0 | hazards mean 44.4 % (DT 0 %); 95.7 % valid JSON, many SLOW_DOWN (206).
Next steps if continued: slow_exit (needs "follow, don't bypass"
reasoning); async LLM calls for real-time use; more hazard types (wrong-way vehicle, blocked exit).

## Phase 4 — reduce collisions (2026-10-23, in progress)
User feedback: with the LLM the hazard scenarios still collide 37.7% of the time — not convincing.
Diagnosis (oracle skill v1, dev seeds 200–259, crash partner = nearest crashed vehicle):
- stall_entry / stall_mid: most crashes are rear-ending a *traffic* vehicle in the outer lane that is queueing
  behind the stalled car (6–7 m/s) while the executor holds 16 m/s waiting for an inner-lane gap.
- slow_entry / slow_mid: the ego hits the crawling hazard directly (no braking allowed, gap never opens).
- slow_exit: the executor stops at the north exit road and the DT rear-ends the crawler there.
Root cause: the v1 executor has no longitudinal safety. v2 (`SafeBypassSkill`, `SKILL_V=2`): car-following cap
(brake also to a stop), speed-dependent lane-change gap, ends 8 m past the obstacle, stays active on the exit road;
plus a non-LLM control `acc` (same cap on the DT always).
Dev result (60 seeds): v2 oracle skill stall 0/43/83 %, slow 13/48/0 % (collisions still 8–100 %); acc 0 % success
everywhere, 52–87 % collisions. **v2 is worse than v1 in stall_entry (100 % collisions) — not adopted yet; needs
crash diagnosis of v2 (likely rear-end by followers when braking, or hitting the obstacle while pulling out at low speed).**
