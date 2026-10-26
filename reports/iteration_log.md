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

### 4.2 v2 diagnosis (dev 200–259, oracle trigger)
- ~half of the v2 crashes: braking on the access road (queue ahead) → ego enters the ring at 1–2 m/s → hit by ring traffic.
- Low-speed lane change then FASTER with the cap computed in the target lane only → ego hits the stalled car before it has moved sideways.
- Rear-end by followers after SLOWER.

### 4.3 Two simulator facts found in the source (they explain most remaining collisions)
1. `MDPVehicle.act("SLOWER")` takes the level below the **current** speed: at 12 m/s the target becomes 0 → a full stop
   (initial deceleration ≈ 13 m/s²). `IDLE` keeps the target speed, so an executor can set `target_speed` itself.
2. `Road.neighbour_vehicles` (used by IDM traffic to find its leader) only searches the **current lane segment**
   (ring segments are 17–20 m). A stopped or slow car just past a segment boundary is invisible to the follower, which
   then rear-ends it. Queued traffic also piles into the stalled car this way. → zero collisions is not reachable for any
   controller that must slow down on the ring; rear-end impacts on an ego that keeps its lane are not the ego's fault.
   New metric `collision_at_fault` (nuPlan-style): the ego is not at fault only when hit from behind (partner > 2 m behind)
   while keeping its lane (no lane-change command in the last 3 steps) outside the merge zone (access road / first 10 m).
   Total collisions are still reported.

### 4.4 v3 executor (`SmoothBypassSkill`, `SKILL_V=3`)
- Executor owns the target speed (sends IDLE): braking target from the needed deceleration
  a = (v² − v_lead²) / 2(gap − 8 m), clipped to [1.5, 6] m/s²; otherwise rate-limited towards an IDM-like safe speed.
- Access road: the DT's own (trained) ring entry. (v3a gated the entry on an outer-lane gap → the ego crept into ring
  traffic; removed.)
- Lane change only with a speed-dependent front/rear gap; the obstacle stays a leader until the lane change is complete;
  ends 8 m past the obstacle; stays active on the exit road; exit-road vehicles are seen as leaders from the ring.
- `FollowSkill`: same longitudinal control, no lane change (for a slow vehicle at the single-lane exit).
- Dev, oracle trigger, success / collision / at-fault %:

| scenario | DT | v1 oracle | v3 oracle bypass | v3 oracle follow | DT + same longitudinal safety (`acc`) |
|---|---|---|---|---|---|
| stall_entry | 0/100/100 | 55/45/– | 50/50/50 | – | 0/100/78 |
| stall_mid | 0/100/100 | 78/22/– | 82/17/17 | – | 0/92/33 |
| stall_exit | 0/100/100 | 87/13/– | 85/13/13 | – | 0/50/33 |
| slow_entry | 0/100/100 | 0/100/– | 82/18/18 | – | 0/90/90 |
| slow_mid | 0/100/100 | 38/62/– | 83/15/15 | – | 0/55/23 |
| slow_exit | 0/100/100 | 0/100/– | 0/98/98 | **0/45/25** (55 % timeout) | 0/42/25 |

  (v1's oracle only triggered on stationary vehicles; the v2/v3 oracle triggers on any vehicle < 5 m/s in the path.)
  The `acc` control has 0 % success everywhere: the safety layer alone does not solve the hazards.
- slow_exit: the crawler enters the single-lane north exit first, so bypassing cannot work (the ego must cut back in
  front of it) → the right maneuver is to follow. This becomes a real decision for the LLM.

### 4.5 Prompt v5 (`SYSTEM_V5`, `describe_v5`, `PROMPT_VER=5` → executor v3 automatically)
- Meta-actions reduced to FOLLOW_POLICY / BYPASS_OBSTACLE / FOLLOW_VEHICLE (SLOW_DOWN / KEEP_SPEED removed; the VLM had
  overused SLOW_DOWN).
- Perception adds the exit (single lane, distance) and a map-zone label per path vehicle ("in the EXIT ZONE" = last
  20 m of the ring before the exit, "on the exit road").
- Rule in the prompt: BYPASS is the default; a STATIONARY vehicle is always bypassed; FOLLOW only if the abnormal vehicle
  is MOVING and in the exit zone / on the exit road; the exit-zone label alone is not abnormal.
- Offline check (`scripts/offline_v5.py`, dev seeds 200–209, t = 0..10, labels = the stated rule), Qwen3-4B:
  - first draft with numeric "X m before your exit": 69 % (the 4B model mis-compares distances → FOLLOW overused);
  - zone labels: 81.5 % (normal traffic in the exit zone became a false cause);
  - + "exit zone alone is not abnormal" example: **94 %** (nominal 100, stall 100/98/100, slow_entry 100, slow_exit 89,
    slow_mid 71 — late states where the crawler slowed to 1–2 m/s are called "stationary"). 100 % valid JSON, 0.82 s.
- Decision log now stores `user_message`, `raw` response, `executed` action and `skill_started` per call.

### 4.6 LLM v5 closed loop, dev seeds 200–229 (30 per scenario), success / collision / at-fault %
- Run A (zone labels): nominal 90/10/10 (= DT on these seeds, 0 overrides); stall 43/57/57, 77/20/20, 80/17/17;
  slow 77/23/23, 80/17/17, slow_exit 0/77/73.
- slow_exit: when the LLM chose FOLLOW_VEHICLE the episode had 0 % at-fault collisions (86 % timeout); it chose
  BYPASS in 21/28 decisions, calling the 3 m/s car "stationary". The 4B model conflates slow with stationary.
- Perception wording tried offline: explicit "(moving)" attribute → 90 % (slow_exit 63 %); predicted manoeuvre labels
  ("entering the single-lane exit ahead of you" / "stopped in the exit zone" / "on the single-lane exit road") → 91 %
  (slow_exit 52 % but the errors are now "none", which is re-queried next step, instead of a committed wrong BYPASS:
  wrong BYPASS on follow labels 2/21). Kept the predicted-manoeuvre labels → run B.
- Run B (predicted-manoeuvre labels): slow_exit FOLLOW chosen in 25 / BYPASS in 3 decisions (was 7 / 21); but FOLLOW
  episodes still had 56 % at-fault collisions → executor problem, not recognition.

### 4.7 Executor fixes found by step-by-step traces (replaying the logged LLM decisions, `scratchpad/trace.py`)
- Traffic crashes into the crawler ahead (IDM blind spot) → the leader stops abruptly; with an 8 m centre margin and the
  0.6 s speed lag the ego touched the (rotated) wreck at < 1 m/s. → stop margin S0 = 10 m + time gap T = 1 s in both
  `brake_target` and `safe_speed`.
- Vehicles between the ego and the obstacle change lanes around it and reveal it late. → the obstacle the LLM designated
  is always a leader (`obstacle_gap`) for both maneuvers.
- Corridor leader detection (vehicles within 2.6 m laterally, cut-ins): no measurable effect; kept on (`CORRIDOR=1`).
- stall_entry (stalled car 12 m after the merge): every variant either collides ~50 % (DT-style entry at speed) or
  waits. Entry hold (stationary obstacle < 25 m after the merge → stop before the merge until the outer lane upstream
  and the inner lane at the entry are free): 0 % collisions, 100 % timeout within the 11 s horizon. A low-speed creep
  during the lane change made it worse (70 % collisions: steering needs speed) → reverted. **Decision: safety first —
  hold at the entry** (a human driver would also wait); reported as a timeout, not hidden.
- Final executor, oracle trigger, dev 200–259 (60 per scenario), success / collision / at-fault / timeout %:
  stall 0/0/0/100, 82/12/12/7, 83/5/5/12; slow 82/17/17/2, 82/8/8/10; slow_exit (follow) 0/10/5/90.
  Hazard mean: success 55 %, collision 8.6 %, at-fault 7.8 % (v1 oracle: 43 % / 57 %). `acc` control: 0 % success.
- LAG_COMP 1.6 (make the speed loop reach the requested deceleration) tried: more rear-end / at-fault crashes in the
  bypass scenarios (stall_mid at-fault 12 → 22 %) → reverted to 1.0.
- Prompt v5 + example with a normal and a crawling vehicle both entering the exit (the 4B model had answered "none"
  there and decided 2 steps late): offline 96.6 % (slow_exit 74 %, slow_mid 98 %, nominal 100 %).
  LLM dev 200–229: slow_exit 0/20/13/80 (was 0/50/43/50); nominal identical to DT (0 overrides).

## Phase 4 test protocol (frozen 2026-10-23 before any test-seed run)
- Seeds 500–549 (50 per scenario), 7 scenarios. Methods: `dt`; `acc` (DT + the same longitudinal safety, no LLM);
  `oracle` (true hazard + the prompt-v5 maneuver rule = perfect recognition reference); `llm` (Qwen3-4B, PROMPT_VER=5,
  executor v3). SKILL_V=3, S0=10, T_GAP=1, LAG_COMP=1, ENTRY_HOLD=1, CORRIDOR=1. Metrics: success, collision,
  at-fault collision, timeout, speed, completion time, min distance, calls, latency.

## Phase 4 final test results (seeds 500–549, 50 per scenario, protocol above)
Hazard mean success / collision / at-fault / timeout %:
- DT 0 / 100 / 100 / 0; acc (no LLM) 0 / 55 / 39 / 45; oracle 45.7 / 9.7 / 9.0 / 44.7;
- **LLM v5 45.0 / 9.7 / 9.0 / 45.3**; previous LLM v4 62.3 / 37.7 / – / 0.
- Per scenario (LLM v5): stall 0/0/0/100, 68/18/18/14, 66/0/0/34; slow 68/30/30/2, 68/6/4/26, 0/4/2/96;
  nominal 100 % (0 overrides). 1681 calls, 100 % valid JSON, 1.66 s mean latency.
- LLM ≈ oracle in every scenario (largest gap: stall_exit 66 vs 70 %) → recognition / maneuver choice is no longer the
  bottleneck; the remaining collisions are executor / traffic (IDM blind spot) limited.
- 20 s horizon sensitivity (DT + oracle only, same seeds): oracle hazard mean 56 / 13 / 12 / 31; slow_exit 52 % success,
  stall_exit 78 %; stall_entry still 98 % timeout (true deadlock); collisions rise slightly with longer exposure.
- Trade-off vs v4: collisions 37.7 → 9.7 %, success 62.3 → 45.0 % (stall_entry waits at the entry; the margins cost time).
- Next steps if continued: a smarter entry for stall_entry (enter in the inner lane directly); fix the IDM blind spot
  in a separate sensitivity study (would change the traffic model, so all baselines must be rerun); VLM with prompt v5.

## Phase 5 (v6): does the model's own scene understanding help the DT? (started 2026-10-25)
Why a new phase: the v5 result cannot be attributed to the model's understanding.
- `describe_v5` already told the model which vehicles are "YOUR PATH AHEAD" vs "OTHER TRAFFIC (not in your path)",
  labelled them "(STATIONARY)", "entering the single-lane exit ahead of you", "stopped in the exit zone".
- `SYSTEM_V5` mapped exactly these labels to maneuvers ("BYPASS is the default ... FOLLOW_VEHICLE ONLY when ...") with
  examples that match the test conditions one to one.
- `run_hazard.py` (v5) chose the maneuver target itself: the slowest of the first 3 path vehicles.
- Every hazard in the 6 test scenarios was on the ego's path, so "intervene whenever something abnormal is listed in
  the path" was always right; no scenario required the model to decide that an abnormal vehicle does NOT matter.

### 5.1 Scenario design (feasibility probe, DT + reference rule, dev seeds 200–219)
Idea: keep the same object (a stalled or crawling car) and change only its relation to the ego's route.
- stall_past_exit (outer lane, 9.6 m AFTER the ego's exit; mirror of stall_exit): DT 95 % success → no intervention
  needed; the v5 path logic would list it as "path ahead" (bug: the path did not stop at the exit).
- stall_upstream (outer lane, 12 m BEFORE the ego's merge point, i.e. behind the ego): DT 100 % → no intervention needed.
- stall_mid_inner (inner lane, same angle as stall_mid): DT 100 % collisions, but the trace shows why: IDM traffic hits the
  stalled car (lane-segment blind spot) and the wreck slides into the OUTER lane ahead of the ego. It becomes a
  different hazard (a wreck in the ego's lane with the other lane blocked), not a clean "other lane" control → excluded,
  reported as a limitation.
- stall_entry dropped: every method times out (entry hold), it cannot separate methods.
Final set: nominal; intervention needed: stall_mid (bypass), stall_exit (bypass), slow_exit (follow);
no intervention needed: stall_past_exit, stall_upstream. Pairs: stall_exit vs stall_past_exit (before/after the exit),
stall_exit vs slow_exit (stationary/crawling at the same place), stall_mid vs stall_upstream (ahead/behind).

### 5.2 New interfaces
- `perceive_v6` / `describe_v6`: every vehicle within 50 m with a stable id, road, lane, position along the road
  (ring: signed arc in the driving direction relative to the ego or to its merge point), speed. No STATIONARY, path,
  exit-zone or manoeuvre labels. Route given as navigation (where the north exit leaves the ring).
- `SYSTEM_V6`: road facts, the policy's training limitation, decision meanings / executor constraints, cost of
  interventions, task ("judge the situation for the ego's own route"). No rule table, no examples.
- Output `{consequence, decision, target_id}`; the executor only checks that the id exists (refusals are logged as
  `rejected_invalid_target`).
- Controls on the same facts + same executor (`src/rules_v6.py`): `naive` (any < 5 m/s vehicle within 40 m in front,
  straight-line geometry → BYPASS) and `designed` (route-aware hand-written rule = the designer's answer key).
- `reference()`: the designed logic applied to the injected vehicle only (privileged), for decision scoring.
- Rule probe (dev 200–219): naive loses 20 points of success in stall_past_exit (unnecessary bypass) and has 90 %
  collisions in slow_exit (bypasses a crawler at the exit); designed = DT where no intervention is needed.

### 5.3 Offline single-step checks (dev states from the designed-rule runs, balanced per scenario/label)
| draft | change | 4B instruct | 8B (no think) |
|---|---|---:|---:|
| p1 | flat list sorted by distance | 41 % | – |
| p2 | grouped by lane + `critical_objects` field | 8.7 % (intervenes everywhere) | – |
| p3 | + kinematics (time to reach each ring vehicle at current speeds, every lane), no `critical_objects` | 53 % | 47 % |
| p3 | same, with `critical_objects` | 21 % | 22 % |
| p4 | + map landmarks in the same 1-D lane coordinate ([YOU], [YOUR MERGE POINT], [YOUR EXIT]); reference no longer cut at 40/50 m | 49 % | – |
- Bug found: the old `_where` still says "ring" for the first metres of the exit road; v6 supervision now stops on
  `road_of(ego)` = exit road, so the reference is not computed in a meaningless frame.
- Error pattern (4B): stall_mid nearly always right (80–90 %), but in the exit area the model ignores a 0 or 3 m/s car in
  its own lane ("no collision, traffic flows") or bypasses moving 16 m/s cars ("may block the merge"). FOLLOW_VEHICLE
  for the crawler before the exit: 0/10 in every draft. 8B without thinking is not better.
- Reasoning model `qwen3:4b-thinking-2507` (p4 input): 30 s per call, all 39 calls hit the 4096-token budget while still
  thinking (~14 k characters) → 0 % valid JSON. Not usable in a closed loop at 2 Hz on an RTX 3070 Ti (thousands of
  calls); dropped. `qwen3:8b` (no think) was not better than the 4B instruct model (above).
- Decision (no further prompt tuning, to avoid fitting rules back in): freeze p4 = lane-grouped facts + kinematics +
  map landmarks, task-only prompt without the `critical_objects` field, model `qwen3:4b-instruct-2507-q4_K_M`
  (same model as Phase 4, so v5 vs v6 differ only in input / prompt / target choice). Dev offline accuracy 49 %
  (p3: 53 % on a stricter reference; the difference is within noise of ~90 states). The point of v6 is to measure the
  model's judgment, not to raise this number with hints.

## Phase 5 test protocol (frozen 2026-10-25, before any run on the test seeds)
- Test seeds 600–649 (50 per scenario; never used before, also not by Phase 4). Dev seeds were 200–219.
- Scenarios: nominal; need: stall_mid, stall_exit, slow_exit; no need: stall_past_exit, stall_upstream. 11 s horizon.
- Methods, all with the same executor v3 (SmoothBypassSkill / FollowSkill, S0 = 10, T_GAP = 1, ENTRY_HOLD = 1):
  `dt`; `naive` (context-free trigger); `designed` (designer's route-aware rule = answer key, upper reference);
  `llm5` (the Phase-4 pipeline unchanged: labels + rule prompt v5, executor picks the target);
  `llm` (v6: facts only, task-only prompt, model returns the target id; Qwen3-4B-instruct, temperature 0).
- Primary outcomes: success / collision / at-fault / timeout per scenario, and the mean over "need" and "no need".
  Mechanism outcomes: per-call decision vs the privileged reference; maneuvers started on the injected vehicle;
  unnecessary maneuvers in "no need" scenarios; invalid-target refusals. Paired exact McNemar tests on the same seeds.
- No change to prompt, input or executor after this point; any bug fix would be logged and all methods rerun.
- Test run 1 of `llm,llm5` aborted at episode ~290/600: KeyError in the v6 *scoring* call inside the llm5 branch (the v5
  pipeline keeps querying on the first metres of the exit road, where the old `_where` still says "ring"). Behaviour of
  no method was affected; fixed `describe_v6` / `reference` for an ego on the exit road (reference = FOLLOW_POLICY there)
  and reran both LLM conditions on the full test set. The aborted run's partial file is not used.

## Phase 5 test results (seeds 600–649, 50 per scenario, protocol above) — `results/hazard_test_v6/tables_v6.md`
Mean success / collision / at-fault / timeout %, "need" = stall_mid, stall_exit, slow_exit; "no need" = stall_past_exit,
stall_upstream:
- DT: need 0 / 100 / 100 / 0; no need 94 / 6 / 6 / 0; nominal 94 / 6 / 6 / 0.
- naive rule + executor: need 45.3 / 35.3 / 34.7 / 19.3; no need 90 / 3 / 3 / 7.
- LLM v5 pipeline: need 41.3 / 11.3 / 10.0 / 47.3; no need 82 / 3 / 3 / 15 (stall_past_exit 64 % success: its "path"
  label lists the car behind the exit, the rule prompt then says BYPASS, 48/50 episodes).
- **LLM v6: need 38.7 / 46.0 / 46.0 / 15.3; no need 92 / 2 / 2 / 6; nominal 96 / 4 / 4 / 0.**
- designed rule (answer key): need 44.7 / 14.7 / 12.0 / 40.7; no need 94 / 6 / 6 / 0.
Mechanism:
- v6 picked the right maneuver AND the injected vehicle in 60 % (stall_mid), 8 % (stall_exit), 0 % (slow_exit) of the
  calls where the reference intervenes; it intervened in 35–52 % of the calls where the reference does not, mostly on
  ordinary 16 m/s vehicles (median target speed 16.0 m/s); ≥ 1 maneuver in 50/50 episodes of every scenario, including
  nominal (1.12 per episode). FOLLOW_VEHICLE on the crawler at the exit: 0/50 episodes (v5: 45/50, designed: 44/50).
- Attribution of v6 successes: stall_exit 27 successes, 20 of them in episodes where the model never targeted the stalled
  car (it "bypassed" a normal car, the lane change avoided the stalled car by chance). stall_mid: 9 of 31.
- Paired McNemar (same seeds), need scenarios: v6 > DT (success 58 vs 0 discordant, p < 1e-17); v6 < naive (success 1 vs 11,
  p = 0.006; no collision 4 vs 20, p = 0.002); v6 < designed (no collision 5 vs 52, p < 1e-10). No need: v6 = DT
  (success 4 vs 6, p = 0.75); v6 > v5 (11 vs 1, p = 0.006).
- 1070 v6 calls, 100 % valid JSON, 0 invalid-target refusals, 0.51 s mean latency (v5: 2354 calls, 1.87 s).
Conclusion: the v6 gain over DT is explained by the executor plus frequent interventions, not by scene understanding:
a context-free trigger with the same executor does significantly better where intervention is needed. v5's lower
collision rate came from the labels + rule prompt (its failure in stall_past_exit shows the judgment sat in the labels).

### 5.4 Supplementary check: a larger pretrained model (offline only, the same 74 dev states as p4)
- `qwen3:30b-a3b-instruct-2507` (MoE, 3 B active, CPU+GPU, 1.9 s/call): 29.7 % (4B: 48.6 %). Where intervention is
  needed both 44 %; where it is not, the 30B model intervened in 81 % of states and in stall_past_exit /
  stall_upstream it targeted the stopped car 20/20 times: it recognises the abnormal object but does not use the
  route relation (behind the merge point / after the exit). Too slow for the closed loop on this GPU; not run there.
