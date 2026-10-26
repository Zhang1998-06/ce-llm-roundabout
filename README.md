# LLM/VLM Chain-of-Causation reasoner + Decision Transformer on a roundabout

PhD candidacy-exam experiment: *propose an architecture with LLMs/VLMs involved, implement it on the roundabout scenario
of my earlier Decision-Transformer research, and evaluate task completion, safety and efficiency.*

A frozen Decision Transformer (DT), trained only on normal traffic, drives the car. A local LLM (or VLM) runs as a
high-level reasoner on top of it. At each policy step it reads a structured scene description and answers in a fixed
Chain-of-Causation (CoC) JSON: `scene → critical_objects → abnormal_cause → causal_reasoning → meta_action`.
- If it names no abnormal cause, the DT keeps driving.
- If it chooses `BYPASS_OBSTACLE`, a temporally extended lane-change maneuver (`BypassSkill`) takes over until the
  obstacle is behind the car.

That description is of the first versions (v4/v5). In the current version (v6), the supervisor reads facts only and
answers `{consequence, decision, target_id}`. The decision is `FOLLOW_POLICY`, `BYPASS` or `FOLLOW_VEHICLE`, and the
executor only checks that the id exists.

```
 env (highway-env roundabout-v0) ──► OccupancyGrid ──► Decision Transformer ──► proposed action ─┐
        │                                                                                       ▼
        └─► describe_v4(): lane-structured text (+ BEV image for the VLM) ──► (V)LM CoC JSON ──► arbiter
                                                                     FOLLOW_POLICY / BYPASS_OBSTACLE / SLOW_DOWN / KEEP_SPEED
```

**Full report (Chinese; section 0 is the current version):** [`reports/CE_experiment_complete.md`](reports/CE_experiment_complete.md). It contains the
architecture, the verbatim prompts, all result tables, the complete iteration log and the exact geometry and
executor action mapping.

## Current version (v6): does the LLM's own understanding help the DT? (test seeds 600–649)

The question: can a pretrained LLM's scene understanding help a frozen DT, trained only on ordinary traffic, handle
out-of-distribution situations without retraining and without hurting nominal driving? Version 5 (below) could not
answer it. Its input already labelled the vehicles ("in your path", STATIONARY, exit-zone labels), its prompt mapped
those labels to maneuvers, and the executor picked the target vehicle.

v6 changes only what is needed to make the judgment the model's own:
- **Scenarios.** Two scenarios add the same stalled car where it does *not* matter: after the ego's exit, or upstream of
  its merge point.
- **Input.** Facts only: ids, lanes, positions along the road, speeds and map landmarks.
- **Prompt.** It defines the task, with no rule table and no examples.
- **Target.** The model returns the target vehicle id.
- **Controls.** Each control uses the same facts and the same executor.

The table shows success / collision / at-fault / timeout %, as the mean over each scenario group. "Need" is a stalled
car mid-ring or before the exit, and a crawler before the exit. "No need" is the same stalled car after the exit or
upstream.

| Method | need | no need | nominal |
|---|---|---|---|
| DT | 0 / 100 / 100 / 0 | 94 / 6 / 6 / 0 | 94 / 6 / 6 / 0 |
| naive trigger (any slow car ahead → bypass) | 45.3 / 35.3 / 34.7 / 19.3 | 90 / 3 / 3 / 7 | 94 / 6 / 6 / 0 |
| LLM v5 (labels + rule prompt) | 41.3 / 11.3 / 10.0 / 47.3 | 82 / 3 / 3 / 15 | 94 / 6 / 6 / 0 |
| **LLM v6 (facts only, model picks the target)** | **38.7 / 46.0 / 46.0 / 15.3** | **92 / 2 / 2 / 6** | **96 / 4 / 4 / 0** |
| designed route-aware rule (answer key) | 44.7 / 14.7 / 12.0 / 40.7 | 94 / 6 / 6 / 0 | 94 / 6 / 6 / 0 |

**Conclusion.** The integrated system beats the DT. Its gain cannot be attributed to the model's understanding, for
three reasons:
- **The naive trigger does better.** With the same executor, it is significantly better where intervention is needed
  (paired McNemar p = 0.006 for success, p = 0.002 for collisions).
- **The model rarely picks the right action and target.** Where intervention is needed, Qwen3-4B chooses the correct
  decision and target in 60 % / 8 % / 0 % of those moments.
- **Its successes are mostly incidental.** It starts a maneuver in every episode, including nominal traffic. Most of its
  successes come from lane changes made while "bypassing" a normal car.

v5's low collision rate came from its labels: it also bypasses the car *after* the exit in 48/50 episodes.

An offline check with Qwen3-30B-A3B on the same development states shows the same pattern. It recognises the stopped
car but intervenes on it even when it is off the route, 20/20 times.

Full write-up (Chinese, section 0): [`reports/CE_experiment_complete.md`](reports/CE_experiment_complete.md).
English LaTeX for Q1 Part 2 and the appendix: [`reports/latex/q1_part2.tex`](reports/latex/q1_part2.tex).
All tables: [`results/hazard_test_v6/tables_v6.md`](results/hazard_test_v6/tables_v6.md).

## Previous version (v5) results (test seeds 500–549) — integration result, not attributable to the model


Six long-tail hazards the DT never saw in training, each placed at the entry, middle or exit of the ring:
- a stalled vehicle (0 m/s), used to design the prompt;
- a crawling vehicle (3 m/s), the generalisation test.

The table shows the mean over the six hazards, 50 episodes per scenario.

| Method | Success % | Collision % | At-fault collision % | Timeout % |
|---|---:|---:|---:|---:|
| DT (baseline) | 0.0 | 100.0 | 100.0 | 0.0 |
| DT + the same longitudinal safety layer, no LLM (control) | 0.0 | 55.0 | 39.0 | 45.0 |
| DT + LLM-CoC v4 (first version, simple executor) | 62.3 | 37.7 | – | 0.0 |
| **DT + LLM-CoC v5 (current, safety-first executor)** | **45.0** | **9.7** | **9.0** | 45.3 |
| Perfect recognition + the same maneuvers (reference) | 45.7 | 9.7 | 9.0 | 44.7 |

**Nominal traffic.** DT + LLM v5 reaches 100 % success with 0 overrides, identical to the DT.

**What the LLM does.** It decides whether there is an abnormal cause in the path. If there is, it chooses between
passing it (`BYPASS_OBSTACLE`) and staying behind it (`FOLLOW_VEHICLE`). Following is the right choice when a moving
slow vehicle is entering the single-lane exit ahead, because it can never be overtaken there.

**Why the gain comes from the LLM.**
- Without the LLM, the same safety layer has 0 % success: it can only slow the car down or stop it.
- With the LLM, the results match the perfect-recognition reference, so the LLM's recognition and maneuver choice are
  not the bottleneck.

**The trade-off between v4 and v5.** Collisions fall from 37.7 % to 9.7 %, while success falls from 62.3 % to 45.0 %.
Episodes that used to end in a crash now end in a timeout instead: the car waits or follows within the 11 s horizon.
A 20 s horizon raises the reference to 56 % success and 13 % collisions.

**Simulator caveat.** IDM traffic in this simulator only looks for a leader inside its current 17–20 m lane segment, so it
rear-ends stopped cars just past a segment boundary. For this reason at-fault collisions (nuPlan-style) are reported in
addition to total collisions.

Full tables (min distance, speed, completion time, LLM calls, latency, 20 s sensitivity):
[`results/hazard_test_v5/final_tables.md`](results/hazard_test_v5/final_tables.md).
First-version tables: [`results/hazard_test/final_tables.md`](results/hazard_test/final_tables.md).

## Code map

| Path | Role |
|---|---|
| `src/coc_supervisor.py` | v6: `perceive_v6` / `describe_v6` (facts-only input), `SYSTEM_V6_NOCO` (frozen prompt), `ask_v6`. Older: prompts `SYSTEM_V5`, `SYSTEM_V4`, scene descriptions (`describe_v5`), `CoCSupervisor.ask()` (Ollama call, JSON parsing, `abnormal_cause = none` → FOLLOW_POLICY) |
| `src/rules_v6.py` | v6 controls on the same facts: `naive_rule`, `designed_rule`; privileged `reference()` for decision scoring |
| `src/hazards.py` | Hazard scenarios (`SCENARIOS`, `Stall` wrapper), reference rule, oracle trigger, executors `SmoothBypassSkill` + `FollowSkill` (v3, current), `BypassSkill` (v1), `SafeBypassSkill` (v2, abandoned); `smooth_acc_action` (no-LLM control) |
| `src/evaluation.py` | Loads the original DT code and frozen checkpoint |
| `src/bev_renderer.py` | Bird's-eye-view image for the VLM |
| `scripts/run_hazard.py` | Closed-loop evaluation: methods `dt, acc, oracle, llm, vlm` (+ `rule, skill, follow`); at-fault classification; logs every LLM call with the full user message and raw response |
| `scripts/make_tables_v6.py`, `scripts/offline_v6.py`, `scripts/replay_v6.py`, `scripts/build_complete_report.py` | v6 tables, offline dev check, step replay of a logged episode, assembly of the complete report |
| `scripts/make_final_tables_v5.py`, `scripts/make_final_tables.py` | Build the v5 / v4 result tables |
| `scripts/offline_v5.py` | Offline single-step accuracy check of prompt v5 |
| `scripts/build_offline_set.py`, `scripts/offline_eval.py` | Offline single-step decision test set used to check prompts before closed-loop runs |
| `scripts/replay_call.py` | Replays one episode and records the exact Ollama request/response |
| `results/hazard_test_v6/` | v6 test data: per-episode outcomes and every supervisor call (full user message + raw response) |
| `results/hazard_test_v5/` | v5 test data: per-episode and per-LLM-call (`decisions_llm.csv`, incl. prompts and raw outputs) |
| `results/hazard_test/` | v4 (first version) test data |
| `results/hazard_dev*/` | Development runs of each iteration |
| `results/phase1_nominal/`, `scripts/run_comparison.py`, `scripts/smoke.py`, `scripts/b0_scan.py`, `scripts/make_figures.py`, `src/vlm_supervisor.py`, `src/llm_supervisor.py` | Phase 1 (abandoned): a VLM YIELD supervisor in normal traffic that never beat the DT; see `reports/experiment_report.md` |

## Reproduce

**Requirements**
- Python 3.8 with `requirements.txt`.
- The DT code, checkpoint and local highway-env fork from the earlier project
  (`uncertaintyDT/CNNDT_baseline`, not included here).
- Ollama with both models pulled.

```bash
ln -s <path>/uncertaintyDT/CNNDT_baseline/highway_env highway_env
export CNNDT_ROOT=<path>/uncertaintyDT/CNNDT_baseline           # DT script + checkpoint (output2/)
ollama pull qwen3:4b-instruct-2507-q4_K_M      # optional for the offline check: qwen3:8b, qwen3:30b-a3b-instruct-2507-q4_K_M

# v6 (current)
S6=nominal,stall_mid,stall_exit,slow_exit,stall_past_exit,stall_upstream
PROMPT_VER=6 METHODS=dt,naive,designed SCEN=$S6 SEEDS=600-649 TAG=test_v6 NP=8 python scripts/run_hazard.py
PROMPT_VER=6 METHODS=llm,llm5 SCEN=$S6 SEEDS=600-649 TAG=test_v6 NP=3 python scripts/run_hazard.py
python scripts/make_tables_v6.py results/hazard_test_v6
python scripts/replay_v6.py stall_mid 602          # step-by-step replay of one logged episode

# v5 (previous)
S=nominal,stall_entry,stall_mid,stall_exit,slow_entry,slow_mid,slow_exit
SKILL_V=3 METHODS=dt,acc,oracle SCEN=$S SEEDS=500-549 TAG=test_v5 NP=6 python scripts/run_hazard.py
PROMPT_VER=5 METHODS=llm SCEN=$S SEEDS=500-549 TAG=test_v5 NP=3 python scripts/run_hazard.py   # implies executor v3
HORIZON_S=20 SKILL_V=3 METHODS=dt,oracle SCEN=$S SEEDS=500-549 TAG=test_v5_h20 NP=6 python scripts/run_hazard.py
python scripts/make_final_tables_v5.py
```
First version (v4): `PROMPT_VER=4 METHODS=llm ... TAG=test` (executor v1 is the default without `SKILL_V`).

Ollama runs at temperature 0, but its output is not guaranteed to be bit-identical across runs. Re-runs can
therefore differ slightly from the logged decisions; the environment and DT are deterministic given the seed.
