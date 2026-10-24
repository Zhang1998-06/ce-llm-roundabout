# LLM/VLM Chain-of-Causation reasoner + Decision Transformer on a roundabout

PhD candidacy-exam experiment: *propose an architecture with LLMs/VLMs involved, implement it on the roundabout scenario
of my earlier Decision-Transformer research, and evaluate task completion, safety and efficiency.*

A frozen Decision Transformer (DT), trained only on normal traffic, drives the car. A local LLM (or VLM) runs as a
high-level reasoner on top of it. At each policy step it reads a structured scene description and answers in a fixed
Chain-of-Causation (CoC) JSON: `scene → critical_objects → abnormal_cause → causal_reasoning → meta_action`.
- If it names no abnormal cause, the DT keeps driving.
- If it chooses `BYPASS_OBSTACLE`, a temporally extended lane-change maneuver (`BypassSkill`) takes over until the
  obstacle is behind the car.

```
 env (highway-env roundabout-v0) ──► OccupancyGrid ──► Decision Transformer ──► proposed action ─┐
        │                                                                                       ▼
        └─► describe_v4(): lane-structured text (+ BEV image for the VLM) ──► (V)LM CoC JSON ──► arbiter
                                                                     FOLLOW_POLICY / BYPASS_OBSTACLE / SLOW_DOWN / KEEP_SPEED
```

**Full report (Chinese; section 0 is the current version):** [`reports/CE_experiment_complete.md`](reports/CE_experiment_complete.md). It contains the
architecture, the verbatim prompts, all result tables, the complete iteration log and the exact geometry and
executor action mapping.

## Results (test seeds 500–549, never used for development)

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
| `src/coc_supervisor.py` | Prompts (`SYSTEM_V5` current, `SYSTEM_V4` first version), scene descriptions (`describe_v5`), `CoCSupervisor.ask()` (Ollama call, JSON parsing, `abnormal_cause = none` → FOLLOW_POLICY) |
| `src/hazards.py` | Hazard scenarios (`SCENARIOS`, `Stall` wrapper), reference rule, oracle trigger, executors `SmoothBypassSkill` + `FollowSkill` (v3, current), `BypassSkill` (v1), `SafeBypassSkill` (v2, abandoned); `smooth_acc_action` (no-LLM control) |
| `src/evaluation.py` | Loads the original DT code and frozen checkpoint |
| `src/bev_renderer.py` | Bird's-eye-view image for the VLM |
| `scripts/run_hazard.py` | Closed-loop evaluation: methods `dt, acc, oracle, llm, vlm` (+ `rule, skill, follow`); at-fault classification; logs every LLM call with the full user message and raw response |
| `scripts/make_final_tables_v5.py`, `scripts/make_final_tables.py` | Build the v5 / v4 result tables |
| `scripts/offline_v5.py` | Offline single-step accuracy check of prompt v5 |
| `scripts/build_offline_set.py`, `scripts/offline_eval.py` | Offline single-step decision test set used to check prompts before closed-loop runs |
| `scripts/replay_call.py` | Replays one episode and records the exact Ollama request/response |
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
ollama pull qwen3:4b-instruct-2507-q4_K_M

S=nominal,stall_entry,stall_mid,stall_exit,slow_entry,slow_mid,slow_exit
SKILL_V=3 METHODS=dt,acc,oracle SCEN=$S SEEDS=500-549 TAG=test_v5 NP=6 python scripts/run_hazard.py
PROMPT_VER=5 METHODS=llm SCEN=$S SEEDS=500-549 TAG=test_v5 NP=3 python scripts/run_hazard.py   # implies executor v3
HORIZON_S=20 SKILL_V=3 METHODS=dt,oracle SCEN=$S SEEDS=500-549 TAG=test_v5_h20 NP=6 python scripts/run_hazard.py
python scripts/make_final_tables_v5.py
```
First version (v4): `PROMPT_VER=4 METHODS=llm ... TAG=test` (executor v1 is the default without `SKILL_V`).

Ollama runs at temperature 0, but its output is not guaranteed to be bit-identical across runs. Re-runs can
therefore differ slightly from the logged decisions; the environment and DT are deterministic given the seed.
