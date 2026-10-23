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

**Full report (Chinese):** [`reports/CE_experiment_complete.md`](reports/CE_experiment_complete.md). It contains the
architecture, the verbatim prompts, all result tables, the complete iteration log and the exact geometry and
executor action mapping.

## Results (test seeds 500–549, never used for development)

Six long-tail hazards were added that the DT never saw in training:

| Hazard | Positions | Notes |
|---|---|---|
| Stalled vehicle (0 m/s) | entry / mid / exit of the ring | Used to design the prompt |
| Crawling vehicle (3 m/s) | entry / mid / exit of the ring | Generalisation test with the **same prompt** |

The table shows success rate in %; collision rate is in parentheses. Each cell is 50 episodes.

| Scenario | DT (baseline) | Hand-written stationary rule (ref.) | **DT + LLM-CoC (Qwen3-4B, text)** |
|---|---:|---:|---:|
| nominal | 100 (0) | 100 (0) | **100 (0)**, 0 false overrides |
| stalled · entry / mid / exit | 0 / 0 / 0 (100) | 56 / 72 / 96 | **56 / 70 / 94** |
| crawling · entry / mid / exit | 0 / 0 / 0 (100) | 0 / 36 / 0 | **68 / 86 / 0** |
| **mean of the 6 hazards** | **0 (100)** | 43.3 (56.3) | **62.3 (37.7)** |

Further numbers for the LLM run:
- Efficiency: the nominal completion time is identical to the DT (7.68 s).
- Calls: 1911 LLM calls, 100 % valid JSON.

A VLM variant was also tested on seeds 500–529 (Qwen3-VL-2B, text + bird's-eye-view image):
- Hazard mean 44.4 % success vs 0 % for the DT.
- Nominal success 93.3 %; it is weaker than the text LLM.

Full metric tables (collision, timeout, min distance, speed, time, calls, latency):
[`results/hazard_test/final_tables.md`](results/hazard_test/final_tables.md).

**Known limitations:**
- The remaining collisions come mostly from the maneuver executor, not from recognition. The LLM matches an
  oracle trigger in the stalled scenarios.
- crawling · exit is unsolved by every method.
- A v2 executor with longitudinal safety (`SafeBypassSkill`, `SKILL_V=2`) is work in progress. It is not used for the
  results above; see `reports/iteration_log.md`, Phase 4.

## Code map

| Path | Role |
|---|---|
| `src/coc_supervisor.py` | Prompts (`SYSTEM_V4` is final), scene descriptions (`describe_v4`), `CoCSupervisor.ask()` (Ollama call, JSON parsing, `abnormal_cause = none` → FOLLOW_POLICY) |
| `src/hazards.py` | Hazard scenarios (`SCENARIOS`, `Stall` wrapper), reference rule, oracle trigger, `BypassSkill` (v1, reported) and `SafeBypassSkill` (v2, WIP) |
| `src/evaluation.py` | Loads the original DT code and frozen checkpoint |
| `src/bev_renderer.py` | Bird's-eye-view image for the VLM |
| `scripts/run_hazard.py` | Closed-loop evaluation: methods `dt, rule, skill, llm, vlm` (+ `acc` with `SKILL_V=2`) |
| `scripts/make_final_tables.py` | Builds `results/hazard_test/final_tables.md` |
| `scripts/build_offline_set.py`, `scripts/offline_eval.py` | Offline single-step decision test set used to check prompts before closed-loop runs |
| `scripts/replay_call.py` | Replays one episode and records the exact Ollama request/response |
| `results/hazard_test/` | Final test data: per-episode, per-LLM-call (`decisions_*.csv`) and summary CSVs |
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
ollama pull qwen3:4b-instruct-2507-q4_K_M && ollama pull qwen3-vl:2b-instruct-q4_K_M

S=nominal,stall_entry,stall_mid,stall_exit,slow_entry,slow_mid,slow_exit
METHODS=dt,rule,skill SCEN=$S SEEDS=500-599 TAG=test NP=4 python scripts/run_hazard.py
PROMPT_VER=4 METHODS=llm SCEN=$S SEEDS=500-549 TAG=test NP=4 python scripts/run_hazard.py
PROMPT_VER=4 METHODS=vlm SCEN=$S SEEDS=500-529 TAG=test NP=2 python scripts/run_hazard.py
python scripts/make_final_tables.py
```

Ollama runs at temperature 0, but its output is not guaranteed to be bit-identical across runs. Re-runs can
therefore differ slightly from the logged decisions; the environment and DT are deterministic given the seed.
