# CE experiment: LLM/VLM Chain-of-Causation supervisor + Decision Transformer on roundabout-v0
**Start here:** `reports/final_report.md` (architecture, prompts, final results, limitations, how to reproduce).
Iteration history (every attempt, diagnosis and number): `reports/iteration_log.md`.
Final data: `results/hazard_test/` (`final_tables.md`, `episodes_*.csv`, `decisions_*.csv`).
Older phase-1 report (nominal-scenario YIELD supervisor, did not beat DT): `reports/experiment_report.md`.
Note: `highway_env` is a symlink to `uncertaintyDT/CNNDT_baseline/highway_env`; the DT code is imported from `CNNDT_baseline`, not copied.

**External dependencies (not in this repo):** the DT code and checkpoint (`uncertaintyDT/CNNDT_baseline/HF-DT-testing_average_return.py`, `output2/`)
and the local highway-env fork (`uncertaintyDT/CNNDT_baseline/highway_env`, symlinked as `./highway_env`). LLMs run on Ollama:
`qwen3:4b-instruct-2507-q4_K_M` (text) and `qwen3-vl:2b-instruct-q4_K_M` (VLM).

**Work in progress:** `SafeBypassSkill` / `acc_action` in `src/hazards.py` (switch `SKILL_V=2`, default 1 = the executor used for the reported results).
Dev-set check in `results/hazard_dev_v2s/` — not yet better than v1, see `reports/iteration_log.md`.
