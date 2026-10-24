### Test seeds 500–549 (50 episodes per scenario and method), horizon 11 s

| Scenario | Method | Success % | Collision % | At-fault collision % | Timeout % | Min dist (m) | Avg speed (m/s) | Completion time (s) | LLM calls/ep | Latency (s/call) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal | DT (baseline) | 100.0 | 0.0 | 0.0 | 0.0 | 7.11 | 15.43 | 7.68 | – | – |
| nominal | DT + same longitudinal safety, no LLM (control) | 94.0 | 0.0 | 0.0 | 6.0 | 6.67 | 14.17 | 8.05 | – | – |
| nominal | Perfect recognition + same maneuvers (reference) | 100.0 | 0.0 | 0.0 | 0.0 | 7.11 | 15.43 | 7.68 | – | – |
| nominal | **DT + LLM-CoC v5 (Qwen3-4B, text)** | 100.0 | 0.0 | 0.0 | 0.0 | 7.11 | 15.43 | 7.68 | 13.4 | 1.80 |
| nominal | DT + LLM-CoC v4 (previous version, executor v1) | 100.0 | 0.0 | – | 0.0 | 7.11 | 15.43 | 7.68 | 13.4 | 2.70 |
| stall_entry | DT (baseline) | 0.0 | 100.0 | 100.0 | 0.0 | 4.88 | 13.64 | – | – | – |
| stall_entry | DT + same longitudinal safety, no LLM (control) | 0.0 | 100.0 | 100.0 | 0.0 | 4.26 | 5.93 | – | – | – |
| stall_entry | Perfect recognition + same maneuvers (reference) | 0.0 | 0.0 | 0.0 | 100.0 | 5.16 | 2.46 | – | – | – |
| stall_entry | **DT + LLM-CoC v5 (Qwen3-4B, text)** | 0.0 | 0.0 | 0.0 | 100.0 | 5.16 | 2.46 | – | 1.0 | 1.17 |
| stall_entry | DT + LLM-CoC v4 (previous version, executor v1) | 56.0 | 44.0 | – | 0.0 | 4.49 | 14.17 | 7.00 | 3.8 | 2.74 |
| stall_mid | DT (baseline) | 0.0 | 100.0 | 100.0 | 0.0 | 4.95 | 14.07 | – | – | – |
| stall_mid | DT + same longitudinal safety, no LLM (control) | 0.0 | 72.0 | 28.0 | 28.0 | 4.58 | 7.60 | – | – | – |
| stall_mid | Perfect recognition + same maneuvers (reference) | 68.0 | 18.0 | 18.0 | 14.0 | 4.52 | 12.32 | 7.78 | – | – |
| stall_mid | **DT + LLM-CoC v5 (Qwen3-4B, text)** | 68.0 | 18.0 | 18.0 | 14.0 | 4.52 | 12.32 | 7.78 | 2.4 | 1.68 |
| stall_mid | DT + LLM-CoC v4 (previous version, executor v1) | 70.0 | 30.0 | – | 0.0 | 4.73 | 14.80 | 7.14 | 2.8 | 2.95 |
| stall_exit | DT (baseline) | 0.0 | 100.0 | 100.0 | 0.0 | 5.02 | 14.57 | – | – | – |
| stall_exit | DT + same longitudinal safety, no LLM (control) | 0.0 | 14.0 | 6.0 | 86.0 | 4.26 | 6.22 | – | – | – |
| stall_exit | Perfect recognition + same maneuvers (reference) | 70.0 | 0.0 | 0.0 | 30.0 | 4.67 | 12.30 | 7.49 | – | – |
| stall_exit | **DT + LLM-CoC v5 (Qwen3-4B, text)** | 66.0 | 0.0 | 0.0 | 34.0 | 4.64 | 11.98 | 7.44 | 5.5 | 1.93 |
| stall_exit | DT + LLM-CoC v4 (previous version, executor v1) | 94.0 | 6.0 | – | 0.0 | 4.75 | 14.89 | 7.49 | 6.2 | 3.00 |
| slow_entry | DT (baseline) | 0.0 | 100.0 | 100.0 | 0.0 | 5.01 | 13.51 | – | – | – |
| slow_entry | DT + same longitudinal safety, no LLM (control) | 0.0 | 92.0 | 70.0 | 8.0 | 4.38 | 5.66 | – | – | – |
| slow_entry | Perfect recognition + same maneuvers (reference) | 68.0 | 30.0 | 30.0 | 2.0 | 4.27 | 13.29 | 7.97 | – | – |
| slow_entry | **DT + LLM-CoC v5 (Qwen3-4B, text)** | 68.0 | 30.0 | 30.0 | 2.0 | 4.27 | 13.29 | 7.97 | 3.1 | 1.71 |
| slow_entry | DT + LLM-CoC v4 (previous version, executor v1) | 68.0 | 32.0 | – | 0.0 | 4.28 | 14.61 | 7.07 | 3.0 | 2.69 |
| slow_mid | DT (baseline) | 0.0 | 100.0 | 100.0 | 0.0 | 5.10 | 14.50 | – | – | – |
| slow_mid | DT + same longitudinal safety, no LLM (control) | 0.0 | 42.0 | 22.0 | 58.0 | 4.47 | 7.40 | – | – | – |
| slow_mid | Perfect recognition + same maneuvers (reference) | 68.0 | 6.0 | 4.0 | 26.0 | 4.56 | 11.99 | 7.69 | – | – |
| slow_mid | **DT + LLM-CoC v5 (Qwen3-4B, text)** | 68.0 | 6.0 | 4.0 | 26.0 | 4.56 | 11.99 | 7.69 | 1.9 | 1.60 |
| slow_mid | DT + LLM-CoC v4 (previous version, executor v1) | 86.0 | 14.0 | – | 0.0 | 4.81 | 14.78 | 7.47 | 2.5 | 2.86 |
| slow_exit | DT (baseline) | 0.0 | 100.0 | 100.0 | 0.0 | 4.98 | 14.81 | – | – | – |
| slow_exit | DT + same longitudinal safety, no LLM (control) | 0.0 | 10.0 | 8.0 | 90.0 | 4.33 | 7.55 | – | – | – |
| slow_exit | Perfect recognition + same maneuvers (reference) | 0.0 | 4.0 | 2.0 | 96.0 | 4.13 | 7.31 | – | – | – |
| slow_exit | **DT + LLM-CoC v5 (Qwen3-4B, text)** | 0.0 | 4.0 | 2.0 | 96.0 | 4.13 | 7.31 | – | 6.4 | 1.89 |
| slow_exit | DT + LLM-CoC v4 (previous version, executor v1) | 0.0 | 100.0 | – | 0.0 | 4.14 | 14.59 | – | 6.5 | 3.03 |
| hazard mean | DT (baseline) | 0.0 | 100.0 | 100.0 | 0.0 | 4.99 | 14.18 | – | – | – |
| hazard mean | DT + same longitudinal safety, no LLM (control) | 0.0 | 55.0 | 39.0 | 45.0 | 4.38 | 6.73 | – | – | – |
| hazard mean | Perfect recognition + same maneuvers (reference) | 45.7 | 9.7 | 9.0 | 44.7 | 4.55 | 9.95 | 7.73 | – | – |
| hazard mean | **DT + LLM-CoC v5 (Qwen3-4B, text)** | 45.0 | 9.7 | 9.0 | 45.3 | 4.55 | 9.89 | 7.72 | 3.4 | 1.66 |
| hazard mean | DT + LLM-CoC v4 (previous version, executor v1) | 62.3 | 37.7 | – | 0.0 | 4.53 | 14.64 | 7.23 | 4.1 | 2.88 |

At-fault: the ego is not at fault only when hit from behind while keeping its lane outside the merge zone (nuPlan-style); not computed for the v4 run. Completion time: successful episodes only.

### Sensitivity: 20 s horizon (protocol is 11 s), DT and the perfect-recognition reference, same seeds

| Scenario | Method | Success % | Collision % | At-fault % | Timeout % |
|---|---|---:|---:|---:|---:|
| nominal | DT (baseline) | 94.0 | 6.0 | 6.0 | 0.0 |
| nominal | Perfect recognition + same maneuvers (reference) | 94.0 | 6.0 | 6.0 | 0.0 |
| stall_entry | DT (baseline) | 0.0 | 100.0 | 100.0 | 0.0 |
| stall_entry | Perfect recognition + same maneuvers (reference) | 2.0 | 0.0 | 0.0 | 98.0 |
| stall_mid | DT (baseline) | 0.0 | 100.0 | 100.0 | 0.0 |
| stall_mid | Perfect recognition + same maneuvers (reference) | 68.0 | 18.0 | 18.0 | 14.0 |
| stall_exit | DT (baseline) | 0.0 | 100.0 | 100.0 | 0.0 |
| stall_exit | Perfect recognition + same maneuvers (reference) | 78.0 | 0.0 | 0.0 | 22.0 |
| slow_entry | DT (baseline) | 0.0 | 100.0 | 100.0 | 0.0 |
| slow_entry | Perfect recognition + same maneuvers (reference) | 68.0 | 30.0 | 30.0 | 2.0 |
| slow_mid | DT (baseline) | 0.0 | 100.0 | 100.0 | 0.0 |
| slow_mid | Perfect recognition + same maneuvers (reference) | 68.0 | 20.0 | 18.0 | 12.0 |
| slow_exit | DT (baseline) | 0.0 | 100.0 | 100.0 | 0.0 |
| slow_exit | Perfect recognition + same maneuvers (reference) | 52.0 | 10.0 | 6.0 | 38.0 |
| hazard mean | DT (baseline) | 0.0 | 100.0 | 100.0 | 0.0 |
| hazard mean | Perfect recognition + same maneuvers (reference) | 56.0 | 13.0 | 12.0 | 31.0 |
