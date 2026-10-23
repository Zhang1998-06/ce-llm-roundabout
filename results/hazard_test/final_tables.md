
### DT + LLM-CoC (Qwen3-4B, text) — complete run, 350 episodes

| Scenario | n | Method | Success % | Collision % | Timeout % | Min dist (m) | Avg speed (m/s) | Completion time (s) | (V)LM calls/ep | Latency (s/call) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal | 50 | DT (baseline) | 100.0 | 0.0 | 0.0 | 7.11 | 15.43 | 7.68 | – | – |
| nominal | 50 | Stationary-vehicle rule (ref.) | 100.0 | 0.0 | 0.0 | 7.11 | 15.43 | 7.68 | – | – |
| nominal | 50 | Oracle trigger + executor (ref.) | 100.0 | 0.0 | 0.0 | 7.11 | 15.43 | 7.68 | – | – |
| nominal | 50 | **DT + LLM-CoC (Qwen3-4B, text)** | **100.0** | 0.0 | 0.0 | 7.11 | 15.43 | 7.68 | 13.4 | 2.70 |
| stall_entry | 50 | DT (baseline) | 0.0 | 100.0 | 0.0 | 4.88 | 13.64 | – | – | – |
| stall_entry | 50 | Stationary-vehicle rule (ref.) | 56.0 | 42.0 | 2.0 | 4.18 | 14.14 | 7.00 | – | – |
| stall_entry | 50 | Oracle trigger + executor (ref.) | 56.0 | 44.0 | 0.0 | 4.49 | 14.17 | 7.00 | – | – |
| stall_entry | 50 | **DT + LLM-CoC (Qwen3-4B, text)** | **56.0** | 44.0 | 0.0 | 4.49 | 14.17 | 7.00 | 3.8 | 2.74 |
| stall_mid | 50 | DT (baseline) | 0.0 | 100.0 | 0.0 | 4.95 | 14.07 | – | – | – |
| stall_mid | 50 | Stationary-vehicle rule (ref.) | 72.0 | 28.0 | 0.0 | 4.59 | 14.42 | 7.36 | – | – |
| stall_mid | 50 | Oracle trigger + executor (ref.) | 70.0 | 30.0 | 0.0 | 4.73 | 14.80 | 7.14 | – | – |
| stall_mid | 50 | **DT + LLM-CoC (Qwen3-4B, text)** | **70.0** | 30.0 | 0.0 | 4.73 | 14.80 | 7.14 | 2.8 | 2.95 |
| stall_exit | 50 | DT (baseline) | 0.0 | 100.0 | 0.0 | 5.02 | 14.57 | – | – | – |
| stall_exit | 50 | Stationary-vehicle rule (ref.) | 96.0 | 4.0 | 0.0 | 4.11 | 15.10 | 7.74 | – | – |
| stall_exit | 50 | Oracle trigger + executor (ref.) | 96.0 | 4.0 | 0.0 | 4.75 | 14.91 | 7.49 | – | – |
| stall_exit | 50 | **DT + LLM-CoC (Qwen3-4B, text)** | **94.0** | 6.0 | 0.0 | 4.75 | 14.89 | 7.49 | 6.2 | 3.00 |
| slow_entry | 50 | DT (baseline) | 0.0 | 100.0 | 0.0 | 5.01 | 13.51 | – | – | – |
| slow_entry | 50 | Stationary-vehicle rule (ref.) | 0.0 | 100.0 | 0.0 | 5.01 | 13.51 | – | – | – |
| slow_entry | 50 | Oracle trigger + executor (ref.) | 0.0 | 100.0 | 0.0 | 5.01 | 13.51 | – | – | – |
| slow_entry | 50 | **DT + LLM-CoC (Qwen3-4B, text)** | **68.0** | 32.0 | 0.0 | 4.28 | 14.61 | 7.07 | 3.0 | 2.69 |
| slow_mid | 50 | DT (baseline) | 0.0 | 100.0 | 0.0 | 5.10 | 14.50 | – | – | – |
| slow_mid | 50 | Stationary-vehicle rule (ref.) | 36.0 | 64.0 | 0.0 | 4.78 | 14.49 | 7.92 | – | – |
| slow_mid | 50 | Oracle trigger + executor (ref.) | 32.0 | 68.0 | 0.0 | 4.82 | 14.42 | 7.91 | – | – |
| slow_mid | 50 | **DT + LLM-CoC (Qwen3-4B, text)** | **86.0** | 14.0 | 0.0 | 4.81 | 14.78 | 7.47 | 2.5 | 2.86 |
| slow_exit | 50 | DT (baseline) | 0.0 | 100.0 | 0.0 | 4.98 | 14.81 | – | – | – |
| slow_exit | 50 | Stationary-vehicle rule (ref.) | 0.0 | 100.0 | 0.0 | 4.09 | 14.69 | – | – | – |
| slow_exit | 50 | Oracle trigger + executor (ref.) | 0.0 | 100.0 | 0.0 | 4.15 | 14.70 | – | – | – |
| slow_exit | 50 | **DT + LLM-CoC (Qwen3-4B, text)** | **0.0** | 100.0 | 0.0 | 4.14 | 14.59 | – | 6.5 | 3.03 |

### DT + VLM-CoC (Qwen3-VL-2B, text+BEV) — complete run, 210 episodes

| Scenario | n | Method | Success % | Collision % | Timeout % | Min dist (m) | Avg speed (m/s) | Completion time (s) | (V)LM calls/ep | Latency (s/call) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal | 30 | DT (baseline) | 100.0 | 0.0 | 0.0 | 7.14 | 15.45 | 7.67 | – | – |
| nominal | 30 | Stationary-vehicle rule (ref.) | 100.0 | 0.0 | 0.0 | 7.14 | 15.45 | 7.67 | – | – |
| nominal | 30 | Oracle trigger + executor (ref.) | 100.0 | 0.0 | 0.0 | 7.14 | 15.45 | 7.67 | – | – |
| nominal | 30 | **DT + VLM-CoC (Qwen3-VL-2B, text+BEV)** | **93.3** | 6.7 | 0.0 | 6.21 | 14.80 | 7.82 | 13.0 | 8.63 |
| stall_entry | 30 | DT (baseline) | 0.0 | 100.0 | 0.0 | 4.95 | 13.75 | – | – | – |
| stall_entry | 30 | Stationary-vehicle rule (ref.) | 60.0 | 36.7 | 3.3 | 4.28 | 14.27 | 7.00 | – | – |
| stall_entry | 30 | Oracle trigger + executor (ref.) | 60.0 | 40.0 | 0.0 | 4.50 | 14.32 | 7.00 | – | – |
| stall_entry | 30 | **DT + VLM-CoC (Qwen3-VL-2B, text+BEV)** | **56.7** | 43.3 | 0.0 | 4.50 | 14.12 | 7.00 | 2.2 | 7.85 |
| stall_mid | 30 | DT (baseline) | 0.0 | 100.0 | 0.0 | 4.94 | 14.10 | – | – | – |
| stall_mid | 30 | Stationary-vehicle rule (ref.) | 80.0 | 20.0 | 0.0 | 4.61 | 14.66 | 7.31 | – | – |
| stall_mid | 30 | Oracle trigger + executor (ref.) | 76.7 | 23.3 | 0.0 | 4.79 | 15.04 | 7.09 | – | – |
| stall_mid | 30 | **DT + VLM-CoC (Qwen3-VL-2B, text+BEV)** | **76.7** | 23.3 | 0.0 | 4.79 | 15.04 | 7.09 | 2.2 | 8.03 |
| stall_exit | 30 | DT (baseline) | 0.0 | 100.0 | 0.0 | 5.03 | 14.61 | – | – | – |
| stall_exit | 30 | Stationary-vehicle rule (ref.) | 93.3 | 6.7 | 0.0 | 4.10 | 15.18 | 7.64 | – | – |
| stall_exit | 30 | Oracle trigger + executor (ref.) | 93.3 | 6.7 | 0.0 | 4.81 | 15.04 | 7.34 | – | – |
| stall_exit | 30 | **DT + VLM-CoC (Qwen3-VL-2B, text+BEV)** | **70.0** | 20.0 | 10.0 | 4.78 | 13.63 | 7.52 | 6.6 | 7.44 |
| slow_entry | 30 | DT (baseline) | 0.0 | 100.0 | 0.0 | 5.04 | 13.58 | – | – | – |
| slow_entry | 30 | Stationary-vehicle rule (ref.) | 0.0 | 100.0 | 0.0 | 5.04 | 13.58 | – | – | – |
| slow_entry | 30 | Oracle trigger + executor (ref.) | 0.0 | 100.0 | 0.0 | 5.04 | 13.58 | – | – | – |
| slow_entry | 30 | **DT + VLM-CoC (Qwen3-VL-2B, text+BEV)** | **33.3** | 66.7 | 0.0 | 4.39 | 12.03 | 7.30 | 3.9 | 3.72 |
| slow_mid | 30 | DT (baseline) | 0.0 | 100.0 | 0.0 | 5.11 | 14.52 | – | – | – |
| slow_mid | 30 | Stationary-vehicle rule (ref.) | 36.7 | 63.3 | 0.0 | 4.83 | 14.55 | 7.86 | – | – |
| slow_mid | 30 | Oracle trigger + executor (ref.) | 30.0 | 70.0 | 0.0 | 4.89 | 14.44 | 7.83 | – | – |
| slow_mid | 30 | **DT + VLM-CoC (Qwen3-VL-2B, text+BEV)** | **30.0** | 66.7 | 3.3 | 4.26 | 11.05 | 8.33 | 4.2 | 3.70 |
| slow_exit | 30 | DT (baseline) | 0.0 | 100.0 | 0.0 | 5.00 | 14.84 | – | – | – |
| slow_exit | 30 | Stationary-vehicle rule (ref.) | 0.0 | 100.0 | 0.0 | 4.14 | 14.72 | – | – | – |
| slow_exit | 30 | Oracle trigger + executor (ref.) | 0.0 | 100.0 | 0.0 | 4.16 | 14.73 | – | – | – |
| slow_exit | 30 | **DT + VLM-CoC (Qwen3-VL-2B, text+BEV)** | **0.0** | 100.0 | 0.0 | 4.32 | 12.73 | – | 7.3 | 3.69 |
