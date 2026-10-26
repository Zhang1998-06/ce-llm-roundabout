# v6 results (results/hazard_test_v6)

## 1. Closed loop: success / collision / at-fault collision / timeout (%)

| Method | nominal | stall_mid | stall_exit | slow_exit | stall_past_exit | stall_upstream | mean (need) | mean (no need) |
|---|---|---|---|---|---|---|---|---|
| DT | 94 / 6 / 6 / 0 | 0 / 100 / 100 / 0 | 0 / 100 / 100 / 0 | 0 / 100 / 100 / 0 | 88 / 12 / 12 / 0 | 100 / 0 / 0 / 0 | 0 / 100 / 100 / 0 | 94 / 6 / 6 / 0 |
| DT + naive trigger rule + executor | 94 / 6 / 6 / 0 | 66 / 20 / 20 / 14 | 70 / 6 / 4 / 24 | 0 / 80 / 80 / 20 | 80 / 6 / 6 / 14 | 100 / 0 / 0 / 0 | 45 / 35 / 35 / 19 | 90 / 3 / 3 / 7 |
| DT + LLM v5 (labels + rule prompt, executor picks target) | 94 / 6 / 6 / 0 | 66 / 20 / 20 / 14 | 58 / 4 / 4 / 38 | 0 / 10 / 6 / 90 | 64 / 6 / 6 / 30 | 100 / 0 / 0 / 0 | 41 / 11 / 10 / 47 | 82 / 3 / 3 / 15 |
| DT + LLM v6 (facts only, model picks target) | 96 / 4 / 4 / 0 | 62 / 24 / 24 / 14 | 54 / 34 / 34 / 12 | 0 / 80 / 80 / 20 | 84 / 4 / 4 / 12 | 100 / 0 / 0 / 0 | 39 / 46 / 46 / 15 | 92 / 2 / 2 / 6 |
| DT + designed rule + executor (answer key) | 94 / 6 / 6 / 0 | 66 / 20 / 20 / 14 | 68 / 6 / 4 / 26 | 0 / 18 / 12 / 82 | 88 / 12 / 12 / 0 | 100 / 0 / 0 / 0 | 45 / 15 / 12 / 41 | 94 / 6 / 6 / 0 |

Episodes per cell: 50–50. Seeds identical across methods (paired).

## 2. Efficiency and interventions (mean per episode)

| Method | scenario | avg speed m/s | completion time s (successes) | maneuver starts | supervisor calls | latency s |
|---|---|---:|---:|---:|---:|---:|
| dt | nominal | 15.1 | 7.76 | 0.00 | 0.0 | nan |
| dt | stall_mid | 13.7 | nan | 0.00 | 0.0 | nan |
| dt | stall_exit | 14.2 | nan | 0.00 | 0.0 | nan |
| dt | slow_exit | 14.4 | nan | 0.00 | 0.0 | nan |
| dt | stall_past_exit | 15.1 | 7.74 | 0.00 | 0.0 | nan |
| dt | stall_upstream | 15.1 | 7.77 | 0.00 | 0.0 | nan |
| naive | nominal | 15.1 | 7.76 | 0.00 | 12.3 | 0.00 |
| naive | stall_mid | 12.2 | 7.86 | 1.00 | 4.7 | 0.00 |
| naive | stall_exit | 12.3 | 7.79 | 0.96 | 5.7 | 0.00 |
| naive | slow_exit | 12.1 | nan | 0.96 | 6.6 | 0.00 |
| naive | stall_past_exit | 12.8 | 8.20 | 0.96 | 8.4 | 0.00 |
| naive | stall_upstream | 15.1 | 7.77 | 1.00 | 8.0 | 0.00 |
| llm5 | nominal | 15.1 | 7.76 | 0.02 | 13.0 | 1.87 |
| llm5 | stall_mid | 12.2 | 7.86 | 1.00 | 2.4 | 1.77 |
| llm5 | stall_exit | 11.4 | 7.60 | 0.96 | 5.6 | 1.94 |
| llm5 | slow_exit | 7.4 | nan | 0.96 | 6.4 | 1.94 |
| llm5 | stall_past_exit | 11.9 | 8.12 | 0.98 | 8.1 | 1.92 |
| llm5 | stall_upstream | 15.1 | 7.77 | 0.20 | 11.5 | 1.76 |
| llm | nominal | 14.5 | 7.77 | 1.12 | 3.9 | 0.50 |
| llm | stall_mid | 12.3 | 7.81 | 1.14 | 3.2 | 0.47 |
| llm | stall_exit | 13.0 | 7.59 | 1.36 | 3.2 | 0.49 |
| llm | slow_exit | 12.3 | nan | 1.32 | 3.3 | 0.52 |
| llm | stall_past_exit | 13.0 | 7.98 | 1.44 | 3.1 | 0.45 |
| llm | stall_upstream | 15.1 | 7.67 | 1.70 | 4.8 | 0.56 |
| designed | nominal | 15.1 | 7.76 | 0.00 | 12.3 | 0.00 |
| designed | stall_mid | 12.2 | 7.86 | 1.00 | 3.2 | 0.00 |
| designed | stall_exit | 12.2 | 7.69 | 0.98 | 4.7 | 0.00 |
| designed | slow_exit | 7.7 | nan | 0.98 | 5.1 | 0.00 |
| designed | stall_past_exit | 15.0 | 7.74 | 0.06 | 12.3 | 0.00 |
| designed | stall_upstream | 15.1 | 7.77 | 0.00 | 13.0 | 0.00 |

## 3. Decision quality per supervisor call (privileged reference, never shown to any supervisor)

| Method | scenario | calls | ref says intervene | correct decision + target when ref intervenes | intervenes when ref says no | of which on the injected vehicle | invalid target refused |
|---|---|---:|---:|---:|---:|---:|---:|
| naive | nominal | 614 | 0 | – % | 0 % (0) | 0 | 0 |
| naive | stall_mid | 236 | 126 | 40 % | 0 % (0) | 0 | 0 |
| naive | stall_exit | 287 | 101 | 47 % | 0 % (0) | 0 | 0 |
| naive | slow_exit | 331 | 124 | 0 % | 3 % (7) | 4 | 0 |
| naive | stall_past_exit | 420 | 0 | – % | 11 % (48) | 48 | 0 |
| naive | stall_upstream | 400 | 0 | – % | 12 % (50) | 50 | 0 |
| llm5 | nominal | 651 | 0 | – % | 0 % (1) | 0 | 0 |
| llm5 | stall_mid | 121 | 0 | – % | 41 % (50) | 50 | 0 |
| llm5 | stall_exit | 278 | 94 | 40 % | 3 % (5) | 5 | 0 |
| llm5 | slow_exit | 322 | 124 | 32 % | 4 % (7) | 5 | 0 |
| llm5 | stall_past_exit | 407 | 0 | – % | 12 % (49) | 48 | 0 |
| llm5 | stall_upstream | 575 | 0 | – % | 2 % (10) | 0 | 0 |
| llm | nominal | 194 | 0 | – % | 35 % (67) | 0 | 0 |
| llm | stall_mid | 159 | 60 | 60 % | 26 % (26) | 3 | 0 |
| llm | stall_exit | 158 | 49 | 8 % | 52 % (57) | 6 | 0 |
| llm | slow_exit | 164 | 35 | 0 % | 49 % (63) | 3 | 0 |
| llm | stall_past_exit | 153 | 0 | – % | 48 % (74) | 19 | 0 |
| llm | stall_upstream | 242 | 0 | – % | 48 % (117) | 15 | 0 |
| designed | nominal | 614 | 0 | – % | 0 % (0) | 0 | 0 |
| designed | stall_mid | 160 | 50 | 100 % | 0 % (0) | 0 | 0 |
| designed | stall_exit | 235 | 49 | 100 % | 0 % (0) | 0 | 0 |
| designed | slow_exit | 255 | 44 | 100 % | 2 % (5) | 0 | 0 |
| designed | stall_past_exit | 614 | 3 | 100 % | 0 % (0) | 0 | 0 |
| designed | stall_upstream | 651 | 0 | – % | 0 % (0) | 0 | 0 |

Note: llm5 calls in stall_mid count as 'intervenes when ref says no' because v5's path perception reaches ~70 m along the
route at t = 0, while the v6 facts (and so the reference) stop at 50 m: these are early detections of the injected car,
not judgment errors. In stall_past_exit / stall_upstream every intervention is unnecessary.

## 4. Episode level: first maneuver started on the injected vehicle

| Method | scenario | episodes with a maneuver on it | BYPASS | FOLLOW_VEHICLE | episodes with any maneuver |
|---|---|---:|---:|---:|---:|
| naive | stall_mid | 50/50 | 50 | 0 | 50/50 |
| naive | stall_exit | 47/50 | 47 | 0 | 48/50 |
| naive | slow_exit | 45/50 | 45 | 0 | 48/50 |
| naive | stall_past_exit | 48/50 | 48 | 0 | 48/50 |
| naive | stall_upstream | 50/50 | 50 | 0 | 50/50 |
| llm5 | stall_mid | 50/50 | 50 | 0 | 50/50 |
| llm5 | stall_exit | 48/50 | 38 | 10 | 48/50 |
| llm5 | slow_exit | 46/50 | 1 | 45 | 48/50 |
| llm5 | stall_past_exit | 48/50 | 48 | 0 | 48/50 |
| llm5 | stall_upstream | 0/50 | 0 | 0 | 10/50 |
| llm | stall_mid | 36/50 | 36 | 0 | 50/50 |
| llm | stall_exit | 10/50 | 10 | 0 | 50/50 |
| llm | slow_exit | 3/50 | 3 | 0 | 50/50 |
| llm | stall_past_exit | 19/50 | 19 | 0 | 50/50 |
| llm | stall_upstream | 15/50 | 15 | 0 | 50/50 |
| designed | stall_mid | 50/50 | 50 | 0 | 50/50 |
| designed | stall_exit | 49/50 | 49 | 0 | 49/50 |
| designed | slow_exit | 44/50 | 0 | 44 | 49/50 |
| designed | stall_past_exit | 3/50 | 3 | 0 | 3/50 |
| designed | stall_upstream | 0/50 | 0 | 0 | 0/50 |

## 5. Paired comparisons (same scenario and seed), exact McNemar test

| Comparison | scenarios | outcome | A better | B better | p |
|---|---|---|---:|---:|---:|
| llm vs dt | need | success | 58 | 0 | 6.94e-18 |
| llm vs dt | need | no collision | 81 | 0 | 8.27e-25 |
| llm vs dt | no need | success | 4 | 6 | 0.754 |
| llm vs dt | no need | no collision | 4 | 0 | 0.125 |
| llm vs dt | nominal | success | 1 | 0 | 1 |
| llm vs dt | nominal | no collision | 1 | 0 | 1 |
| llm vs naive | need | success | 1 | 11 | 0.00635 |
| llm vs naive | need | no collision | 4 | 20 | 0.00154 |
| llm vs naive | no need | success | 5 | 3 | 0.727 |
| llm vs naive | no need | no collision | 1 | 0 | 1 |
| llm vs naive | nominal | success | 1 | 0 | 1 |
| llm vs naive | nominal | no collision | 1 | 0 | 1 |
| llm vs llm5 | need | success | 5 | 9 | 0.424 |
| llm vs llm5 | need | no collision | 2 | 54 | 4.43e-14 |
| llm vs llm5 | no need | success | 11 | 1 | 0.00635 |
| llm vs llm5 | no need | no collision | 1 | 0 | 1 |
| llm vs llm5 | nominal | success | 1 | 0 | 1 |
| llm vs llm5 | nominal | no collision | 1 | 0 | 1 |
| llm vs designed | need | success | 1 | 10 | 0.0117 |
| llm vs designed | need | no collision | 5 | 52 | 6.4e-11 |
| llm vs designed | no need | success | 4 | 6 | 0.754 |
| llm vs designed | no need | no collision | 4 | 0 | 0.125 |
| llm vs designed | nominal | success | 1 | 0 | 1 |
| llm vs designed | nominal | no collision | 1 | 0 | 1 |
