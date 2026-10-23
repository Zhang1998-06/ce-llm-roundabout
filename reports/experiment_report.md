# DT vs. DT + VLM on the roundabout: experiment report

## 1. Setup
- GPU: NVIDIA RTX 3070 Ti, 8 GB. VLM on GPU (`ollama ps`: 100 % GPU), DT on CPU. Peak whole-GPU memory during the run: 2872 MiB.
- VLM: `qwen3-vl:2b-instruct-q4_K_M` via a local, user-space Ollama 0.40.8, temperature 0, num_predict 128, num_ctx 2048, JSON output.
- DT: frozen `CNNDT_baseline/output2` checkpoint (see `repository_audit.md`). No retraining.
- Environment: `roundabout-v0`, OccupancyGrid obs (4, 50, 41), 5 discrete meta-actions, 2 Hz.
- VLM intervention: queried every 4 DT steps (= 2.0 s of simulation time) while the ego is in the entry region (lanes ser/ses/se and <= 70 m from the roundabout centre); latest valid mode held between queries; YIELD and DT = FASTER -> SLOWER, otherwise DT action. The DT history stores the executed action (identical to B0, where executed = DT).
- Evaluation: 30 paired seeds (0-29), fixed in `configs/experiment.yaml` before any comparison; smoke seeds 1000-1003 were used only for debugging.
- VLM input: matplotlib BEV of the current state restricted to the DT's observation window (+-50 m x +-41 m around the ego), plus a text list of the 6 nearest visible vehicles (relative position, speed, relative speed, "on ring" flag) and the DT's proposed action. No future information.
- Simulation pauses during VLM inference, so this is a simulation-time closed-loop result, not a real-time one.

## 2. Method
B0 is the unchanged DT. B1 uses the same checkpoint and code, plus an inference-time supervisor: the frozen VLM returns NORMAL/YIELD, and only a YIELD with a DT FASTER action changes the executed action. Nothing is trained, so no retraining or fine-tuning is needed.

## 3. Results (30 paired episodes, measured)
| Method | Success Rate (%) | Collision Rate (%) | Average Speed (m/s) | Completion Time (s) | Average Reward |
|---|---:|---:|---:|---:|---:|
| Original DT | 96.7 | 3.3 | 15.09 | 7.78 | 20.99 |
| DT + VLM | 0.0 | 0.0 | 0.28 | n/a (no successes) | 18.33 |

- Original DT: 29 successes, 1 collision (seed 22), 0 timeouts. DT + VLM: 0 successes, 0 collisions, 30 timeouts.
- Paired outcomes: DT fail -> VLM success 0; DT success -> VLM fail 29; both success 0; both fail 1.
- VLM: 180 calls, 91 overrides (50.6 % of calls), JSON validity 100 %, mean latency 0.48 s, YIELD in 180/180 calls.
- Reward for DT + VLM is not much lower because the reward only has a small speed term and no penalty for stalling.
- **B1 did not improve over B0; it was much worse.** The collision the DT had was avoided only because the ego never entered the roundabout.

## 4. Qualitative analysis / diagnosis
- Example (`figures/vlm_decision_example.png`, seed 0, step 0): DT proposes FASTER, the VLM answers YIELD ("Circulating traffic presents a plausible entry conflict"), the action becomes SLOWER. Here traffic really is on the ring, so the call is defensible.
- The failure is systematic, not a single bad call: the first override happens at step 0 in all 30 episodes (the ego starts at 8 m/s, SLOWER sets the target speed to 0), and the ego is below 1 m/s from step 3 on, stopped about 40 m from the roundabout centre (before the entry).
- The 2B VLM answers YIELD every time, including 54 of 180 replies whose own reason says there is no conflict. Once stopped, the DT proposes FASTER again, the VLM keeps saying YIELD, and the ego never restarts. The prompt asks for "plausible" conflict, and the always-dense ring traffic in this environment makes that nearly always true.
- Caveat: the all-YIELD behaviour is a property of this model, prompt and integration rule; I did not tune the prompt or rule after seeing results, as instructed. A second-order contributor is that the integration rule has no release condition.

## 5. Limitations
30 paired episodes only, one VLM, one prompt, one integration rule; synthetic BEV (not camera perception); a 2B VLM has weak spatial reasoning; 0.5 s latency per call (vs 0.5 s DT step); no training; only a narrow high-level intervention. This does not show anything about real-world safety, and the result does not rule out that a stronger VLM, or a rule with a release condition, would help; it was not tested.

## 6. CE relevance
The experiment is a working closed-loop VLM-assisted planning stack: a learned DT policy, a frozen VLM supervisor with image + text input, a bounded intervention interface, and real measurements. Its negative result is informative for the CE: a small VLM used as a yield gate can fail systematically by over-yielding. Future work could add traffic-rule retrieval (RAG) as extra context for the VLM, which was not implemented here.

## Reproduce
```
export OLLAMA_MODELS=/media/zhang/990/ollama_local/models
/media/zhang/990/ollama_local/bin/ollama serve &
python scripts/smoke.py            # debugging only
python scripts/run_comparison.py   # B0 + B1, raw CSVs, comparison table
python scripts/make_figures.py     # figures + diagnostics
```
