# CE 实验报告:LLM/VLM 链式因果推理 + Decision Transformer 处理 roundabout 长尾危险

> **当前主结果是第 0 节(第二版 v5,安全优先)。** 第 1–9 节是第一版 v4,保留作对比。

日期:2026-10-23 · 代码目录:`/media/zhang/990/roudabout/CE-llms-roundabout/ce_vlm_experiment`
完整迭代记录(每一次尝试、诊断和数据):`reports/iteration_log.md`

---

## 0. 第二版 v5:安全优先(当前主结果,2026-10-23)

第一版(下面第 1–9 节,保留作对比)在 6 个危险场景平均成功率 62.3%,但**碰撞率仍有 37.7%**,说服力不够。第二版的目标是在
LLM 仍然决定"何时、做什么机动"的前提下,把碰撞降下来。完整过程见 `reports/iteration_log.md` 的 Phase 4。

### 0.1 结论(测试 seed 500–549,每个场景 50 个回合,开发阶段从未使用)

| 方法(6 个危险场景平均) | 成功 % | 碰撞 % | 自车有责碰撞 % | 超时 % |
|---|---:|---:|---:|---:|
| DT(原始基线) | 0.0 | 100.0 | 100.0 | 0.0 |
| DT + 同样的纵向安全层,不用 LLM(对照) | 0.0 | 55.0 | 39.0 | 45.0 |
| 第一版 DT + LLM v4(执行器 v1) | 62.3 | 37.7 | – | 0.0 |
| **第二版 DT + LLM v5(执行器 v3)** | **45.0** | **9.7** | **9.0** | 45.3 |
| 完美识别 + 同样的机动(参考上限) | 45.7 | 9.7 | 9.0 | 44.7 |

- **安全:** 碰撞率从 DT 的 100% 降到 **9.7%**,比第一版的 37.7% 降低约 3/4。抛锚车·入口 / 出口前、慢速车·出口三个场景的碰撞率是 0 / 0 / 4%。
- **正常交通:** 100% 成功,0 次误介入,和 DT 完全一样。
- **LLM 的作用:** 不用 LLM、只加同样的纵向安全层,成功率是 0%。安全层只能让车停下或慢下来,不能完成任务。LLM v5 的结果和"完美识别"参考几乎完全一样(45.0 vs 45.7),说明识别和机动选择(绕行 / 跟随)基本没有错误。
- **代价:** 成功率从第一版的 62.3% 降到 45.0%。失败的回合变成了超时(停下或跟随,不撞车)。主要原因有三个:
  - 抛锚车·入口改为在入口等待,碰撞 44% → 0%,但成功 56% → 0%。
  - 平缓刹车加大了安全余量,抛锚车·出口前和慢速车·环中有 26–34% 的回合在 11 s 内没走完。
  - 20 s 回合的敏感性分析(见 0.4 节的表):慢速车·出口的跟随在 20 s 内有 52% 能走完(11 s 时是 0%),抛锚车·出口前达到 78%(11 s 时 70%)。
    但碰撞率也有所上升:慢速车·出口 4 → 10%,慢速车·环中 6 → 20%,因为车在交通里停留的时间更长了;20 s 下危险场景平均是 56% 成功、13% 碰撞。
    这说明相当一部分超时是 11 s 时限造成的;但抛锚车·入口在 20 s 下仍然是 98% 超时,是真正的死锁。

### 0.2 改了什么(都由诊断驱动)

1. **两个模拟器事实(从源码确认):**
   - `SLOWER` 按**当前车速**降档,12 m/s 时会直接把目标速度设成 0,相当于急刹到停。
   - IDM 交通车找前车时只看**当前车道分段**(环上每段只有 17–20 m),看不见刚过分段点的停车或慢车,会直接追尾。
   所以在这个模拟器里,任何需要在环上减速的控制器都做不到零碰撞,于是新增了"自车有责碰撞"指标(nuPlan 风格)。
   被追尾、且自车在本车道内、不在汇入区时,判为自车无责;总碰撞率照常报告。
2. **执行器 v3(`SmoothBypassSkill` / `FollowSkill`,`src/hazards.py`):**
   - 执行器直接设定目标速度(发送 IDLE),按需要的减速度平缓刹车,停车余量 10 m + 1 s 时距。
   - LLM 指定的异常车始终算作前车。
   - 换道时目标车道的前后间隙随速度差变化。
   - 越过障碍 8 m 后结束;在出口路上也继续生效。
   - 静止障碍就在汇入点后 25 m 以内时,在入口等待。
3. **prompt v5(`SYSTEM_V5` / `describe_v5`):**
   - 机动从 4 个减到 3 个:FOLLOW_POLICY / BYPASS_OBSTACLE / **FOLLOW_VEHICLE**。去掉了 VLM 曾滥用的 SLOW_DOWN 和 KEEP_SPEED。
   - LLM 需要做一个**因果判断**:慢车正在驶入单车道出口时,自车不可能超到它前面,只能跟随;静止车停在出口区时,可以绕过去再出口。
   - 感知给出地图区域和预测动作这类事实标签("entering the single-lane exit ahead of you" / "stopped in the exit zone"),不给出决策。
   - 离线单步准确率:第一稿用数字距离只有 69%,最终版 96.6%。

### 0.3 系统 prompt v5(原文,`src/coc_supervisor.py: SYSTEM_V5`)

```
You are the high-level reasoning module of a self-driving car (the EGO) in a two-lane roundabout.
A learned low-level policy drives the car. It handles all normal traffic well: entering the ring, following,
vehicles close beside it in the other lane. It was never trained on abnormal situations.
You override it ONLY when there is a concrete ABNORMAL CAUSE.

ROAD: the ego enters the ring on the OUTER lane and must leave at the north exit. The INNER lane is on the ego's left.
Normal ring traffic moves at 12-16 m/s. The NORTH EXIT ROAD IS A SINGLE LANE: every car that leaves there, including
the ego, must pass through it one after another.

WHAT COUNTS AS AN ABNORMAL CAUSE: something IN THE EGO'S PATH AHEAD (its own lane ahead, the ring lane it is about to
enter, or the exit road it will take) that will block it or that it will hit: e.g. a stationary or broken-down vehicle,
an obstacle, a vehicle far slower than traffic, a vehicle coming the wrong way.
NOT a cause: normal moving traffic (even close), anything behind the ego, vehicles beside the ego in the other lane,
stationary vehicles that are not in the ego's path.

MANEUVERS (meta_action). Both maneuvers are executed automatically and brake smoothly when needed.
- FOLLOW_POLICY: let the learned policy drive (default; required when abnormal_cause is "none").
- BYPASS_OBSTACLE: THE DEFAULT MANEUVER for an abnormal vehicle in your path. Passes it in the other ring lane: the car
  keeps moving, changes lane as soon as the gap is free, and returns control to the policy once the obstacle is behind.
  Also for a vehicle "stopped in the exit zone": it does not move, so you pass it and still take the exit.
- FOLLOW_VEHICLE: stay behind the abnormal vehicle at a safe distance (slow down, stop if needed), no lane change.
  ONLY when the abnormal vehicle is "entering the single-lane exit ahead of you" or "on the single-lane exit road":
  it will be in the single exit lane before you, so you can never get in front of it; passing it is impossible.
A vehicle entering the exit at normal speed (12-16 m/s) is NORMAL traffic -> "none".

CHAIN OF CAUSATION — fill the JSON fields in order:
1. scene: ego location, speed, distance to the exit.
2. critical_objects: the vehicle(s) in the ego's path ahead (gap, speed, label); copy numbers exactly.
3. abnormal_cause: "none", or the concrete cause in the ego's path (which vehicle, gap, why abnormal).
4. causal_reasoning: if abnormal_cause is "none": "normal traffic, the policy keeps driving". Otherwise: what happens
   if the policy keeps driving; can the vehicle be passed before the exit; which maneuver removes the cause.
5. meta_action: one of FOLLOW_POLICY, BYPASS_OBSTACLE, FOLLOW_VEHICLE.

EXAMPLES
- Path ahead: vehicle 20 m ahead, 15.8 m/s; a vehicle beside in the inner lane -> "none" -> FOLLOW_POLICY.
- Path ahead clear; a stationary vehicle 15 m behind in the other lane -> "none" (not in my path) -> FOLLOW_POLICY.
- Path ahead: STATIONARY vehicle 30 m ahead in my lane -> cause: it blocks my lane -> BYPASS_OBSTACLE.
- Ego on the access road; the ring lane I enter has a STATIONARY vehicle 10 m after the entry -> cause: it blocks the
  lane I will enter -> BYPASS_OBSTACLE.
- Path ahead: STATIONARY vehicle 25 m ahead, stopped in the exit zone -> cause: it blocks my lane; it does not move, so
  I can pass it and still exit -> BYPASS_OBSTACLE.
- Path ahead: vehicle 30 m ahead at 3.0 m/s (no exit label) -> cause: crawling vehicle in my lane, still on the ring;
  I can pass it -> BYPASS_OBSTACLE.
- Path ahead: vehicle 25 m ahead at 15.2 m/s, entering the single-lane exit ahead of you -> "none" (normal speed) ->
  FOLLOW_POLICY.
- Path ahead: vehicle 18 m ahead at 3.0 m/s, entering the single-lane exit ahead of you -> cause: crawling vehicle
  entering the single exit lane before me; I can never get in front of it -> FOLLOW_VEHICLE.
- Path ahead: vehicle 37 m ahead at 12.5 m/s, entering the single-lane exit ahead of you; vehicle 46 m ahead at 3.0 m/s,
  entering the single-lane exit ahead of you -> cause: the 3.0 m/s vehicle is crawling into the single exit lane (the
  12.5 m/s one is normal); I can never get in front of it -> FOLLOW_VEHICLE.

Keep each field under 25 words. Return only JSON:
{"scene": "...", "critical_objects": "...", "abnormal_cause": "...", "causal_reasoning": "...", "meta_action": "..."}
```

每次调用的 user message = `describe_v5(env, DT 动作)` + `"\n\nReason step by step in the JSON fields, then give meta_action."`。
调用参数:`qwen3:4b-instruct-2507-q4_K_M`,temperature 0,format json,num_predict 320,num_ctx 4096。
测试集共 1681 次调用,JSON 100% 有效,平均延迟 1.66 s(3 个进程并发)。

### 0.4 结果表(测试 seed 500–549)

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


### 0.5 真实调用示例(测试集,日志原文,不是回放)

**stall_mid, seed 501, t = 0**(原始日志 `results/hazard_test_v5/decisions_llm.csv`;DT 建议 FASTER,执行 FASTER,启动执行器:是;回合结局:成功)

user message:
```
EGO: south access road (before the ring), speed 8.0 m/s. Learned policy wants: FASTER.
Your exit (north exit road, single lane) starts 80 m ahead along your path.
You will enter the ring OUTER lane in 24 m.
YOUR PATH AHEAD (access lane and the outer ring lane up to 40 m after the entry): vehicle 31 m ahead along your path, speed 16.1 m/s; vehicle 52 m ahead along your path, speed 0.0 m/s (STATIONARY)
OTHER TRAFFIC (not in your path): 26 m away on ring inner lane, speed 16.0 m/s
```
模型原始输出:
```json
{"scene": "Ego on south access, 8.0 m/s; exit 80 m ahead; enter ring outer lane in 24 m; path ahead: vehicle 31m at 16.1 m/s, 52m at 0.0 m/s (stationary).", "critical_objects": "vehicle 52 m ahead, speed 0.0 m/s (stationary) in my path", "abnormal_cause": "stationary vehicle 52 m ahead in my path blocks lane", "causal_reasoning": "Policy would hit stationary vehicle; can pass in outer lane; bypass removes cause.", "meta_action": "BYPASS_OBSTACLE"}
```

**slow_exit, seed 500, t = 5**(原始日志 `results/hazard_test_v5/decisions_llm.csv`;DT 建议 IDLE,执行 IDLE,启动执行器:是;回合结局:超时(无碰撞))

user message:
```
EGO: ring outer lane, speed 15.9 m/s. Learned policy wants: IDLE.
Your exit (north exit road, single lane) starts 49 m ahead along your path.
YOUR PATH AHEAD (your ring lane, then the exit road): vehicle 38 m ahead, speed 12.4 m/s, entering the single-lane exit ahead of you; vehicle 46 m ahead, speed 3.0 m/s, entering the single-lane exit ahead of you
OTHER TRAFFIC (not in your path): 5 m away on ring inner lane, speed 15.9 m/s; 10 m away on ring outer lane, speed 12.1 m/s; 20 m away on ring inner lane, speed 12.6 m/s; 30 m away on ring inner lane, speed 15.9 m/s
```
模型原始输出:
```json
{"scene": "Ego outer lane, 15.9 m/s, exit 49 m ahead.", "critical_objects": "vehicle 38m ahead 12.4m/s entering exit; vehicle 46m ahead 3.0m/s entering exit", "abnormal_cause": "crawling vehicle 46m ahead entering exit; cannot pass, blocks path", "causal_reasoning": "Policy would hit stationary vehicle; cannot pass it in exit lane; must follow it to avoid collision.", "meta_action": "FOLLOW_VEHICLE"}
```

注意:这个例子的选择是对的(FOLLOW_VEHICLE),但 `causal_reasoning` 把 3.0 m/s 的车说成了"stationary vehicle"。4B 模型经常混淆"慢"和"静止"。

**nominal, seed 501, t = 0**(原始日志 `results/hazard_test_v5/decisions_llm.csv`;DT 建议 FASTER,执行 FASTER,启动执行器:否;回合结局:成功)

user message:
```
EGO: south access road (before the ring), speed 8.0 m/s. Learned policy wants: FASTER.
Your exit (north exit road, single lane) starts 80 m ahead along your path.
You will enter the ring OUTER lane in 24 m.
YOUR PATH AHEAD (access lane and the outer ring lane up to 40 m after the entry): vehicle 31 m ahead along your path, speed 16.1 m/s
OTHER TRAFFIC (not in your path): 26 m away on ring inner lane, speed 16.0 m/s
```
模型原始输出:
```json
{"scene": "Ego on south access, 8.0 m/s; exit 80 m ahead; enter ring outer lane in 24 m; vehicle 31 m ahead at 16.1 m/s.", "critical_objects": "vehicle 31 m ahead, 16.1 m/s, label: none", "abnormal_cause": "none", "causal_reasoning": "Normal traffic; policy can drive; no obstruction in path.", "meta_action": "FOLLOW_POLICY"}
```

### 0.6 第二版的局限

- 抛锚车·入口:安全了(0% 碰撞),但完不成任务(在入口等待,20 s 内仍然等不到空档)。
- 剩余的碰撞主要在慢速车·入口(30%)和抛锚车·环中(18%)。两者都和 IDM 交通车互相追尾、堵住车道有关。预知参考的结果也一样,所以问题不在 LLM。
- "自车有责"是简化规则,不是正式的责任判定。总碰撞率照常报告。
- 只测了文本 LLM。VLM 在第一版里明显更弱,第二版没有再测。
- 执行器参数(停车余量、时距、换道间隙)是在开发集上通过诊断确定的。测试集只运行了一次,运行前测试方案已经冻结,写在 iteration log 里。

---

## 1. 任务与结论(第一版 v4)

**老师的要求:** 提出一个有 LLM/VLM 参与的架构或算法,在之前研究用过的 roundabout 场景上实现,并从任务完成、安全、效率三方面评估。

**结论(在调参从未用过的 seed 上测得)**
- **正常交通:** 原始 DT 成功率 100%(seed 500–549)。DT + LLM 一次都没有介入,结果和 DT 完全一样,没有任何退化。
- **长尾危险 1(抛锚静止车,3 个位置):** 原始 DT 全部撞车,成功率 0%。DT + LLM 达到 56% / 70% / 94%,和"已知答案触发"的参考上限基本相同(56 / 70 / 96%)。
- **长尾危险 2(慢速车,prompt 和代码完全没有为它修改,用来检验泛化):** 原始 DT 是 0%。DT + LLM 在两个位置达到 68% 和 86%,而手写的"静止车规则"只有 0% 和 36%。第三个位置(慢速车卡在单车道出口分叉处)所有方法都是 0%,物理上无法绕开。
- **VLM 版(Qwen3-VL-2B,文本加鸟瞰图,完整结果,每个场景 30 个回合):** 6 个危险场景的平均成功率是 44.4%(DT 是 0%),所有危险场景都超过 DT,但明显不如文本 LLM:抛锚车 57 / 77 / 70%,慢速车 33 / 30 / 0%。正常场景 30 个回合里误伤了 2 个(93.3%)。每次调用的延迟约为 LLM 的 2–3 倍。

提升完全来自 LLM 的**识别和决策**:判断"路径前方有没有异常、是什么异常、要不要绕行"。执行器只负责完成 LLM 选定的机动,本身不判断危险。用"已知答案"去触发同一个执行器,得到的成绩就是识别完全正确时的上限,LLM 在危险 1 上达到了这个上限。对于危险 2,只认静止车的规则和已知答案触发都识别不到,而 LLM 靠同一个 prompt 里的常识定义("比车流慢得多的车也是异常")就能处理。

---

## 2. 架构

```
           ┌─────────────────────────────────────────────────────────────┐
 env 状态 ─▶│ 感知摘要 describe_v4(按路径组织的对象列表,纯几何,不做判断)   │
 (+BEV图)   └───────────────┬─────────────────────────────────────────────┘
                            ▼
           ┌─────────────────────────────────────────────────────────────┐
           │ 高层推理器 LLM/VLM(Chain-of-Causation, JSON)              │
           │ scene → critical_objects → abnormal_cause → causal_reasoning │
           │ → meta_action ∈ {FOLLOW_POLICY, BYPASS_OBSTACLE,            │
           │                  SLOW_DOWN, KEEP_SPEED}                     │
           │ 约束:abnormal_cause = "none" ⇒ 强制 FOLLOW_POLICY            │
           └───────┬──────────────────────────────┬──────────────────────┘
        FOLLOW_POLICY                       BYPASS_OBSTACLE
                   ▼                              ▼
  ┌────────────────────────────┐   ┌──────────────────────────────────────────┐
  │ Decision Transformer(冻结) │──▶│ 绕行机动执行器 BypassSkill(时间扩展的 option) │
  │ 占据栅格 → 元动作            │   │ 纵向控制交给 DT;禁止换回原车道/刹停;       │
  └──────────────┬─────────────┘   │ 相邻车道有空隙就换道;障碍物到身后就结束      │
                 │                 └────────────────────┬─────────────────────┘
                 └────────── 执行的元动作 ◀───────────────┘ → highway-env
                             (同时写回 DT 的动作历史)
```

- **低层:** 原研究冻结的 DT(`uncertaintyDT/CNNDT_baseline`,checkpoint `output2`)。输入 4×41×50 的占据栅格,输出 5 个离散元动作。DT 没有接收 LLM 输出的接口,也没有重新训练。
- **接口:** LLM 不改 DT 的输入,而是在动作层面介入:选择让 DT 继续开,或者启动一个机动。实际执行的动作会写回 DT 的历史动作序列(`HIST_EXECUTED`),这样 DT 之后的决策知道刚才发生了什么。
- **调用频率:** 每个策略步(2 Hz)查询一次,从起点一直到驶入北出口道路为止。机动进行期间不查询。

---

## 3. 场景、方法与评估设定

**环境:** 原研究本地修改过的 highway-env `roundabout-v0`。双车道环岛(内侧半径约 20 m,外侧约 24 m),策略频率 2 Hz,回合时长 11 s(22 步)。目标速度只有 0/8/16 m/s 三档。自车从南入口以 8 m/s 出发,从外侧车道进环,在北出口离开。成功的定义是到达出口且没有碰撞(沿用原研究的 exit evaluator)。

**场景(`src/hazards.py`):**

| 场景 | 内容 |
|---|---|
| `nominal` | 原始环境,不做修改 |
| `stall_entry` / `stall_mid` / `stall_exit` | 一辆**静止的抛锚车**停在自车必经的外侧车道上:入口后 12 m(se→ex)、环中段(ex→ee,10 m)、出口前(ee→nx,8 m) |
| `slow_entry` / `slow_mid` / `slow_exit` | 同样三个位置,换成一辆以 **3 m/s 低速行驶**的车(IDM,不换道)。**这是泛化测试,设计 prompt 和执行器时都没有使用** |

**对比方法:**

| 方法 | 作用 |
|---|---|
| DT(baseline) | 原研究的冻结策略 |
| DT + LLM-CoC | **本方法**,Qwen3-4B-Instruct-2507(q4_K_M,本地 Ollama),只用文本 |
| DT + VLM-CoC | **本方法**,Qwen3-VL-2B-Instruct(q4_K_M),文本加 BEV 图像 |
| 静止车规则(参考) | 手写规则,用模拟器真值:前方 25 m 内同车道有静止车就 LANE_LEFT。只能处理静止车 |
| 已知答案触发 + 执行器(参考) | 用模拟器真值("40 m 内外侧车道有静止车")触发同一个 BypassSkill,用来衡量 LLM 识别得好不好 |

**seed 划分:** 开发和调参用 200–499,最终评估用 500–599(之前从未使用)。最终评估:LLM 每个场景 50 个回合(500–549),VLM 每个场景 30 个回合(500–529),为了节省时间而缩减。对比时参考方法只取相同的 seed。

**指标:**
- **任务完成:** 成功率、超时率。
- **安全:** 碰撞率、整个回合中与其他车辆的最小中心距离(m)。
- **效率:** 平均速度、成功回合的完成时间、每回合 (V)LM 调用次数、每次调用延迟。

---

## 4. 最终结果(测试 seed)

完整表格(含全部指标):`results/hazard_test/final_tables.md` 和 `final_tables.csv`,用 `python scripts/make_final_tables.py` 生成。

### 4.1 DT + LLM-CoC(Qwen3-4B,文本),seed 500–549,每个场景 50 个回合(完整结果)

| 场景 | DT 成功 | 静止车规则 | 已知答案 + 执行器 | **DT + LLM 成功** | LLM 碰撞 | LLM 最小距离 (m) | LLM 平均速度 (m/s) | LLM 完成时间 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal | 100 | 100 | 100 | **100** | 0 | 7.11 | 15.43 | 7.68 |
| stall_entry | 0 | 56 | 56 | **56** | 44 | 4.49 | 14.17 | 7.00 |
| stall_mid | 0 | 72 | 70 | **70** | 30 | 4.73 | 14.80 | 7.14 |
| stall_exit | 0 | 96 | 96 | **94** | 6 | 4.75 | 14.89 | 7.49 |
| slow_entry | 0 | 0 | 0 | **68** | 32 | 4.28 | 14.61 | 7.07 |
| slow_mid | 0 | 36 | 32 | **86** | 14 | 4.81 | 14.78 | 7.47 |
| slow_exit | 0 | 0 | 0 | 0 | 100 | 4.14 | 14.59 | – |

单位:成功率和碰撞率是百分比。DT 在所有危险场景中碰撞率都是 100%。

**汇总(6 个危险场景,共 300 个回合):** DT 成功率 0%,碰撞率 100%;DT + LLM 成功率 62.3%,碰撞率 37.7%。正常场景两者都是 100%。

**LLM 调用统计:** 共 1911 次调用,JSON 全部有效(100%)。其中 FOLLOW_POLICY 1596 次,BYPASS_OBSTACLE 315 次。正常场景每回合约 13 次调用,**介入 0 次**。危险场景每回合 2.5–6.5 次调用,启动绕行后不再查询。
**延迟:** 闭环里平均 2.9 s/次,但这是 3 个并行进程加上 VLM 同时占用同一块 RTX 3070 Ti 时测得的,包含排队时间。离线单次调用约 0.66 s。

### 4.2 DT + VLM-CoC(Qwen3-VL-2B,文本 + BEV),seed 500–529,每个场景 30 个回合(完整结果)

| 场景 | DT 成功 | 静止车规则 | 已知答案 + 执行器 | **DT + VLM 成功** | VLM 碰撞 | VLM 超时 | VLM 平均速度 (m/s) | 延迟 (s/次) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal | 100 | 100 | 100 | **93.3** | 6.7 | 0 | 14.80 | 8.6 |
| stall_entry | 0 | 60.0 | 60.0 | **56.7** | 43.3 | 0 | 14.12 | 7.9 |
| stall_mid | 0 | 80.0 | 76.7 | **76.7** | 23.3 | 0 | 15.04 | 8.0 |
| stall_exit | 0 | 93.3 | 93.3 | **70.0** | 20.0 | 10.0 | 13.63 | 7.4 |
| slow_entry | 0 | 0 | 0 | **33.3** | 66.7 | 0 | 12.03 | 3.7 |
| slow_mid | 0 | 36.7 | 30.0 | **30.0** | 66.7 | 3.3 | 11.05 | 3.7 |
| slow_exit | 0 | 0 | 0 | 0 | 100 | 0 | 12.73 | 3.7 |

**汇总(6 个危险场景,共 180 个回合):** DT 成功率 0%,DT + VLM 44.4%,碰撞率 53.3%。
**调用统计:** 共 1184 次调用,有效 JSON 95.7%。其中 FOLLOW_POLICY 741 次,BYPASS_OBSTACLE 237 次,SLOW_DOWN 206 次。正常场景里共介入 15 次。延迟在闭环里与 LLM 同时运行、共用 GPU 时平均 6.5 s/次;离线单次调用约 1.3 s。

---

## 5. 分析

1. **LLM 的价值在于识别"训练里没见过的异常"。** DT 在训练分布内很强(正常场景 100%),但一遇到路径上的静止车或慢速车就 100% 撞车,因为它从没见过这种情况。LLM 按 CoC 顺序推理:先找出路径上的关键物体,再判断是否异常、什么原因,最后才选机动。这样能在不重新训练 DT 的前提下处理这些情况。
2. **对正常驾驶没有副作用。** 正常场景 50 个回合、约 670 次调用里,LLM 一次都没有介入,成功率、速度和完成时间都与 DT 完全相同。这主要归功于 `abnormal_cause` 字段的约束(第 7 节 v3)。
3. **识别接近完美。** 在抛锚车场景,LLM 触发和"已知答案"触发同一个执行器,结果几乎完全相同(56/70/94 对 56/70/96)。这说明剩下的失败来自执行层面,比如换道时旁边车道没有空隙、入口后距离太短,而不是 LLM 认错。
4. **泛化能力(最能体现 LLM 价值的结果)。** 慢速车场景是在 prompt 冻结之后才加入的。只认静止车的规则和已知答案触发因为速度阈值识别不到它,只有 0–36%;LLM 用同一个 prompt 达到了 68% 和 86%。要让规则也处理这种情况,就得为每一种新危险手写新规则,而 LLM 不需要。
5. **安全与效率。** 在危险场景,碰撞率从 100% 降到 6–44%(slow_exit 除外)。成功回合的完成时间是 7.0–7.5 s,和 DT 在正常场景里差不多,绕行没有明显拖慢速度。没有出现超时,因为执行器不会刹停。最小距离略低于正常场景(4.3–4.8 m 对 7.1 m),这是在近距离绕过障碍物的结果。
6. **VLM 对比 LLM。** 2B VLM 在所有危险场景都超过 DT(平均 44.4% 对 0%),所以"有 VLM 参与、比 baseline 强"也成立。但它明显不如 4B 文本 LLM(危险场景平均 62.3%),主要有三点:(a) 正常场景误介入 15 次,误伤了 2/30 个回合;(b) 经常选择 SLOW_DOWN(206 次),导致车速下降(慢速车场景平均只有 11–12 m/s),出口前抛锚车场景出现 10% 超时;(c) 在慢速车场景只有 30–33%,LLM 是 68–86%。在这个任务里,结构化文本已经包含了所需的全部信息,鸟瞰图没有带来额外收益,更小的 VLM 反而推理更弱。如果要体现视觉输入的价值,需要换成文本里没有的信息,比如直接从图像检测障碍物,或使用更大的 VLM。

---

## 6. 方法细节(关键信息)

### 6.1 感知摘要 `describe_v4`(`src/coc_supervisor.py`)
这一步只做几何整理,不做任何判断。在环上,按车道角度计算沿环的弧长距离,报告自车车道前方最近的 2 辆车;在进环道上,报告同车道前方的车,以及外侧环道入口后 40 m 内的车(标为"在你的路径上")。其余 40 m 内的车作为"其他交通"列出。速度低于 0.5 m/s 时标注 `(STATIONARY)`。

示例(stall_mid,自车在进环道上):
```
EGO: south access road (before the ring), speed 14.5 m/s. Learned policy wants: IDLE.
You will enter the ring OUTER lane in 14 m.
YOUR PATH AHEAD (access lane and the outer ring lane up to 40 m after the entry): vehicle 41 m ahead along your path, speed 0.0 m/s (STATIONARY)
OTHER TRAFFIC (not in your path): 16 m away on ring inner lane, speed 16.0 m/s; 20 m away on ring outer lane, speed 16.0 m/s; 27 m away on ring inner lane, speed 15.9 m/s; 34 m away on ring inner lane, speed 15.3 m/s
```
示例(nominal,自车在环上):
```
EGO: ring outer lane, speed 16.0 m/s. Learned policy wants: IDLE.
YOUR PATH AHEAD (your ring lane): clear for 50 m
OTHER TRAFFIC (not in your path): 8 m away on ring inner lane, speed 16.0 m/s; 13 m away on ring outer lane, speed 11.2 m/s; 18 m away on ring inner lane, speed 14.0 m/s; 31 m away on ring inner lane, speed 15.9 m/s
```

### 6.2 系统 prompt(最终版 v4,原文)
```
You are the high-level reasoning module of a self-driving car (the EGO) in a two-lane roundabout.
A learned low-level policy drives the car. It handles all normal traffic well: entering the ring, following,
vehicles close beside it in the other lane. It was never trained on abnormal situations.
You override it ONLY when there is a concrete ABNORMAL CAUSE.

ROAD: the ego enters the ring on the OUTER lane and leaves at the north exit. The INNER lane is on the ego's left.
Both ring lanes lead to the exit. Normal ring traffic moves at 12-16 m/s.

VEHICLE DYNAMICS (important): speed is set in three levels: 0, 8 and 16 m/s. Ring traffic drives 16 m/s, so a car
that slows down or stops on the ring is likely to be hit from behind. A lane change needs the car to be moving.

WHAT COUNTS AS AN ABNORMAL CAUSE: something IN THE EGO'S PATH AHEAD (its own lane ahead, or the ring lane it is about to
enter) that will block it or that it will hit: e.g. a stationary or broken-down vehicle, an obstacle, a vehicle far
slower than traffic, a vehicle coming the wrong way.
NOT a cause: normal moving traffic (even close), anything behind the ego, vehicles beside the ego in the other lane,
stationary vehicles that are not in the ego's path.

MANEUVERS (meta_action)
- FOLLOW_POLICY: let the learned policy drive (default; required when abnormal_cause is "none").
- BYPASS_OBSTACLE: start the bypass maneuver around the obstacle in the ego's path. It is executed automatically:
  it keeps the car moving, changes to the free ring lane as soon as the gap is free (also right after entering the
  ring), and returns control to the policy once the obstacle is behind. Choose it as soon as an obstacle is in the path,
  even if the lane change is not possible yet.
- SLOW_DOWN: reduce speed one level (never to a stop). Only for a cause that a lane change cannot solve.
- KEEP_SPEED: hold the current speed.

CHAIN OF CAUSATION — fill the JSON fields in order:
1. scene: ego location and speed.
2. critical_objects: the vehicle(s) in the ego's path ahead (gap, speed); copy numbers exactly from the input.
3. abnormal_cause: "none", or the concrete cause in the ego's path (which vehicle, gap, why abnormal).
4. causal_reasoning: if abnormal_cause is "none": "normal traffic, the policy keeps driving". Otherwise: what happens
   if the policy keeps driving, and which maneuver removes the cause.
5. meta_action: one of FOLLOW_POLICY, BYPASS_OBSTACLE, SLOW_DOWN, KEEP_SPEED.

EXAMPLES
- Path ahead: vehicle 20 m ahead, 15.8 m/s; a vehicle beside in the inner lane -> "none" -> FOLLOW_POLICY.
- Path ahead clear; a stationary vehicle 15 m behind in the other lane -> "none" (not in my path) -> FOLLOW_POLICY.
- Path ahead: STATIONARY vehicle 30 m ahead in my lane -> cause: it blocks my lane -> BYPASS_OBSTACLE.
- Ego on the access road; the ring lane I enter has a STATIONARY vehicle 10 m after the entry -> cause: it blocks the
  lane I will enter -> BYPASS_OBSTACLE.

Keep each field under 25 words. Return only JSON:
{"scene": "...", "critical_objects": "...", "abnormal_cause": "...", "causal_reasoning": "...", "meta_action": "..."}
```
VLM 版在系统 prompt 末尾追加:
```
You also get a bird's-eye-view image of the current scene: blue box = ego, orange boxes = other vehicles, labels = speed in m/s; the ego drives upward in the image. Use the image together with the numbers.
```
用户消息是感知摘要,后面加一句 `Reason step by step in the JSON fields, then give meta_action.`(VLM 版附带 BEV PNG)。

**调用参数:** Ollama `/api/chat`,`format: json`,temperature 0,`num_predict` 320,`num_ctx` 4096。

**输出处理:**
- JSON 无效或调用出错 → FOLLOW_POLICY(由 DT 继续开)。
- `abnormal_cause` 以 "none" 开头 → 强制 FOLLOW_POLICY。
- BYPASS_OBSTACLE → 以路径前 3 辆车里最慢的那辆作为障碍物,启动执行器。
- SLOW_DOWN → 降一档,但最低保持 8 m/s。
- KEEP_SPEED → IDLE。

### 6.3 CoC 输出示例(真实闭环记录,开发集)
```json
{"scene": "EGO on south access, 8.0 m/s; entering outer ring in 24 m; path ahead: vehicle 31m at 16.2 m/s, 52m at 0.0 m/s (stationary).",
 "critical_objects": "vehicle 31 m ahead, 16.2 m/s; vehicle 52 m ahead, 0.0 m/s",
 "abnormal_cause": "stationary vehicle 52 m ahead blocks path ahead",
 "causal_reasoning": "Policy would drive faster, risking collision with stationary vehicle; bypass avoids impact.",
 "meta_action": "BYPASS_OBSTACLE"}
```
```json
{"scene": "EGO on south access, 8.0 m/s; entering outer ring lane in 24 m.",
 "critical_objects": "vehicle 36 m ahead, speed 3.0 m/s",
 "abnormal_cause": "stationary vehicle blocks path ahead",
 "causal_reasoning": "Policy would hit stationary vehicle; bypass avoids collision by changing lane.",
 "meta_action": "BYPASS_OBSTACLE"}
```
第二条来自慢速车场景:动作选对了,但模型把 3 m/s 的车说成了 "stationary",描述不准确。
```json
{"scene": "EGO on south access, 8.0 m/s; entering outer ring lane in 24 m.",
 "critical_objects": "vehicle 31 m ahead, 16.2 m/s",
 "abnormal_cause": "none",
 "causal_reasoning": "Normal traffic; policy can follow without risk.",
 "meta_action": "FOLLOW_POLICY"}
```

### 6.4 绕行机动执行器 `BypassSkill`(`src/hazards.py`)
这是一个通用的车道绕行 option,本身不判断危险,只在 LLM 启动后运行:
- 进环道上和换道完成后:执行 DT 的动作,但去掉换道,也不允许刹到 0。
- 在环上:检查相邻车道自车侧后 6 m 到侧前 3.6 m 的空隙,有空隙就换道,没有就保持 DT 的纵向动作继续行驶(不刹车)。
- 障碍物到自车身后 3 m 以上时结束,控制权交还 DT。

---

## 7. 迭代过程摘要(详见 `reports/iteration_log.md`)

| 阶段 | 做法 | 结果 | 结论 |
|---|---|---|---|
| 1 | 正常场景,VLM/LLM 只判断 YIELD/NORMAL,YIELD 时把 FASTER 改成 SLOWER 或 IDLE | 测试 seed 上没有超过 DT(96.7%);27 种时机变体在开发集上几乎全部净亏 | DT 在正常场景的失败是时机上的偶然,决策时刻的状态和成功回合几乎一样,任何基于状态的监督器都分不开 |
| 2 | 预测安全护盾(用模拟器副本前推 2–3 步) | 开发集 99.0% 对 96.3% | 能超过 DT,但提升来自物理推演,不是 LLM,放弃 |
| 3.1 | 换成长尾危险(抛锚车);CoC prompt v1,输入原始相对坐标 | DT 0%,规则 48–80%;LLM 在正常场景造成撞车 | 4B 模型读不懂原始坐标 |
| 3.2 | 先离线单步测试再上闭环;v2 按车道整理感知 + 示例;v3 加 `abnormal_cause` 约束,换道以选项形式给出 | 离线:正常误介入 72% → 21% → **0%**,危险 13/35 → 35/35;闭环正常 92% = DT,危险 0/32/40% | 识别对了,但多步执行失败:刹停后换不了道、判断时有时无 |
| 3.3 | 机动级执行器的可行性(已知答案触发) | 简单绕行的上限约为 45 / 60–80 / 75–80% | 场景本身难;执行器能解决 LLM 的执行问题 |
| 3.4 | v4:BYPASS_OBSTACLE 机动接口 + 动力学常识 | 入口和出口达到上限,环中段因执行器在进环道上不加速而超时 | 执行器缺陷,不是 LLM 的问题 |
| 3.5 | 执行器的纵向控制交还 DT | 开发集 48/72/76%,慢速车 68/72/0%(未调参) | 冻结方案,开始最终评估 |

---

## 8. 局限

- **`slow_exit` 无法通过:** 慢速车正好卡在单车道出口分叉处,所有方法都是 0%,LLM 选择了 BYPASS。理想做法是减速跟车(超时但不碰撞),这需要模型知道"这辆车也要走同一个出口",而当前输入里没有这个信息。我没有为它调参。
- **剩余失败来自执行层面:** 换道时旁边车道没有空隙、入口后距离太短,"已知答案"的上限也只有 56–96%。
- **感知摘要用的是模拟器真实状态:** 相当于完美感知。现实中需要换成真实的感知模块。
- **样本量:** 测试集每个场景 50(LLM)或 30(VLM)个回合,95% 置信区间大约 ±13–17 个百分点。但 0% 对 56–94% 的差距远大于这个误差。
- **VLM 用的是 2B 小模型**(受 8 GB 显存限制),视觉输入的潜力没有被充分测试。
- **只测试了两种危险类型和一个环岛布局。**
- **延迟:** 每次调用约 0.7 s(LLM,离线)到 1.3 s(VLM),远慢于 2 Hz 控制周期的要求。这里是按步同步等待,实时部署需要异步处理或换更小的模型。

---

## 9. 复现

```bash
cd /media/zhang/990/roudabout/CE-llms-roundabout/ce_vlm_experiment
PY=/home/zhang/anaconda3/envs/HF-DT/bin/python          # Ollama 运行在 127.0.0.1:11434
S=nominal,stall_entry,stall_mid,stall_exit,slow_entry,slow_mid,slow_exit
# 参考方法(无 LLM)
METHODS=dt,rule,skill SCEN=$S SEEDS=500-599 TAG=test NP=14 $PY scripts/run_hazard.py
# 本方法
PROMPT_VER=4 METHODS=llm SCEN=$S SEEDS=500-549 TAG=test NP=3 $PY scripts/run_hazard.py
PROMPT_VER=4 METHODS=vlm SCEN=$S SEEDS=500-529 TAG=test NP=3 $PY scripts/run_hazard.py
$PY scripts/make_final_tables.py
# 离线单步测试(调 prompt 时用)
$PY scripts/build_offline_set.py 200 210 results/offline_dev.pkl
$PY scripts/offline_eval.py results/offline_dev.pkl qwen3:4b-instruct-2507-q4_K_M 4 0 300
```

**文件清单**

| 文件 | 内容 |
|---|---|
| `src/coc_supervisor.py` | CoC prompt v1–v4、感知摘要 `describe*`、`path_ahead`、`CoCSupervisor` |
| `src/hazards.py` | 危险场景包装器 `Stall`、参考规则、`BypassSkill` |
| `src/evaluation.py`、`src/bev_renderer.py` | DT 加载和 BEV 渲染(沿用) |
| `scripts/run_hazard.py` | 闭环评估 |
| `scripts/make_final_tables.py` | 最终表格 |
| `scripts/build_offline_set.py`、`scripts/offline_eval.py` | 离线单步测试 |
| `results/hazard_test/` | 最终回合数据 `episodes_*.csv`、每一步的 CoC 决策 `decisions_*.csv`、`final_tables.*` |
| `results/hazard_dev*/` | 各轮开发集数据 |
| `reports/iteration_log.md` | 全部迭代记录 |

`reports/experiment_report.md` 是第 1 阶段(正常场景 YIELD 监督器)的旧报告,保留作参考。

---

# 附录 A:完整结果表

## A.1 第二版 v5(当前)

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

## A.2 第一版 v4


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


---

# 附录 B:完整迭代记录


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

---

# 附录 C:危险位置与 BypassSkill(第一版执行器 v1)动作映射(按源代码核对)

## C.1 危险车位置(`src/hazards.py` SCENARIOS + `highway_env/envs/roundabout_env.py`)

坐标系为 highway-env 屏幕坐标(x 向右、y 向下,所以 +y 是南),环岛中心在 (0, 0)。外车道(lane index 1)半径 24 m,内车道半径 20 m。
自车从南入口路驶入(`ses→se` 终点在 (5.5, 25.5)),在外环行驶 `se→ex→ee→nx`,经过东侧,从北出口 `nx→nxs` 驶出。
外环上这段路线全长 55.3 m(17.59 + 20.11 + 17.59)。

| 场景 | 车道段 (lane_id) | 段内纵向位置 s | 坐标 (x, y) m | 极角 | 距入环节点 se(弧长) | 距出口节点 nx(弧长) |
|---|---|---:|---|---:|---:|---:|
| stall/slow_entry | ("se","ex",1) 外车道 | 12.0 m | (19.08, 14.56) | 37.4° | 12.0 m | 43.3 m |
| **stall/slow_mid** | **("ex","ee",1) 外车道** | **10.0 m** | **(24.00, 0.05)** | **0.1°(正东)** | **27.6 m** | **27.7 m** |
| stall/slow_exit | ("ee","nx",1) 外车道 | 8.0 m | (17.52, −16.40) | −43.1° | 45.7 m | 9.6 m |

- 各段的极角范围:`se→ex` 为 66°→24°,`ex→ee` 为 24°→−24°,`ee→nx` 为 −24°→−66°。
- **mid 位于 `ex→ee` 段(长 20.11 m)的中点,也就是环岛正东侧,正好在自车环上路线的中点。** "ex"/"ee" 是东出口/东入口的连接节点,自车不在那里离开。
- stall 场景:在该点放一辆速度 0 的 `Vehicle`。slow 场景:放一辆 3 m/s 的 `IDMVehicle`,`enable_lane_change=False`,沿路线开往 "nxs"。
- 两种场景都会先删除距该点 8 m 以内的其他车,再重新计算观测。

## C.2 BypassSkill 动作映射(`src/hazards.py: BypassSkill.act`;运行时参数 `BypassSkill(17.0, 6.0, slow_on_access=False)`)

启动:当 (V)LM 输出 `BYPASS_OBSTACLE` 且 `path_ahead` 非空时,把 `path_ahead` 前 3 辆车中速度最低的那辆设为 obstacle 并启动。
启动后每一步都由 skill 产生动作,**不再调用 (V)LM**,直到 skill 结束。
(`v_target=17.0` 和 `slow_on_access` 在 `act()` 中没有使用;实际起作用的参数只有 `gap = 6.0`。)

每一步按以下顺序处理(`ts` 为自车目标速度档 0/8/16 m/s,动作编号:0 LANE_LEFT、1 IDLE、2 LANE_RIGHT、3 FASTER、4 SLOWER):

1. **结束检查:** obstacle 相对自车、沿自车朝向的纵向距离 < −3 m(已在自车后方 3 m 以上)时 → `stop()`,本步直接执行 DT 动作,之后恢复调用 (V)LM。
2. **`keep(a_dt)` 规则**(只改 DT 动作中的换道和"减速到停"):

   | DT 提议 | 条件 | 执行器改成 |
   |---|---|---|
   | LANE_LEFT / LANE_RIGHT | 任意 | **IDLE**(换道只能由执行器发起) |
   | SLOWER | ts ≤ 8(会降到 0 档,或已是 0 档) | **IDLE**(不允许停车) |
   | SLOWER | ts = 16 | SLOWER(16→8,允许) |
   | IDLE / FASTER | ts < 1(已停) | **FASTER** |
   | IDLE / FASTER | ts ≥ 1 | 不变 |

   边界情况:当 ts = 0 且 DT 提议换道或 SLOWER 时,`keep` 返回 IDLE,而不是 FASTER(先判断了换道和 SLOWER)。
3. **入口路上**(自车还没进环),或**已经发出过换道**(`done_change=True`)→ 执行 `keep(a_dt)`。纵向控制仍由 DT 负责。
4. **在环上且还没换道:** 检查另一条环车道上的车辆。按自车半径计算弧长差 g(g > 0 表示对方在自车前方;代码已验证符号)。
   - 若有车满足 −6 m ≤ g ≤ +3.6 m(`-gap ≤ g ≤ 0.6·gap`)→ 执行 `keep(a_dt)`,继续行驶,等待空档。
   - 若该窗口无车 → 置 `done_change=True`,发出**一次** LANE_LEFT(外→内)或 LANE_RIGHT(内→外)。
   - `done_change` 在发出换道命令时就被置位,不检查换道是否真的完成。

**结束方式:**
- 正常结束只有一种,就是第 1 步:越过障碍物 3 m。
- 没有超时机制。
- 另外,`run_hazard.py` 只在自车路线未走完、且不在北出口路上时才调用高层模块(包括 skill)。自车驶入北出口路后,即使 skill 仍处于 active,也直接执行 DT 动作。
- 碰撞或回合结束时,episode 本身终止。

示例(stall_mid / seed 500,replay):t = 0 启动 skill;t = 0–5 执行 `keep(a_dt)`(FASTER、IDLE…);t = 6 内车道空档满足条件,发出 LANE_LEFT;此后执行 `keep(a_dt)`,越过静止车后 skill 结束。

### C.3 第二版执行器 v3(`SmoothBypassSkill` / `FollowSkill`)与 v1 的区别
- **纵向:** 执行器自己设定 `target_speed` 并发送 IDLE,不用 SLOWER。SLOWER 会按当前车速降档,12 m/s 时直接降到 0。
  - 需要刹车时,减速度 a = (v² − v_前²) / 2(间距 − 10 m − 1 s·v),限制在 1.5–6 m/s²。
  - 否则按 IDM 式安全速度,每步最多变化 −1 / +2 m/s。
- **前车:** 本车道最近的车、横向距离 2.6 m 以内的切入车,以及 **LLM 指定的那辆异常车**(始终算在内)。
- **换道(仅绕行):** 目标车道前方间隙 > 7 + 1.5·max(0, v_自 − v_它),后方间隙 > 9 + 1.5·dv + dv²/2(dv = 后车速度 − 自车速度)。
  换道完成之前,原车道的前车仍然算作前车。
- **入口:** 一般情况下仍由 DT 自己入环。只有静止障碍在汇入点后 25 m 以内时,才在入口停车等待,直到外环上游和内车道都空出来。
- **结束:** 障碍物落到自车后方 8 m 以外时结束。跟随机动在障碍物离开 60 m 以外时结束。在北出口路上也继续生效。
