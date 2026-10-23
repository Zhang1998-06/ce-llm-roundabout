# LLM 调用实例:stall_mid / seed 500 / t = 0

来源标注:**[原始日志]** = 正式测试 `results/hazard_test/decisions_llm.csv` 第 860 行或 `episodes_llm.csv`; **[代码常量]** = `src/coc_supervisor.py`;**[replay]** = 用同一 seed 回放(`scripts/replay_call.py`)截获,不是原始日志。原始运行只记录了 JSON 解析后的 5 个字段,没有记录 user message、原始返回字符串和最终执行动作。

## 1. Scenario / seed / timestep [原始日志]
- scenario `stall_mid`,seed **500**,episode timestep **t = 0**(2 Hz,0.0 s);方法 `llm`(qwen3:4b,PROMPT_VER=4)
- 本回合结果 [原始日志]:success=1,collision=0,完成时间 7.0 s,LLM 调用 3 次,覆盖 1 次

## 2. System prompt [代码常量 SYSTEM_V4;replay 截获的请求与之相同;文本 LLM 不加 VISION_NOTE]
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
请求参数 [replay 截获]:
```json
{"model": "qwen3:4b-instruct-2507-q4_K_M", "stream": false, "format": "json", "keep_alive": "60m", "options": {"temperature": 0, "num_predict": 320, "num_ctx": 4096}}
```

## 3. User message [replay —— 原始日志未记录]
前几行是 `describe_v4(env, 'FASTER')` 的输出,最后一句是 `ask()` 固定追加的。t = 0 的环境状态只由 `reset(seed=500)` 决定,原始输出里的数字(24 m;32 m / 16.0;52 m / 0.0)也和这里一致。
```
EGO: south access road (before the ring), speed 8.0 m/s. Learned policy wants: FASTER.
You will enter the ring OUTER lane in 24 m.
YOUR PATH AHEAD (access lane and the outer ring lane up to 40 m after the entry): vehicle 32 m ahead along your path, speed 16.0 m/s; vehicle 52 m ahead along your path, speed 0.0 m/s (STATIONARY)
OTHER TRAFFIC (not in your path): 27 m away on ring inner lane, speed 15.9 m/s; 39 m away on ring inner lane, speed 15.9 m/s

Reason step by step in the JSON fields, then give meta_action.
```

## 4. Ollama 返回的五字段 JSON [原始日志]
日志保存的是解析后的 5 个字段,下面按 prompt 规定的顺序列出;原始字符串的键顺序和空白格式没有保存。valid=True,延迟 2.53 s。
```json
{
 "scene": "EGO on south access, 8.0 m/s; entering outer ring lane in 24 m.",
 "critical_objects": "vehicle 32 m ahead, 16.0 m/s; vehicle 52 m ahead, 0.0 m/s (STATIONARY)",
 "abnormal_cause": "stationary vehicle 52 m ahead blocks path ahead",
 "causal_reasoning": "Policy would drive faster, risking collision with stationary vehicle; bypass avoids hit.",
 "meta_action": "BYPASS_OBSTACLE"
}
```
作为对照,replay 得到的原始字符串**不是原始输出**:它的 `causal_reasoning` 与日志不同,说明 Ollama 在 temperature=0 下也不是逐字确定的。
```json
{"scene": "EGO on south access, 8.0 m/s; entering outer ring lane in 24 m.", "critical_objects": "vehicle 32 m ahead, 16.0 m/s; vehicle 52 m ahead, 0.0 m/s (STATIONARY)", "abnormal_cause": "stationary vehicle 52 m ahead blocks path ahead", "causal_reasoning": "Policy would drive faster, but stationary vehicle blocks path; bypassing avoids collision.", "meta_action": "BYPASS_OBSTACLE"}
```

## 5. 动作
- DT 原本建议的动作:**FASTER** [原始日志]
- 是否启动 BypassSkill:**是**,在 t = 0 启动,障碍物 = 路径前 3 辆车里最慢的那辆(52 m 外的静止车) [由原始日志 meta_action=BYPASS_OBSTACLE 和代码逻辑决定,replay 确认]
- t = 0 最终执行的动作:**FASTER** [replay];执行器在入口路上把纵向控制交还给 DT,所以不算覆盖
- 该回合唯一一次覆盖:t = 6 执行器发出 LANE_LEFT [replay],与原始日志 n_overrides=1 一致
- 回放执行序列 [replay]:FASTER IDLE IDLE IDLE IDLE IDLE LANE_LEFT IDLE IDLE IDLE IDLE IDLE IDLE IDLE IDLE IDLE IDLE IDLE IDLE IDLE IDLE IDLE

## 6. 回放和原始回合的一致性
success、collision、完成时间、平均速度、最小间距、调用次数、覆盖次数全部相同(replay:7.0 s,15.716 m/s,3.389 m;原始:7.0 s,15.716 m/s,3.389 m)。只有延迟不同。

证据文件:`results/hazard_test/replay_llm_stall_mid_seed500.json`(完整请求和响应、逐步执行动作、BypassSkill 启动时刻)。
