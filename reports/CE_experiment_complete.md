# CE 实验报告:LLM/VLM 链式因果推理 + Decision Transformer 处理 roundabout 长尾危险

日期:2026-10-23 · 代码目录:`/media/zhang/990/roudabout/CE-llms-roundabout/ce_vlm_experiment`
完整迭代记录(每一次尝试、诊断和数据):`reports/iteration_log.md`

---

## 1. 任务与结论

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

# 附录 A:完整结果表(全部指标)

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

---

# 附录 C:危险位置与 BypassSkill 动作映射(按源代码核对)

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
