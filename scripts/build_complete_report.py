"""Assemble reports/CE_experiment_complete.md (the single deliverable) from its parts:
final_report.md + appendix A (all result tables) + B (iteration log) + C (geometry / v1 executor) + D (v6 prompt).
Appendices A.1/A.2 and C are taken from the previous complete file, unchanged.
    python scripts/build_complete_report.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
import coc_supervisor as C

RED = '<span style="color:#c00000">{}</span>'
old = open("reports/CE_experiment_complete.md").read()
a1 = old[old.index("## A.1 "):old.index("# 附录 B")].rstrip().rstrip("-").rstrip()
a1 = a1.replace("## A.1 第二版 v5(当前)", "## A.1 第二版 v5(" + RED.format("旧版本") + ")", 1)
appc = old[old.index("# 附录 C"):]
mark = "附录 D:第三版"                                          # appendix D is rebuilt below
appc = appc[:appc.rindex("\n#", 0, appc.index(mark))] if mark in appc else appc
tables = open("results/hazard_test_v6/tables_v6.md").read().split("\n", 1)[1]
log = open("reports/iteration_log.md").read().split("\n", 1)[1]

D = ["# " + RED.format("附录 D:第三版 v6 的 prompt 原文、输入格式与代码位置"), "",
     RED.format("以下为新增内容。") + "", "",
     "## D.1 系统 prompt(原文,`src/coc_supervisor.py: SYSTEM_V6_NOCO`,冻结版本)", "",
     "```text", C.SYSTEM_V6_NOCO, "```", "",
     "## D.2 用户消息格式(`describe_v6`,冻结开关 V6_KIN=1、V6_LANDMARK=1)", "",
     "每一步的用户消息就是 `describe_v6()` 的输出,不加任何其他文字;例子见第 0.6 节。各行含义:",
     "- `EGO`:自车所在道路、车道和速度;在入口路上时给出到汇入点的距离。",
     "- `ROUTE`:导航,即北出口在环上的位置。",
     "- `POLICY PROPOSES`:DT 的提议动作和当前目标速度。",
     "- `PERCEPTION`:50 m 内的车辆,按道路和车道分组,从上游到下游排列;每辆车给 id、沿环位置和速度;"
     "环上车辆另给按当前速度多久到达它的位置;同一坐标里插入地标 [YOU]、[YOUR MERGE POINT]、[YOUR EXIT]。", "",
     "## D.3 代码位置", "",
     "| 内容 | 位置 |", "|---|---|",
     "| 场景 | `src/hazards.py: SCENARIOS`(stall_past_exit、stall_upstream 为新增) |",
     "| 事实观测 / prompt / 查询 | `src/coc_supervisor.py: perceive_v6, describe_v6, SYSTEM_V6_NOCO, ask_v6` |",
     "| 对照规则与参考标签 | `src/rules_v6.py: naive_rule, designed_rule, reference` |",
     "| 闭环、目标 ID 校验、日志 | `scripts/run_hazard.py`(`VER == 6` 分支;方法 `llm`、`llm5`、`naive`、`designed`) |",
     "| 离线单步检查(开发集) | `scripts/offline_v6.py` |",
     "| 表格 | `scripts/make_tables_v6.py` → `results/hazard_test_v6/tables_v6.md` |",
     "| 逐步回放 | `scripts/replay_v6.py` |", ""]

out = [open("reports/final_report.md").read().rstrip(), "", "---", "",
       "# 附录 A:完整结果表", "",
       "## A.0 " + RED.format("第三版 v6(当前,测试 seed 600–649)"), "", tables.strip(), "", a1, "", "---", "",
       "# 附录 B:完整迭代记录", "", log.strip(), "", "---", "", appc.rstrip(), "", "---", "", "\n".join(D)]
open("reports/CE_experiment_complete.md", "w").write("\n".join(out) + "\n")
print("written", sum(1 for _ in open("reports/CE_experiment_complete.md")), "lines")
