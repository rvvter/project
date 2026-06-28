"""
src/graph/workflow.py

AgentForge 的 LangGraph 工作流定义。

完整流程图：
  START → 考点规划师 → 人工审批
        →（确认）→ 知识讲解师 → 模拟面试官 → 弱项分析师
        →（还有考点）→ 知识讲解师（循环）
        →（全部完成）→ END
        →（拒绝）→ 考点规划师（重新生成）

核心设计决策：
  - SqliteSaver：每个节点执行后持久化到磁盘（断电也不丢进度）
  - interrupt()：在人工审批节点暂停，等待用户确认
  - 路由函数是纯 Python 函数（控制流中无 LLM 调用）
  - 所有业务逻辑在 agents/ 中，此文件只做连线
"""

import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agents.curriculum_planner import curriculum_planner_node
from agents.explainer import explainer_node
from agents.human_approval import human_approval_node
from agents.progress_coach import progress_coach_node
from agents.quiz_generator import quiz_generator_node
from graph.state import AgentState, session_is_complete

# 注意：LangGraph 从 SQLite checkpoint 反序列化 dataclass 时
# 会得到普通 dict。graph/state.py 中的所有状态访问函数
# 通过 isinstance 检查和 from_dict() 类方法兼容两种形式。


# ─────────────────────────────────────────────────────────────────────────────
# 路由函数
# ─────────────────────────────────────────────────────────────────────────────

def route_after_approval(state: dict) -> str:
    """
    人工审批之后的分支决策。

    返回：
        "explainer"           : 用户确认了计划，开始复习
        "curriculum_planner"  : 用户拒绝，重新生成计划
    """
    if state.get("approved", False):
        return "explainer"
    return "curriculum_planner"


def route_after_coach(state: dict) -> str:
    """
    弱项分析之后的分支决策。

    返回：
        "explainer"  : 还有考点未复习
        "end"        : 全部考点已覆盖
    """
    if session_is_complete(state):
        return "end"
    return "explainer"


# ─────────────────────────────────────────────────────────────────────────────
# 图构建
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(
    db_path: str = "data/checkpoints.db",
    interrupt_before: list | None = None,
):
    """
    构建并编译 AgentForge 复习备考图。

    参数：
        db_path:          SQLite checkpoint 数据库路径。
        interrupt_before: 需要暂停的节点名列表（供 Streamlit UI 集成使用）。

    SqliteSaver 将 checkpoint 持久化到指定的数据库文件。
    数据库文件不存在时会自动创建。

    人工审批节点使用 interrupt() 暂停执行，
    等待用户输入后再继续。
    """
    # 确保 data 目录存在
    Path("data").mkdir(exist_ok=True)

    # 允许通过环境变量覆盖数据库路径
    if db_path == "data/checkpoints.db":
        db_path = os.getenv("CHECKPOINT_DB", "data/checkpoints.db")

    builder = StateGraph(AgentState)

    # ── 注册所有节点 ────────────────────────────────────────────────
    builder.add_node("curriculum_planner", curriculum_planner_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("explainer", explainer_node)
    builder.add_node("quiz_generator", quiz_generator_node)
    builder.add_node("progress_coach", progress_coach_node)

    # ── 静态边 ──────────────────────────────────────────────────────
    builder.add_edge(START, "curriculum_planner")
    builder.add_edge("curriculum_planner", "human_approval")
    builder.add_edge("explainer", "quiz_generator")
    builder.add_edge("quiz_generator", "progress_coach")

    # ── 条件边 ──────────────────────────────────────────────────────
    # 审批后：开始复习 or 重新规划
    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "explainer": "explainer",
            "curriculum_planner": "curriculum_planner",
        },
    )

    # 分析后：下个考点 or 结束
    builder.add_conditional_edges(
        "progress_coach",
        route_after_coach,
        {
            "explainer": "explainer",
            "end": END,
        },
    )

    # ── 编译并挂载 SQLite checkpoint ───────────────────────────────
    # 关键：直接创建连接，不要用 context manager。
    # checkpointer 必须在进程生命周期内保持连接。
    # graph 是模块级变量，生命周期长于 build_graph()。
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or [],
    )


# 模块级 graph 实例，供 main.py 和 streamlit_app.py 导入
graph = build_graph()
