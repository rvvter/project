"""
graph/state.py

AgentForge —— 共享状态定义。

这是整个项目最重要的文件。
每个 Agent（考点规划师、知识讲解师、模拟面试官、弱项分析师）
都从这个状态读取数据，并将更新写回这个状态。

LangGraph 在每个节点执行后，将此状态对象通过 SQLite
进行检查点持久化（checkpoint），这也是系统能够：
  - 中断恢复（crash 后继续）
  - 人工审批流程（暂停等待用户确认）
的基础能力。

核心设计决策已在行内注释中说明。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ─────────────────────────────────────────────────────────────────────────────
# 嵌套数据结构
#
# 这些是 AgentState 中使用的复杂对象。
# 使用 @dataclass 而非普通 dict 的两个原因：
#   1. 类型安全——不会因为拼写错误导致难以排查的 Bug
#   2. 自文档化——结构本身就是文档
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InterviewTopic:
    """
    复习路线图中的一个考点。

    由考点规划师创建。
    知识讲解师和模拟面试官读取。
    弱项分析师更新其状态。

    状态流转：
        pending      → 尚未开始复习
        in_progress  → 正在讲解中
        completed    → 面试通过（得分 >= 0.5）
        needs_review → 需要重新复习（得分 < 0.5）
    """
    title: str                    # 考点名称（如 "MySQL 索引原理与优化"）
    description: str              # 考点简介（一句话说明涵盖内容）
    estimated_minutes: int        # 建议复习时长（分钟）

    # 带默认值的字段必须放在无默认值字段之后
    tags: list[str] = field(default_factory=list)           # 标签（如 ["数据库", "高频"]）
    interview_tips: str = ""                                 # 面试加分建议
    prerequisites: list[str] = field(default_factory=list)   # 前置考点名称列表
    status: str = "pending"                                  # 状态：pending / in_progress / completed / needs_review

    def to_dict(self) -> dict:
        """转为普通 dict，用于 JSON 序列化。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "InterviewTopic":
        """从普通 dict 重建 InterviewTopic 实例（JSON 反序列化后使用）。"""
        return cls(
            title=data["title"],
            description=data["description"],
            estimated_minutes=data["estimated_minutes"],
            tags=data.get("tags", []),
            interview_tips=data.get("interview_tips", ""),
            prerequisites=data.get("prerequisites", []),
            status=data.get("status", "pending"),
        )


@dataclass
class InterviewPlan:
    """
    考点规划师生成的完整复习计划。

    包含一个有序的考点列表。顺序很重要——
    前面的考点是后面考点的前置知识。
    """
    job_target: str               # 求职目标（如 "腾讯后台开发暑期实习"）
    total_weeks: int              # 计划总周数
    topics: list[InterviewTopic]  # 考点列表（按学习顺序排列）
    weekly_hours: int = 10        # 每周建议学习小时数

    def to_dict(self) -> dict:
        """转为 JSON 可序列化的 dict。"""
        return {
            "job_target": self.job_target,
            "total_weeks": self.total_weeks,
            "weekly_hours": self.weekly_hours,
            "topics": [t.to_dict() for t in self.topics],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InterviewPlan":
        """从普通 dict 重建 InterviewPlan 实例。"""
        return cls(
            job_target=data.get("job_target", data.get("goal", "")),
            total_weeks=data["total_weeks"],
            weekly_hours=data.get("weekly_hours", 10),
            topics=[InterviewTopic.from_dict(t) for t in data.get("topics", [])],
        )

    def completed_count(self) -> int:
        """已完成的考点数量。"""
        return sum(1 for t in self.topics if t.status == "completed")

    def is_complete(self) -> bool:
        """所有考点都已结束（完成或需重学）时返回 True。"""
        return all(t.status in ("completed", "needs_review") for t in self.topics)


@dataclass
class InterviewQuestion:
    """
    一道面试题，包含用户回答和评分结果。

    由模拟面试官创建。
    弱项分析师读取以定位薄弱环节。
    """
    question: str           # 面试题文本
    expected_answer: str    # 参考答案（评分基准）

    # 以下字段在用户回答后由评分 Agent 填充
    user_answer: str = ""       # 用户的回答文本
    correct: bool = False       # 是否正确
    feedback: str = ""          # 一句话具体反馈
    score: float = 0.0          # 得分（0.0 ~ 1.0）
    difficulty: str = "medium"  # 难度：easy / medium / hard

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InterviewResult:
    """
    一次模拟面试的完整结果。

    存储在 AgentState.interview_results 中，每个考点一条记录。
    弱项分析师读取此数据来决定下一步行动。
    """
    topic: str                        # 考点名称
    questions: list[InterviewQuestion] # 面试题列表
    score: float                      # 平均得分（0.0 ~ 1.0）
    weak_areas: list[str]             # 薄弱概念列表
    timestamp: str = ""               # UTC ISO 格式时间戳

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "score": self.score,
            "weak_areas": self.weak_areas,
            "timestamp": self.timestamp,
            "questions": [q.to_dict() for q in self.questions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InterviewResult":
        """
        从普通 dict 重建 InterviewResult 实例。

        当 LangGraph 从 SQLite checkpoint 中通过 msgpack
        反序列化出 dict 时调用此方法。在恢复中断的会话时触发。
        """
        return cls(
            topic=data.get("topic", ""),
            questions=[],  # 题目数据在进度分析时不需要，跳过
            score=float(data.get("score", 0.0)),
            weak_areas=data.get("weak_areas", []),
            timestamp=data.get("timestamp", ""),
        )

    def passed(self) -> bool:
        """得分 >= 0.5 视为通过。"""
        return self.score >= 0.5

    def strong_pass(self) -> bool:
        """得分 >= 0.75 视为优秀，可进入下一个考点。"""
        return self.score >= 0.75


# ─────────────────────────────────────────────────────────────────────────────
# 主状态类
#
# AgentState 是 LangGraph 管理的 TypedDict。
# 图中的每个节点接收完整的状态，并返回部分更新（仅包含变更的键）。
#
# 为什么用 TypedDict 而不是普通类？
#   LangGraph 要求 dict 兼容的对象。TypedDict 在保持
#   dict 兼容的同时，提供了类型安全保障。
# ─────────────────────────────────────────────────────────────────────────────

from typing import TypedDict


class AgentState(TypedDict):
    """
    AgentForge 系统图的共享状态。

    重要：部分更新机制
    当节点返回 {"approved": True} 时，LangGraph 将其合并到
    现有状态中，而不是替换整个状态 dict。
    节点只需返回它们修改了的键。

    唯一的例外是 `messages` 字段——它使用 `add_messages` 归并器，
    新消息会追加到列表末尾而非覆盖。这对保留对话历史至关重要。
    """

    # ── 对话历史 ────────────────────────────────────────────────────
    # add_messages 是归并器，新消息追加而非覆盖。
    # 保留一次会话中所有 Agent 调用的完整对话历史。
    messages: Annotated[list[BaseMessage], add_messages]

    # ── 会话标识 ────────────────────────────────────────────────────
    # 本次复习会话的唯一 ID。
    # 同时作为 LangGraph 的 thread_id（用于 checkpoint）
    # 和 MCP 内存存储的 key。
    session_id: str

    # ── 用户的求职目标 ──────────────────────────────────────────────
    # 会话开始时设置，整个会话期间不变。
    # 示例："准备腾讯后台开发暑期实习面试"
    job_target: str

    # ── 考点规划师输出 ──────────────────────────────────────────────
    # 由考点规划师节点填充。
    # 后续所有 Agent 读取此字段。
    # 在规划师运行之前为 None。
    study_plan: InterviewPlan | None

    # ── 人工审批 ────────────────────────────────────────────────────
    # 用户在人工审批节点确认复习计划后置为 True。
    # 图形在 approved 为 True 之前不会继续执行。
    approved: bool

    # ── 复习循环位置 ────────────────────────────────────────────────
    # 当前正在复习第几个考点？
    # 0 = 第一个考点，len(study_plan.topics) = 全部完成。
    # 弱项分析师节点在每次面试后递增此值。
    current_topic_index: int

    # ── 面试历史 ────────────────────────────────────────────────────
    # 每个已复习考点对应一条 InterviewResult。
    # 随着会话推进逐步增长。
    interview_results: list[InterviewResult]

    # ── 累计薄弱环节 ────────────────────────────────────────────────
    # 已去重的薄弱概念列表。
    # 跨所有面试累计。弱项分析师使用它来安排补强复习。
    weak_areas: list[str]

    # ── 文件系统路径 ────────────────────────────────────────────────
    # 用户的面试笔记所在目录。
    # 传递给 MCP 文件系统工具。
    study_materials_path: str

    # ── 错误处理 ────────────────────────────────────────────────────
    # 如果节点失败，将错误信息写入此字段而非抛出异常。
    # 图形将路由到错误处理节点。
    # 正常运行时为 None。
    error: str | None


# ─────────────────────────────────────────────────────────────────────────────
# 状态工厂函数
#
# 始终使用此函数创建新的会话状态。
# 不要手动构造 AgentState，工厂函数确保所有字段
# 都有合理的默认值，避免遗漏必填字段。
# ─────────────────────────────────────────────────────────────────────────────

def initial_state(
    job_target: str,
    session_id: str,
    study_materials_path: str = "study_materials/sample_notes",
) -> dict:
    """
    为新复习会话创建初始状态。

    返回普通 dict（而非 AgentState 实例），因为 LangGraph
    接受普通 dict 作为初始状态，内部会根据 TypedDict schema
    验证键名。

    参数：
        job_target:           用户的求职目标。
                              示例："准备腾讯后台开发暑期实习面试"
        session_id:           本次会话的唯一 ID。
                              用作 LangGraph 的 thread_id。
                              格式：短 UUID 或人类可读字符串。
        study_materials_path: 存放 .md 笔记的目录路径。
                              默认为示例笔记目录。

    返回：
        符合 AgentState schema 的 dict，所有字段已初始化。
    """
    return {
        "messages": [],                     # 暂无消息
        "session_id": session_id,
        "job_target": job_target,
        "study_plan": None,                 # 规划师尚未运行
        "approved": False,                  # 用户尚未确认
        "current_topic_index": 0,           # 从第 0 个考点开始
        "interview_results": [],            # 暂无面试记录
        "weak_areas": [],                   # 尚无疑难杂症
        "study_materials_path": study_materials_path,
        "error": None,                      # 暂无错误
    }


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数：安全的状态访问器
#
# 这些辅助函数让 Agent 节点安全地从状态中读取数据，
# 不会因为缺失值或 None 值而崩溃。
# ─────────────────────────────────────────────────────────────────────────────

def get_current_topic(state: dict) -> "InterviewTopic | None":
    """
    获取当前正在复习的考点，如果会话已完成则返回 None。

    在 Agent 节点中的用法：
        topic = get_current_topic(state)
        if topic is None:
            return {"error": "没有当前考点"}

    兼容 dict 和 dataclass 两种形式的 study_plan
    （checkpoint 反序列化可能返回 dict）。
    """
    study_plan = state.get("study_plan", state.get("roadmap"))
    if study_plan is None:
        return None

    # checkpoint 反序列化后，study_plan 可能是普通 dict
    if isinstance(study_plan, dict):
        topics_raw = study_plan.get("topics", [])
    else:
        topics_raw = study_plan.topics

    idx = state.get("current_topic_index", 0)
    if idx >= len(topics_raw):
        return None

    t = topics_raw[idx]
    # 单个 topic 也可能被反序列化为 dict
    if isinstance(t, dict):
        return InterviewTopic.from_dict(t)
    return t


def get_latest_interview_result(state: dict) -> InterviewResult | None:
    """
    获取最近一次面试结果，如果没有面试记录则返回 None。

    兼容 dict 和 dataclass 两种形式——
    checkpoint 恢复后，interview_results 可能是普通 dict 列表。

    在弱项分析师节点中用于分析刚完成的面试表现。
    """
    results = state.get("interview_results", state.get("quiz_results", []))
    if not results:
        return None

    latest = results[-1]

    # msgpack checkpoint 反序列化后可能是普通 dict
    if isinstance(latest, dict):
        return InterviewResult.from_dict(latest)

    return latest


def session_is_complete(state: dict) -> bool:
    """
    所有考点都已复习完毕时返回 True（无论通过还是需重学）。

    由条件边函数使用，决定是继续循环到讲解师还是结束会话。

    兼容 dict 和 dataclass 两种形式的 study_plan。
    """
    study_plan = state.get("study_plan", state.get("roadmap"))
    if study_plan is None:
        return True
    topics = (
        study_plan.get("topics", [])
        if isinstance(study_plan, dict)
        else study_plan.topics
    )
    idx = state.get("current_topic_index", 0)
    return idx >= len(topics)
