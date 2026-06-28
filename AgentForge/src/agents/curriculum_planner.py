"""
src/agents/curriculum_planner.py

考点规划师 Agent。

职责：接收用户的求职目标，生成一份结构化的复习路线图
（InterviewPlan），包含有序的考点列表、时间估计和前置依赖。

这是系统中最简单的 Agent：
  - 无 MCP 工具调用
  - 一次 LLM 调用 + 结构化 JSON 输出
  - 确定性解析为 InterviewPlan dataclass

它展示了每个 Agent 遵循的基础模式：
  读取状态 → 调用 LLM → 解析输出 → 返回状态更新
"""

import json
import os

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import InterviewPlan, InterviewTopic
from llm_factory import build_llm, get_model_name


# ─────────────────────────────────────────────────────────────────────────────
# 模型配置
#
# 从环境变量读取，方便不修改代码切换模型。
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = get_model_name()


# ─────────────────────────────────────────────────────────────────────────────
# 系统 Prompt
#
# System Prompt 是给 Agent 的「岗位描述」。
# 它在所有调用中保持不变，只有 HumanMessage 随请求变化。
#
# Prompt 设计要点：
#   1. "只返回 JSON，不要 markdown 代码块"——确保可解析
#   2. 显式 schema + 字段名和类型——降低 LLM 输出偏差
#   3. 明确规则约束输出空间——避免不合理的计划
#   4. 4~8 个考点——足够展示循环流程，又不会让演示过长
# ─────────────────────────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """你是一位资深技术面试导师，曾在大厂担任面试官多年。
你的任务是根据用户的求职目标，生成一份结构化的复习路线图。

求职目标示例：
- "准备腾讯后台开发暑期实习面试"
- "准备字节跳动 AI 工程师校招面试"
- "准备美团 Java 后端春招面试"

只返回纯 JSON，不要加任何解释文字，不要用 markdown 代码块。
JSON 必须严格匹配以下 schema：

{
  "job_target": "求职目标原文",
  "total_weeks": <1 到 8 的整数>,
  "weekly_hours": <每周建议学习小时数，8 到 20>,
  "topics": [
    {
      "title": "考点名称（如：MySQL 索引原理与 B+树优化）",
      "description": "该考点的具体内容和面试常问角度，一句话说明",
      "estimated_minutes": <30 到 180 的整数>,
      "tags": ["标签1", "标签2"],
      "interview_tips": "面试时回答该问题的加分建议，一句话",
      "prerequisites": ["前置考点名称（如有），否则空数组"],
      "status": "pending"
    }
  ]
}

规则：
- 考点从基础到进阶排序，前面的是后面的前置知识
- 每个考点标注面试频率标签（高频/中频/必考/加分项等）
- 每个考点必须包含 interview_tips 面试加分建议
- 生成 4 到 8 个考点，覆盖该岗位的核心技术栈
- tags 用于分类（如：["数据库", "高频"]）
- 所有 status 必须为 "pending"
"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM 工厂
#
# 每次调用创建新实例而非模块级单例。
# 避免长时间运行进程中的隐性状态问题，
# 也便于测试时使用不同配置。
# ─────────────────────────────────────────────────────────────────────────────

def build_planner_llm():
    """
    为考点规划师创建 LLM 客户端。

    temperature=0.1：结构化输出需要极低温度。
    高温会引入随机性，让 JSON 解析变得不可靠。
    创造性任务（如讲解）使用 0.3~0.4。
    """
    return build_llm(temperature=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# 输出解析器
#
# 与节点函数分离，便于独立测试（无需运行 LLM 调用）。
# ─────────────────────────────────────────────────────────────────────────────

def parse_plan_json(json_string: str) -> InterviewPlan:
    """
    将 LLM 输出的 JSON 字符串解析为 InterviewPlan dataclass。

    参数：
        json_string: LLM 返回的原始字符串。

    返回：
        填充完毕的 InterviewPlan 实例。

    异常：
        ValueError: JSON 无效或缺少必填字段。
                    节点函数捕获此异常并返回错误状态。
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(
            "LLM 返回了无效 JSON。\n"
            f"错误信息：{e}\n"
            f"原始输出（前 300 字符）：{json_string[:300]}"
        )

    # 验证顶层必填字段
    required = ["job_target", "total_weeks", "topics"]
    for field in required:
        if field not in data:
            # 兼容旧版字段名
            if field == "job_target" and "goal" in data:
                data["job_target"] = data["goal"]
            else:
                raise ValueError(f"LLM JSON 缺少必填字段：'{field}'")

    if not isinstance(data["topics"], list) or len(data["topics"]) == 0:
        raise ValueError("LLM JSON 的 'topics' 必须是非空列表")

    # 构建 InterviewTopic 对象列表
    topics = []
    for i, t in enumerate(data["topics"]):
        for field in ["title", "description", "estimated_minutes"]:
            if field not in t:
                raise ValueError(f"第 {i} 个 topic 缺少必填字段：'{field}'")
        topics.append(InterviewTopic(
            title=t["title"],
            description=t["description"],
            estimated_minutes=int(t["estimated_minutes"]),
            tags=t.get("tags", []),
            interview_tips=t.get("interview_tips", ""),
            prerequisites=t.get("prerequisites", []),
            status=t.get("status", "pending"),
        ))

    return InterviewPlan(
        job_target=data.get("job_target", data.get("goal", "")),
        total_weeks=int(data["total_weeks"]),
        weekly_hours=int(data.get("weekly_hours", 10)),
        topics=topics,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph 节点
#
# 这是 LangGraph 调用的函数。接收完整状态 dict，
# 返回仅包含变更键的部分更新 dict。
# ─────────────────────────────────────────────────────────────────────────────

def curriculum_planner_node(state: dict) -> dict:
    """
    LangGraph 节点：考点规划师

    读取：
        state["job_target"]（兼容旧 state["goal"]）：用户的求职目标

    写入：
        state["study_plan"]  : 成功时填充 InterviewPlan
        state["messages"]    : LLM 消息（prompt + 响应）
        state["error"]       : 失败时写入错误信息，成功时为 None

    执行流程：
        1. 从状态中提取求职目标
        2. 构建 LLM 消息（系统 prompt + 人类消息）
        3. 调用 LLM
        4. 解析 JSON 响应为 InterviewPlan
        5. 返回部分状态更新
    """
    # 兼容旧字段名 goal
    job_target = state.get("job_target", state.get("goal", "")).strip()

    if not job_target:
        return {"error": "未提供求职目标。请在运行前设置 state['job_target']。"}

    print(f"\n[考点规划师] 正在为求职目标生成复习路线图：'{job_target}'")

    llm = build_planner_llm()

    # 构建消息列表。
    # SystemMessage = Agent 的指令（不变）
    # HumanMessage  = 具体请求（每次不同）
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"请为以下求职目标创建复习路线图：{job_target}"),
    ]

    # 调用 LLM。这是同步阻塞调用。
    # 底层通过 HTTP POST 发送请求到 DeepSeek API 或 Ollama。
    print(f"[考点规划师] 正在调用 {MODEL_NAME}...")
    try:
        response = llm.invoke(messages)
    except Exception as e:
        print(f"[考点规划师] LLM 调用失败：{e}")
        return {
            "error": f"LLM 调用失败：{e}",
            "messages": messages,
        }

    # response 是 AIMessage 对象
    # response.content 是模型返回的字符串
    try:
        study_plan = parse_plan_json(response.content)
    except ValueError as e:
        print(f"[考点规划师] 解析错误：{e}")
        return {
            "error": str(e),
            "messages": messages + [response],
        }

    # 成功，打印生成结果
    print(f"[考点规划师] 已生成复习路线图：{len(study_plan.topics)} 个考点，"
          f"{study_plan.total_weeks} 周计划")
    for i, topic in enumerate(study_plan.topics, 1):
        tags_str = f" [{', '.join(topic.tags)}]" if topic.tags else ""
        prereqs = f"（前置：{', '.join(topic.prerequisites)}）" if topic.prerequisites else ""
        tips = f" | 面试加分：{topic.interview_tips}" if topic.interview_tips else ""
        print(f"  {i}. {topic.title}{tags_str}{prereqs}，{topic.estimated_minutes} 分钟{tips}")

    # 返回部分状态更新。
    # 只包含我们修改了的键。
    # LangGraph 将此合并到完整状态中。
    return {
        "study_plan": study_plan,
        "messages": messages + [response],
        "error": None,
    }
