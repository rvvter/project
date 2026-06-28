"""
src/agents/explainer.py

知识讲解师 Agent。

给定路线图中的一个考点，此 Agent 执行以下步骤：
  1. 列出可用的复习笔记（MCP：list_study_files）
  2. 搜索相关内容（MCP：search_notes）
  3. 读取最相关的文件全文（MCP：read_study_file）
  4. 将上下文存入会话记忆（MCP：memory_set）
  5. 生成一份清晰、有据可依的讲解

核心特性：讲解内容基于你自己的笔记，而非仅凭 LLM 训练数据。
如果你的笔记里有相关内容，讲解会引用它。
如果笔记没有覆盖该考点，Agent 会用通用知识补充并说明。

架构模式：
  此 Agent 展示了工具调用循环（tool-calling loop）——
  所有使用外部工具的 Agent 的基础模式。LLM 自行决定
  调用哪些工具以及调用顺序。我们执行工具调用并将结果
  反馈回去，循环直到 LLM 给出最终答案。
"""

import json
import os

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from graph.state import get_current_topic
from llm_factory import build_llm, get_model_name
from mcp_servers.filesystem_server import (
    list_study_files,
    read_study_file,
    search_notes,
)
from mcp_servers.memory_server import memory_get, memory_set


# ─────────────────────────────────────────────────────────────────────────────
# 模型配置
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = get_model_name()


# ─────────────────────────────────────────────────────────────────────────────
# MCP 工具封装为 LangChain 工具
#
# @tool 装饰器将普通 Python 函数转为 LLM 可以调用的 LangChain 工具。
# 函数的 docstring 成为工具描述——LLM 据此判断是否调用以及如何使用。
#
# 关键：docstring 必须清晰具体。模糊的描述会导致
# 错误的工具选择或错误的参数值。
# ─────────────────────────────────────────────────────────────────────────────

@tool
def tool_list_files() -> list[str]:
    """
    列出笔记目录中所有可用的复习笔记文件。
    返回文件名列表，如 ['MySQL索引原理.md', '操作系统基础.md']。
    在任何文件读取操作之前先调用此函数，了解有哪些材料可用。
    """
    return list_study_files()


@tool
def tool_read_file(filename: str) -> str:
    """
    读取指定笔记文件的完整内容。
    参数：
        filename: 由 tool_list_files() 返回的准确文件名。
                  示例：'MySQL索引原理.md' 或 '操作系统基础.md'
    返回文件全文，若文件不存在则返回错误信息。
    """
    return read_study_file(filename)


@tool
def tool_search_notes(query: str) -> str:
    """
    在所有复习笔记中搜索关键词或短语。
    用于在读取文件前，先定位哪个文件涵盖了特定概念。
    参数：
        query: 搜索词（不区分大小写）。示例：'B+树'、'索引优化'
    返回包含匹配行及其文件位置的 JSON 字符串。
    """
    results = search_notes(query)
    if not results:
        return "未找到匹配内容。"
    return json.dumps(results, indent=2, ensure_ascii=False)


@tool
def tool_memory_get(session_id: str, key: str) -> str:
    """
    从会话记忆中读取一个值。
    参数：
        session_id: 当前会话 ID（从状态中获取）。
        key:        要查找的记忆键名。示例：'explained_topics'
    返回存储的值，若未找到则返回 'null'。
    """
    return memory_get(session_id, key)


@tool
def tool_memory_set(session_id: str, key: str, value: str) -> str:
    """
    将一个值存入会话记忆，供后续 Agent 读取。
    参数：
        session_id: 当前会话 ID（从状态中获取）。
        key:        描述性键名。示例：'explained_topics'
        value:      字符串值。复杂数据用 JSON 格式。
    返回确认消息。
    """
    return memory_set(session_id, key, value)


# 所有工具列表，传递给 llm.bind_tools()
EXPLAINER_TOOLS = [
    tool_list_files,
    tool_read_file,
    tool_search_notes,
    tool_memory_get,
    tool_memory_set,
]

# 工具名 → 函数映射，用于分发
TOOL_MAP = {t.name: t for t in EXPLAINER_TOOLS}


# ─────────────────────────────────────────────────────────────────────────────
# 系统 Prompt
#
# 指导 LLM 如何使用工具，而非仅仅告知工具存在。
# "方法"部分至关重要——没有它，LLM 可能跳过工具调用，
# 仅凭训练数据生成讲解。
# ─────────────────────────────────────────────────────────────────────────────

EXPLAINER_SYSTEM_PROMPT = """你是一位资深大厂面试官，正在为求职者深入讲解技术面试考点。

你的核心任务是用自己的专业知识，生成一份结构完整、面试可直接引用的话术级讲解。
本地笔记是参考资料——如有相关笔记优先引用，没有则完全用你的专业知识展开。

工作流程（快速完成，不要超过2轮工具调用）：
1. 可选：调用 tool_search_notes(topic) 快速查看笔记是否有相关内容
2. 可选：如有相关笔记，调用 tool_read_file(filename) 读取关键段落
3. 直接生成完整讲解（这是你唯一的输出，不要额外调用 tool_memory_set）

讲解必须包含以下四个板块，缺一不可，直接输出讲解内容，不要加"讲解已生成"之类的开头：

### 1. 核心概念（200-300字）
- 用生活化类比引入，让零基础也能听懂
- 清晰阐述这个概念解决什么问题、为什么被发明出来
- 底层原理是什么，用通俗语言解释

### 2. 面试回答话术（150-200字）
- 给出"面试时你可以这样回答"的示范文本
- 包含关键术语和逻辑链条
- 让求职者可以直接背诵使用

### 3. 加分亮点（100-150字）
- 横向对比（如：B+树 vs B树、TCP vs UDP、进程 vs 线程）
- 实际生产环境中的踩坑经验
- 源码层面的实现细节（如果有价值）

### 4. 常见误区（50-100字）
- 面试中最容易被追问或答错的一个点
- 给出正确理解和错误理解的对比

重要：你的唯一输出就是这四个板块的讲解内容。不要输出确认语、不要调用 tool_memory_set、
不要说"讲解完毕"。如果笔记中有相关内容就引用，没有就用专业知识展开。
"""


# ─────────────────────────────────────────────────────────────────────────────
# 工具执行
# ─────────────────────────────────────────────────────────────────────────────

def execute_tool_call(tool_call: dict) -> str:
    """
    执行 LLM 请求的单个工具调用，将结果以字符串形式返回。

    参数：
        tool_call: 来自 LLM 响应的 dict，包含键 'name', 'args', 'id'。

    返回：
        放进 ToolMessage 的字符串结果。
        永不抛出异常——错误以字符串形式返回，让 LLM
        能看到问题并尝试恢复。
    """
    name = tool_call["name"]
    args = tool_call["args"]

    if name not in TOOL_MAP:
        return f"错误：未知工具 '{name}'。可用工具：{list(TOOL_MAP.keys())}"

    try:
        result = TOOL_MAP[name].invoke(args)
        # 确保结果是字符串，以便放入 ToolMessage
        if isinstance(result, (list, dict)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)
    except Exception as e:
        return f"执行 {name}({args}) 时出错：{type(e).__name__}：{e}"


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph 节点
# ─────────────────────────────────────────────────────────────────────────────

def explainer_node(state: dict) -> dict:
    """
    LangGraph 节点：知识讲解师

    读取：
        state["study_plan"]（兼容 state["roadmap"]）：获取当前考点
        state["current_topic_index"]                  ：第几个考点
        state["session_id"]                           ：供记忆工具使用

    写入：
        state["messages"]  ：对话 + 工具调用历史
        state["error"]     ：失败时写入错误信息

    工具调用循环：
        1. LLM 产生响应（可能包含工具调用请求）
        2. 如果有工具调用：逐一执行，追加 ToolMessage
        3. 用更新后的消息列表再次调用 LLM
        4. 重复直到 LLM 产生无工具调用的响应
        5. 最终响应即是讲解内容
    """
    # ── 获取当前考点 ───────────────────────────────────────────────────
    topic = get_current_topic(state)
    if topic is None:
        return {
            "error": "没有找到当前考点。请确保考点规划师已运行。",
        }

    session_id = state.get("session_id", "unknown")
    print(f"\n[知识讲解师] 考点：'{topic.title}'")
    print(f"[知识讲解师] 描述：{topic.description}")

    # ── 设置 LLM 并绑定工具 ────────────────────────────────────────────
    # bind_tools() 告知 LLM 有哪些工具可用。
    # LLM 收到工具 schema（名称、描述、参数类型）作为上下文的一部分，
    # 可以请求调用其中任何工具。
    llm = build_llm(temperature=0.3).bind_tools(EXPLAINER_TOOLS)

    # ── 构建初始消息 ────────────────────────────────────────────────────
    messages = [
        SystemMessage(content=EXPLAINER_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"请为我讲解这个面试考点：'{topic.title}'\n"
            f"考点简介：{topic.description}\n"
            f"面试加分建议：{topic.interview_tips}\n"
            f"标签：{', '.join(topic.tags) if topic.tags else '无'}\n"
            f"会话 ID（供记忆工具使用）：{session_id}"
        )),
    ]

    # ── 工具调用循环 ────────────────────────────────────────────────────
    # 安全限制：防止 LLM 持续请求工具而不给出最终答案的死循环
    max_iterations = 8
    final_response = None

    for iteration in range(max_iterations):
        print(f"[知识讲解师] 第 {iteration + 1}/{max_iterations} 次 LLM 调用...")
        try:
            response = llm.invoke(messages)
        except Exception as e:
            print(f"[知识讲解师] LLM 调用失败：{e}")
            return {
                "messages": messages,
                "error": f"讲解师 LLM 调用失败：{e}",
            }
        messages.append(response)

        # 无工具调用 = LLM 已完成，这是最终答案
        if not response.tool_calls:
            final_response = response
            print(f"[知识讲解师] 完成，共 {iteration + 1} 次 LLM 调用")
            break

        # 处理本次响应中的每个工具调用
        print(f"[知识讲解师] 请求了 {len(response.tool_calls)} 个工具调用：")
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"  → {tool_name}({tool_args})")

            result = execute_tool_call(tool_call)

            # 日志中截断过长的结果（消息中不截断）
            log_result = result[:100] + "..." if len(result) > 100 else result
            print(f"    ← {log_result}")

            # 追加 ToolMessage，LLM 将其视为工具的响应
            messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call["id"],   # 必须与请求 ID 匹配
            ))

    # 处理达到最大迭代次数但无最终答案的情况
    if final_response is None:
        error_msg = (
            f"讲解师达到最大迭代次数（{max_iterations}），"
            "未能生成最终讲解。可能模型陷入了工具调用循环。"
        )
        print(f"[知识讲解师] 警告：{error_msg}")
        return {
            "messages": messages,
            "error": error_msg,
        }

    explanation_length = len(final_response.content)
    print(f"[知识讲解师] 讲解内容：{explanation_length} 个字符")

    return {
        "messages": messages,
        "error": None,
        # 显式透传核心状态，LangGraph 1.1.0 状态传播的兼容方案
        "study_plan": state.get("study_plan", state.get("roadmap")),
        "current_topic_index": state.get("current_topic_index", 0),
        "session_id": state.get("session_id", ""),
    }
