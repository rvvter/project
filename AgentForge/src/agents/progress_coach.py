"""
src/agents/progress_coach.py

弱项分析师 Agent。

读取面试结果，生成个性化的辅导反馈，
更新路线图中的考点状态，定位薄弱环节并给出针对性的补强建议。

这是复习循环中做决策的节点：
  - 得分 >= 0.75：标记为 completed，继续下一个考点
  - 得分 >= 0.5：标记为 completed（基本通过）
  - 得分 < 0.5：标记为 needs_review，留在弱项列表中
  - 所有考点完成：打印总结报告
"""

import json
import os
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from graph.state import InterviewResult, InterviewPlan, get_latest_interview_result
from llm_factory import build_llm, get_model_name
from mcp_servers.memory_server import memory_set


MODEL_NAME = get_model_name()
PASS_THRESHOLD = 0.5
STRONG_PASS_THRESHOLD = 0.75


COACHING_PROMPT = """你是一位鼓励型的面试辅导教练，正在帮求职者分析模拟面试结果。

请根据以下信息给出简要、温暖的辅导反馈（中文，2-3 句话）：
  - 刚刚面试的考点名称
  - 得分（0.0 = 0%，1.0 = 100%）
  - 定位到的薄弱环节

只返回纯 JSON：
{{
  "summary": "2-3 句话的总结，包括做得好的地方和需要加强的地方",
  "encouragement": "一句鼓励的话，为下一步打气"
}}

评价要具体，引用考点和薄弱环节的实际名称。
永远不要打击信心。低分意味着"需要多练"，不是"你不行"。
"""


def get_coaching_message(topic: str, score: float, weak_areas: list[str]) -> dict:
    """调用 LLM 生成个性化辅导反馈。"""
    llm = build_llm(temperature=0.4)

    context = {
        "topic": topic,
        "score_percent": f"{score:.0%}",
        "weak_areas": weak_areas if weak_areas else ["未识别出明显薄弱环节"],
    }

    try:
        response = llm.invoke([
            SystemMessage(content=COACHING_PROMPT),
            HumanMessage(content=json.dumps(context, ensure_ascii=False)),
        ])
    except Exception as e:
        print(f"[弱项分析师] LLM 调用失败：{e}")
        return {
            "summary": f"你在「{topic}」上的得分为 {score:.0%}。继续加油！",
            "encouragement": "每一个考点都是你面经的积累。",
        }

    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {
            "summary": f"你在「{topic}」上的得分为 {score:.0%}。",
            "encouragement": "继续加油，每次练习都在进步！",
        }


def progress_coach_node(state: dict) -> dict:
    """
    LangGraph 节点：弱项分析师

    读取：
        state["interview_results"]（兼容 state["quiz_results"]）：最新面试结果
        state["study_plan"]（兼容 state["roadmap"]）            ：更新考点状态
        state["current_topic_index"]                            ：刚完成的考点索引
        state["session_id"]                                     ：供 MCP 记忆持久化

    写入：
        state["study_plan"]          ：考点状态更新
        state["current_topic_index"] ：递增，指向下一个考点
        state["messages"]            ：辅导反馈消息
        state["error"]               ：失败时写入错误信息
    """
    latest = get_latest_interview_result(state)
    if latest is None:
        return {"error": "没有面试记录。请确保模拟面试官已运行。"}

    study_plan = state.get("study_plan", state.get("roadmap"))
    if study_plan is None:
        return {"error": "没有找到复习计划。请确保考点规划师已运行。"}

    idx = state.get("current_topic_index", 0)
    session_id = state.get("session_id", "unknown")
    score = latest.score

    print(f"\n[弱项分析师] 考点：'{latest.topic}'")
    print(f"[弱项分析师] 得分：{score:.0%}")
    if latest.weak_areas:
        print(f"[弱项分析师] 薄弱环节：{', '.join(latest.weak_areas)}")

    # ── 获取辅导反馈 ────────────────────────────────────────────────────
    coaching = get_coaching_message(latest.topic, score, latest.weak_areas)

    # ── 更新考点状态 ────────────────────────────────────────────────────
    topics = study_plan.get("topics", []) if isinstance(study_plan, dict) else study_plan.topics
    if idx < len(topics):
        topic = topics[idx]
        if score >= STRONG_PASS_THRESHOLD:
            new_status = "completed"
            status_emoji = "✅"
        elif score >= PASS_THRESHOLD:
            new_status = "completed"
            status_emoji = "✔️"
        else:
            new_status = "needs_review"
            status_emoji = "🔄"
        if isinstance(topic, dict):
            topic["status"] = new_status
        else:
            topic.status = new_status

    # ── 前进到下一个考点 ─────────────────────────────────────────────────
    next_idx = idx + 1
    all_done = next_idx >= len(topics)

    # ── 通过 MCP 记忆持久化进度 ──────────────────────────────────────────
    progress_data = json.dumps({
        "topic": latest.topic,
        "score": score,
        "weak_areas": latest.weak_areas,
        "status": new_status if idx < len(topics) else "done",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False)
    memory_set(session_id, f"progress_topic_{idx}", progress_data)

    # ── 打印辅导反馈 ────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"👨‍🏫 导师：{coaching['summary']}")
    print(f"💪 {coaching['encouragement']}")

    if all_done:
        completed = sum(
            1 for t in topics
            if (t.get("status") if isinstance(t, dict) else t.status) == "completed"
        )
        total = len(topics)
        results = state.get("interview_results", state.get("quiz_results", []))
        avg = sum(
            r.score if hasattr(r, 'score') else r.get('score', 0)
            for r in results
        ) / max(len(results), 1)
        print(f"\n🎉 本轮复习全部完成！{completed}/{total} 个考点已通过。")
        print(f"📊 总均分：{avg:.0%}")

        # 汇总所有薄弱环节
        all_weak = state.get("weak_areas", [])
        if all_weak:
            print(f"📝 需要重点补强的方向：{', '.join(all_weak)}")
    else:
        next_topic = topics[next_idx]
        next_title = next_topic.get("title") if isinstance(next_topic, dict) else next_topic.title
        next_status = (
            next_topic.get("status") if isinstance(next_topic, dict) else next_topic.status
        )
        print(f"\n➡️ 下一个考点：'{next_title}'（状态：{next_status}）")
    print(f"{'─'*60}\n")

    return {
        "study_plan": study_plan,
        "current_topic_index": next_idx,
        "messages": [AIMessage(content=coaching["summary"])],
        "error": None,
    }
