"""
streamlit_app.py

AgentForge —— Streamlit Web 界面。

运行与 main.py 相同的 LangGraph 图，仅 I/O 机制不同——
不使用终端输入输出，改用 Streamlit 组件和会话状态。

启动方式：
    streamlit run streamlit_app.py

架构：
    App 是一个有 5 个界面的状态机：
    目标输入 → 计划审批 → 讲解展示 → 模拟面试 → 完成总结

    使用 interrupt_before=["quiz_generator"] 编译独立的
    图实例（ui_graph），使图形在面试前暂停，
    控制权交给 Streamlit。UI 直接处理面试 I/O
    （调用 generate_questions 和 grade_answer），
    然后通过 graph.update_state() 将面试结果注入 checkpoint，
    并从弱项分析师恢复执行。

    这保证了：
    - quiz_generator_node 和 run_interview 零改动
    - 终端界面（main.py）完全不受影响
    - LangGraph 图代码完全一致，只有 I/O 层不同
"""

import sys
from pathlib import Path

# ── 路径设置 ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import uuid
import streamlit as st
from langgraph.types import Command

from graph.workflow import build_graph
from graph.state import initial_state, InterviewPlan, InterviewResult, InterviewQuestion
from observability.langfuse_setup import get_langfuse_config, flush_langfuse
from agents.quiz_generator import generate_questions, grade_answer


# ── 构建 UI 专用图（在面试前暂停）───────────────────────────────────────
# 这让图形在 quiz_generator 之前停下来，
# 让 UI 可以处理面试交互而不需要调用 input() 阻塞 Streamlit。
ui_graph = build_graph(
    db_path="data/checkpoints_ui.db",
    interrupt_before=["quiz_generator"],
)


# ── 页面配置 ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI 技术面试备考系统",
    page_icon="🤖",
    layout="centered",
)


# ── 会话状态初始化 ──────────────────────────────────────────────────────

def init_state():
    """初始化 Streamlit 会话状态中的默认值。"""
    defaults = {
        "screen": "GOAL_INPUT",
        "session_id": None,
        "graph_config": None,
        "study_plan": None,
        "current_topic_index": 0,
        "quiz_questions": [],
        "current_question_idx": 0,
        "graded_answers": [],
        "current_quiz_missing_concepts": [],
        "interview_results": [],
        "weak_areas": [],
        "explanation": "",
        "topic_title": "",
        "topic_description": "",
        "topic_tags": [],
        "topic_interview_tips": "",
        "coaching_message": "",
        "error": None,
        "job_target": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()


# ── 辅助函数 ────────────────────────────────────────────────────────────

def go_to(screen: str):
    """切换到指定界面。"""
    st.session_state.screen = screen


def get_study_plan() -> InterviewPlan | None:
    """安全获取当前复习计划（兼容 dict 和 dataclass）。"""
    r = st.session_state.study_plan
    if r is None:
        return None
    if isinstance(r, dict):
        return InterviewPlan.from_dict(r)
    return r


def extract_explanation(messages: list) -> str:
    """获取讲解师的讲解内容——取最后一条内容丰富的 AIMessage（>200字），
    避免误取到前面考点规划师生成的 JSON 数据。"""
    from langchain_core.messages import AIMessage
    # 从后往前找第一条有实质内容的消息（200字以上的自然语言讲解）
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content.strip()
            # 跳过 JSON 数据（以 { 或 [ 开头）
            if content.startswith("{") or content.startswith("["):
                continue
            if len(content) > 200:
                return content
    return ""


def extract_coaching(messages: list) -> str:
    """获取弱项分析师的辅导反馈——从后往前找非讲解、非 JSON 的短消息。"""
    from langchain_core.messages import AIMessage
    explanation = extract_explanation(messages)
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content.strip()
            # 跳过讲解内容、JSON 数据、过短的消息
            if content == explanation:
                continue
            if content.startswith("{") or content.startswith("["):
                continue
            if len(content) < 15:
                continue
            return content
    return ""


# 缓存每个考点的讲解内容，用于上一考点导航
if "topic_explanations" not in st.session_state:
    st.session_state.topic_explanations = {}
if "completed_topics" not in st.session_state:
    st.session_state.completed_topics = set()


def go_to_prev_topic():
    """回到上一个考点，显示之前缓存的讲解内容。"""
    sp = get_study_plan()
    new_idx = st.session_state.current_topic_index - 1
    if new_idx < 0 or sp is None:
        return

    st.session_state.current_topic_index = new_idx
    topic = sp.topics[new_idx]
    st.session_state.topic_title = topic.title if hasattr(topic, "title") else topic.get("title", "")
    st.session_state.topic_description = topic.description if hasattr(topic, "description") else topic.get("description", "")
    st.session_state.topic_tags = topic.tags if hasattr(topic, "tags") else topic.get("tags", [])
    st.session_state.topic_interview_tips = topic.interview_tips if hasattr(topic, "interview_tips") else topic.get("interview_tips", "")
    # 恢复缓存的讲解内容
    cached = st.session_state.topic_explanations.get(new_idx, "")
    st.session_state.explanation = cached


def skip_to_next_topic():
    """跳过当前考点，直接进入下一个。"""
    sp = get_study_plan()
    new_idx = st.session_state.current_topic_index + 1
    if sp is None or new_idx >= len(sp.topics):
        return

    # 标记当前为已完成
    st.session_state.completed_topics.add(st.session_state.current_topic_index)

    st.session_state.current_topic_index = new_idx
    topic = sp.topics[new_idx]
    st.session_state.topic_title = topic.title if hasattr(topic, "title") else topic.get("title", "")
    st.session_state.topic_description = topic.description if hasattr(topic, "description") else topic.get("description", "")
    st.session_state.topic_tags = topic.tags if hasattr(topic, "tags") else topic.get("tags", [])
    st.session_state.topic_interview_tips = topic.interview_tips if hasattr(topic, "interview_tips") else topic.get("interview_tips", "")
    st.session_state.explanation = st.session_state.topic_explanations.get(new_idx, "")
    st.session_state.quiz_questions = []
    st.session_state.current_question_idx = 0
    st.session_state.graded_answers = []
    st.session_state.current_quiz_missing_concepts = []


def get_topic_info(result: dict, idx: int):
    """从结果或会话状态中获取指定索引考点的标题和描述。"""
    plan = result.get("study_plan", result.get("roadmap")) or st.session_state.study_plan
    sp = plan
    if isinstance(sp, dict):
        sp = InterviewPlan.from_dict(sp)
    if sp and idx < len(sp.topics):
        topic = sp.topics[idx]
        title = topic.title if hasattr(topic, "title") else topic.get("title", "")
        desc = topic.description if hasattr(topic, "description") else topic.get("description", "")
        tags = topic.tags if hasattr(topic, "tags") else topic.get("tags", [])
        tips = topic.interview_tips if hasattr(topic, "interview_tips") else topic.get("interview_tips", "")
        return title, desc, tags, tips
    return "", "", [], ""


def new_session():
    """重置所有会话状态，开始全新会话。"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()


# ── 图形交互 ────────────────────────────────────────────────────────────

def start_session(job_target: str):
    """
    开始新会话。运行：考点规划师 → 人工审批（中断）。
    """
    session_id = str(uuid.uuid4())[:8]
    config = get_langfuse_config(session_id)
    st.session_state.session_id = session_id
    st.session_state.graph_config = config
    st.session_state.job_target = job_target

    state = initial_state(job_target, session_id)

    with st.spinner("🤖 正在为你生成复习路线图..."):
        result = ui_graph.invoke(state, config=config)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        st.session_state.study_plan = payload.get("study_plan", payload.get("roadmap"))
        go_to("ROADMAP_APPROVAL")
    elif result.get("error"):
        st.session_state.error = result["error"]
    else:
        st.session_state.error = "意外：规划师之后没有触发中断。"


def approve_plan(approved: bool):
    """
    审批决策后恢复执行。

    如果确认：
        图形执行：人工审批节点 → 知识讲解师
        然后在 interrupt_before=["quiz_generator"] 处暂停。
        提取讲解内容并生成面试题。

    如果拒绝：
        图形执行：人工审批节点 → 考点规划师 → 中断。
        显示新生成的复习计划。
    """
    decision = "yes" if approved else "no"

    with st.spinner("正在准备讲解内容..." if approved else "正在重新生成计划..."):
        result = ui_graph.invoke(
            Command(resume=decision),
            config=st.session_state.graph_config,
        )

    if "__interrupt__" in result:
        # 计划被拒绝，新计划已生成
        payload = result["__interrupt__"][0].value
        st.session_state.study_plan = payload.get("study_plan", payload.get("roadmap"))
        go_to("ROADMAP_APPROVAL")
        return

    # 图形在面试官之前暂停，讲解师已完成
    messages = result.get("messages", [])
    explanation = extract_explanation(messages)
    st.session_state.explanation = explanation

    plan = result.get("study_plan", result.get("roadmap")) or st.session_state.study_plan
    st.session_state.study_plan = plan
    idx = result.get("current_topic_index", 0)
    st.session_state.current_topic_index = idx
    # 缓存讲解内容，供上一考点导航使用
    st.session_state.topic_explanations[idx] = explanation

    title, desc, tags, tips = get_topic_info(result, idx)
    st.session_state.topic_title = title
    st.session_state.topic_description = desc
    st.session_state.topic_tags = tags
    st.session_state.topic_interview_tips = tips

    # 预生成面试题
    with st.spinner("正在生成模拟面试题..."):
        questions = generate_questions(title, explanation, n=3)

    st.session_state.quiz_questions = questions
    st.session_state.current_question_idx = 0
    st.session_state.graded_answers = []
    st.session_state.current_quiz_missing_concepts = []

    go_to("EXPLAINING")


def advance_after_interview(interview_result: InterviewResult):
    """
    UI 处理的模拟面试完成后：
    1. 将面试结果注入 checkpoint，就像面试官节点已经运行过一样。
    2. 从弱项分析师恢复执行 → （讲解师 or 结束）。
    3. 如果讲解师运行（还有考点），在下一个面试官之前再次暂停。
    """
    config = st.session_state.graph_config
    existing = st.session_state.interview_results
    all_weak = list(set(st.session_state.weak_areas + interview_result.weak_areas))

    # 告知 LangGraph：面试官已经运行完毕，结果在此。
    # 这将 checkpoint 状态设置为如同 quiz_generator_node 正常返回了一样。
    ui_graph.update_state(
        config,
        {
            "interview_results": existing + [interview_result],
            "weak_areas": all_weak,
            "study_plan": st.session_state.study_plan,
            "current_topic_index": st.session_state.current_topic_index,
            "error": None,
        },
        as_node="quiz_generator",
    )

    # 恢复执行：运行弱项分析师，然后要么讲解师（下一考点）
    # 要么 END（全部完成）。
    # 由于 interrupt_before=["quiz_generator"]，如果还有考点，
    # 图形会在下一轮面试官之前再次暂停。
    with st.spinner("正在生成下一考点..."):
        result = ui_graph.invoke(None, config=config)

    # 从消息中提取辅导反馈
    messages = result.get("messages", [])
    coaching = extract_coaching(messages)
    st.session_state.coaching_message = coaching

    # 更新累计状态
    st.session_state.interview_results = result.get("interview_results", result.get("quiz_results", existing + [interview_result]))
    st.session_state.weak_areas = result.get("weak_areas", all_weak)
    new_idx = result.get("current_topic_index", st.session_state.current_topic_index + 1)
    st.session_state.current_topic_index = new_idx
    st.session_state.study_plan = result.get("study_plan", result.get("roadmap", st.session_state.study_plan))

    sp = get_study_plan()

    # 会话完成
    if sp is None or new_idx >= len(sp.topics):
        flush_langfuse()
        go_to("COMPLETE")
        return

    # 还有考点，图形在下一次面试官前暂停
    # 从结果消息中提取下一个考点的讲解内容
    explanation = extract_explanation(messages)
    st.session_state.explanation = explanation
    # 缓存讲解内容，供上一考点导航使用
    st.session_state.topic_explanations[new_idx] = explanation

    title, desc, tags, tips = get_topic_info(result, new_idx)
    st.session_state.topic_title = title
    st.session_state.topic_description = desc
    st.session_state.topic_tags = tags
    st.session_state.topic_interview_tips = tips

    with st.spinner("正在生成模拟面试题..."):
        questions = generate_questions(title, explanation, n=3)

    st.session_state.quiz_questions = questions
    st.session_state.current_question_idx = 0
    st.session_state.graded_answers = []
    st.session_state.current_quiz_missing_concepts = []

    go_to("EXPLAINING")


def resume_session(session_id: str):
    """恢复已有会话——从 SQLite checkpoint 加载状态并跳转到对应界面。"""
    config = get_langfuse_config(session_id)
    st.session_state.session_id = session_id
    st.session_state.graph_config = config

    # 尝试从 checkpoint 恢复图形状态
    try:
        state_snapshot = ui_graph.get_state(config)
    except Exception as e:
        st.session_state.error = f"无法加载会话「{session_id}」：{e}"
        return

    if state_snapshot is None or state_snapshot.values is None:
        st.session_state.error = f"未找到会话「{session_id}」，请检查 ID 是否正确。"
        return

    state = state_snapshot.values
    # 兼容旧字段名
    study_plan = state.get("study_plan", state.get("roadmap"))
    if study_plan is None:
        st.session_state.error = "该会话尚未生成复习计划，无法恢复。"
        return

    # 还原 Streamlit 会话状态
    st.session_state.study_plan = study_plan
    st.session_state.current_topic_index = state.get("current_topic_index", 0)
    st.session_state.interview_results = state.get("interview_results", state.get("quiz_results", []))
    st.session_state.weak_areas = state.get("weak_areas", [])
    st.session_state.job_target = state.get("job_target", state.get("goal", ""))
    st.session_state.approved = state.get("approved", False)

    idx = st.session_state.current_topic_index
    sp = get_study_plan()
    if sp and idx < len(sp.topics):
        topic = sp.topics[idx]
        st.session_state.topic_title = topic.title if hasattr(topic, "title") else topic.get("title", "")
        st.session_state.topic_description = topic.description if hasattr(topic, "description") else topic.get("description", "")
        st.session_state.topic_tags = topic.tags if hasattr(topic, "tags") else topic.get("tags", [])
        st.session_state.topic_interview_tips = topic.interview_tips if hasattr(topic, "interview_tips") else topic.get("interview_tips", "")

    # 跳转到对应界面
    if not st.session_state.approved:
        go_to("ROADMAP_APPROVAL")
    elif idx >= len(sp.topics) if sp else True:
        go_to("COMPLETE")
    else:
        # 恢复后直接进入讲解界面（需要重新调用讲解师）
        start_session_for_resume(session_id, state)
        return

    st.session_state.error = None


def start_session_for_resume(session_id: str, state: dict):
    """恢复会话后重新运行讲解师获取讲解内容。"""
    import os
    config = st.session_state.graph_config
    job_target = state.get("job_target", state.get("goal", ""))

    # 直接从 checkpoint 状态继续 invoke，触发 explainer
    with st.spinner("🔄 正在恢复你的复习进度..."):
        result = ui_graph.invoke(None, config=config)

    messages = result.get("messages", [])
    explanation = extract_explanation(messages)
    st.session_state.explanation = explanation

    idx = result.get("current_topic_index", st.session_state.current_topic_index)
    st.session_state.current_topic_index = idx
    st.session_state.topic_explanations[idx] = explanation

    plan = result.get("study_plan", result.get("roadmap")) or st.session_state.study_plan
    st.session_state.study_plan = plan

    title, desc, tags, tips = get_topic_info(result, idx)
    st.session_state.topic_title = title
    st.session_state.topic_description = desc
    st.session_state.topic_tags = tags
    st.session_state.topic_interview_tips = tips

    # 预生成面试题
    with st.spinner("正在生成面试题..."):
        questions = generate_questions(title, explanation, n=3)
    st.session_state.quiz_questions = questions
    st.session_state.current_question_idx = 0
    st.session_state.graded_answers = []
    st.session_state.current_quiz_missing_concepts = []
    st.session_state.coaching_message = ""

    go_to("EXPLAINING")


# ── 界面 1：目标输入 ────────────────────────────────────────────────────

def screen_goal_input():
    st.title("🤖 AI 技术面试备考系统")
    st.markdown(
        "输入你的求职目标，系统将为你：\n"
        "1. 📋 规划专属复习路线图\n"
        "2. 📖 结合你的笔记讲解每个考点\n"
        "3. 💬 模拟真实面试并评分\n"
        "4. 📊 定位薄弱环节并给出补强建议"
    )

    with st.form("goal_form"):
        job_target = st.text_input(
            "你的求职目标是什么？",
            placeholder="例如：准备腾讯后台开发暑期实习面试",
        )
        submitted = st.form_submit_button("🚀 开始规划复习路线", type="primary")

    if submitted:
        if not job_target.strip():
            st.error("请输入求职目标。")
        else:
            start_session(job_target.strip())
            st.rerun()

    # ── 继续之前的会话 ────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📂 继续之前的会话", expanded=False):
        st.markdown("输入你的会话 ID，从上一次中断的位置继续复习。")
        st.caption("会话 ID 在页面底部可以找到。")
        resume_id = st.text_input("会话 ID", placeholder="例如：ec7c7c81", key="resume_input")
        if st.button("▶️ 继续复习", use_container_width=True):
            if resume_id.strip():
                resume_session(resume_id.strip())
                st.rerun()
            else:
                st.error("请输入会话 ID")

    if st.session_state.error:
        st.error(f"错误：{st.session_state.error}")
        if st.button("重试"):
            st.session_state.error = None
            st.rerun()


# ── 界面 2：复习计划审批 ────────────────────────────────────────────────

def screen_roadmap_approval():
    st.title("📋 你的复习路线图")
    sp = get_study_plan()

    if sp is None:
        st.error("未找到复习计划。")
        if st.button("重新开始"):
            new_session()
            st.rerun()
        return

    job_target = getattr(sp, 'job_target', getattr(sp, 'goal', '未知目标'))
    st.markdown(f"### 🎯 目标岗位：{job_target}")
    st.markdown(f"📅 计划时长：**{sp.total_weeks} 周** @ 每周 **{sp.weekly_hours} 小时**")
    st.markdown("---")

    # 为每个考点生成精美卡片
    for i, topic in enumerate(sp.topics, 1):
        title = topic.title if hasattr(topic, "title") else topic.get("title", "")
        desc = topic.description if hasattr(topic, "description") else topic.get("description", "")
        mins = topic.estimated_minutes if hasattr(topic, "estimated_minutes") else topic.get("estimated_minutes", "?")
        tags = topic.tags if hasattr(topic, "tags") else topic.get("tags", [])
        tips = topic.interview_tips if hasattr(topic, "interview_tips") else topic.get("interview_tips", "")
        prereqs = topic.prerequisites if hasattr(topic, "prerequisites") else topic.get("prerequisites", [])

        # 难度标签颜色映射
        tag_colors = {
            "必考": "#dc2626", "高频": "#ea580c", "中频": "#2563eb",
            "加分项": "#7c3aed", "算法": "#0891b2", "数据库": "#059669",
            "操作系统": "#d97706", "计算机网络": "#4f46e5", "缓存": "#db2777",
            "Linux": "#65a30d", "设计模式": "#9333ea", "项目": "#0f766e",
        }
        tags_html = ""
        for tag in tags:
            color = tag_colors.get(tag, "#6366f1")
            tags_html += f'<span style="background:{color};color:#fff;padding:2px 10px;border-radius:12px;font-size:12px;margin-right:6px;">{tag}</span>'

        prereq_text = ""
        if prereqs:
            prereq_text = f'<span style="color:#94a3b8;font-size:12px;"> 🔗 前置：{" · ".join(prereqs)}</span>'

        tips_html = ""
        if tips:
            tips_html = f'<div style="margin-top:10px;padding:8px 14px;background:linear-gradient(135deg,#fef3c7,#fef9c3);border-left:3px solid #f59e0b;border-radius:0 8px 8px 0;font-size:13px;color:#92400e;">💡 {tips}</div>'

        # 根据索引交替左右配色
        accent = ["#6366f1", "#8b5cf6", "#3b82f6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#ec4899"][(i - 1) % 8]

        card_html = f"""
        <div style="
            background:#fff;
            border-radius:14px;
            padding:22px 26px;
            margin:16px 0;
            border-left:5px solid {accent};
            box-shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 12px rgba(0,0,0,0.04);
            transition:box-shadow 0.2s;
        " onmouseover="this.style.boxShadow='0 2px 8px rgba(0,0,0,0.10),0 6px 20px rgba(0,0,0,0.06)'" onmouseout="this.style.boxShadow='0 1px 3px rgba(0,0,0,0.06),0 4px 12px rgba(0,0,0,0.04)'">
            <div style="display:flex;align-items:flex-start;justify-content:space-between;">
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="
                        background:{accent};
                        color:#fff;
                        min-width:32px;height:32px;
                        border-radius:10px;
                        display:flex;align-items:center;justify-content:center;
                        font-weight:700;font-size:15px;
                    ">{i}</div>
                    <div>
                        <div style="font-size:17px;font-weight:700;color:#1e293b;margin-bottom:2px;">{title}</div>
                        <div style="font-size:12px;color:#64748b;">⏱️ {mins} 分钟</div>
                    </div>
                </div>
                <div style="text-align:right;">
                    {tags_html}
                    <div style="margin-top:4px;">{prereq_text}</div>
                </div>
            </div>
            <div style="margin-top:14px;font-size:14px;color:#475569;line-height:1.7;padding-left:44px;">
                {desc}
            </div>
            {tips_html}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ✅ 这份复习计划看起来如何？")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 确认，开始复习", type="primary", use_container_width=True):
            approve_plan(True)
            st.rerun()
    with col2:
        if st.button("🔄 不满意，重新生成", use_container_width=True):
            approve_plan(False)
            st.rerun()


# ── 界面 3：讲解展示 ────────────────────────────────────────────────────

def screen_explaining():
    sp = get_study_plan()
    total = len(sp.topics) if sp else 1
    idx = st.session_state.current_topic_index

    st.progress(idx / total, text=f"考点 {idx + 1} / {total}")
    st.title(f"📖 {st.session_state.topic_title}")

    # 标签和面试技巧
    tags = st.session_state.topic_tags
    if tags:
        tag_html = " ".join([f"`{t}`" for t in tags])
        st.markdown(tag_html)
    tips = st.session_state.topic_interview_tips
    if tips:
        st.info(f"💡 **面试加分建议：** {tips}")

    st.caption(st.session_state.topic_description)
    st.caption(f"💾 会话 ID：`{st.session_state.session_id}`（可用于下次继续）")
    st.markdown("---")

    if st.session_state.explanation:
        st.markdown("### 📝 考点讲解")
        st.markdown(st.session_state.explanation)
    else:
        st.warning("暂无讲解内容，将使用考点信息生成面试题。")

    st.markdown("---")

    # 导航按钮
    is_completed = idx in st.session_state.completed_topics

    col_prev, col_quiz, col_next = st.columns([1, 1.2, 1])

    with col_prev:
        if idx > 0:
            if st.button("⬅️ 上一考点", use_container_width=True):
                go_to_prev_topic()
                st.rerun()

    with col_quiz:
        if is_completed:
            if st.button("🔄 重新做题", use_container_width=True):
                st.session_state.coaching_message = ""
                go_to("QUIZZING")
                st.rerun()
        else:
            if st.button("💬 开始模拟面试 →", type="primary", use_container_width=True):
                st.session_state.coaching_message = ""
                go_to("QUIZZING")
                st.rerun()

    with col_next:
        if idx < total - 1:
            if is_completed:
                if st.button("➡️ 下一考点", use_container_width=True, type="primary"):
                    skip_to_next_topic()
                    st.rerun()
            else:
                st.button("➡️ 下一考点", use_container_width=True, disabled=True,
                          help="请先完成面试或重新做题")


# ── 界面 4：模拟面试 ────────────────────────────────────────────────────

def screen_quizzing():
    questions = st.session_state.quiz_questions
    q_idx = st.session_state.current_question_idx
    total_q = len(questions)
    sp = get_study_plan()
    total_topics = len(sp.topics) if sp else 1
    topic_idx = st.session_state.current_topic_index

    st.progress(topic_idx / total_topics, text=f"考点 {topic_idx + 1} / {total_topics}")
    if total_q > 0 and q_idx < total_q:
        st.progress(q_idx / total_q, text=f"题目 {q_idx + 1} / {total_q}")
    elif total_q > 0:
        st.progress(1.0, text=f"题目 {total_q} / {total_q} ✅")

    st.title(f"💬 模拟面试：{st.session_state.topic_title}")
    st.markdown("---")

    # 显示已评分的回答
    for i, graded in enumerate(st.session_state.graded_answers):
        status = "✅" if graded.correct else "❌"
        with st.expander(f"{status} 第{i+1}题：{graded.question[:80]}...", expanded=False):
            st.markdown(f"**你的回答：** {graded.user_answer}")
            st.markdown(f"**得分：** {graded.score:.0%}")
            st.markdown(f"**反馈：** {graded.feedback}")

    # 当前题目
    if q_idx < total_q:
        q = questions[q_idx]
        question_text = q.get("question", "")
        difficulty = q.get("difficulty", "medium")
        difficulty_label = {"easy": "简单", "medium": "中等", "hard": "困难"}.get(difficulty, "中等")

        st.markdown(f"**第 {q_idx + 1} 题 [{difficulty_label}]：**")
        st.markdown(question_text)

        with st.form(f"answer_form_{q_idx}"):
            answer = st.text_area(
                "你的回答：",
                placeholder="在此输入你的回答...",
                height=120,
                key=f"answer_input_{q_idx}",
            )
            submitted = st.form_submit_button("📤 提交回答 →", type="primary")

        if submitted:
            user_answer = answer.strip() or "（未作答）"
            expected = q.get("expected_answer", "")

            with st.spinner("🤔 评分中..."):
                grade = grade_answer(question_text, expected, user_answer)

            graded_q = InterviewQuestion(
                question=question_text,
                expected_answer=expected,
                user_answer=user_answer,
                correct=bool(grade.get("correct", False)),
                feedback=grade.get("feedback", ""),
                score=float(grade.get("score", 0.0)),
                difficulty=difficulty,
            )
            st.session_state.graded_answers.append(graded_q)
            # 捕获 LLM 识别出的缺失概念
            missing = grade.get("missing_concept", "").strip()
            if missing:
                st.session_state.current_quiz_missing_concepts.append(missing)
            st.session_state.current_question_idx = q_idx + 1
            st.rerun()

    else:
        # 所有题目已完成
        st.markdown("---")
        graded = st.session_state.graded_answers
        avg_score = sum(q.score for q in graded) / len(graded) if graded else 0.0
        # 本轮面试中已去重的薄弱环节
        weak_areas = list(dict.fromkeys(
            st.session_state.current_quiz_missing_concepts
        ))

        st.success("✅ 面试完成！")
        st.metric("你的得分", f"{avg_score:.0%}")

        interview_result = InterviewResult(
            topic=st.session_state.topic_title,
            questions=graded,
            score=avg_score,
            weak_areas=weak_areas,
        )

        if st.button("➡️ 下一考点 →", type="primary"):
            # 标记当前考点为已完成
            st.session_state.completed_topics.add(st.session_state.current_topic_index)
            advance_after_interview(interview_result)
            st.rerun()


# ── 界面 5：完成总结 ────────────────────────────────────────────────────

def screen_complete():
    st.title("🎉 本轮复习全部完成！")
    st.markdown("---")

    sp = get_study_plan()
    interview_results = st.session_state.interview_results

    if sp:
        job_target = getattr(sp, 'job_target', getattr(sp, 'goal', '未知目标'))
        st.markdown(f"**目标岗位：** {job_target}")

    if interview_results:
        avg = sum(
            (r.score if hasattr(r, "score") else r.get("score", 0))
            for r in interview_results
        ) / len(interview_results)
        st.metric("总均分", f"{avg:.0%}")
        st.markdown("---")
        st.markdown("### 📊 各考点表现")
        for r in interview_results:
            if isinstance(r, dict):
                r = InterviewResult.from_dict(r)
            status = "✅" if r.score >= 0.5 else "🔄"
            weak = f"，薄弱：{', '.join(r.weak_areas[:2])}" if r.weak_areas else ""
            st.markdown(f"{status} **{r.topic}**：{r.score:.0%}{weak}")

    if st.session_state.weak_areas:
        st.markdown("---")
        st.markdown("### 📝 需要重点补强的方向")
        for w in st.session_state.weak_areas[:5]:
            st.markdown(f"- {w}")

    st.markdown("---")
    st.markdown(f"**会话 ID：** `{st.session_state.session_id}`")
    st.caption("终端恢复：`python main.py --resume <session-id>`")

    if st.button("🔄 开始新的复习会话", type="primary"):
        new_session()
        st.rerun()


# ── 错误横幅 ────────────────────────────────────────────────────────────

def display_error():
    if st.session_state.error:
        st.error(f"出错了：{st.session_state.error}")
        if st.button("← 重新开始"):
            new_session()
            st.rerun()


# ── 路由 ────────────────────────────────────────────────────────────────

screen = st.session_state.screen

if screen == "GOAL_INPUT":
    screen_goal_input()
elif screen == "ROADMAP_APPROVAL":
    display_error()
    screen_roadmap_approval()
elif screen == "EXPLAINING":
    display_error()
    screen_explaining()
elif screen == "QUIZZING":
    display_error()
    screen_quizzing()
elif screen == "COMPLETE":
    screen_complete()
else:
    st.error(f"未知界面：{screen}")
    if st.button("重置"):
        new_session()
        st.rerun()
