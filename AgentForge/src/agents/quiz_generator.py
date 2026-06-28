"""
src/agents/quiz_generator.py

模拟面试官 Agent。

职责：
  1. 基于已讲解的考点生成模拟面试题
  2. 通过交互式输入让用户逐题作答
  3. 使用 LLM-as-Judge 对每道题的回答进行评分
  4. 返回 InterviewResult，包含得分和定位到的薄弱环节

架构模式：
  两个独立的 LLM 调用，各司其职：
    - 出题调用：创造性任务，温度较高，生成面试题
    - 评分调用：分析性任务，温度极低，产生稳定评分
  将两者分离，避免评分的判断受出题风格的影响。
"""

import json
import os
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from graph.state import InterviewQuestion, InterviewResult, get_current_topic
from llm_factory import build_llm, get_model_name


# ─────────────────────────────────────────────────────────────────────────────
# 模型配置
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = get_model_name()


# ─────────────────────────────────────────────────────────────────────────────
# 面试题生成
# ─────────────────────────────────────────────────────────────────────────────

GENERATION_PROMPT = """你是一位资深技术面试官，需要为求职者出 {n} 道模拟面试题。

给定一个考点及其讲解内容，生成能够测试真正理解深度的问题，
而非仅靠背诵就能回答的表面问题。

好的面试题要求求职者：
  - 将概念应用到新场景中
  - 解释 WHY（为什么这样设计），而非仅仅 WHAT（是什么）
  - 识别边界条件和常见误区
  - 横向对比相关技术方案

只返回纯 JSON，不要加解释文字，不要 markdown 代码块：
{{
  "questions": [
    {{
      "question": "清晰具体的面试问题，以问号结尾",
      "expected_answer": "参考答案（1-3 句话）",
      "difficulty": "easy|medium|hard"
    }}
  ]
}}

规则：
  - 至少包含一道关于常见误区或踩坑经历的问题
  - expected_answer 要简洁但完整
  - 避免是非题，要追问解释或演示
  - 问题语言风格：专业但不刻板，贴近真实面试场景
"""

GRADING_PROMPT = """你是一位严格公正的面试官，正在评估求职者的回答。

面试题：{question}
参考答案：{expected_answer}
求职者回答：{student_answer}

请严格公正地评分。首先判断回答是否有效：

【无效回答——直接给 0 分】
- 求职者写"不知道""不会""不知该互动""随便""..."等明显放弃的回答
- 求职者回答与题目完全无关
- 求职者只写了一两个无关的字
→ score=0.0, correct=false, feedback="回答无效，请认真对待面试。"

【有效回答——按以下标准评分】
- 核心正确，逻辑完整，有深度：0.8-0.95
- 核心正确，仅细节不完整：0.7-0.8
- 概念正确但表述不够精准：0.5-0.7
- 部分正确，有遗漏：0.3-0.5
- 核心错误或严重遗漏：0.1-0.3

只返回纯 JSON，不要加解释文字，不要 markdown 代码块：
{{
  "correct": true,
  "score": 0.85,
  "feedback": "一句话具体反馈，指出哪里好和哪里需要改进",
  "missing_concept": "被忽略的关键概念，若回答完整则为空字符串"
}}
"""


def generate_questions(topic: str, explanation: str, n: int = 3) -> list[dict]:
    """
    调用 LLM 为一个考点生成 n 道模拟面试题。

    参数：
        topic:       考点名称。
        explanation: 讲解师产出的讲解内容（作为出题上下文）。
        n:           生成题目数量。

    返回：
        面试题 dict 列表，每道题包含 question、expected_answer、difficulty。
        若 LLM 输出无法解析，回退到一道通用题目。
    """
    llm = build_llm(temperature=0.4)

    prompt = GENERATION_PROMPT.format(n=n)
    try:
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"考点：{topic}\n\n讲解内容：\n{explanation}"),
        ])
    except Exception as e:
        print(f"[模拟面试官] 出题时 LLM 调用失败：{e}")
        # 返回最小回退题目，保证面试流程可以继续
        return [{
            "question": f"请用自己的话解释「{topic}」的核心概念，并说明其应用场景。",
            "expected_answer": f"对 {topic} 的清晰解释，展示对核心原理和应用场景的理解。",
            "difficulty": "medium",
        }]

    try:
        data = json.loads(response.content)
        questions = data.get("questions", [])
        if questions and isinstance(questions, list):
            return questions
    except (json.JSONDecodeError, KeyError):
        pass

    # 解析失败时的回退题目
    print("[模拟面试官] 警告：无法解析出题结果，使用回退题目")
    return [{
        "question": f"请用自己的话解释「{topic}」的核心概念及它的应用场景。",
        "expected_answer": f"对 {topic} 的清晰解释，展示对核心原理和应用场景的理解。",
        "difficulty": "medium",
    }]


def grade_answer(question: str, expected: str, student_answer: str) -> dict:
    """
    使用 LLM 对求职者的回答进行评分。

    参数：
        question:       面试题内容。
        expected:       参考答案。
        student_answer: 求职者的回答。

    返回：
        包含 correct（bool）、score（float）、feedback（str）、
        missing_concept（str）的 dict。
        若 LLM 输出无法解析，返回安全的默认评分。
    """
    # 极低温度——评分需要一致性和分析性
    llm = build_llm(temperature=0.1)

    prompt = GRADING_PROMPT.format(
        question=question,
        expected_answer=expected,
        student_answer=student_answer,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
    except Exception as e:
        print(f"[模拟面试官] 评分时 LLM 调用失败：{e}")
        # 返回部分分数，让会话可以继续
        return {
            "correct": False,
            "score": 0.5,
            "feedback": "评分服务暂时不可用，请自行对照参考答案检查。",
            "missing_concept": "",
        }

    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        # 评分解析失败时的安全默认值
        return {
            "correct": False,
            "score": 0.0,
            "feedback": "自动评分失败，请自行对照参考答案检查。",
            "missing_concept": "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 交互式面试运行器
# ─────────────────────────────────────────────────────────────────────────────

def run_interview(topic: str, explanation: str) -> InterviewResult:
    """
    为一个考点运行完整的交互式模拟面试。

    生成面试题 → 逐题收集回答 → 评分 → 汇总结果。

    参数：
        topic:       被面试的考点名称。
        explanation: 讲解师的输出（作为面试出题上下文）。

    返回：
        包含题目、得分和薄弱环节的 InterviewResult。
    """
    print(f"\n{'='*60}")
    print(f"💬 模拟面试：{topic}")
    print(f"{'='*60}")
    print("请用你自己的话回答以下问题。按 Enter 提交。\n")

    questions_data = generate_questions(topic, explanation, n=3)
    graded_questions = []
    total_score = 0.0
    weak_areas = []

    for i, q_data in enumerate(questions_data, 1):
        question_text = q_data["question"]
        expected = q_data["expected_answer"]
        difficulty = q_data.get("difficulty", "medium")

        difficulty_label = {"easy": "简单", "medium": "中等", "hard": "困难"}.get(difficulty, "中等")

        print(f"第 {i} 题 [{difficulty_label}]：{question_text}")
        user_answer = input("你的回答：").strip()

        # 处理空回答
        if not user_answer:
            user_answer = "（未作答）"

        print("评分中...")
        grade = grade_answer(question_text, expected, user_answer)

        score = float(grade.get("score", 0.0))
        correct = bool(grade.get("correct", False))
        feedback = grade.get("feedback", "")
        missing = grade.get("missing_concept", "")

        total_score += score

        # 显示结果
        status = "✅" if correct else "❌"
        print(f"{status} 得分：{score:.0%}，{feedback}\n")

        if missing:
            weak_areas.append(missing)

        graded_questions.append(InterviewQuestion(
            question=question_text,
            expected_answer=expected,
            user_answer=user_answer,
            correct=correct,
            feedback=feedback,
            score=score,
            difficulty=difficulty,
        ))

    # 计算总体得分
    avg_score = total_score / len(questions_data) if questions_data else 0.0
    correct_count = sum(1 for q in graded_questions if q.correct)

    print(f"{'='*60}")
    print(f"面试完成！总分：{avg_score:.0%} "
          f"（{correct_count}/{len(graded_questions)} 道正确）")
    if weak_areas:
        print(f"需要补强的知识点：{', '.join(set(weak_areas))}")
    print(f"{'='*60}\n")

    return InterviewResult(
        topic=topic,
        questions=graded_questions,
        score=avg_score,
        weak_areas=list(set(weak_areas)),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph 节点
# ─────────────────────────────────────────────────────────────────────────────

def quiz_generator_node(state: dict) -> dict:
    """
    LangGraph 节点：模拟面试官

    读取：
        state["study_plan"]（兼容 state["roadmap"]）：获取当前考点
        state["current_topic_index"]                  ：第几个考点
        state["messages"]                             ：提取讲解内容

    写入：
        state["interview_results"]（兼容 state["quiz_results"]）：追加面试结果
        state["weak_areas"]                                      ：累计薄弱环节（已去重）
        state["error"]                                           ：失败时写入错误信息
    """
    topic = get_current_topic(state)
    if topic is None:
        return {"error": "没有找到当前考点。请确保考点规划师和讲解师已运行。"}

    # 从消息历史中提取最近一次讲解
    # 讲解师的最终回复是最后一条无 tool_calls 的 AIMessage
    messages = state.get("messages", [])
    explanation = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            explanation = msg.content
            break

    if not explanation:
        print("[模拟面试官] 警告：未找到讲解内容，使用考点信息生成通用面试题")
        explanation = f"考点：{topic.title}。{topic.description}"

    print(f"\n[模拟面试官] 正在为考点生成面试题：'{topic.title}'")
    interview_result = run_interview(topic.title, explanation)

    # 累计结果
    existing_results = state.get("interview_results", state.get("quiz_results", []))
    all_weak_areas = list(set(
        state.get("weak_areas", []) + interview_result.weak_areas
    ))

    return {
        "interview_results": existing_results + [interview_result],
        "weak_areas": all_weak_areas,
        "error": None,
        # 显式透传核心状态，LangGraph 1.1.0 状态传播的兼容方案
        "study_plan": state.get("study_plan", state.get("roadmap")),
        "current_topic_index": state.get("current_topic_index", 0),
        "session_id": state.get("session_id", ""),
    }
