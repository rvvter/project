"""
src/agents/human_approval.py

人工审批节点——人在回路中（Human-in-the-Loop）。

此节点位于考点规划师和知识讲解师之间。
它在开始复习之前暂停图形执行，等待用户确认（或拒绝）复习计划。

interrupt() 的工作原理：
  1. LangGraph 执行到达此节点
  2. 调用 interrupt()，图形执行在此暂停
  3. LangGraph 将完整 checkpoint 保存到 SQLite
  4. 控制权返回给调用方（main.py）
  5. main.py 显示复习计划并收集用户输入
  6. main.py 调用 graph.invoke(Command(resume=user_input), config)
  7. 执行在此恢复，decision = 用户输入
  8. 节点根据用户决定返回状态更新

为什么这对生产环境很重要：
  在 Web 应用中，步骤 4-6 会变为：
    4. HTTP 响应带着复习计划返回浏览器
    5. 用户提交表单
    6. 新的 HTTP 请求恢复图形执行
  LangGraph 代码在终端模式和 Web 模式下完全相同，
  变化的只是输入收集机制。
"""

from langgraph.types import interrupt

from graph.state import InterviewPlan


def human_approval_node(state: dict) -> dict:
    """
    LangGraph 节点：人工审批

    读取：
        state["study_plan"]（兼容 state["roadmap"]）：展示给用户的复习计划

    写入：
        state["approved"]：True = 用户确认，False = 用户拒绝

    当 approved=False 时，条件边路由回考点规划师重新生成计划。
    当 approved=True 时，图形继续前进到知识讲解师。
    """
    study_plan: InterviewPlan | None = state.get("study_plan", state.get("roadmap"))

    if study_plan is None:
        # 没有计划可审批，自动通过
        print("[人工审批] 未找到复习计划，自动跳过审批")
        return {"approved": True}

    print("\n[人工审批] 暂停，等待用户审核复习计划...")

    # interrupt() 在此暂停图形执行。
    # 传给 interrupt() 的 dict 是 "payload"。
    # main.py 读取它来确定该向用户展示什么。
    # 当 Command(resume=...) 被调用后恢复执行。
    job_target = study_plan.job_target if hasattr(study_plan, 'job_target') else study_plan.get("job_target", study_plan.get("goal", "未知目标"))
    topics_count = len(study_plan.topics) if hasattr(study_plan, 'topics') else len(study_plan.get("topics", []))

    decision = interrupt({
        "type": "plan_approval",
        "job_target": job_target,
        "study_plan": study_plan,
        "topics_count": topics_count,
        "prompt": (
            "这份复习计划看起来如何？\n"
            "  输入 'yes' 开始复习\n"
            "  输入 'no' 重新生成计划\n"
            "  或输入修改意见（如：'增加计算机网络考点'）"
        ),
    })

    # decision 是用户输入的内容（通过 Command(resume=...) 传入）
    approved = str(decision).lower().strip() in ("yes", "y", "ok", "approve", "确认", "是")

    if approved:
        print("[人工审批] ✅ 复习计划已确认，开始备考！")
    else:
        print(f"[人工审批] ❌ 计划被拒绝（原因：{decision}），重新生成...")

    # LangGraph 1.1.0 在 Command(resume=...) 后不会完整携带
    # interrupt 前的 checkpoint 状态。因此显式返回完整状态，
    # 确保下游 Agent 能收到 study_plan、session_id 等字段。
    return {
        "approved": approved,
        "study_plan": study_plan,
        "job_target": state.get("job_target", state.get("goal", "")),
        "session_id": state.get("session_id", ""),
        "current_topic_index": state.get("current_topic_index", 0),
        "interview_results": state.get("interview_results", state.get("quiz_results", [])),
        "weak_areas": state.get("weak_areas", []),
        "study_materials_path": state.get("study_materials_path", "study_materials/sample_notes"),
        "error": None,
    }
