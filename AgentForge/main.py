"""
main.py

AgentForge —— AI 技术面试备考系统 入口文件。

运行交互式复习会话，可选 Langfuse 可观测性追踪。

使用方法：
  python main.py "准备腾讯后台开发暑期实习面试"
  python main.py --resume <会话ID>
"""

import sys
import uuid
from pathlib import Path

# Windows 终端默认使用 GBK 编码，需要强制 UTF-8 输出以支持中文和 emoji
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 将 src/ 添加到 Python 搜索路径，确保项目模块可以被导入
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from langgraph.types import Command

from graph.workflow import graph
from graph.state import initial_state, InterviewPlan, InterviewResult
from observability.langfuse_setup import get_langfuse_config, flush_langfuse


def print_session_summary(result: dict) -> None:
    """打印复习会话的完成总结报告。"""
    # SqliteSaver 反序列化后，study_plan 和 interview_results 可能是普通 dict。
    # 先转换回 dataclass 再访问属性。
    raw_plan = result.get("study_plan", result.get("roadmap"))
    if raw_plan is None:
        return

    study_plan = (
        InterviewPlan.from_dict(raw_plan)
        if isinstance(raw_plan, dict)
        else raw_plan
    )

    raw_results = result.get("interview_results", result.get("quiz_results", []))
    interview_results = [
        InterviewResult.from_dict(r) if isinstance(r, dict) else r
        for r in raw_results
    ]

    if not interview_results:
        return

    print(f"\n{'='*60}")
    print("📊 复习总结报告")
    print(f"{'='*60}")
    job_target = study_plan.job_target if hasattr(study_plan, 'job_target') else study_plan.goal if hasattr(study_plan, 'goal') else '未知目标'
    print(f"求职目标：{job_target}")
    print(f"已完成考点：{len(interview_results)}/{len(study_plan.topics)}")

    total_score = sum(r.score for r in interview_results)
    avg = total_score / len(interview_results)
    print(f"总均分：{avg:.0%}\n")

    for r in interview_results:
        status = "✅" if r.score >= 0.5 else "🔄"
        weak = f"，薄弱：{', '.join(r.weak_areas)}" if r.weak_areas else ""
        print(f"  {status} {r.topic}：{r.score:.0%}{weak}")

    all_weak = result.get("weak_areas", [])
    if all_weak:
        print(f"\n📝 需要重点补强的方向：{', '.join(all_weak)}")

    print(f"{'='*60}\n")


def run_session(job_target: str, session_id: str | None = None) -> None:
    """运行一次完整的交互式复习会话，可选 Langfuse 追踪。"""
    is_resume = session_id is not None
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    # get_langfuse_config() 构建完整的运行配置：
    #   - thread_id 用于 SQLite checkpoint
    #   - Langfuse 回调处理器（如果设置了 LANGFUSE_PUBLIC_KEY）
    config = get_langfuse_config(session_id)

    print(f"\n{'='*60}")
    print("🤖 AI 技术面试备考系统")
    print(f"会话 ID：{session_id}")
    if is_resume:
        print("正在恢复之前的会话...")
    else:
        print(f"求职目标：{job_target}")
    print(f"{'='*60}")

    # 新会话提供初始状态，恢复会话传 None（LangGraph 从 checkpoint 加载）
    state = None if is_resume else initial_state(job_target, session_id)

    try:
        result = graph.invoke(state, config=config)
    except Exception as e:
        if is_resume:
            print(f"\n[错误] 无法恢复会话 '{session_id}'：{e}")
            print("如果会话 ID 错误或 checkpoint 数据库已被删除，请开启新会话。")
            return
        raise

    # ── 处理人在回路中（Human-in-the-Loop）中断 ────────────────────────
    # 当图形执行到 interrupt() 时暂停，result 中包含 "__interrupt__"。
    # 我们收集用户输入后恢复执行。
    while "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value

        # SqliteSaver 序列化后，载荷中的 study_plan 可能是普通 dict。
        raw_plan = interrupt_payload.get("study_plan", interrupt_payload.get("roadmap"))
        study_plan = (
            InterviewPlan.from_dict(raw_plan)
            if isinstance(raw_plan, dict)
            else raw_plan
        )

        # 向用户展示复习计划
        if study_plan:
            job_target = study_plan.job_target if hasattr(study_plan, 'job_target') else getattr(study_plan, 'goal', interrupt_payload.get("job_target", "未知"))
            print(f"\n{'='*60}")
            print("📋 复习路线图")
            print(f"{'='*60}")
            print(f"目标岗位：{job_target}")
            print(f"计划时长：{study_plan.total_weeks} 周 "
                  f"@ {study_plan.weekly_hours} 小时/周\n")
            for i, topic in enumerate(study_plan.topics, 1):
                prereqs = (f"（前置：{', '.join(topic.prerequisites)}）"
                           if topic.prerequisites else "")
                tags = f" [{', '.join(topic.tags)}]" if topic.tags else ""
                tips = f"\n      💡 {topic.interview_tips}" if topic.interview_tips else ""
                print(f"  {i}. {topic.title}{tags} "
                      f"({topic.estimated_minutes} 分钟){prereqs}{tips}")
                print(f"     {topic.description}")

        print(f"\n{interrupt_payload.get('prompt', '继续？')}")
        user_input = input("> ").strip()

        # 用用户的决定恢复图形执行
        result = graph.invoke(Command(resume=user_input), config=config)

    # ── 处理错误 ─────────────────────────────────────────────────────────
    if result.get("error"):
        print(f"\n[错误] {result['error']}")
        return

    print_session_summary(result)

    # 退出前刷新 Langfuse，确保所有追踪数据已发送
    flush_langfuse()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "AgentForge：AI 技术面试备考系统。\n"
            "一个四 Agent 协作系统：考点规划师为你规划复习路线，\n"
            "知识讲解师结合你的笔记进行讲解，模拟面试官出题并评分，\n"
            "弱项分析师定位薄弱环节并调整计划。"
        ),
        epilog=(
            "使用示例：\n"
            "  python main.py \"准备腾讯后台开发暑期实习面试\"\n"
            "  python main.py --resume a3f1b2c4\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "job_target", nargs="?",
        default="准备腾讯后台开发暑期实习面试",
        help="你的求职目标（默认：准备腾讯后台开发暑期实习面试）",
    )
    parser.add_argument(
        "--resume", metavar="SESSION_ID",
        help="恢复已有会话（通过 8 位会话 ID）",
    )
    args = parser.parse_args()

    if args.resume:
        run_session(job_target="", session_id=args.resume)
    else:
        run_session(job_target=args.job_target)
