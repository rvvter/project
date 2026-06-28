"""
src/observability/langfuse_setup.py

AgentForge 的 Langfuse 可观测性配置。

提供一个函数 get_langfuse_config()，返回带有
Langfuse 回调处理器的 LangGraph 运行配置。

在 main.py 中的使用方式：
    from observability.langfuse_setup import get_langfuse_config
    config = get_langfuse_config(session_id)
    graph.invoke(state, config=config)

挂载后自动捕获：
  - 每个 Agent 节点的执行过程（开始时间、结束时间、状态）
  - 每次 LLM 调用（模型、prompt、响应、token 数、延迟）
  - 每次工具调用（名称、参数、结果、延迟）
  - 会话元数据（session_id、user_id、标签）

无需修改 Agent 代码。
所有观测通过回调系统自动完成。
"""

import os


def _langfuse_configured() -> bool:
    """
    检查环境中是否配置了 Langfuse 凭证。

    如果密钥缺失或为空，返回 False——
    系统将在无可观测性的情况下运行而非崩溃。
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    return bool(public_key and secret_key)


def get_langfuse_handler(session_id: str, user_id: str = "local"):
    """
    为一次会话创建 Langfuse 回调处理器。

    参数：
        session_id: 复习会话 ID（用作 Langfuse 的 session_id）。
                    将一次复习会话的所有追踪归为一组。
        user_id:    可选的用户标识，用于 UI 中筛选。

    返回：
        配置好的 CallbackHandler；若 Langfuse 未设置则返回 None。
        调用方应优雅地处理 None。
    """
    if not _langfuse_configured():
        return None

    try:
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
            session_id=session_id,
            user_id=user_id,
            # 标签出现在 Langfuse UI 中，便于过滤追踪记录
            tags=["agentforge", "interview-prep"],
            metadata={
                "model": os.getenv("DEEPSEEK_MODEL", os.getenv("OLLAMA_MODEL", "deepseek-chat")),
                "framework": "langgraph",
            },
        )
        return handler
    except ImportError:
        print("[可观测性] langfuse 未安装。运行：pip install langfuse")
        return None
    except Exception as e:
        print(f"[可观测性] 创建 Langfuse 处理器失败：{e}")
        return None


def get_langfuse_config(
    session_id: str,
    user_id: str = "local",
    extra_config: dict | None = None,
) -> dict:
    """
    构建完整的 LangGraph 运行配置，含 Langfuse 可观测性。

    这是 main.py 中使用的主函数。它合并了：
      - thread_id（用于 checkpoint）
      - Langfuse 回调处理器（如果已配置）
      - 你传入的任何额外配置

    参数：
        session_id:   复习会话 ID。
        user_id:      可选的用户标识。
        extra_config: 需要合并的额外 LangGraph 配置。

    返回：
        可直接传入 graph.invoke() 的 config dict。

    示例：
        config = get_langfuse_config(session_id)
        result = graph.invoke(state, config=config)
        # 所有 Agent 调用现在都会出现在 Langfuse UI 中
    """
    config = {
        "configurable": {"thread_id": session_id},
    }

    # 合并额外配置
    if extra_config:
        config.update(extra_config)

    # 挂载 Langfuse 处理器（如果可用）
    handler = get_langfuse_handler(session_id, user_id)
    if handler:
        config["callbacks"] = [handler]
        print(f"[可观测性] 追踪会话 {session_id} → "
              f"{os.getenv('LANGFUSE_HOST', 'http://localhost:3000')}")
    else:
        print("[可观测性] Langfuse 未配置，将在无追踪的情况下运行。")

    return config


def flush_langfuse() -> None:
    """
    进程退出前刷新所有待发送的 Langfuse 事件。

    Langfuse 在后台线程中异步发送追踪数据。
    在 main.py 末尾调用此函数，确保进程退出前
    所有追踪数据已发送。

    如果 Langfuse 未配置，此函数为无操作。
    """
    if not _langfuse_configured():
        return

    try:
        from langfuse import Langfuse
        Langfuse().flush()
    except Exception:
        pass  # 尽力刷新，不在退出时报错
