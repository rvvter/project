"""
src/llm_factory.py

统一的 LLM 工厂模块。

所有 Agent 通过此模块获取 LLM 实例，不再各自创建 ChatOllama。
通过修改 .env 中的 LLM_PROVIDER 即可一键切换后端：
  - deepseek  : DeepSeek API（推荐，中文友好，便宜）
  - openai    : OpenAI API
  - ollama    : 本地 Ollama（需要 GPU）

使用方式：
    from llm_factory import build_llm
    llm = build_llm(temperature=0.1)
    llm = build_llm(temperature=0.4, json_mode=True)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── 当前使用的 LLM 提供商 ────────────────────────────────────────────
# 可选值：deepseek / openai / ollama
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")

# ── DeepSeek 配置 ────────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# ── OpenAI 配置 ───────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ── Ollama 配置 ───────────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def build_llm(temperature: float = 0.1, json_mode: bool = False):
    """
    创建 LLM 实例的工厂函数。

    所有 Agent 统一通过此函数获取 LLM，方便切换后端。

    参数：
        temperature: 温度参数。
                     0.1 = 结构化输出（规划、评分）
                     0.4 = 创造性输出（讲解、生成题目）
        json_mode:   是否启用 JSON 模式。
                     规划师和评分 Agent 需要开启。

    返回：
        BaseChatModel 实例（ChatOpenAI 或 ChatOllama）。
    """
    if LLM_PROVIDER == "deepseek":
        return _build_deepseek(temperature)
    elif LLM_PROVIDER == "openai":
        return _build_openai(temperature)
    elif LLM_PROVIDER == "ollama":
        return _build_ollama(temperature, json_mode)
    else:
        raise ValueError(f"不支持的 LLM_PROVIDER: {LLM_PROVIDER}，可选值：deepseek / openai / ollama")


def _build_deepseek(temperature: float):
    """
    创建 DeepSeek API 客户端。

    DeepSeek 的 Chat API 兼容 OpenAI SDK，直接使用 ChatOpenAI 即可。
    """
    from langchain_openai import ChatOpenAI

    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "请在 .env 中设置 DEEPSEEK_API_KEY。\n"
            "获取 Key: https://platform.deepseek.com/api_keys"
        )

    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=temperature,
    )


def _build_openai(temperature: float):
    """
    创建 OpenAI API 客户端。
    """
    from langchain_openai import ChatOpenAI

    if not OPENAI_API_KEY:
        raise ValueError("请在 .env 中设置 OPENAI_API_KEY")

    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=temperature,
    )


def _build_ollama(temperature: float, json_mode: bool):
    """
    创建本地 Ollama 客户端。

    需要 Ollama 已安装并在本地运行。
    安装: https://ollama.com
    """
    from langchain_ollama import ChatOllama

    kwargs = {
        "model": OLLAMA_MODEL,
        "base_url": OLLAMA_BASE_URL,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["format"] = "json"

    return ChatOllama(**kwargs)


def get_model_name() -> str:
    """
    获取当前使用的模型名称，用于日志输出。

    返回：
        模型名称字符串，如 "deepseek-chat" 或 "qwen2.5:7b"。
    """
    if LLM_PROVIDER == "deepseek":
        return DEEPSEEK_MODEL
    elif LLM_PROVIDER == "openai":
        return OPENAI_MODEL
    elif LLM_PROVIDER == "ollama":
        return OLLAMA_MODEL
    return "unknown"
