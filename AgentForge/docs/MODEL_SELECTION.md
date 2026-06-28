# 模型选择指南

AgentForge 支持三种 LLM 后端，通过 `.env` 中的 `LLM_PROVIDER` 一键切换。

---

## 为什么模型选择对 Agent 很重要

Agent 通过生成结构化 JSON 来调用工具。如果模型产生格式错误的 JSON 或虚构工具名称，工具调用将静默失败，Agent 陷入循环，最终触发 `max_iterations` 上限。**对于可靠的 Agent 行为，建议使用 7B 及以上参数的模型。**

---

## 三种后端对比

| 后端 | 硬件要求 | 成本 | 中文质量 | 推荐场景 |
|---|---|---|---|---|
| **DeepSeek API** | 无 | ¥0.1/次会话 | ⭐⭐⭐⭐⭐ | 日常 Demo、面试展示 |
| **OpenAI API** | 无 | ~$0.01/次会话 | ⭐⭐⭐ | 备用方案 |
| **Ollama 本地** | 8GB+ 显存 | 免费 | ⭐⭐⭐ | 离线使用、隐私场景 |

---

## DeepSeek API（推荐）

无需 GPU，中文质量最佳，成本极低。

```bash
# .env 配置
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

获取 Key：https://platform.deepseek.com/api_keys

---

## OpenAI API（备选）

兼容性最广，英文场景表现最佳。

```bash
# .env 配置
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://api.openai.com/v1
```

---

## Ollama 本地（离线/隐私优先）

适合有机器的用户，零成本。

### 推荐模型

| 显存 | 推荐模型 | 拉取命令 |
|---|---|---|
| 8 GB | `qwen2.5:7b` | `ollama pull qwen2.5:7b` |
| 8 GB | `qwen3:8b` | `ollama pull qwen3:8b` |
| 24 GB | `qwen2.5-coder:32b` | `ollama pull qwen2.5-coder:32b` |
| 24 GB | `qwen3:32b` | `ollama pull qwen3:32b` |

**最低要求：7B 参数。** 低于 7B 的模型在工具调用和 JSON 输出上可靠性不足，不建议用于 Agent 场景。

```bash
# .env 配置
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Agent 专用的温度设置

不同类型的 Agent 调用使用不同的温度参数，在代码中已预设：

| 场景 | 温度 | 说明 |
|---|---|---|
| 考点规划师 | 0.1 | 结构化 JSON 输出，需要极低温度保证格式正确 |
| 模拟面试官（评分） | 0.1 | 评分需要一致性和分析性 |
| 知识讲解师 | 0.3 | 讲解需要一定创造性但不可跑题 |
| 弱项分析师 | 0.4 | 辅导反馈需要温暖自然的语气 |
| 模拟面试官（出题） | 0.4 | 题目需要多样性 |

**经验法则：** 凡产出结构化 JSON 的 Agent，温度不超过 0.1；创造性任务可在 0.3~0.4 之间。

---

## LLM 工厂切换机制

所有 Agent 统一通过 `llm_factory.build_llm(temperature)` 获取 LLM 实例：

```python
from llm_factory import build_llm

# 结构化输出（规划、评分）
llm = build_llm(temperature=0.1)

# 创造性输出（讲解、出题）
llm = build_llm(temperature=0.4)
```

切换后端只需在 `.env` 中改一行，代码零改动。工厂函数负责根据 `LLM_PROVIDER` 环境变量自动选择正确的客户端。
