# 🤖 AI 技术面试备考系统

一个基于多智能体协作的 AI 技术面试备考平台。输入求职目标，系统自动规划复习路线、讲解考点、模拟面试并评分、定位薄弱环节。

**🔗 在线体验：[点击这里]([AI 技术面试备考系统 · Streamlit (appapppy-5n9jcpkij3bjbjynmejwae.streamlit.app)](https://appapppy-5n9jcpkij3bjbjynmejwae.streamlit.app/))**（部署后替换此链接）

---

## ✨ 功能演示

```
求职目标：「准备腾讯后台开发暑期实习面试」

┌─────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐
│考点规划师│ →  │知识讲解师│ →  │模拟面试官 │ →  │弱项分析师│
│生成6个考点│    │结合笔记讲解│   │出题+评分  │    │定位薄弱环节│
└─────────┘    └─────────┘    └──────────┘    └──────────┘
                                                     │
                                              有考点 → 循环
                                              全完成 → 总结报告
```

- 📋 **智能规划**：根据求职目标自动生成复习路线图（考点排序、时长估计、面试加分建议）
- 📖 **深度讲解**：每个考点四板块讲解（核心概念→面试话术→加分亮点→常见误区）
- 💬 **模拟面试**：LLM 出题 + 实时评分 + 逐题反馈
- 📊 **弱项分析**：定位知识薄弱点，生成针对性补强建议
- 💾 **进度保存**：每个考点自动保存，支持断点续学

---

## 🚀 快速开始

### 本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY（从 https://platform.deepseek.com 获取）

# 3. 终端版
python main.py "准备腾讯后台开发暑期实习面试"

# 4. Web 版（推荐）
streamlit run streamlit_app.py
# 浏览器打开 http://localhost:8501
```

### 在线部署

1. Fork 本仓库到你的 GitHub
2. 在 [share.streamlit.io](https://share.streamlit.io) 用 GitHub 登录
3. 新建 App，选择本仓库，Main file path: `streamlit_app.py`
4. 在 Secrets 中配置 `DEEPSEEK_API_KEY`
5. 获得公开链接

---

## 🏗️ 技术架构

| 层级 | 技术 | 说明 |
|---|---|---|
| 编排框架 | LangGraph 1.1.0 | 状态图编排 + SQLite 检查点持久化 |
| LLM 后端 | DeepSeek API | 默认后端，支持一键切换 OpenAI/Ollama |
| 工具协议 | MCP 1.26.0 | 文件系统工具、会话记忆 |
| 前端 | Streamlit 1.43.2 | 响应式 Web 界面 |
| 可观测性 | Langfuse 4.0.1 | 可选，完整 LLM 调用链追踪 |

```
┌─────────────────────────────────────────────┐
│              Streamlit Web UI                │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│              LangGraph 状态图                 │
│                                              │
│  curriculum_planner → human_approval         │
│         ↓ 确认           ↓ 拒绝（重生成）     │
│  explainer → quiz_generator → progress_coach │
│         ↑                        ↓           │
│         └──── 循环（下一考点）────┘           │
│                                              │
│  SqliteSaver: 每步自动写入 SQLite             │
│  interrupt(): 人在回路中审批                   │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌──────────────┐    ┌──────────────────┐
│  MCP Tools   │    │  DeepSeek API     │
│  文件系统工具  │    │  (兼容 OpenAI/Ollama)│
│  会话记忆     │    └──────────────────┘
└──────────────┘
```

---

## 📂 项目结构

```
AgentForge/
├── main.py                  # 终端入口
├── streamlit_app.py         # Web 界面入口
├── .env.example             # 环境变量模板
├── requirements.txt         # 依赖列表
├── src/
│   ├── llm_factory.py       # 统一 LLM 工厂（DeepSeek/OpenAI/Ollama）
│   ├── agents/
│   │   ├── curriculum_planner.py  # 考点规划师
│   │   ├── explainer.py           # 知识讲解师（工具调用循环）
│   │   ├── quiz_generator.py      # 模拟面试官（出题+评分）
│   │   ├── progress_coach.py      # 弱项分析师
│   │   └── human_approval.py      # 人工审批（人在回路中）
│   ├── graph/
│   │   ├── state.py               # 共享状态定义
│   │   └── workflow.py            # LangGraph 图构建
│   ├── mcp_servers/
│   │   ├── filesystem_server.py   # 笔记文件系统 MCP
│   │   └── memory_server.py       # 会话记忆 MCP
│   └── observability/
│       └── langfuse_setup.py      # Langfuse 可观测性
├── docs/
│   ├── ARCHITECTURE.md      # 架构详解
│   └── MODEL_SELECTION.md   # 模型选择指南
├── study_materials/         # 复习笔记（Markdown）
└── data/                    # SQLite 检查点（自动生成，不提交）
```

---

## ⚙️ 配置说明

编辑 `.env` 文件：

```bash
# LLM 后端（三选一）
LLM_PROVIDER=deepseek        # deepseek / openai / ollama
DEEPSEEK_API_KEY=sk-xxx      # DeepSeek API Key

# 可选：Langfuse 可观测性
LANGFUSE_PUBLIC_KEY=         # 留空则不启用
LANGFUSE_SECRET_KEY=
```

---

## 📄 许可证

MIT License
