# AgentForge 架构参考

本文档详解 AgentForge 的技术架构、设计决策与数据流。

---

## 系统概览

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph 系统                             │
│                                                             │
│  考点规划师 → 人工审批 → 知识讲解师 → 模拟面试官 → 弱项分析师   │
│       ↑          ↓             ↑              ↓            │
│    (拒绝)    (确认/拒绝)    (循环)        (继续/结束)         │
└─────────────────────────────────────────────────────────────┘
         │ MCP（工具）              │ LLM 后端
         ▼                          ▼
┌──────────────────┐    ┌─────────────────────┐
│  文件系统 MCP     │    │  DeepSeek API        │
│  内存 MCP        │    │  (兼容 Ollama/OpenAI) │
└──────────────────┘    └─────────────────────┘
         │                         │
         ▼                         ▼
┌──────────────────────────────────────────────────────────────┐
│                 Langfuse（可选可观测性）                        │
│         完整追踪 · Token 统计 · 每个 Agent 的延迟               │
└──────────────────────────────────────────────────────────────┘
```

---

## 为什么选 LangGraph

LangGraph 将 Agent 工作流建模为有向图：

- **节点（Node）** 是 Python 函数（各 Agent 节点）
- **边（Edge）** 定义路由——静态边或条件边
- **状态（State）** 是所有节点共享的 TypedDict
- **检查点（Checkpoint）** 每个节点执行后自动保存到 SQLite

用简单的 while 循环也能跑，但一旦崩溃进度全丢，也无法实现人在回路中（Human-in-the-Loop）审批。LangGraph 把容错和人工介入变成了原生能力。

### 图状态

```python
class AgentState(TypedDict):
    messages:             list[BaseMessage]    # 对话历史（追加模式）
    session_id:           str                  # 会话唯一标识
    job_target:           str                  # 求职目标
    study_plan:           InterviewPlan | None # 复习路线图
    approved:             bool                 # 是否已确认
    current_topic_index:  int                  # 当前考点位置
    interview_results:    list[InterviewResult]# 面试历史
    weak_areas:           list[str]            # 累计薄弱环节
    study_materials_path: str                  # 笔记存放路径
    error:                str | None           # 错误信息
```

节点返回**部分更新**——只包含它修改了的键。LangGraph 将其合并到完整状态中。

### 检查点机制

```python
conn = sqlite3.connect(db_path, check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = builder.compile(checkpointer=checkpointer)
```

- 每个节点执行后自动写入 SQLite
- `check_same_thread=False` 是因为 LangGraph 内部节点函数和检查点写入运行在不同线程上
- 连接在进程生命周期内保持打开
- 崩溃后通过 `--resume <session-id>` 恢复

### 人在回路中（Human-in-the-Loop）

`interrupt()` 在人工审批节点中暂停图形执行，并将完整检查点持久化。

```
1. 考点规划师生成路线图 → 图形暂停
2. 控制权返回 main.py / Streamlit
3. 向用户展示路线图，等待输入
4. graph.invoke(Command(resume=user_input)) 恢复执行
5. 决策传入人工审批节点，继续或重新生成
```

---

## 为什么选 MCP

MCP（Model Context Protocol）标准化了 Agent 与工具之间的交互接口。

| 原语 | 示例 | 使用场景 |
|---|---|---|
| **工具** | `read_study_file(filename)` | Agent 需要执行操作 |
| **资源** | `notes://index` | Agent 需要读取结构化数据 |

### 安全设计

`read_study_file()` 会先将请求路径解析为绝对路径，验证其确实在笔记目录内，否则拒绝访问。路径遍历攻击（如 `../../.env`）会被拦截并返回错误信息——让 LLM 看到问题而非直接崩溃。

---

## 为什么选 DeepSeek API

原项目基于 Ollama 本地模型，需要 8GB+ 显存。改造后默认使用 DeepSeek API：

- **零硬件门槛**：4GB 显存的笔记本也能跑
- **中文友好**：DeepSeek 对中文理解和生成质量显著优于同等规模的英文模型
- **成本极低**：一次完整复习会话（4 个 Agent × 8 个考点）约消耗 50K token，花费不到 0.1 元
- **一键切换**：`.env` 中改 `LLM_PROVIDER=ollama` 即可切回本地模型

### LLM 工厂模式

所有 Agent 通过 `llm_factory.build_llm()` 获取 LLM 实例，而非各自创建。切换后端只需改一个环境变量：

```python
# 统一工厂，一行切换后端
def build_llm(temperature=0.1, json_mode=False):
    if LLM_PROVIDER == "deepseek":
        return _build_deepseek(temperature)
    elif LLM_PROVIDER == "openai":
        return _build_openai(temperature)
    elif LLM_PROVIDER == "ollama":
        return _build_ollama(temperature, json_mode)
```

---

## 可观测性

Langfuse 通过 LangChain 的回调系统集成。挂载一个处理器即可自动捕获：

- 每个 Agent 节点的执行（开始/结束/状态）
- 每次 LLM 调用（模型、prompt、响应、token 数、延迟）
- 每次工具调用（名称、参数、结果、延迟）

```python
config = {
    "configurable": {"thread_id": session_id},
    "callbacks": [langfuse_handler],
}
graph.invoke(state, config=config)
```

`flush_langfuse()` 在进程退出前确保所有异步追踪数据已发送。

---

## 一次完整会话的数据流

```
1. 用户输入求职目标："准备腾讯后台开发暑期实习面试"
   → initial_state() 创建 AgentState

2. 考点规划师（curriculum_planner_node）
   → LLM 调用：求职目标 → InterviewPlan JSON
   → state["study_plan"] = InterviewPlan(8 个考点)

3. 人工审批（human_approval_node）
   → interrupt() 暂停图形
   → main.py 展示路线图，等待用户输入
   → 输入 "yes" → state["approved"] = True

4. 知识讲解师（explainer_node）[每个考点重复]
   → MCP：list_study_files() → 列出笔记
   → MCP：search_notes("TCP协议") → 定位相关文件
   → MCP：read_study_file("计算机网络.md") → 读取全文
   → LLM：生成有据可依的讲解（300-500 字）
   → MCP：memory_set(session_id, "explained_topics", ...)

5. 模拟面试官（quiz_generator_node）
   → LLM：生成 3 道面试题
   → input()：逐题收集回答
   → LLM：逐一评分（LLM-as-Judge）
   → state["interview_results"].append(InterviewResult)

6. 弱项分析师（progress_coach_node）
   → LLM：生成辅导反馈
   → 更新考点状态（completed / needs_review）
   → 得分 ≥ 0.75：优秀，前进
   → 得分 < 0.5：标记为需重学，加入薄弱列表
   → state["current_topic_index"] += 1

7. 条件路由（route_after_coach）
   → 还有考点 → 回到步骤 4（继续下一个考点）
   → 全部完成 → END → 打印总结报告
```

---

## 测试策略

| 层级 | 命令 | 速度 | 依赖 |
|---|---|---|---|
| 单元测试 | `pytest tests/ -m "not eval"` | ~3 秒 | 无 |
| 评估测试 | `pytest tests/test_eval.py -m eval` | ~90 秒 | LLM API |

- **单元测试**：Mock 所有 LLM 调用，测试解析、路由、工具逻辑
- **评估测试**：运行真实 Agent，使用 DeepEval + LLM-as-Judge 打分

核心测试模式：
- Agent 节点测试：mock LLM，验证状态转换
- 路由测试：直接调用路由函数，传入构造的状态
- MCP 测试：直接调用工具函数（无需子进程）
