# M5：医疗 Agent Core 与领域 Tool

## 已实现链路

```text
Browser / API Client
  -> C++ ChatController
  -> Agent Scheduler（独立有界池）
  -> AgentApplicationService
  -> PythonAgentClient
  -> Python TreeSemAgent
       -> Agent Loop / Tool Registry / ResponsePolicy
       -> OpenAI-compatible LLM
       -> C++ Internal Tool API
  -> MySQL Agent Run + final chat messages
```

C++ Gateway 仍是唯一外部入口和业务事实源。Python Agent 不连接 MySQL、ONNX 或模型对象；预测、解释、历史和比较只能调用 C++ Tool API。LLM 不能把自己生成的数值当成模型事实。

## Agent Loop

每次请求创建一个短生命周期 Run。上下文只包含系统提示词、最近 12 条最终消息、当前预测摘要和本轮用户消息。模型可以返回最终回答或结构化 Tool Call。

硬终止条件包括：最多 5 Step、8 次 Tool Call、25 秒总 deadline、连续重复 Tool Call，以及上游取消。GET Tool 的传输错误最多重试一次；有副作用的预测 Tool 不自动重试。LLM 的连接错误、429 和 5xx 也只能在总 deadline 内重试一次。

五个原生 Tool：

| Tool | C++ Internal API |
|---|---|
| `predict_sample` | `POST /internal/v1/predictions` |
| `get_prediction` | `GET /internal/v1/predictions/:id` |
| `get_explanation` | `GET /internal/v1/explanations/:id` |
| `get_prediction_history` | `GET /internal/v1/sessions/:id/history` |
| `compare_predictions` | `POST /internal/v1/comparisons` |

`session_id`、actor、subject 和 capability 都由运行上下文绑定，不进入 LLM 可填写的 Tool 参数。Tool 输入输出由 Pydantic 校验，并限制响应大小、JSON 结构和有限数值。

## 医疗回答边界

- treeSem 是辅助解释模型，不是诊断或处方系统。
- 模型标签、概率、版本、解释和历史必须来自本轮 Tool Result。
- 回答引用的 `prediction_id` 必须属于本轮成功 Tool Result。
- Tool 失败时说明数据暂时不可用，不补写或猜测结果。
- 不保存思维链、LLM 中间消息、Tool 参数或完整 Tool Result。
- 数据库只保存用户消息、助手最终回答以及 Tool 名称、状态、耗时和引用 ID 的摘要。

## 持久化和幂等

迁移 `002_m5_agent.sql` 创建 `treesem_agent_runs` 和 `treesem_chat_messages`。用户消息与 Run 创建在同一事务；助手最终消息与 Run 完成在同一事务。Agent 失败时保留用户消息并把 Run 标记为 `failed`，不写伪造助手回答。

`POST /api/v1/chat` 要求 8～64 字符 `Idempotency-Key`。相同 Session、key 和 payload 返回原结果；相同 key 配不同 payload 返回 409；仍在执行返回 `agent_run_in_progress`。

## Python 服务配置

Python 3.10 独立环境，固定 FastAPI、Uvicorn、HTTPX 和 Pydantic 版本，不依赖 LangChain/LangGraph。LLM 通过 OpenAI-compatible `/chat/completions` 协议接入。

关键配置：

- `TREESEM_AGENT_LLM_BASE_URL`、`TREESEM_AGENT_LLM_MODEL`、`TREESEM_AGENT_LLM_API_KEY`
- `TREESEM_AGENT_BACKEND_URL`，默认 `http://127.0.0.1:8080`
- `TREESEM_AGENT_MAX_STEPS=5`、`TREESEM_AGENT_MAX_TOOL_CALLS=8`
- `TREESEM_AGENT_TOTAL_TIMEOUT_MS=25000`
- `TREESEM_AGENT_SERVICE_SECRET`，M6 required 模式必须配置

C++ 侧使用 `TREESEM_AGENT_ENABLED`、`TREESEM_AGENT_URL`、连接/总超时以及独立 Worker/Queue 配置。Agent 队列过载返回 503，不占用推理或数据库 Scheduler。

## 验证边界

单元测试使用确定性 Scripted/Fake LLM，覆盖单 Tool、多 Tool、未知 Tool、参数错误、重复调用和 grounding 校验。C++ 测试覆盖 Run 幂等、消息持久化、失败状态和 Agent Client 错误映射。真实 LLM 只作为手工 smoke test，不作为 CI 依赖。

M7 再把医疗知识 RAG 作为 MCP Server 接入；M8 再实现可信 Skill。M5 不把普通本地 Tool 假装成 MCP，也不宣称 Agent 回答具备临床诊断效力。
