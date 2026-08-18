# M9：可观测性、Agent Evaluation 与容量决策

## 交付结果

M9 把一次请求从 C++ Gateway 串联到 Python Agent、C++ Internal Tool 和 Knowledge MCP。Gateway 接受合法 W3C `traceparent`，否则创建根 Trace；所有响应返回 `X-Request-Id` 和 `X-Trace-Id`。异步中间件顺序已调整为“先注入上下文，再构造 responder”，因此跨 Worker 回包仍使用原请求上下文。

三个服务输出安全 JSON span。日志只含服务、operation、父子 Span、耗时、结果和稳定错误码，不记录临床输入、聊天正文、Token、Tool 参数或完整结果。`scripts/trace_query.py` 可按 trace ID 汇总本地或 Compose 日志并恢复父子调用链。

`/internal/metrics` 输出 Prometheus 文本，覆盖 HTTP、隔离调度池、模型推理/fallback/shadow、数据库连接池、Agent/Tool/LLM、知识检索和认证拒绝。路由标签使用模板，Tool 标签使用固定名称，绝不使用 prediction/session/request ID。C++ Registry 与 Python histogram 都使用固定桶，内存不会随请求数增长。production 环境缺少 Metrics Bearer Token 会启动失败。

## 评测证据

`PythonServices/TreeSemAgent/evaluation/cases.json` 由 12 类场景、每类 5 个变体组成，共 60 条非患者合成多轮用例。它覆盖预测、解释、历史、比较、RAG、三个 Skill、no-answer、提示注入、紧急问题和跨角色引用。Fake LLM 模式实际经过 Agent Loop、Tool Registry、Skill 与 grounding policy，是 CI 硬门槛，不是只检查静态 JSON。

当前确定性结果：60/60 通过，关键安全失败为 0。真实 OpenAI-compatible 模式没有凭据时写明 `not_run`；有凭据时记录模型、评测集 SHA、Skill Catalog、知识索引、成功率和失败案例，绝不伪造报告。

## 压测与 MQ 决策

`load/k6/` 提供即时预测和混合业务流量脚本，记录延迟分位数、QPS、错误率和 503。`scripts/verify_full.py` 把完整测试、确定性 Agent 评测、Compose 演示和可选 k6 串成统一入口。当前环境没有 k6，且 Docker daemon 无访问权限，因此本轮只完成脚本与 Compose 静态验证，未把未执行的容器压测写成通过。

RabbitMQ 决策见 [ADR-0001](adr/0001-rabbitmq-decision.md)。交互式预测和 Chat 需要同步结果，现有隔离有界池已经提供背压；MQ 不会降低模型或 LLM 的执行时间。只有批量离线、跨 HTTP 生命周期、重启续跑或跨机器 Worker 等需求出现后才重新评估。

## 关键调用链

```text
HTTP request
  -> RequestContext / W3C Trace
  -> before middleware
  -> bounded scheduler
  -> model / database / Python Agent
  -> native Tool or MCP child span
  -> EventLoop one-shot response
  -> after middleware / metrics / safe trace log
```

## 面试重点

- Trace 和 Metrics 的区别，以及为什么 request ID 不能作为 Metrics label。
- 异步回包如何保持同一上下文，为什么中间件顺序会影响正确性。
- 如何用队列深度、连接等待和分段延迟定位 p99。
- 为什么固定桶、有界队列和低基数标签都是“内存有界”的一部分。
- Agent 为什么必须做确定性流程测试和真实模型质量测试两条轨道。
