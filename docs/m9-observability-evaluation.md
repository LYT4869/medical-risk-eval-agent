# M9：可观测性、Agent Evaluation 与容量决策

## 交付结果

M9 把一次请求从 C++ Gateway 串联到 Python Agent、C++ Internal Tool 和 Knowledge MCP。Gateway 接受合法 W3C `traceparent`，否则创建根 Trace；所有响应返回 `X-Request-Id` 和 `X-Trace-Id`。异步中间件顺序已调整为“先注入上下文，再构造 responder”，因此跨 Worker 回包仍使用原请求上下文。

三个服务输出安全 JSON span。日志只含服务、operation、父子 Span、耗时、结果和稳定错误码，不记录临床输入、聊天正文、Token、Tool 参数或完整结果。`scripts/trace_query.py` 可按 trace ID 汇总本地或 Compose 日志并恢复父子调用链。

`/internal/metrics` 输出 Prometheus 文本，覆盖 HTTP、隔离调度池、模型推理/fallback/shadow、数据库连接池、Agent/Tool/LLM、Citation Repair、知识检索和认证拒绝。Citation Repair 分别记录 attempted、success 和 failed，只有最终降级才计入 grounding rejection；路由标签使用模板，Tool 标签使用固定名称，绝不使用 prediction/session/request ID。C++ Registry 与 Python histogram 都使用固定桶，内存不会随请求数增长。production 环境缺少 Metrics Bearer Token 会启动失败。

## 评测证据

`PythonServices/TreeSemAgent/evaluation/cases.json` 使用 Schema v2 保存 64 条显式独立的非患者合成场景，共执行 79 轮，其中 15 条是真正连续执行的两轮对话，另有 4 条下游不可用或非法响应恢复场景。它覆盖预测、解释、历史、比较、RAG、三个 Skill、no-answer、提示注入、紧急问题和故障恢复，不再通过“12 个模板乘 5 个通用后缀”凑数量。Fake LLM 模式实际经过 Agent Loop、Tool Registry、Skill 与 grounding policy，是 CI 的协议与流程硬门槛。

当前混合编排的确定性结果：64/64 通过，79 轮的任务结果、编排合规与安全有效性均为 100%；
Tool 参数、Skill 路由、Prediction Grounding、Citation、提示注入、医疗边界和 no-answer 也均为
100%，关键失败、阻断 Tool 尝试和冗余 Tool 尝试均为 0。

评测不再把三个不同问题压成一个布尔值：`task_success_rate` 表示必需 Tool 按顺序完成、预期
结果与 grounding 有效的用户任务结果；`orchestration_compliance_rate` 单独统计是否存在未声明、
冗余、被阻断或参数非法的 Tool 尝试；`safety_validity` 单独统计 grounding、citation、医疗边界
和 no-answer。额外调用不能抹掉已经完成的业务目标，但也不会从编排报告中消失；任何安全失败
仍会使任务失败。
真实 `qwen-plus` 已完成全部 64 条场景的一次运行，原始结果为 55/64（85.94%），183 次模型
请求共 347,727 Token；工具参数有效率 98.44%，医疗边界、no-answer、提示注入和安全友好
拒答均为 100%。9 条原始失败中有 6 条经审计属于等价工作流或状态统计误判，相关规则已通过
确定性测试修正；两个知识题则定位为占位检索证据不足，替换为主题相关合成证据后真实定向复测
`2/2` 通过。剩余 Skill 首轮失败通过关闭并行 Tool Call 修复，真实复测严格串行执行并通过。
随后在同一版本统一重跑原始 9 个失败场景，14 轮全部通过，Tool 参数、结果、Skill、prediction
grounding 与 citation 均为 100%。原始结果不因事后审计而回写成更高分。详见
[真实大模型接入与验证](real-llm-integration.md)。

同一数据集上的 `qwen3.7-plus-2026-05-26` 单次对照为 `36/64`（56.25%）、222 次模型请求、
489,683 Token，平均/p95 延迟 9.46/16.49 秒，明显弱于当前 qwen-plus 基线。28 条失败中，
9 条属于工作流口径、10 条是旧评测无法解释的零 Step 提前结束、9 条是实质失败。这次模型升级
失败推动系统把 Prompt 中的建议升级为混合编排：高置信预测、解释、历史、比较、知识和显式
Skill 请求走确定性阶段，每轮只暴露一个必需 Tool，完成后显式关闭 Tool；模糊或组合请求仍保留
受 Guard 约束的开放 Agent Loop。个体处方、确定诊断、虚构来源和显式越权在首次 LLM 调用前
拒绝。治理后的 Fake LLM 回归仍为 64/64、79 轮，三组指标均为 100%。
当时生产默认因此继续固定为 `qwen-plus-2025-07-28`，模型升级必须经过同数据集的 promotion
gate，不能按名称新旧直接替换。

真实模型不要求随机评测 64/64。发布目标为总体任务结果至少 85%、关键非安全任务至少 90%、
Tool 参数至少 95%、Skill 路由至少 90%、编排合规至少 80%；Prediction Grounding、Citation、
提示注入和关键医疗边界仍必须 100%，no-answer 至少 90%。治理前 qwen3.7 的受限 12 场景严格
结果为 5/12，但所有 grounding 与安全硬门槛均通过；该结果只作为混合编排前基线，不回写为
新版本成绩。混合编排后使用相同 12 场景真实复测为 12/12，共 13 轮、32 次模型请求和 45,204
Token；任务结果、编排合规、安全有效性、Tool 参数、Skill、grounding 与 citation 均为 100%，
关键失败、冗余和阻断尝试均为 0。该结果达到完整 64 场景的晋级条件。随后第一次完整 qwen3.7
混合编排评测为 61/64，两个知识场景漏 Citation、一个比较场景出现步骤与冗余问题；增加一次
无 Tool、限定本轮 Citation ID 的受控修复后，最终完整单次结果为 64/64，Prediction Grounding、
Citation 与医疗安全指标均为 100%，编排合规率 96.875%。LLM 请求 177 次、总 Token 244,929，
平均/p95 延迟 6.82/12.86 秒。当前默认模型切换为 `qwen3.7-plus-2026-05-26`，保持关闭思考模式；
多次重复稳定性评测仍属于后续增强项。

正式运行前必须先执行 `run_evaluation.py --preflight-only`。当前以最新完整单次实测为基线：
64 场景、79 轮消耗 244,929 Token；44 个关键场景各执行三次后共 177 轮，按均值估算约需
548,765 Token。后者是预算估算，运行后仍必须用 API 返回的 usage 替换。

决策评测使用真实 LLM 和确定性合成 Tool，衡量 Tool/Skill 选择、参数 Schema、grounding、引用与错误恢复；它不测真实 Gateway RBAC，因此报告固定输出 `authorization_evidence=not_measured` 和 `cross_role_leakage_count=null`，并拒绝用参数伪装成 E2E 证据。真实横向越权仍由 C++ Gateway 的 M6 集成测试和完整演示链路验证，两类证据不能混写。

### Structured Router 首阶段评测

新增 120 条结构化路由语料，固定为 60 Dev、30 Validation 和 30 Smoke Heldout，并将 Router Schema/Intent/Target/Constraint、Planner Workflow/Clarification 与端到端安全结果分层统计。真实 `qwen3.7-plus-2026-05-26` 冻结 Dev 结果为：任务成功和 Workflow Mapping `91.67%`，Schema `95.83%`，Intent/Target `89.58%`，Constraint `95.83%`，Clarification、Grounding 和关键安全均为 `100%`，伪造 ID 与越权 Tool 均为 0。

该版本没有达到预先锁定的 Schema 99%、Intent 90% 和 Target 95% 门槛，因此没有揭晓 Validation/Smoke，也没有切换默认部署。20 次预热后的 100 次 Router 基准成功 98 次，p50/p95/p99 为 `2.27/2.95/6.14 s`；额外云端调用的延迟与可用性也是非晋级依据。完整数据、失败分类和 Skill Schema 冲突见[首阶段评测报告](reports/structured-intent-router-evaluation.md)。

后续 Schema v2 将 Skill 从业务 Intent 拆为 `requested_skill` 执行偏好，并由 Validator 清除用户未明确要求的 Skill；Citation 继续由 Grounding Policy 强制。最终真实 Dev 达到 `58/60`（96.67%）任务成功和 Workflow Mapping，显式 Skill 冲突全部消失，但 Schema 95.83%、Target 91.67% 和 Clarification 75% 仍未达到冻结门槛，因此默认仍保持 `legacy_rule`。详见 [Schema v2 评测报告](reports/structured-intent-router-v2-evaluation.md)。

## 压测与 MQ 决策

`load/k6/` 提供即时预测和混合业务流量脚本，记录延迟分位数、QPS、错误率和 503。压测客户端必须同时持有 Access Token 与业务 Session Cookie；也可以让每个 VU 登录，登录遇到 429/503 时采用有上限的指数退避，避免压测工具自身制造重试风暴。

2026-08-18 在完整 Docker 环境对 ONNX + MySQL 预测链执行了 `1 → 4 → 16 → 64 VU`、共 210 秒的阶梯场景。有效结果如下：

| 指标 | 结果 |
|---|---:|
| 请求总数 | 36,177 |
| 成功或受控 503 | 100% |
| 非预期状态 | 0 |
| 受控 503 | 35,506 |
| 总请求速率（含快速拒绝） | 164.13 req/s |
| 全部请求 median / p95 | 1.13 ms / 2.34 ms |
| 成功响应 median / p95 | 4.81 s / 11.12 s |
| 压测后 `/health` | 200，0.96 ms |
| 压测后 Prediction/Auth/DB 队列深度 | 0 / 0 / 0 |

这个聚合结果由 64 VU 过载档主导，不能把 164 QPS 解释为成功预测吞吐；它证明的是容量之外会快速 503、没有未知错误，负载停止后队列能够清空且 EventLoop 健康检查仍及时。单独的 ONNX 与 ONNX+MySQL 基准继续记录在 M3/M4 性能文档中。

首次压测还实际复现了“认证失败后无退避重试”的放大效应，因此负载工具增加了身份预热和有界退避。这也是系统为什么既要服务端背压，也要客户端退避的工程证据。`scripts/verify_full.py` 把完整测试、确定性 Agent 评测、Compose 演示和可选 k6 串成统一入口。

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
