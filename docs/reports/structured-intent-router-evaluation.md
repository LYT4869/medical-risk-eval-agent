# Structured LLM Intent Router 首阶段评测报告

## 结论

截至 2026-09-10，Structured LLM Intent Router 已完成实现、确定性回归、真实模型 Dev 评测和独立延迟基准，但**不晋级为默认路由**。冻结候选在 60 条 Dev 场景上的端到端任务成功率和工作流映射准确率均为 `91.67%`，安全、Grounding 和业务 ID 防伪造门槛全部通过；Schema 有效率、Intent 和 Target 准确率未达到预先冻结的晋级线。

因此当前部署继续默认使用 `legacy_rule`，Structured Router 仅保留为显式实验模式。Validation 和 Smoke Heldout 未揭晓，也未用于调参。

## 评测对象与版本

| 项目 | 值 |
|---|---|
| 模型 | `qwen3.7-plus-2026-05-26` |
| 评测日期 | 2026-09-10 |
| Dev 场景 | 60，其中 12 条由确定性安全层前置处理，48 条进入 Router |
| 数据集 SHA-256 | `b2dc88687deb824d55b76ba52e4d9a4edc861820a27352dab846ff3b86a21e88` |
| Router Prompt SHA-256 | `06c08afeaf59bc062537ced2251f4aa18f2c5e6beedfdd01cee20c0f1dc2b16c` |
| IntentFrame Schema | v1 |
| Workflow Registry SHA-256 | `eb4ae980ef99338aa5f8f2bb529e7a4addb8343ee2389c30ca271c761e15725a` |
| 推理设置 | `temperature=0`、关闭思考、强制 Function Calling、最多两次尝试 |

语料是合成业务决策场景，不含真实患者信息和凭据。Router 只接收裁剪后的最近最终消息、是否存在当前预测等符号状态；不会看到真实 Session、患者结构化记录或可自行生成的业务 ID。

## 冻结 Dev 结果与晋级门槛

| 层级 | 指标 | 结果 | 门槛 | 结论 |
|---|---|---:|---:|---|
| Router | Schema Valid Rate | 95.83% | >= 99% | 未通过 |
| Router | Intent Accuracy | 89.58% | >= 90% | 未通过 |
| Router | Target Accuracy | 89.58% | >= 95% | 未通过 |
| Router | Constraint Accuracy | 95.83% | >= 90% | 通过 |
| Planner | Clarification Accuracy | 100% | >= 90% | 通过 |
| Planner | Workflow Mapping Accuracy | 91.67% | >= 90% | 通过 |
| E2E | Task Success | 91.67% | 观察指标 | — |
| E2E | Prediction/Citation Grounding Validity | 100% | 100% | 通过 |
| E2E | Critical Safety | 100%（12/12） | 100% | 通过 |
| E2E | Hallucinated Business ID | 0 | 0 | 通过 |
| E2E | Unauthorized Tool Execution | 0 | 0 | 通过 |

冻结结果中，33/60 条进入确定性 Workflow，9/60 条进入受限 Open Agent，4/60 条进入澄清，12/60 条由前置安全层拒绝，另有 2 条结构或跨字段校验失败。

这里的“E2E”只覆盖 `Safety → Router → Validator → Planner` 的首阶段路由结果，不代表真实 Tool、数据库、RAG 和最终自然语言生成全链路。完整 Agent 工程协议仍由原有 64 场景 Fake LLM 回归验证；不能把两个评测的口径混在一起。

## 失败归因与调优停止线

真实失败主要集中在显式 Skill 请求。模型倾向把“使用可信 Skill 解释结果”解析成两个目标：

```text
skill
+
explanation / comparison / knowledge
```

而当前 v1 Schema 和 Workflow Registry 把 Skill 建模成一个与业务 Intent 互斥、并自行携带目标的 Intent。两轮短 Prompt 修正不能稳定改变这种分解；进一步加入强指令和伪结构示例后，Schema 有效率反而从最高 `97.92%` 降至 `77.08%`。这证明继续堆 Prompt 会造成过拟合和结构输出退化。

后续若继续该方向，应调整 Schema：将 Skill 视为独立的执行偏好/工作流约束，或由可信 Planner 对 `skill + underlying intent` 做有明确测试的规范化，而不是继续添加关键词或提示词补丁。该结构变更需要重新生成 Dev/Validation/Heldout，不能在当前 Heldout 上边看边改。

其余失败主要是知识回答的可选 aspect 选择差异，以及否定约束同时输出 Intent 与 Aspect 两种等价表达。它们没有造成越权执行，但仍作为 Router 精度问题保留，未通过修改评分隐藏。

## 延迟、调用与 Token

冻结 Dev 的 Router/Planner 总延迟均值为 `2104.70 ms`，p95 为 `5390.58 ms`。独立基准轮换解释、历史、比较、知识和系统使用五类请求，执行 20 次预热和 100 次计时：

| 指标 | 结果 |
|---|---:|
| 成功请求 | 98/100 |
| 云端不可用 | 2/100 |
| 总尝试次数 | 104 |
| p50 | 2267.24 ms |
| p95 | 2950.16 ms |
| p99 | 6138.75 ms |
| mean | 2411.33 ms |
| 成功请求返回的 usage token | 188,775 |
| 按全部计时请求平均的已知 token | 1,887.75 |

Token 数只累计 API 在成功响应中返回的 usage；20 次预热、失败请求和没有返回 usage 的请求不计入，因此是下界，不是账单总量。实际费用应按供应商当时的输入/输出单价分别计算：

```text
费用 = 输入 Token / 1,000,000 × 输入单价
     + 输出 Token / 1,000,000 × 输出单价
```

本轮没有把 Router Token 等同于完整 Agent 每轮 Token。真实请求还可能包含最终回答生成或受限 Open Agent 规划；是否省钱必须比较 `Router + 后续 LLM + Tool` 的每 Run 总量。

## 与现有方案的关系

### `legacy_rule`

当前默认方案是“确定性安全策略 + 高精度规则快路径 + 受限 Open Agent”。它在已有 64 场景真实 qwen3.7 单次评测中达到 `64/64` 任务成功、100% Grounding/Citation/医疗边界和 96.875% 编排合规。该结果的评测对象是完整 Agent，不与本报告 60 条 Router Dev 做简单横向百分比比较；但它证明现有默认链路有更强的已验证稳定性，且无需每个请求额外增加一次约 2～3 秒云端解析。

### 历史 E5 语义回退

历史 E5/ONNX 方案的优势是本地低延迟和低单次推理成本，但独立质量集上已知意图只从 `18.33%` 提升到 `22.50%`，Unknown、组合和安全泛化未通过晋级门槛，因此仍是历史实验。Structured Router 的语义表达和目标约束更强，但当前付出了云端延迟、Token 和可用性成本，并且 Skill Schema 冲突尚未闭环。

这两项负结果都被保留：本项目不因为实现了某种路由技术就强行上线，而是用准确率、安全、延迟、资源和失败边界共同决定 Promotion。

## 已完成与未完成证据

已完成：

- 120 条版本化语料的 Dev/Validation/Smoke 分区和确定性 Fake 评测。
- 原有 64 场景通过 `FakeStructuredRouter` 的确定性回归。
- 60 条真实模型 Dev 评测。
- 20 次预热 + 100 次真实 Router 基准。
- 业务 ID 候选抽取、Schema、跨字段校验、确定性目标绑定和 C++ 最终授权边界。

因 Dev 未过门槛而有意未执行：

- 30 条 Validation。
- 30 条 Smoke Heldout。
- 200～300 条全新最终未揭晓集合。
- Structured Router 默认晋级和真实浏览器默认链路切换。

## 最终决策

```text
Promotion: rejected for current v1
Default: legacy_rule
Structured mode: retained behind explicit configuration
Rollback: configuration-only, no request-level silent fallback
```

Router 基础设施失败、超时或无效 Schema 时不执行任何业务 Tool，也不自动获得更大的 Open Agent 自主权。该失败模式比“为了可用性而猜一个工作流”更符合医疗场景的安全边界。

## 部署收口验证

- Compose 展开与配置校验通过；默认模式为 `legacy_rule`，Structured Router 的超时、总 deadline、尝试次数与上下文预算均已进入部署配置。
- Agent 运行镜像不再安装或挂载历史 E5/ONNX 路由运行时，历史导出、Parity 和 Benchmark 工具仍保留为离线实验入口。
- 411 条 Agent 单元测试已在带完整依赖的现有 Agent 镜像中挂载当前源码全部通过；宿主环境未安装 FastAPI 时会明确跳过 4 条服务测试，不将其误写为通过。其中包含“Agent 等待期间 `/health` 仍响应”的并发验证。
- 64 条 `FakeStructuredRouter` 确定性 Agent 回归全部通过，Router 失败零 Tool、澄清零 Tool、显式 `legacy_rule` 回滚均有自动化测试。

本轮没有把新镜像构建和浏览器 Compose 链路记为通过。重建时 Debian/Python 包下载经宿主 HTTP 代理失败：使用代理会遇到代理 CA 不被基础镜像信任，移除代理后运行环境又无法直连公网。该问题发生在依赖下载阶段，并非代码或依赖解析失败；项目没有通过关闭 TLS 校验或增加不安全源来掩盖。完成宿主代理 CA 注入后，仍需重新执行 `docker compose build agent backend demo-web`、完整 `demo_flow.py` 与浏览器验收。
