# 真实大模型接入与验证

## 当前配置

treeSem Agent 已通过 OpenAI-compatible Chat Completions 协议接入
`qwen-plus`。API Key、工作空间地址和代理地址只保存在未提交的 `.env`
与进程环境中，不进入 Git、日志、Trace 或响应。

Agent 使用 `temperature=0`，单次最多生成 1024 Token。容器只继承访问云端
所需的 `HTTP_PROXY` / `HTTPS_PROXY`，`NO_PROXY` 必须包含
`backend,knowledge,model-adapter,mysql,127.0.0.1,localhost`，保证内部 Tool
调用不经过外部代理。

## 工程改动

- OpenAI-compatible Client 显式限制温度和输出长度，并统计请求、输入、输出和总 Token。
- 支持解析 JSON 代码围栏，避免模型输出 Markdown 包装后丢失结构化字段。
- 预测 grounding 由本轮可信 Tool Result 确定性生成；模型伪造或跨 Run ID 仍会被拒绝。
- Citation 取模型声明与正文中合法引用的并集；模型遗漏正文尾注时由编排层补齐。
- Skill Catalog 明确要求先激活匹配 Skill，再调用该 Skill 声明的领域 Tool。
- 真实评测可按 case 限量执行，并控制关键场景重复次数，避免直接运行完整矩阵造成意外消耗。

## 2026-08-24 真实验证结果

### 完整端到端 Chat

调用链：

```text
C++ Gateway /api/v1/chat
  -> Python Agent
  -> activate_skill(pph_evidence_education)
  -> MCP search_medical_knowledge
  -> Knowledge Retriever
  -> 带 3 条 Citation 的回答
```

结果：

```text
HTTP                         200
Agent status                completed
Agent steps                 3
LLM requests                3
Prompt tokens               5749
Completion tokens           249
Total tokens                5998
End-to-end latency          8465.358 ms
Agent latency               7864.053 ms
MCP client latency          1022.728 ms
Knowledge retrieval latency 857.031 ms
```

同一个 W3C Trace 串联了 C++ Gateway、Python Agent、MCP Client 和 Knowledge
Server。日志中只保存 operation、耗时、结果和 Trace 标识，不保存聊天正文、
Token、临床输入或完整 Tool Result。

### 六场景真实冒烟评测

场景覆盖预测、医生解释、医疗知识 RAG、解释 Skill、Prompt Injection 和紧急
医疗边界。每个场景只执行一次，完整报告位于本地忽略目录：

```text
artifacts/evaluation/qwen-plus-smoke-20260824.json
```

结果：

```text
Task success                    5 / 6 = 83.33%
Tool argument valid rate        100%
Skill routing accuracy          100%
Prediction grounding validity  100%
Citation validity              100%
Medical-boundary pass rate      100%
Exact tool-sequence accuracy     50%
LLM requests                     12
Total tokens                  20396
```

唯一未通过项是医生解释场景：模型调用了 `get_explanation`、`get_prediction`
和 `search_medical_knowledge`，结果安全且具备有效 grounding，但超出了评测预期的
最小 Tool 序列。这说明下一轮应优化 Tool 选择效率和 Prompt，而不是放宽引用、权限
或医疗安全门槛。

### 最小 Tool 路由优化

随后在不增加硬编码意图路由的前提下，明确了系统提示词和 Tool 描述的职责边界：

- 只解释重要特征或决策路径时，单独调用 `get_explanation`。
- 只有询问标签、概率、置信度或模型版本时才增加 `get_prediction`。
- 只有询问一般知识、证据、指标、术语或模型限制时才增加知识检索。
- 用户明确要求完整稳定流程时，仍允许激活解释 Skill 并执行其完整 Tool 集合。

对 `explain__doctor_zh` 使用真实 `qwen-plus` 连续执行三次：

```text
Runs                          3
Task success                  100%
Exact tool-sequence accuracy  100%
Tool argument valid rate      100%
Prediction grounding          100%
Total tokens                  11916
```

对应本地报告：

```text
artifacts/evaluation/qwen-plus-explain-routing-20260824.json
```

生产 Compose 链路又完成一次 Patient 预测后解释：返回 HTTP 200，只调用一次
`get_explanation`，两步结束，grounding 与刚创建的 prediction ID 一致，且没有
不必要的知识检索。

优化后重新执行六个代表场景，结果如下：

```text
Task success                    6 / 6 = 100%
Tool argument valid rate        100%
Skill routing accuracy          100%
Prediction grounding validity  100%
Citation validity              100%
Medical-boundary pass rate      100%
Exact tool-sequence accuracy     66.67%
LLM requests                     12
Total tokens                  21938
```

精确序列低于 100% 的两条不是任务失败：紧急场景允许不等待检索直接提示急救，
普通 PPH 教育场景允许先激活可信教育 Skill，再执行知识检索。评测将这两种路径记录为
等价工作流，同时继续单独保留精确 Tool 序列指标。优化后完整报告位于：

```text
artifacts/evaluation/qwen-plus-smoke-optimized-20260824.json
```

优化前六场景报告继续保留为基线。正式数据集现已升级为 64 个独立场景、79 轮执行。

### Schema v2 分层回归

从正式数据集中分层抽取 15 条场景、18 个连续对话轮次，覆盖预测、解释、历史、比较、
模型知识、临床知识、三个 Skill、安全、紧急问题、无答案和两类下游故障。首次完整运行的
真实结果为：

```text
Task success                    11 / 15 = 73.33%
Tool argument valid rate                93.33%
Tool outcome valid rate                 93.33%
Skill routing accuracy                  86.67%
Prediction grounding validity           93.33%
Citation validity                       93.33%
Medical-boundary pass rate             100.00%
LLM requests                                  37
Prompt / completion tokens       68669 / 2623
Total tokens                               71292
Latency mean / p95               4.58 s / 6.62 s
```

四条失败分别暴露了四类问题，而不是简单归因于“模型不稳定”：多轮历史信息没有把上一轮
Tool Result 持久化进下一轮上下文；循证教育请求绕过了显式 Skill 激活；安全策略成功拦截了
伪造预测但以执行失败结束；无答案场景直接安全拒答，却被评测器错误要求必须先检索。

修正遵循最小边界：允许比较前重新获取历史、允许受信比较 Skill 包装合法领域流程、强化
显式 Skill 请求必须先激活的提示和 Tool 描述，并允许明确的无答案请求直接安全拒答。四个失败
场景随后均获得一次通过证据。这里不能写成“一次 15/15”，因为它们不是在同一版本、同一次
完整运行中产生的结果。之后又将策略拒绝改为确定性的安全用户回复，不再把 grounding 拒绝
暴露为内部执行失败。

本地报告：

```text
artifacts/evaluation/qwen-plus-staged-v2-20260824.json
artifacts/evaluation/qwen-plus-staged-v2-failures-after-fix-20260824.json
artifacts/evaluation/qwen-plus-staged-v2-history-after-fix-20260824.json
```

这些文件处于 Git 忽略目录，不包含 API Key。

### Schema v2 完整单次评测

2026-08-25 使用 `qwen-plus` 对全部 64 条独立场景执行一次真实评测，共 79 轮：

```text
Task success                    55 / 64 = 85.94%
Tool argument valid rate                 98.44%
Tool outcome valid rate                  96.88%
Skill routing accuracy                   98.44%
Prediction grounding validity            98.44%
Citation validity                        95.31%
Medical-boundary pass rate              100.00%
No-answer accuracy                      100.00%
Prompt-injection pass rate              100.00%
Security graceful-response rate         100.00%
LLM requests                                  183
Prompt / completion tokens       336952 / 10775
Total tokens                              347727
Latency mean / p95                4.77 s / 7.52 s
```

原始失败为 9 条。逐条审计后，其中 5 条是模型在多轮场景中安全地重新读取历史或预测，属于
评测器把等价工作流误判为失败；1 条是故障恢复场景把 Skill 激活成功也错误计入领域 Tool 状态。
这 6 条已通过规则测试修正，但不反向篡改原始真实报告。

另外两个模型知识题最初虽然返回了 citation 对象，但 Fake Knowledge 的 excerpt 只有通用占位文本，
并不包含标准化或模型指标的实际证据。模型不能用该证据支撑答案时，系统按 fail-closed 拒答是
正确行为，不能归因为“qwen 固定漏写 citation”。将夹具替换成对应主题的非患者合成证据后，
两题定向真实复测 `2/2` 通过，citation validity 为 100%，共消耗 9,122 Token。最后一个显式
Skill 场景结合调用次数判断为模型并行规划 Skill 激活和领域 Tool，因此请求设置
`parallel_tool_calls=false`。随后真实定向复测按 5 个 Step 串行完成 Skill 激活和 4 次 Tool Call，
所有路由、参数、结果与 grounding 指标均为 100%，消耗 8,840 Token。

为避免把分散的定向结果拼成结论，最后在同一代码、Prompt、数据集版本下统一重跑原始 9 个
失败场景。结果为 `9/9`，共 14 轮、40 次模型请求和 77,287 Token；任务成功、Tool 参数、Tool
结果、Skill 路由、prediction grounding 与 citation validity 均为 100%。精确 Tool 序列为
55.56%，因为部分场景采用了评测明确允许的可信 Skill 包装或安全上下文刷新；这些路径单独保留
统计，但不应误写成任务失败。原始 64 场景的 85.94% 仍不追溯修改。

完整原始报告与三条定向复测报告：

```text
artifacts/evaluation/qwen-plus-full64-single-20260825.json
artifacts/evaluation/qwen-plus-three-failures-20260825.json
artifacts/evaluation/qwen-plus-two-evidence-20260825.json
artifacts/evaluation/qwen-plus-skill-serial-20260825.json
artifacts/evaluation/qwen-plus-original-nine-regression-20260825.json
```

上述报告都位于 Git 忽略目录且不包含 API Key。`authorization_evidence=not_measured` 表示这套
合成 Tool 决策评测不测 Gateway RBAC，不能把它写成“跨角色泄漏为 0”。

### qwen3.7 模型升级对照与代码治理

2026-08-25 将模型替换为 `qwen3.7-plus-2026-05-26`，关闭思考模式，并在同一套 64 场景、
79 轮数据集上执行一次完整真实评测。原始结果保持不变：

```text
Task success                    36 / 64 = 56.25%
LLM requests                                  222
Total tokens                              489,683
Latency mean / p95                 9.46 s / 16.49 s
```

28 条失败进一步分为：9 条只是不符合旧的精确工作流统计、10 条在零 Tool/零 Step 阶段提前结束
且旧报告没有稳定原因码、9 条是 grounding、Tool 结果或任务目标没有满足的实质失败。报告文件为
`artifacts/evaluation/qwen37-full64-20260825.json`，SHA-256 为
`329e601263a4317a520fb3262e0e89ecdcb6848a831a9671345b568941704d8e`。

这次对照说明“更新的模型”不等于“在当前 Agent 协议上更稳定”。qwen3.7 更倾向先规划完整
工作流、激活 Skill、追加只读 Tool 或重复检索，导致调用轮次、Token 和延迟同时上升。仅继续
加 Prompt 会把安全和终止条件交给模型解释，因此改为请求级代码治理：

- `AgentRunGuard` 只对高置信意图暴露最小 Tool 集合，普通请求不能激活 Skill。
- 成功知识检索后移除检索 Tool；空结果或错误只允许一次重试。
- 成功历史/比较后收窄后续 Tool，显式伪造或越权请求在首次 LLM 调用前拒绝。
- 同轮“激活 Skill + 领域 Tool”先完成激活，对同轮领域调用返回结构化错误，再允许模型修正。
- Agent 终止和评测报告使用稳定错误码，不保存 Prompt、用户消息、Tool 参数或上游正文。

治理后的 Fake LLM 全量门槛为 64/64 场景、79/79 轮通过，Prediction Grounding 与 Citation
Validity 均为 100%，关键失败为 0。该结果证明编排协议已确定性收敛，不能替代后续真实模型
复测。当前生产默认继续固定为北京地域的 `qwen-plus-2025-07-28`，默认关闭思考模式；只有
qwen3.7 通过受限 12 场景晋级门槛后，才考虑再次运行付费的完整矩阵。

## 复现方式

先把真实配置写入未提交的 `.env`：

```text
TREESEM_AGENT_LLM_MODE=real
TREESEM_AGENT_LLM_BASE_URL=<OpenAI-compatible /v1 URL>
TREESEM_AGENT_LLM_MODEL=qwen-plus-2025-07-28
TREESEM_AGENT_LLM_API_KEY=<local secret>
TREESEM_AGENT_LLM_TEMPERATURE=0
TREESEM_AGENT_LLM_MAX_OUTPUT_TOKENS=1024
```

重建 Agent 并检查就绪状态：

```bash
docker compose up -d --build --no-deps agent
docker compose ps agent
```

先运行零费用预算预检：

```bash
python evaluation/run_evaluation.py --preflight-only
```

当前改用完整 64 场景、79 轮的 347,727 Token 实测作为预算基线：单次完整执行按实测为
347,727 Token，44 个关键场景各执行三次后共 177 轮，按平均值约为 779,085 Token。后一个
数字仍是预算估算，不包含上游重试和输出波动，运行前需要留出余量。
该 runner 使用确定性 Tool fixture 评估 LLM 决策，不得将
`cross_role_leakage_count=null` 改写为真实权限零泄漏。
