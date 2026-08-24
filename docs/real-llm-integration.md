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

优化前六场景报告继续保留为基线；完整 60 场景真实评测仍未运行。

## 复现方式

先把真实配置写入未提交的 `.env`：

```text
TREESEM_AGENT_LLM_MODE=real
TREESEM_AGENT_LLM_BASE_URL=<OpenAI-compatible /v1 URL>
TREESEM_AGENT_LLM_MODEL=qwen-plus
TREESEM_AGENT_LLM_API_KEY=<local secret>
TREESEM_AGENT_LLM_TEMPERATURE=0
TREESEM_AGENT_LLM_MAX_OUTPUT_TOKENS=1024
```

重建 Agent 并检查就绪状态：

```bash
docker compose up -d --build --no-deps agent
docker compose ps agent
```

先通过 `--case-id` 执行少量真实场景；完整 60 场景和关键场景三次重复只有在
确认免费额度、成本预算和冒烟质量后才运行。
