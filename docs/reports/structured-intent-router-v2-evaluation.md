# Structured LLM Intent Router Schema v2 评测报告

## 结论

截至 2026-09-11，IntentFrame v2 已完成 Dev 阶段验证。v2 将 Skill 从业务 Intent 中拆出，改为可选的可信执行偏好；最终一次真实 `qwen3.7-plus-2026-05-26` Dev 评测在 60 条场景上达到 `96.67%` 端到端任务成功率和 Workflow Mapping，较冻结 v1 的 `91.67%` 提升 5 个百分点。显式 Skill 场景不再出现 `skill + explanation/comparison/knowledge` 冲突。

v2 仍不晋级为默认路由：Schema Valid Rate 为 `95.83%`，Target Accuracy 为 `91.67%`，没有达到预先冻结的 `99%` 和 `95%` 门槛；同时一次 Dev 运行不能证明云端可用性稳定。默认继续使用 `legacy_rule`，v2 保留在 `structured_llm` 和 `structured_shadow` 模式。Validation 和 Smoke Heldout 未揭晓。

## 最小结构修正

v1 将 `skill` 与 explanation、comparison、knowledge 放在同一个 Intent 枚举中，但 Skill 表达的是执行方法，而不是业务目标。v2 改为：

```json
{
  "schema_version": 2,
  "goals": [
    {
      "intent": "explanation",
      "target": {"type": "current_prediction"}
    }
  ],
  "requested_skill": "explain_prediction"
}
```

只增加一个受枚举限制的 `requested_skill`，没有增加新的 Router、线程池、模型或策略 DSL。三个合法值为：

- `explain_prediction`
- `compare_prediction_history`
- `pph_evidence_education`

Validator 校验 Skill 与底层 Intent/Target 的固定兼容关系。若模型为普通请求自行填写 Skill，而用户没有明确提到技能、流程或对应 Skill 名称，Validator 会清除该执行偏好并继续执行普通 Workflow。这里的词面检查只是“用户是否明确授权使用额外执行偏好”的窄边界，不承担业务意图识别。

知识检索后的 Citation 继续由既有 Grounding Policy 强制校验，不再作为 Router 的 requested aspect。Router 只识别知识意图和 scope。

## 澄清状态修正

真实诊断发现，模型面对“解释那个”一类缺少目标的请求时，会正确输出 `needs_clarification=true` 和 unresolved reference，但可能自然地返回空 `goals`。v1/v2 初版强制至少一个 Goal，会把正确的澄清状态误判为非法结构。

最终契约允许空 `goals`，但必须同时满足：

```text
needs_clarification = true
unresolved_references 非空
requested_skill = null
```

否则仍在 Schema 加载阶段拒绝。Planner 根据 unresolved reference 生成确定性澄清，不执行任何 Tool。

## 最终 Dev 结果

| 层级 | 指标 | v2 结果 | 冻结门槛 | 结论 |
|---|---|---:|---:|---|
| Router | Schema Valid Rate | 95.83% | >= 99% | 未通过 |
| Router | Intent Accuracy | 91.67% | >= 90% | 通过 |
| Router | Target Accuracy | 91.67% | >= 95% | 未通过 |
| Router | Aspect Accuracy | 97.92% | 观察指标 | — |
| Router | Constraint Accuracy | 97.92% | >= 90% | 通过 |
| Router | Skill Accuracy | 97.92% | 观察指标 | — |
| Planner | Clarification Accuracy | 75.00% | >= 90% | 未通过 |
| Planner | Workflow Mapping Accuracy | 96.67% | >= 90% | 通过 |
| E2E | Task Success | 96.67%（58/60） | 观察指标 | — |
| E2E | Grounding Validity | 100% | 100% | 通过 |
| E2E | Critical Safety | 100%（12/12） | 100% | 通过 |
| E2E | Hallucinated Business ID | 0 | 0 | 通过 |
| E2E | Unauthorized Tool Execution | 0 | 0 | 通过 |

版本证据：

| 项目 | 值 |
|---|---|
| 模型 | `qwen3.7-plus-2026-05-26` |
| 数据集 SHA-256 | `d65d4f86a859c775b325d8dc1998eb6ab1042dee621faa35a0ebd9d30b8ad443` |
| Prompt SHA-256 | `e471850b8b28399bb810460cadd0da461acb3804e60ae0ae32a8c851fd651684` |
| Workflow Registry SHA-256 | `45c7b14e592ce5d6f8e0e6600704f60d56114e923242a065311ab3efc3dc7c59` |
| IntentFrame | Schema v2 |
| 真实 Router 请求 | 48；另有 12 条确定性安全拒绝 |
| Router 调用尝试 | 51 |
| 已知成功响应 Token | 91,603 |

已知 Token 不包含失败请求没有返回的 usage，因此仍是账单下界。

## 失败分层

最终只剩两条端到端失败：

1. 一条 explanation + knowledge 组合请求发生 `intent_router_unavailable`，属于云端或网络可用性问题。
2. 一条缺少样本索引的预测请求两次返回无效结构，属于格式稳定性问题。

另外四条只影响 Router 严格字段指标，没有改变最终执行：

- 一条普通比较请求被模型附加 comparison Skill，Validator 清除未被用户明确要求的 Skill，最终仍进入普通比较 Workflow。
- 两条缺少预测目标的请求输出空 Goals 和正确 unresolved reference，最终进入确定性澄清。
- 一条否定请求多输出了语义安全但冗余的 excluded aspect，Workflow 不变。

因此评测继续区分：

```text
Router 严格字段
→ Validator/Planner 语义执行
→ 端到端任务与安全结果
```

不通过修改评分隐藏模型的严格字段差异，也不把已被确定性代码安全纠正的冗余误报为业务失败。

## 延迟与成本

| 指标 | v2 最终 Dev |
|---|---:|
| 平均 Router/Planner 延迟 | 2.34 s |
| p95 | 5.48 s |
| 每场景平均已知 Token | 1,526.72 |
| Workflow Rate | 63.33% |
| Open Agent Rate | 8.33% |

与 v1 相比，已知平均 Token 从 `1,585.45` 降至 `1,526.72`，但 p95 未明显改善。Schema 修正解决的是语义建模，不解决云端排队、网络抖动和重试造成的长尾。

## 发布决策

```text
Default: legacy_rule
Experiment: structured_llm Schema v2
Gray verification: structured_shadow
Validation/Smoke Heldout: sealed
```

下一次晋级前应先用 Shadow 采集稳定性证据，确认 Schema 和澄清格式达到门槛，再运行 Validation；不能因为一次 Dev 达到 96.67% 就直接切换默认链路。

## 评测后的工程加固

本次冻结结果之后完成了两项不改变发布结论的工程加固：

- Router 的第二次格式修复不再只给通用提示，而是提供固定、脱敏的错误类别和最多四个 Schema 字段路径；不回传第一次模型原文、字段值或异常正文。
- Structured Workflow 和受限 Open Agent 在把 `get_explanation` 结果注入最终 LLM 前，按照已验证的 `requested_aspects` 投影数据。例如只请求 `decision_path` 时，只保留路径、`prediction_id` 和 `model_version`，不注入重要特征或其他解释元数据。

完整 Tool API、业务持久化结果和 Grounding 元数据均不受投影影响。默认 `legacy_rule` 没有类型化 `requested_aspects`，因此不伪造同等裁剪能力；上述加固的真实模型收益需要下一次同口径评测确认，本文不回写已经冻结的 96.67% 指标。
