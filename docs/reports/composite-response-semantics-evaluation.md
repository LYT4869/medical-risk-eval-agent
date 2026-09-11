# 复合请求与最终回答语义诊断

## 结论

截至 2026-09-11，现有 `IntentFrame v2` 暂时没有出现“目标拆分后把用户总体诉求拆没”的充分证据，因此不升级到包含 `primary_goal`、`goal_relations` 和 `response_constraints` 的 Schema v3。

本轮使用 12 条合成复合请求，固定正确的结构化意图输入，单独检查 Router 之后的 Workflow/Open Agent 执行与最终回答。最终一次真实模型运行保存了全部合成场景回答；下表是修正同义表达误判后，按最终评分口径对这些已保存输出进行的离线重算，没有再次调用模型：

| 指标 | 结果 |
|---|---:|
| 业务任务成功率 | 91.67%（11/12） |
| Tool 选择、参数与结果有效率 | 91.67% |
| Prediction Grounding | 100% |
| Citation Grounding | 100% |
| 子目标完成率 | 88.89%（16/18） |
| 回答约束遵守率 | 91.67%（11/12） |
| 综合回答质量 | 91.67%（11/12） |
| 模型调用 | 20 次，25,235 Token |
| 平均 / p95 延迟 | 4.64 s / 5.75 s |

其中唯一失败场景是“比较最近两次预测，再结合模型资料解释 AUC 是否能说明个体差异”。Agent 在开放执行分支中达到 Step 上限，没有形成最终回答，因此该场景的两个子目标、回答约束和综合质量均按失败计入，未从语义指标分母中排除。

该结果说明当前系统保留原始用户消息、结构化子目标和裁剪后的 Tool Evidence，通常足以让最终模型恢复总体语义。当前更值得处理的是开放执行分支的偶发长规划和最终响应协议稳定性，而不是立刻扩大 Router Schema。

## 为什么需要独立评测

原有 64 条 Agent 评测主要回答：

- 是否调用了必要 Tool；
- Tool 参数、顺序和结果是否有效；
- Prediction ID 与 Citation ID 是否可信；
- 安全拒绝和医疗边界是否生效。

这些指标无法识别一种重要失败：Tool 全部执行成功，但最终回答遗漏了其中一个目标，或者没有遵守“只看路径、结论先行、区分模型事实与资料边界”等表达约束。

因此本轮在不修改 Agent 生产 Schema 的前提下，增加三个互相独立的诊断口径：

1. `subgoal_completion_rate`：每个声明的子目标既完成必要 Tool 链，也在最终回答中出现对应事实证据。
2. `response_constraint_adherence_rate`：检查明确可观察的必需内容、禁止内容、开头要求、长度限制及内部响应包装泄漏。
3. `integrated_answer_quality_rate`：只有全部子目标完成、全部回答约束满足，并呈现必要的跨目标综合关系时才通过。

业务任务成功率保持原定义，未与新指标混在一起。这样可以明确区分：

```text
语义解析是否正确
  ↓
Workflow / Tool 是否执行正确
  ↓
最终回答是否完整满足原请求
```

## 数据集设计

评测集位于：

```text
PythonServices/TreeSemAgent/evaluation/composite_response_cases.json
```

12 条场景覆盖：

- 最近两次预测比较 + 两条决策路径解释；
- 预测解释 + 模型知识边界；
- 历史记录 + 当前预测摘要；
- 比较结果 + AUC 总体指标边界；
- 只输出决策路径或只输出重要特征；
- 上一次预测的目标绑定；
- 预测解释、历史比较和循证教育 Skill；
- 中文与英文、患者与医生表达；
- 结论先行、简短表达、非因果说明和字段裁剪。

语料和 Tool 返回均为合成数据，不使用真实患者记录。评测中的结构化目标是人工声明的 Oracle Frame，不调用真实 Router，因此结果用于诊断 Router 下游是否丢失总体语义，不能用于证明 Router 自身准确率。

## 评测口径修正

第一轮自动评分过度依赖固定短语，将以下语义等价表达误判为失败：

```text
预期：不代表病情恶化
实际：并不直接等同于病情恶化

预期：不建立因果关系
实际：并不确立病因

预期：排序能力
实际：排序性能
```

因此只修正了评分器和人工声明的同义证据，没有修改 Agent Prompt 或执行逻辑。修正后的评分仍然保留以下客观失败：

- 没有最终回答；
- 必要 Tool 未执行；
- 明确要求的业务事实缺失；
- 输出中出现内部 `grounding_prediction_ids` / `grounding_source_ids` JSON 包装；
- 明确的禁止字段泄漏；
- 可观察的长度或开头约束不满足。

这次修正也暴露了评测工程的一条原则：自然语言质量不能用单个关键词做结论，规则评分应限于可观察事实和格式约束；更主观的“是否通俗、是否自然”仍需要人工抽检或独立 Judge，并在最终 Heldout 集上验证。

## 有价值的失败

### 1. Tool 正确但比较事实偶尔遗漏

一次校准运行中，比较与两条解释 Tool 均成功，但最终回答解释了路径变化，没有给出已取得的概率差。这属于真实的子目标表达遗漏，而不是 Router 丢失目标，因为正确的复合 Frame 和 Tool Result 已经到达最终回答阶段。

### 2. 开放执行分支达到 Step 上限

最终运行中，“比较 + 模型知识”需要历史、比较、知识检索和最终作答。模型没有在有限 Step 内结束，触发 `step_limit`。这说明复杂组合请求仍存在开放规划尾部风险。

当前不立即为每种组合新增 Workflow。后续只有相同组合在更大的未揭晓集合中稳定复现，才考虑增加一条高价值确定性 Recipe，或对 Target Binding 做少量条件依赖插入，避免 Workflow 数量无界增长。

### 3. 最终 JSON 包装偶发重复输出

校准运行中，模型曾先输出自然语言，再附带一份代码块形式的最终 JSON。客户端只在整个内容都是 JSON 时提取 `answer`，因此这种混合形式会进入最终回答，造成重复和冗长。

评测器现已将内部 Grounding 包装暴露视为约束失败。生产修复应单独建立测试，选择严格拒绝混合响应或安全提取唯一 JSON 对象，不能用删字符串的方式掩盖协议错误。

## Schema v3 的触发条件

暂不新增完整的：

```text
Original Request
Overall Goal
Subgoals: completed / pending
Response Constraints
Current Evidence
```

原因是原始请求已经保存在最终回答上下文中；completed/pending 属于 Executor Run State；Current Evidence 属于 Workflow/Tool 执行层。全部塞进 Router Frame 会混合语义解析、运行状态和证据管理三种职责。

只有后续全新 Heldout 集稳定出现以下模式，才进入最小 Schema v3 设计：

- Router 和 Tool 链正确；
- Grounding 正确；
- 最终回答反复遗漏顶层目标关系或明确表达约束；
- 通过 Finalizer Prompt 或确定性数据裁剪仍无法解决；
- 失败率超过预先设定的可接受线，而不是偶发单例。

即便触发，也只优先考虑：

```text
primary_goal
goal_relations
response_constraints
```

不会让 Router 生成 `completed/pending` 或 `current_evidence`。

## 运行方式

确定性回归：

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 PythonServices/TreeSemAgent/evaluation/run_evaluation.py \
  --mode deterministic \
  --routing-mode structured_llm \
  --cases PythonServices/TreeSemAgent/evaluation/composite_response_cases.json \
  --critical-repeats 1
```

真实模型诊断使用同一命令，将 `--mode` 改为 `real`。真实运行前必须加载本地未提交的 LLM 环境变量。

## 证据边界

- 这是 12 条 Dev 诊断集，不是最终统计显著性结论。
- 自动评分只覆盖显式事实与可观察约束，不等价于医疗回答质量评价。
- 使用合成 Tool Fixture，不测 C++ Gateway 的真实 RBAC 和跨患者隔离。
- 本轮固定 Oracle Frame，不测真实 Router 准确率。
- 真实模型仍有随机性；需要全新 Heldout 集验证后，才能作为 Schema 升级依据。
