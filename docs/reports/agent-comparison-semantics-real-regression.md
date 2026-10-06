# 比较数值与临床语义：真实模型定向回归

日期：2026-09-15；代码提交：`d9dd302`；模型：`qwen3.7-plus-2026-05-26`，thinking=false。

## 结论

本轮验证“程序约束事实、LLM 自由表达”，不是完整 Agent 质量分数。6 条已揭晓定向场景全部正确表达比较方向和百分点；从 20% 到 70% 表达为增加 50 个百分点，从 70% 到 20% 表达为减少 50 个百分点，不变场景表达为 0 个百分点。没有再把百分点写成相对百分比。

但结构化数值不能独自解决所有自由文本越界：三目标综合回答仍生成了“主要关联到产时出血数值较高”的无依据扩展，违反了用户“不讲关键特征”的约束，也把树路径条件扩展成了主要关联和“较高”。另一个临床边界回答主结论正确，但附加了与问题关系较弱的通用 PPH 复发与照护材料。不能将本轮写成 6/6 无告警通过。

人工结果为：3 条干净通过，1 条正确 no-answer，1 条通过但有相关性告警，1 条语义失败。数值与单位这个单独目标是 6/6。

## 场景结果

| 场景 | 执行与数值 | 人工语义检查 |
|---|---|---|
| 20% → 70% | 上升 50 个百分点；置信度下降 10 个百分点 | 通过；限定为模型编码，不作临床解释 |
| 70% → 20% | 下降 50 个百分点 | 通过；没有混成“下降 50%” |
| 40% → 40% | 不变，差异 0 个百分点 | 通过；类别保持模型负类 |
| 模型类别能否叫临床阴转阳 | 比较事实正确；MCP 检索返回空结果 | 正确 no-answer；明确不能将模型编码直接当临床状态。旧自动 Gold 强制要求 citation，因此自动失败不代表回答失败 |
| 比较＋路径＋模型知识 | Tool、实际引用、数值和三个 Goal 均完成 | 失败/告警；末段恢复“主要关联某特征、数值较高”，超出路径证据和排除约束 |
| 概率变化能否证明恶化或决定治疗 | 增加 50 个百分点；首次引用遗漏后格式修复成功 | 主结论通过；但附加的 PPH 复发照护材料相关性较弱，保留告警 |

## 测试环境事故与证据边界

第一次运行误用了不包含官方 MCP SDK 的 Agent 测试环境：前三条纯 Native 比较有效；后三条 MCP 调用约 31 ms 即报不可用，属于 Harness 环境错误，不计入模型质量。原始结果没有覆盖。随后使用同时包含 Agent 和官方 MCP SDK 的环境，仅重跑这三条；模型、代码和用例事实保持不变。Clinical 场景的测试 Capability 按 Patient 既定权限加入 `clinical_patient`，没有扩大生产权限。

有效回归由第一次运行的前三条和 MCP replay 三条组成。Native 预测、历史和解释仍是严格 ID 的合成夹具；Router、最终回答、官方 MCP Streamable HTTP 和本地真实知识索引参与真实执行。这不是 Gateway、RBAC、ONNX 或 MySQL 全栈验证。

## 调用与 Token

实际发生 26 个上游请求、60,825 个已知 Token（输入 54,831，输出 5,994），unknown usage 为 0。其中有效质量证据为 18 个请求、43,050 Token；另 8 个请求、17,775 Token 属于错误测试环境产生的无效 MCP 阶段。这里记录的是 Token，不是人民币费用。

## 冻结证据

```text
comparison-semantics-cases-20260915.json
f43d0fcf6ed66b1e3ce2a095c1d2d368f9bda82fa3cdfb89b13c969164aae701
run_comparison_semantics_real.py
758c8aeb7afa9cb4c23624e047dc4cad9b3b4f5b217791af2b8d8140a2896334
comparison-semantics-real-20260915.json
534b3b5076f5cb33dc51ae80aefae71116a3c3e2095b77bf40f4e9c82824f876
comparison-semantics-mcp-replay-cases-20260915.json
13b63b35c3a99b43a89222d26fb68793fd8091d907d4a5722a3baae085a911c0
run_comparison_semantics_mcp_replay.py
4fbd5b92face9ef61bfa09495e4b49855fc9a2140e6547604e8d231e6b7d8368
comparison-semantics-mcp-replay-real-20260915.json
6fa332a8171a8dc80d847fa83b1e7c84b50ed4e872987618f8defe22333bbfef
Agent source SHA
755f4a702b0b65c76c0400f80b5b79f31199e3da09e6444eb8baf791c261dea0
```

生成用例、完整回答和只含必要观测字段的原始证据保存在本 worktree 的忽略目录 `artifacts/evaluation/`。不提交 API Key、思维链或完整检索 excerpt。

## 下一步判断

本轮数值契约不需要回退，也不应升级成整段回答模板。后续若继续修复，应针对“Finalizer 恢复已排除内容”和“检索结果与问题相关性”分别处理：优先强化结构化证据裁剪和 no-answer/relevance 判定，不继续往 Router Schema 添加字段，也不靠结尾免责声明掩盖无依据结论。
