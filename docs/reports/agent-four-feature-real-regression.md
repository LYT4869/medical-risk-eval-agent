# 四项 Agent 改动的真实定向验证

日期：2026-09-14；源码检查点 `87e8baa`；隔离分支 `test/composite-response-semantics`。

## 结论

本轮只测试，不修改 Agent、Router、Tool 防重复、知识源或默认部署。四项改动不能统一宣布全部闭环：

| 改动 | 本轮证据 | 结论 |
|---|---|---|
| 目标绑定与契约一致性 | 上次摘要/解释、明确 ID 顺序、记录缺失正确；8 条契约测试通过 | 已覆盖定向绑定；格式/跨字段修复由确定性测试证明，本轮真实 Router 未触发修复 |
| Subgoal 进展与综合回答 | 三目标开放请求取得 history、comparison、explanation、knowledge，最终上下文三目标均 completed | 任务和原始要求没有在拆分中丢失，但最终回答仍有不必要且证据不足的特征关联扩展 |
| 按要求裁剪证据 | 中英文 path-only、features-only；组合请求的 explanation 也未传 important_features | 数据裁剪有效；不能据此保证 LLM 不从路径中的特征名自行扩展 |
| 新标签知识与引用 | 实际官方 MCP 获取新索引；比较后解释 0/1 编码并保留临床未知边界 | 该定向新证据回答通过；纯模型知识问答仍存在响应策略误拦和无依据 prediction 引用问题 |

**这是定向回归，不是新 Heldout、全量晋级或整个系统的成功率。** 不切默认、不合并、不推送；旧 48 条及历史 6 条结果未改。

## 测试条件

- 当前云模型 `qwen3.7-plus-2026-05-26`，thinking=false、temperature=0。
- 使用真实结构化 Router 和真实最终回答；Router 输出上限 384，单次 6 秒、总预算 13 秒、最多两次；最终回答上限 1024；Run 总预算 25 秒、最多 5 个执行 step、8 次 Tool。
- 18 条先冻结：12 条原 48 场景子集保持原消息、fixture、Gold 不变；6 条独立新证据条件。其中两条沿用原问题但通过实际 MCP 检索，不能直接改写原分数。
- Native Tool 使用严格 ID 的**合成业务记录**，不代表真实 Gateway/ONNX/MySQL/RBAC 全栈通过。
- 新证据使用官方 MCP SDK Streamable HTTP，测试服务仅监听 `127.0.0.1:18092`，独立测试签名密钥，不使用生产 Secret；索引 `knowledge-c3132c6f8032ad67`，固定模型 revision、既有校准阈值。
- 初始 6 条 MCP Token 仅有 `model_public`，专门验证模型知识域。Router 若选 clinical，结果为空；这不等于正式 Patient 的 clinical_patient 检索故障。后续单独补齐 Patient 的两个 scope 做对照，但保留原用例的 Doctor 角色；不是 Patient-role 完整权限验证。
- 旧知识环境的可选 torchvision 兼容问题仍使用进程内 text-only 诊断方式绕开；没有修复生产环境或宣称正常部署启动 Gate 通过。
- 不持久化思维链、中间规划正文、Token 或完整检索 excerpt；只保存合成问题、最终回答、来源元数据和证据字段路径。

## 原证据条件：逐条人工检查

| 原场景 | 结果 | 检查依据 |
|---|---|---|
| 021 上次摘要 | 通过 | history 后读真实上一条 `…40`，label 0、probability 0.4 |
| 013 当前仅路径 | 通过 | 正确当前 `…27`；最终证据没有 important_features |
| 014 当前仅特征 | 通过 | 正确当前 `…2a`；最终证据没有 decision_path |
| 017 英文仅路径 | 通过 | 当前 `…33`；路径、分支及叶输出正确 |
| 019 上次路径 | 通过 | 读上次 `…3a`，不是最新记录 |
| 007 明确顺序比较 | 通过 | `…16→…15`，0.2→0.7，增加 50 个百分点，confidence 下降 0.1 |
| 026 比较＋标签含义 | 未完整回答 | 比较正确，但仍用 PPH/AUC 替代标签定义，未指出编码证据缺口 |
| 028 比较＋临床含义 | 通过 | 上次→最新下降 0.02；明确不能等同病情变化 |
| 031 路径＋阈值知识 | 通过，覆盖告警 | 路径阈值和标准化正确；不代表临床阈值定义有证据 |
| 037 无当前预测 | 部分 | 无虚构，但“请说明哪条记录”不如直接说明尚无当前预测及下一步 |
| 039 无上次预测 | 通过 | 查 history 后说明没有可定位的上次记录，空 grounding |
| 041 指定记录不存在 | 通过，评分接线遗漏 | 尝试正确 missing ID、安全回答、空 grounding；一次性 harness 未派生评估用 missing_prediction_id，原 minimum_contract=false 不是 Agent 失败 |

原始自动最低契约 11/12，失败为 041；人工检查发现 026 实际漏答、037 部分回答、031 有覆盖告警。**自动成绩不能代替语义检查，也不能把纠正评分接线视为修好了 Agent。** 原 raw 保留，未追写。

## 新证据条件：真实 MCP 与最终回答

| 场景 | 结果 | 问题或证据 |
|---|---|---|
| live_026 比较＋标签含义 | 通过 | 得到编码章节；答出负类 0/正类 1、-1→0、诊断与临床定义未知；引用 `cite_c2022db258759ade4229` |
| live_028 比较＋临床变化 | 部分，机制扩展告警 | Router 选 clinical，而初始 Token 仅 model_public，故空结果；比较及缺证据说明正确，但又扩展“模型输出变化反映树路径改变”，不能据此断言神经概率机制 |
| live_probability 概率/置信度术语 | 未完成 | 检索到对应资料，最终返回知识回答不可用；后续诊断确认响应策略误拦 |
| live_mechanism 神经分类与树 | 未完成 | 检索到对应资料，最终降级；后续复现捕获了未经 Native Tool 验证的预测 ID 引用 |
| live_missing_dictionary 结局与窗口 | 未完成 | Router 选 clinical，受 model_public-only 测试 Token 限制无结果，且最终说明不可用；不能据此断言权威字典已补齐或正式 clinical 检索损坏 |
| live_composite 三目标综合任务 | 执行通过，回答约束/证据告警 | 四 Tool 均成功，三目标均 completed；原始请求、排除特征约束和裁剪后的路径仍在上下文。但正文追加“当前高概率主要与较高产时出血数值相关”，没有主要特征重要性证据，也超出仅路径要求 |

新证据组原始最低契约 2/6，其中三目标用例自动通过但人工有告警。引用 ID 只使用了实际返回的候选，但 **ID 合法≠每一句声明都有证据**。

## 四条诊断：不要把所有降级都归因于 Prompt

原 18 条冻结之后，只增加观测，保持生产代码和模型配置不变；4 条诊断结果独立保存，不替换原失败。
诊断可以证明可复现的失败机制；原运行没有捕获的最终校验原因，不能据此逐条反推为完全相同的事件。

| 对照 | 捕获结果 | 解释 |
|---|---|---|
| 概率术语＋当前预测上下文 | `missing_prediction_grounding` | 模型输出合法 envelope、有效 citation、空 prediction grounding；一般术语被误当患者预测事实 |
| 同问题，去掉当前上下文 | 仍 `missing_prediction_grounding` | 排除“只是当前预测上下文污染”的单一解释 |
| 神经/树机制＋当前上下文 | `unavailable_prediction` | 机制说明与 citation 正确，但 grounding 附上上下文中的当前 ID；没有 Native Tool 证据，拒绝是正确的 |
| clinical 问题，补正式 Patient 两个 scope，角色仍为 Doctor | 返回 RCOG 资料、响应校验接受 | 回答仍诚实指出资料没有模型概率与临床结局的对应证据；是 scope 对照和诚实部分覆盖，不是 Patient-role audience 路径验证或临床定义补齐 |

### 已定位的响应策略问题

`ResponsePolicy` 的 numeric-fact 检查并非只识别患者数值，它把 probability、概率、置信度、标签、label 等词本身也当作需要 prediction ID 的声明。
即使本轮只有模型知识目标、没有 Native 预测结果，正确解释术语也被拒绝。这个错误不在 MCP，也不是输出 JSON 不合法。

不能简单改成“有 citation 就放过所有概率”，否则会允许患者事实借通用知识引用过关。后续应区分模型一般知识与患者具体预测的证据要求，保留假 ID/具体患者事实拒绝。

### 已定位的上下文引用风险

纯知识问题不需要患者结果，模型却可能把请求中已有的 current_prediction ID 放入 grounding。它只是定位上下文，不是本轮已验证的 Tool 事实；此次拒绝正确。
后续应减少知识-only finalizer 的无关患者上下文，并维持 prediction 引用必须来自 Native Tool 的原则，而不是为了通过率放宽政策。

### 还没有解决的问题

原证据不足的 026 仍漏答；最终文本可能越过已裁剪证据做关联扩展；缺当前记录的澄清还不够贴切。它们没有因新标签资料或本地测试通过而自动消失。

## 验证、计量与复现

- Agent 单元测试 519 条：515 通过、4 跳过；其中契约一致性 8 条、结构化执行 29 条新鲜单独回归通过。
- Knowledge 在已有官方 MCP SDK 检查环境 12/12 通过；上游 lifespan forward-reference 警告如实保留。
- 官方 MCP discovery 和带短期签名 Token 的 call_tool 预检通过；无生产 Secret。
- 18 条真实回归：39 次物理请求，输入 71,214、输出 7,727，已知 Token **78,941**；1 次未知 usage。
- 4 条诊断：8 次物理请求，输入 19,667、输出 2,662，已知 Token **22,329**；未知 usage 0。
- 合计 **47 次请求、101,270 已知 Token**。未知 usage 不算成 0，平台最终账单可能更高；不是准确费用记录。
- 预算是单 Run 准入限制，不是硬费用上限；初始 18 条和诊断各自保存预算/配置，未在执行途中改 Prompt 或评估数据。
- 本轮不做全量压测、Heldout、模型数值 parity、真实数据库资源授权或部署 Gate，不更新简历成功率和性能数字。

生成证据在本 worktree 的 `artifacts/evaluation/` 忽略目录；模型源码 SHA256：`c62fdfcadf96b78d4b98aa03afafed7c1bf0c46796227cf853d6da7af66b8115`。

```text
feature4-targeted-cases-20260914.json
1afab4711f8a27b3f0155d8848e5ebfecc6833b2b36e213059477bd2004cbbb3
feature4-targeted-real-20260914.json
a5ac2bf6e7e358b9bc0e62c11683fb36a375b3583def9bf245e7d7ffd3b615c3
run_feature4_smoke.py
723ca8a1c6588ec914160c7d8224ebfbe30972834a9f1d999d1f5ed7e0dc014b
feature4-diagnostic-cases-20260914.json
ce7a27db384384fc542ff3cfa5a6954b7b6a8fb89df486caa261a250717e86bf
feature4-diagnostic-real-20260914.json
b2329927a42656af046796668cf8012ef0cbfd635da4811ea49f606f4a3b3c34
run_feature4_diagnostic.py
b000ec7c675bb09ab820d6cc68207a50ef6205e110df815ddacdd0bccc9abd9e
```

一次性 harness 及 MCP 启动器仅用于验证，不作为新业务子系统。复现须使用相同冻结用例、源码 SHA、模型/Prompt 配置和索引，不能把变更后的补测写回原结果。

## 后续顺序

先解决模型知识与患者事实的证据边界、知识-only 上下文和最终回答越界扩展，再做小规模修复回归；Tool Action Signature 改良仍留在待办，不与这一批混改。

前序：[契约回归](agent-contract-consistency-regression.md)、[标签知识检索验证](agent-label-knowledge-regression.md)。
