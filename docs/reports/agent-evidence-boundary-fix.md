# 一般知识与患者事实：证据边界修复

日期：2026-09-14；基线 `5edd731`；隔离分支 `test/composite-response-semantics`。

## 结论与范围

已修复有合法知识引用的一般模型术语回答被要求提供患者 prediction ID 的错误。
知识-only 回答不再自动带入当前预测或整段历史临床聊天；可唯一识别的知识追问仅携带公共术语名称，不携带历史患者值。
具体患者断言、假 prediction/citation、没有合法引用及未获得 knowledge-only 范围的负例仍被拒绝。

这里的“负例拒绝”指已验证的测试集合，不是对所有自由文本的完整语义证明。
本轮没有新增逐句 Claim Checker、独立安全模型、Router 字段、Tool 防重复策略或默认部署切换。
没有合并、推送、重启用户演示服务，也没有把定向回归成绩写进简历。

## 实现：修正证据种类，而非放松所有校验

1. **范围由程序确定。**结构化执行仅当已验证的所有 Goal 都属于 knowledge 时授予 knowledge-only；组合业务目标、Other 和未确定目标没有这个例外。Legacy 复用已确定的 Knowledge scope 或明确的教育 Skill，不由最终回答自行声明范围。
2. **引用仍须真实取得。**原有完整 ID、当前 Run 可用 prediction/citation 集合检查保留。知识-only 且有合法实际引用时，概率、标签等术语及模型总体指标不再自动要求患者 ID；未知范围不能因为拿到一个 citation 就自动升级。
3. **具体患者值仍有有限检测。**检测患者指代之后的预测术语与数值/明确类别声明，不依赖“是、升至”等动词枚举。逗号不截断同一患者断言；早先的总体 Accuracy 加上“不是你的预测概率”不被错误归属患者。引用 ID 中的数字不视为患者数值。这是有限 tripwire，不是完整医疗内容分类器。
4. **知识上下文最小化。**不注入 current_prediction 与原始历史聊天。对于“再解释它”，只从最近四条消息中提取闭集公共术语名称，如 macro-F1；当前问题已明确术语时不补无关旧主题。需要历史指代却不能唯一识别时直接澄清，不执行 Tool。明确工作流的检索 Query 和最终上下文使用同一公共主题；Router 原有上下文不变。
5. **保留证据裁剪。**最终回答指令进一步说明树路径不等于特征重要性，也不能证明神经概率变化原因。不恢复被排除的字段，不建设通用文本事实证明系统。

核心代码：[响应策略](../../PythonServices/TreeSemAgent/agent/policy.py)、[调用链与上下文](../../PythonServices/TreeSemAgent/agent/loop.py)、[已验证任务范围](../../PythonServices/TreeSemAgent/agent/intent_validation.py)、[公共主题提示](../../PythonServices/TreeSemAgent/agent/knowledge_context.py)、[回归测试](../../PythonServices/TreeSemAgent/tests/test_evidence_boundary.py)。

## 测试驱动与复核中发现的问题

- 先复现：有/无当前预测的一般术语解释均降级，合法模型指标百分比也被拒绝。
- 补反例：你的概率 70%、当前置信度 0.9、你的标签 1，以及假引用，不能借通用知识来源通过。
- 独立审查发现“升至”“你上次”“positive_probability”的患者值漏过；补 RED 测试后统一修正指代、术语与数值的关系，而非逐个补变化动词。
- 第一版删除全部历史导致 macro-F1 的“它”追问丢失；改为公共主题提示或澄清，未恢复患者记录和聊天正文。
- 一个 Legacy Unknown 测试因 Tool 未获授权而通过，并未验证证据范围。替换为明确拥有合法来源但 knowledge_only=false 的响应策略负例，并检查拒绝原因是 missing_prediction_grounding。没有改变历史 Gold 或评分。
- 复核又发现逗号截断患者主题、无逗号免责声明误拦；分别先观察测试失败，再修正数值归属顺序。英文 probability 公开别名也补齐。

## 确定性验证

| 验证 | 结果 | 证明边界 |
|---|---|---|
| 新边界测试 | 15 项通过 | 术语/总体指标、患者值、引用、主题提示、澄清及业务绑定 |
| 全部 Agent 测试 | 534 项：530 通过、4 跳过 | 当前 Python Agent 测试集，不是 C++/MySQL 全栈 |
| 重点回归 | 55 项通过，连续 20 轮 | 边界、共享契约、结构化执行及进展状态的确定性重复验证 |
| 独立只读代码审查 | 已报告的有限阻断问题闭环 | 不宣称完整自由文本语义或所有医疗表达覆盖 |

本地执行：

```bash
TREESEM_TRACE_STDOUT=false PYTHONPATH=PythonServices/TreeSemAgent \
  python -m unittest discover -s PythonServices/TreeSemAgent/tests
TREESEM_TRACE_STDOUT=false \
  PYTHONPATH=PythonServices/TreeSemAgent:PythonServices/TreeSemAgent/tests \
  python -m unittest test_evidence_boundary test_contract_consistency \
  test_structured_agent_loop test_execution_state
```

## 真实 LLM 与官方 MCP：分阶段保留原结果

模型 `qwen3.7-plus-2026-05-26`，thinking=false、temperature=0。
真实 Router、最终回答与官方 MCP Streamable HTTP；Native Tool 为严格 ID 的合成业务记录，不是 Gateway/ONNX/MySQL/RBAC 真实全栈。
MCP 使用本地真实索引 `knowledge-c3132c6f8032ad67`、model_public 测试 scope 与独立测试密钥；无生产 Secret。
测试环境的可选 torchvision 不兼容仍使用进程内 text-only 绕开，未宣称部署环境修复。

### 阶段一：5 条

阶段源码 SHA：`91d913f2e9fa0ead7d902a20ebbea4d65fd1bd0e95c76527b7d6186fb67ec8f0`。

| 问题 | 人工检查 |
|---|---|
| 概率/置信度，有当前上下文 | 通过；实际定义引用、空 prediction grounding，不再误拦 |
| 同问题，无当前上下文 | 通过；说明不需要患者预测也可以解释模型术语 |
| 树叶是否导致神经概率变化 | 通过；区分输出链路，有实际引用，没有借用当前 ID |
| 标签编码与未登记临床定义 | 通过；0/1、-1→0，以及结局/窗口未知边界有对应来源 |
| 比较＋路径＋知识综合 | 执行通过，回答告警；三个 Goal 完成，但追加“低风险→高风险、主要受产时出血影响”，缺少类别临床语义/主要影响证据且违背排除要求 |

原自动最低契约 5/5，**不等于人工语义全部通过**。阶段结果保留，不追写。

### 最终版本：3 条补测

最终源码 SHA：`b653410a35680830cb237310791d52d2535433bb844752563e1ee1ba50e2c11a`，与本次提交的 Agent 源码相同。

| 问题 | 人工检查 |
|---|---|
| 概率/置信度术语 | 通过；定义、argmax、负类关系和临床校准限制有实际引用；空 prediction grounding |
| 神经分类与树机制 | 通过；真实引用，空 prediction grounding，没有把当前定位 ID 当 Tool 事实 |
| 三目标综合回答 | history→compare→explanation→实际 MCP 均成功，三个 Goal 完成；本次未再说“主要受某特征影响”。但后文把“50 个百分点”写作“这 50% 的概率增加”，保留单位告警；“风险评分”“阴性→阳性”“高值”也存在语义范围告警，不算无告警语义满分 |

最终自动最低契约 3/3；所有返回 citation ID 均属于实际检索候选，所有 prediction grounding 均属于本轮合成 Native 记录。
**ID 合法不等于每句文本都有充分证据。**本次主要修复确已真实复现通过，但不据此宣称综合表达已经完全稳定。

独立复核补充：positive_probability 不能自动升级为已登记的临床“风险评分”；在临床结局定义未知时，“阴性→阳性”应限定为模型负类→正类；160 高于树阈值 80 只证明分支条件成立，不证明临床异常高值。这些告警和单位问题一起保留，不以自动契约通过覆盖。

### 用量与冻结证据

| 阶段 | 物理请求 | 输入 Token | 输出 Token | 已知总 Token | 未知 usage |
|---|---:|---:|---:|---:|---:|
| 阶段一 5 条 | 15 | 33,580 | 3,391 | 36,971 | 0 |
| 最终 3 条 | 10 | 22,281 | 2,317 | 24,598 | 0 |
| 合计 | 25 | 55,861 | 5,708 | 61,569 | 0 |

不是完整质量测评，也不是准确费用记录。准入预算分别为 20 请求/45,000 已知 Token 和 12 请求/30,000 已知 Token，不是硬费用上限。
冻结证据位于本 worktree 的忽略目录 `artifacts/evaluation/`，不提交真实上游中间规划、Token 或完整检索 excerpt。

```text
evidence-boundary-cases-20260914.json
db8e697d2397f29c6cdd7f245544c197ae523fa030531952b8ab2e5111cbd5be
evidence-boundary-real-20260914.json
3e2e0ef6bd7d05402cb7978fe4bea5ca0242193fd7c51de397d8560f38002dd0
run_evidence_boundary_smoke.py
9e6e69dec70fe24c64603067600bbfa338903ee0ff90ab756ed0fa0e660380d1
evidence-boundary-final-cases-20260914.json
5bb7af115f2dbcb585bb6a2e377a221d85a38c3f70ce8716f723082f8cffbc90
evidence-boundary-final-real-20260914.json
4ad401dc54ccc937dd083eefca158d884a953ed885e3e00c4f33d0b449c5eadc
run_evidence_boundary_final.py
fff3b06848dab305561ba4ecc4bfac3dec93da5fadb028c68edf5581d1976d32
```

## 剩余限制与后续

- 公共主题提示是有限词表；未知/多主题追问澄清，不是任意知识主题理解。
- 无知识召回时不自动授予“通用术语可以无引用回答”的例外，不宣称所有 no-answer 问题闭环。
- 自由文本仍可能产生类别语义、重要性或百分比措辞漂移；有限患者值检测不等于逐句事实证明。后续如继续改进，优先固定业务数值和单位的展示，不为刷满分建设复杂校验子系统。
- Tool Action Signature 仍待办，未与本轮修复混改。
- 本轮未重跑完整 Heldout、真实数据库权限、ONNX parity、压测或用户 Web 部署。默认策略和用户演示服务未改变。

前序：[四项改动真实验证](agent-four-feature-real-regression.md)。
