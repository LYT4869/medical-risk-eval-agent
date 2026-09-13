# Agent 执行收敛与最终回答协议试验

## 目的和边界

2026-09-13，本轮不扩大 IntentFrame v2，也不让 Router 生成 Overall Goal、执行状态或证据。
目标是让已经取得足够证据的执行流程及时结束，并防止最终回答包装泄漏、缺少引用和遗漏组合目标。

试验在 `test/composite-response-semantics` 独立分支进行，基于 `7de7d8a`。
当前主分支、默认路由和部署不自动切换。使用 qwen3.7-plus-2026-05-26，关闭思考模式，输出预算 1024 Token。

所有业务、知识返回均为合成夹具；结构化语义输入使用 Oracle Frame。
**这不是完整 Gateway E2E，不测真实 Router 准确率、资源授权、真实模型数值或真实知识召回。**

## 实现了什么

| 原问题 | 本轮处理 | 确定性验证 |
|---|---|---|
| 比较加知识检索需要模型逐步规划，取得事实后仍可能继续调用 | 主流程仍预定义；恰好一个只读业务目标加一个知识目标时，拼接既有 Recipe，复用 Target Binding/History 依赖 | 不合成预测写操作、Skill 激活或未知组合；未知组合仍交给受保护 Open Agent |
| 开放流程耗尽 Step 后没有最终回答 | 程序维护请求级 ExecutionState；目标齐全、工具配额耗尽或剩最后一步时进入无 Tool Finalizer | 最后一步保留作答；同轮批量调用也不能越过配额；失败保留安全的部分执行摘要 |
| 自然语言之后附带 JSON 包装，或者响应字段错误 | 严格检查最终作答信封；允许至多一次无 Tool 修复；两次仍失败则安全失败/降级 | 缺字段、额外字段、重复键、错误类型、混合包装被拒绝；不回传原始非法正文 |
| 拆分后只回答最后一个目标 | Finalization Context 保留原始问题、现有子目标、completed/pending、排除约束和裁剪证据 | 不扩 Router Schema；用户和 Tool 数据进入普通数据消息，不提升为 System 指令 |
| Tool 调用自己的超时可能长于 Run 总预算 | 确定性和开放执行均受剩余总 deadline 限制 | 慢 Tool 被取消，不等到自身长超时耗尽 |
| 相对目标可能误绑定旧历史分页 | 只有无 cursor 的历史首页可绑定上一次/最近两次 | 旧页解释不会把 previous 目标标为完成；请求状态彼此隔离 |

执行状态只记录安全名称、成功/错误、目标索引、剩余预算和 LLM 阶段。临床值、参数、Token、原始消息和完整响应不进入执行摘要日志。
事实获取完成并不自动证明回答完整；评测继续分别报告工具执行、子目标表达、约束和引用。

最终作答使用三个字段的 JSON 信封。真实客户端保留内部“已完成信封解包”标志，避免将合法答案开头的 `[1]` 或 JSON 示例再次误解析。
旧入口和开放执行早期回答保留安全的普通文本兼容；**最终作答阶段**不接受无结构普通文本。
完整 fenced JSON 保留兼容解包，混合正文加包装不接受。修复绝不重新执行业务 Tool。

## 同一失败切片的前后重复诊断

场景：比较最近两次预测，并查询 AUC 的解释边界。每个版本连续运行三次。

| 指标 | 修改前（三次） | 初版修改后（三次） |
|---|---:|---:|
| 每次 API 请求数 | 5 / 5 / 5 | 1 / 1 / 1 |
| Token | 7,554 / 8,411 / 8,279 | 1,416 / 1,396 / 1,407 |
| 延迟 | 14.72 / 20.55 / 18.66 s | 5.53 / 4.24 / 4.57 s |
| Tool 获取成功 | 3/3 | 3/3 |
| 混合正文与内部 JSON 包装 | 三次均出现 | 三次均未出现 |

此切片平均 Token 减少约 82.6%。这是“开放逐步规划改为复用确定性依赖”的局部效果，
不是全项目 Token 降低 82.6%，也不是云端延迟的严格因果基准。调用时点不同且样本只有三次。
修改前本次重复没有复现 Step 上限，因此不能声称原失败必现；它复现的是多轮规划和包装不稳定。

本地原始诊断位于忽略目录 `artifacts/evaluation/composite-planning-probe-{1,2,3}.json`
和 `composite-finalization-after-slice-{1,2,3}.json`。

## 冻结 36 条未揭晓集合：第一轮真实结果

数据：`PythonServices/TreeSemAgent/evaluation/composite_heldout_cases.json`。
6 类组合各 6 条，共 72 个子目标，覆盖中英文、Patient/Doctor、当前/上一次、路径/特征裁剪、比较、摘要、历史和知识。
不覆盖所有安全类别或真实 Skill 路由，不能替代原 64 场景全链路质量评测。

首次真实运行前冻结 SHA256：

```text
2d052a60ae68abf2cda847dc90cd4412205fea25a3245be8353de6fccbf9b6a9
```

| 指标（不修改评分口径） | 首轮真实结果 |
|---|---:|
| 必要业务流程按冻结序列通过 | 34/36，94.44% |
| 子目标完成（含正文证据） | 58/72，80.56% |
| 回答约束 | 36/36，100% |
| 全部子目标和约束同时通过 | 22/36，61.11% |
| Prediction / Citation ID 合法性 | 100% / 100% |
| API 请求 / Token | 48 / 61,651 |
| 平均 / p95 延迟 | 4.49 / 7.44 s |

原始完整报告保存于本地忽略目录 `artifacts/evaluation/composite-finalization-heldout36.json`。

两条“流程失败”实际取得了全部正确证据且答案完整，只是 `get_prediction` 和 History 顺序与固定期待不同。
这批问题中有“先当前摘要再历史”的表达，而夹具统一期待 History 在前，存在评分设计问题。
本轮**不在揭晓后修改 Tool 序列或追溯提高 34/36**；实际判断必须审计用户要求和合法依赖，不能用这一分数断言两条发生了业务失败。

14 条正文未出现 citation ID，但响应的 `grounding_source_ids` 中有合法 ID，UI 可通过 citation metadata 展示来源。
因此“引用 ID 合法率 100%”与“正文引用完整”不是同一指标，不能用前者掩盖后者。
本批语义合同明确要求正文引用，故保留 14 条失败，不删除合同或补同义词抬高结果。

另有比较回答把 B-A 的负差值说成按时间的风险下降，而 A 实际为当前、B 为上一次。
数值 token 检查没有捕获全部方向解释错误；这仍是独立质量风险，不能仅凭子目标指标宣称比较叙述正确。

## 揭晓后的最小修正

1. 最终作答同时验证引用数组和正文 citation；正文缺失触发既有一次修复，不重检索。
2. 短 Finalizer 指令明确差值为 B-A，并要求区分接口排序与时间顺序；不让模型重新计算业务事实。
3. 审查补齐低配额边界和部分执行降级，防止比较未执行时降级文案却说“比较数据已经取得”。

没有增加新 Router 字段、Reflection Agent、通用规划框架或全量 Workflow 枚举。
同一 36 条的后续运行只能称为**揭晓后回归**，不能继续作为独立 Heldout 晋级证据。
首轮运行期间补充的解包和低预算边界修复由单元测试验证；首轮数据不冒充最终代码的完整验证。

### 修复后同集回归（非独立 Heldout）

报告：本地忽略目录 `artifacts/evaluation/composite-finalization-regression36.json`，
已标记 `evaluation_split=post_reveal_regression`、`not_independent_heldout=true`。
语料和评分合同保持原 SHA，不在揭晓后更改序列期待。

| 指标 | 修复后同集回归 |
|---|---:|
| 冻结流程口径通过 | 34/36，94.44% |
| 子目标完成（含正文证据） | 72/72，100% |
| 回答约束 / 全子目标和约束通过 | 100% / 100% |
| Prediction / Citation ID 合法性 | 100% / 100% |
| API 请求 / Token | 49 / 64,781 |
| 平均 / p95 延迟 | 4.21 / 6.87 s |

两条严格流程失败仍为 `heldout_open_history_summary_3/4`，均取得正确摘要和历史且答案完整，
只是读取顺序不同。本轮不更改原分数。
两次 36 场景云端评测共计 126,432 Token；本节不把较早 Dev 诊断 Token 混入这两轮计数。

回归运行固定 Agent Python 源文件指纹：

```text
6165804ee20eee6b76df0452f4b6c0c166881ace1427dc1d56e8b7874294b7d0
```

**“全子目标和约束 100%”不是整体回答质量 100%。**
人工审查 12 条比较回答，至少确认 3 条仍有明确的方向/对象矛盾：

- `heldout_comparison_knowledge_3`：“从 0.34 降至 0.82”的表达与数值矛盾。
- `heldout_comparison_knowledge_4`：先说“降至”，后又纠正为上升，答案内部不一致。
- `heldout_comparison_paths_3`：把实际 A/B 记录对调，却仍使用原 B-A 差值。

自动规则检查检测到了概率差值、路径和引用是否存在，没有完整验证自然语言叙述的对象和方向。
这三条是真实剩余问题，不是同义词误判；其余比较中的时间方向歧义也仍需专项人工验证。
不要将本回归的 100% 指标写成“真实医疗任务质量 100%”。

## 本轮采用建议

执行事实、预算收敛、无 Tool 最终作答和一次协议修复有明确局部收益，值得保留。
但目前不据此切换主分支默认路由，也不扩大 Router Schema：

1. 下一步优先由程序在比较证据中标注 A/B 对应的 current/previous 角色、B-A 定义和确定性时间方向差值，
   不把方向转换留给模型自由推导。
2. 用专项合同验证对象/方向叙述，再准备新的未揭晓组合集合。
3. 下一版评测对合法的独立只读顺序和真正必须遵循的依赖分别判定；旧集原成绩不追溯改写。

本轮结果属于局部工程试验，不证明 LLM Structured Router 已超过 legacy_rule，也不改变既有晋级结论。

## 有价值的面试经验

- 为什么不能只拦截重复工具？拦截能保护下游，但无法保证模型及时形成最终回答；需要保留作答预算和切换无 Tool 阶段。
- 为什么没有扩大 Router Schema？此次失败主要发生在执行和输出阶段，原始目标已保留；把证据/状态塞给 Router 会混淆职责。
- 为什么引用 100% 仍然有失败？引用 ID 合法、正文有引用、知识结论有证据，是三个不同检查点；任务率也不能替代回答完整率。
- 为什么不直接把 34/36 改成 36/36？Heldout 揭晓后的评分修正容易造成指标漂移，应保留原数和审计说明，下一套独立集合再采用修正口径。
- 为什么差值正确而方向解释错误？API 的 A/B 顺序不等于时间顺序；数字正确不等于业务叙述正确，需要确定性元数据和专项验证。
- 为什么部分成功也必须标 pending？已取得历史不代表已经比较；安全降级不能把未完成目标包装为成功。

## 验证入口

最终代码本地验证：Agent 单元测试 467 项（4 项可选依赖跳过），其余全部通过；
原 64 条场景在 legacy_rule 与 structured_llm 两种模式下均通过确定性回归；
新增 36 条确定性场景通过。异步结构化执行及请求状态回归连续 20 轮通过。
这些重复单元测试用于发现局部竞态/预算回归，不等价于进程级并发压测。
代码审查发现的预算、分页误绑定、已解包答案二次解析和错误成功文案均有先失败再通过的回归测试。
Python compileall 与 `git diff --check` 通过；本轮不改 C++，未重跑完整 Docker/MySQL/ONNX Gate。

```bash
PYTHONPATH=PythonServices/TreeSemAgent TREESEM_TRACE_STDOUT=false \
  python3 -m unittest discover -s PythonServices/TreeSemAgent/tests -q

PYTHONPATH=PythonServices/TreeSemAgent TREESEM_TRACE_STDOUT=false \
  python3 PythonServices/TreeSemAgent/evaluation/run_evaluation.py \
  --mode deterministic --routing-mode structured_llm \
  --cases PythonServices/TreeSemAgent/evaluation/composite_heldout_cases.json \
  --critical-repeats 1
```

真实运行需要本地未提交的 LLM 配置；不将 Key、完整临床数据或上游非法响应放入文档。
