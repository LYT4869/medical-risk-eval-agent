# 标签编码知识补齐与本地检索验证

日期：2026-09-14；基线 `ae5d737`；隔离分支 `test/composite-response-semantics`。

## 本轮结论

新增一个项目自有、登记 checksum 的模型知识源，说明标签数值编码、概率/置信度及神经网络与解释树的不同输出来源。复用现有 Source Manifest、chunker、混合检索及 citation 生成，不新增 Tool、Router 字段、特判问句或执行框架。

**知识源已接通本地索引和真实检索，不等于云端完整 Agent 已通过新证据测试。** 没有改部署知识索引路径、合并、推送，也没有改旧评测的夹具、Gold 或分数。

## 可以证明与不能证明的事实

- 历史 PPH loader 和 Bundle exporter 将源标签 `-1` 映射为 `0`；Bundle v2 的 `label_schema` 定义负类 0、正类 1、源负类 -1、基线边界 0.5。
- Bundle predictor 用神经网络分类概率 argmax 选 label，完全相等时选择索引 0；`positive_probability` 取索引 1，`confidence` 取选中类别的概率。不能笼统写成“概率大于等于 0.5 必为标签 1”。
- 原生树单独求 `tree_probability`、leaf 和 path；不能把叶节点变化写成神经网络概率变化的原因。解释关联不能作为因果证明。
- 这些是项目实现的数值契约，**不是权威临床结局定义**。观察窗口、结局判定、医疗单位与类别语义仍缺可靠字典，标签 0/1 不能等同于未患病/确诊。

审阅源文件及 SHA256：

```text
历史 trivae/data/pph.py
b1185d54cd0b4a4e7e890be3bf2841946e00867d86b48f2a7e5289916503a96d
TreeSemModelAdapter/treesem_adapter/export_bundle.py
83c8731486aa0a694f429c2e548eb6dcff12c3b93d25618bac1a1118afc3506d
TreeSemModelAdapter/treesem_adapter/model.py
35383c8ede037f078020d7639b6b73910702c2b36f6f76a536d789e49ad5d947
```

历史训练源码、模型二进制和 Bundle 未修改。

## 知识治理与检索证据

新来源 `src_treesem_label_coding`，版本 `encoding-contract-v1`，scope `model_public`，audience `both`，URL `treesem://project/label-coding`。
来源文件 SHA256：`8b8670d9afbab0f7dbdd4529509cac8abcf764c182b53e81d4b85d771fae1070`。
原四个来源的内容、checksum 和 ID 不变。缺外部快照时仍必须按原 checksum 准备，不会自动替换网络内容。

使用缓存的真实 Embedding 与 Cross-Encoder，在离线模式构建全五来源索引：

```text
index_version = knowledge-c3132c6f8032ad67
chunk_count = 40
Embedding revision = 614241f622f53c4eeff9890bdc4f31cfecc418b3
Reranker revision = 1427fd652930e4ba29e8149678df786c240d8825
reranker_min_score = -5.5
all_scope_reranker_min_score = -2.0
rrf_min_score = 0.03
```

模型、chunk 参数和阈值沿用旧校准索引 `knowledge-6f9bf6633fd5b6b9`，未对新问题调阈值；相同输入重复构建返回同一已校验索引。

| 问题 | 患者/医生首条来源 | 对应 citation |
|---|---|---|
| 模型标签 0 和 1 分别是什么意思？ | 标签编码章节 | `cite_c2022db258759ade4229` |
| positive_probability 和 confidence 的区别？ | 概率与置信度章节 | `cite_82e52906cd80a5ea08d9` |
| 树叶变化是否导致神经网络阳性概率变化？ | 神经网络与解释树章节 | `cite_8f77d4fcabbfaf8816c6` |
| 临床结局定义和观察窗口是什么？ | 证据范围与缺失说明 | `cite_c8f9f4babc2ac95c3f1b` |

共 8 次真实本地检索，全部为 hybrid。前三类问题共 6 次，正确对应章节均排第一。这是已知问题的定向 smoke，不是 Heldout，不更新 Recall/MRR 全量成绩。
第四类返回的是“尚缺定义”的证据，不是临床定义已经补齐；相关性分数或非空检索不能表示知识问题已被完整回答。

本地旧环境存在 torchvision 与 torch 的可选图像组件不兼容。诊断进程关闭 Transformers 的可选 torchvision 可用标记后，文本模型加载与真实检索成功。**未修改包或生产配置，因此不能宣称该旧环境已修复或正常启动 Gate 已通过。** 这项进程内调整只用于本次离线验证，后续部署需使用已锁定兼容镜像/环境。

## 回归与可复现边界

- 新测试先观察到“来源未登记”失败，再补来源；真实 checksum 读取、chunk、SQLite FTS5、lexical degraded、citation 和 scope 隔离通过。
- Knowledge core：12 条，11 通过、1 因基础环境缺官方 MCP SDK 跳过。
- 在已有官方 MCP SDK 检查环境再次运行：12/12 通过，覆盖 Tool Schema；上游 Pydantic settings 有一条 lifespan forward-reference 警告。这不代表 MCP 网络端到端已验证。
- Agent：519 条，515 通过、4 跳过。
- 原 legacy/structured 确定性场景均 64/64，数据 SHA 未变；不代表新增云端成功率。
- 云端 LLM 调用 0，收费 Token 0。没有执行 MCP 网络调用、真实 Gateway/业务数据库 E2E、新证据 Agent 最终回答或完整检索评测。

生成文件仅在当前 worktree 的忽略目录保存：

```text
artifacts/knowledge/knowledge-c3132c6f8032ad67/
artifacts/knowledge/label-encoding-retrieval-smoke-20260914.json
smoke SHA256: 9180a191151c1a82911adc7ecb88ee26be6db3c2409780f05722316c217db9f5
index manifest SHA256: a5082631027a46fd579c554a6749a8236e01fba924991e8dc14198e8b519cbe1
```

## 后续验证

在独立新证据条件下验证 Router → 实际知识检索 → Finalizer 的定向回答，检查引用与“模型类别≠诊断”。原缺证据场景仍保留，应继续诚实说明缺口，不能替换其 fixture 后宣称原分数提升。
部署前再校验候选索引、官方 MCP 契约及新知识对原完整检索集的影响；本轮不切默认方案。

见[原契约一致性回归](agent-contract-consistency-regression.md)及[原独立 A/B](independent-router-execution-evaluation.md)。
