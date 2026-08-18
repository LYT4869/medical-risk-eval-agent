# M7：医疗知识 RAG 与 MCP

## 目标与边界

M7 给 Agent 增加可追溯的知识检索，不改变预测主链。模型标签、概率、历史和决策路径仍只能来自 C++ 原生 Tool；RAG 只回答模型说明与通用 PPH 知识，不读取 MySQL、患者特征或聊天历史。

```text
Agent Loop
  ├─ Native Tool Provider -> C++ Internal API -> ONNX / MySQL
  └─ MCP Provider -> Streamable HTTP -> Knowledge Server
                                      -> SQLite FTS5 + FAISS
                                      -> RRF + Cross-Encoder
```

## 语料与索引

每个来源先人工登记 publisher、audience、knowledge scope、版本、URL、许可说明、本地路径和 SHA-256。外部全文放在 Git 忽略的 `corpus/raw/`，避免 URL 内容变化后静默进入索引，也避免提交第三方全文。当前登记的官方来源包括 WHO 2025 consolidated PPH guideline、RCOG 患者材料和 NHS 产后患者材料；只有完成本地快照和 checksum 登记的来源才进入生产 manifest。

Markdown 按标题切块，PDF 不跨页切块，目标 400 token、overlap 60。chunk、source 和内容均有稳定 ID/checksum。Embedding 和 reranker 同时固定 model revision；chunk 参数、语料 checksum 或模型 revision 改变都会生成新的 `index_version`。索引加载时验证所有文件 checksum，损坏时 fail fast。

当前固定 revision：`intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3` 与 `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1@1427fd652930e4ba29e8149678df786c240d8825`。模型只在离线构建阶段下载，运行期要求命中本地缓存。

检索先各取 BM25 与 Dense 候选，再用 RRF 合并并以 Cross-Encoder rerank。开发集分别校准明确 `model/clinical` 与默认 `all` 查询的 no-answer 阈值，阈值随 index manifest 保存；提示注入、个体处方/剂量、确定性诊断、患者数据探测以及 Patient 请求专业治疗流程会在检索前安全 abstain。单路召回或 reranker 失败时可降级；两条召回都失败时返回 `knowledge_unavailable`，不让 LLM 猜答案。

## MCP 和权限

Knowledge Server 使用官方 Python MCP SDK 的无状态 Streamable HTTP，暴露一个只读 Tool `search_medical_knowledge` 和只读 Resource `treesem://knowledge/index-manifest`。输入限制 query 1～500 字符、top_k 1～6，SDK 生成的 Pydantic argument model被收紧为 `extra=forbid`。

C++ 为每个 Agent Run 签发独立 Knowledge Capability JWT，audience 为 `treesem-knowledge`，并绑定 actor、role、session、subject、run、tool 和 scope。Patient 只能使用 `model_public` 与 `clinical_patient`；Doctor 额外获得 technical/professional scope。该密钥与 Access、Internal Capability 和 Agent service secret 分离。

检索片段始终作为不可信 Tool Data 注入，不能覆盖 System Prompt。最终回答里的 `citation_id` 必须来自本轮 MCP 结果；C++ 与 MySQL 只保存引用元数据和 index version，不保存 excerpt。幂等重放返回原引用快照，不重新检索。

## 故障与线程模型

Embedding、SQLite、FAISS 和 rerank 在 Knowledge Server 的独立有界执行器运行，默认 1 Worker + 16 等待槽。队列满返回 `knowledge_overloaded`；MCP 只读请求在传输中断时最多重试一次，并受 Agent 总 deadline 约束。Knowledge Server 下线只影响知识回答，不影响 C++ 预测、历史和比较。

## 测试入口

- `python_knowledge_core_test`：chunk、页码、JWT、scope 隔离、no-answer 和可选官方 MCP SDK Schema。
- `python_agent_core_test`：MCP Tool、citation grounding、Capability 传递及失败降级。
- `evaluation/questions.json`：42 条模型知识、跨语言、权限、无答案和 prompt-injection 固定问题。
- `evaluation/run_evaluation.py`：输出 Recall@5、MRR@10、negative no-answer 和跨 audience 泄漏报告。

真实检索质量报告由固定 revision 模型和本地审核语料离线生成，不依赖 CI 联网；外部来源内容或 checksum 更新后必须重新评测。

当前基线索引为 `knowledge-6f9bf6633fd5b6b9`（4 个来源、35 个 chunk）。42 个固定问题的结果为 Recall@5 `0.9375`、MRR@10 `0.9271`、negative no-answer `1.0`、cross-audience leakage `0`，超过 M7 门槛；不含原文的可提交摘要位于 `PythonServices/TreeSemKnowledge/evaluation/baseline-summary.json`。完整索引和第三方原文继续保持 Git 忽略。
