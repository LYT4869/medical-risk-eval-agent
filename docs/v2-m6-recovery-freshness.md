# TreeSem V2 M6｜持久化 Writer 提议、MySQL/Checkpoint 故障窗口和晚到来源更正（P001 SIM）

> **工程状态以 GitHub 当前 HEAD 的 CI 为准**。这是一条显式 P001 合成记录 + Fake Writer/固定 Assertion Catalog 的可靠性测试，不是医院可信 LIS、真实 Qwen、多 Worker 生产部署或签发链路。本阶段不改变 M5 的医疗语义能力，不把“可持久化”误写成“临床正确”。

## 1. 先明确最危险的故障窗口

M5 的图为 Coverage → （Writer/Reviewer/Publish）→ END。这个合并节点如果调用一个**非确定性 LLM**，在 MySQL 已提交报告但 Graph SQLite Checkpoint 未记录完成时，恢复后可能要求 LLM 再次生成报告。即使保持相同 `operation_id`，生成的文本变化也会造成幂等 Fingerprint 冲突，安全地拒绝错误重放，但执行无法自动推进。

M6 采用 MySQL proposal/outbox **先于**发布的程序边界：

```text
                   已确认 Contract + M3 调查 + M4 EV-001
                                  |
                      prepare（重新验权限/来源）
                                  |
           GET 已暂存提议? ----------+------ 是 → 读取已落盘候选
                                  |
                                 否
                                  |
                  一次 Writer → Assertion Reviewer
                                  |
            POST Stage Proposal（MySQL 原子写、唯一 Run）
                                  |
                  checkpoint 仅 proposal_key, run_id
                                  |
                  publish（由服务端按 key 读取候选）
                                  |
           C++ 在新 MySQL 事务内重验 Contract / Issue /
               原始来源逻辑 ID、数值与版本 / 全部 Assertion
                                  |
                  SAVE immutable Draft v1 + Operation
                                  |
              checkpoint report_id, revision, phase=delivered
                                  |
              新鲜度重核后向医生显示 Draft
```

**业务事实源**是已授权的 C++/MySQL；Graph/Checkpoint 只保留运行位置和 Stable ID，不存医生 Token、患者报告正文、Writer 生成文本。事故中 MySQL COMMIT 成功但 Checkpoint 未推进时，下一轮 `prepare` 首先读取 MySQL 候选，而非调用 LLM；`publish` 只按已落盘 `proposal_key` 再次调用现有 M5 发布接口。若候选在两个竞争 Worker 中不同，数据库只允许**同一 Run 的首个候选提交**；另一候选 409，不静默覆盖，可以显式重新读回首个候选。**这只是候选提交防重，不是生产多 Worker 调度/锁/租约。**

## 2. 存储与接口

新增且唯一的迁移：`db/migrations/011_v2_m6_recovery.sql`，`treesem_v2_m6_writer_proposals`（PK run_id+proposal_key；单 Run unique，合同版本，完整候选 JSON、SHA-256、created_at）。保留 M5 `...reports`/`...report_revisions`/`...operations` 与 M4 来源版本 Overlay；不迁移、更改历史 M0–M5 数据。

- `V2SimReportRecovery.cpp`：`stageM6Proposal` 写一次不可变候选（当前状态仅 `staged_untrusted`）；`getM6Proposal` 只给原 Run 授权内部 Capability 读取；`publishM6Proposal` 从 MySQL 读取候选、交由 M5 `publishM5Report` 的新 SQL 事务做全面断言/来源检查。GET 和 Stage 只按 Run 权限和基本结构检查，**并未认证自由文本**；绝对不能把 Stage 作为正式医生报告。
- `V2SimArtifactCorrections.cpp`：新增医生显式触发的服务端内置 EV-002 合成晚到 LIS 更正（不是医院真实 LIS webhook）。同一 `LIS-P001-HB-20261010` 来源 E-005/v2 Hb 130g/L → E-006/v3 Hb 125g/L；E-005/S-05 标 stale，E-006/S-06 current，I-06 needs_review。保留 E-004/S-04 历史以及 E-002 250mL、E-003 转诊约 800mL 的独立来源差异。
- `V2SimReportStore.cpp`：历史报告 Get 返回完整**不可变旧 revision**，同时附带现行来源状态 `source_freshness=current|stale_requires_investigation`；`signing_eligible=false` 始终为 SIM。医生 explain/shorten 仍要重新核对来源，旧编辑操作重放也不能绕过来源版本校验。Graph 的 cached complete 必须在每次交付时重新授权并读取此实时新鲜度；若已 stale，则不能再次“作为当前报告”交付，返回需补查，而不是隐式重写 Hb 数值。
- 内部短期 Capability：`POST /internal/v2/sim-runs/:run_id/artifacts/m6/proposals`（body 只能 `proposal_key` + `document`），`GET .../artifacts/m6/proposals/:proposal_key`，`POST .../artifacts/m6/publish`（body 只有 `proposal_key` + `operation_id`）。Doctor Access JWT 无法调用内部接口。
- 医生触发 SIM 事件：`POST /api/v2/doctor/patients/:patient_id/sim-runs/:run_id/m6/late-correction`，body **只能空 JSON**。不允许医生/Agent 在请求中自由填 Hb 或伪造 `LIS official` 标记。患者绑定和有效 Assignment 继续由 C++ Gateway 重新校验。

## 3. Checkpointer：官方 PostgreSQL Saver 实测目标，而非自研 MySQL Saver

`v2_m6/service.py` 复用官方 `SqliteSaver`（开发本机）和可选 `langgraph.checkpoint.postgres.PostgresSaver`，使用新环境参数 `TREESEM_M6_POSTGRES_DSN`（仅测试脚本读取），`LANGGRAPH_STRICT_MSGPACK=true` 限制反序列化。新测试依赖见 `requirements-v2-m6.txt`，固定 `langgraph-checkpoint-postgres==3.2.0`，`PostgresSaver.from_conn_string(...)` + `setup()`。官方文档：[checkpoint-postgres README](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-postgres/README.md)。目标 PostgreSQL 存图的 thread_state，MySQL 存报告和业务记录，**无分布式事务、也没有伪装成 exactly-once**。

测试支持“进程/工作目录切换后，复用同一 PostgreSQL thread_id，仍能复原 M6 Graph 阶段”。这只验证基础**跨实例 Checkpoint 持久化与操作重放**；尚无生产多 Worker 同时竞争的 Lease/Fencing、自动工作分配、数据库安全审批、HA/备份配置，不将一个临时 PostgreSQL 容器视为正式医院部署。

## 4. 故障矩阵及预期

| 故障 / 注入位置 | 预期处理 |
|---|---|
| Writer/Reviewer 完成 → MySQL Proposal 已 COMMIT → Graph `prepare` checkpoint 丢失 | GET 稳定 key 找回同一个完整候选，不再次调用 Writer；一 Run 一候选 |
| 两个 Writer 生成了不同候选 | SQL immutable proposal 只允许第一次提交；另一写入冲突、不覆盖；程序可读取获胜候选 |
| Proposal 已保存 → 真实来源更正 EV-002 → 发布 | M5 事务重核发现 E-005/S-05 stale，409 拒绝发布；不能自动替换成 125 |
| MySQL Report v1 已 COMMIT → Graph `publish` checkpoint 未写 | 稳定 op ID 重放已提交结果，报告 revision 和 operation 各保持一条 |
| 医生交付 v1 后 EV-002 到达 | 已归档 v1 可按历史 revision 查看；返回 stale，新来源变更产生 I-06 needs_review；阻止当前解释、修改及再交付为当前报告 |
| 同患者另一 Report/Run、过期 Capability、撤销 Assignment | C++ 每次检查 Run/actor/patient，不能借旧 checkpoint 跨报告读取或绕权 |
| E-002/E-003 异源分歧 | 维持独立未决；不能作为“同源 LIS 修订”覆盖原记录 |
| 真实 Writer 产出无来源确诊 / 不符合固定 catalog 的释义 | Stage 只存低信任候选；M5 Publish 的独立 P001 审核拒绝；**不冒称自然语言支持性成功** |

## 5. 测试与不适用声明

M6 CI 在 GitHub Actions 新增 `v2_m6` 单测和 `v2_m6_mysql_postgres_integration` 集成 Job：临时 MySQL 8 + 原生 C++ JWT Gateway + 官方 PostgreSQL 16 Saver；覆盖 M1→M3→M4→M6、不同故障注入以及 EV-002 后的历史报告检索/编辑冻结，并要求原有 M0–M5/V1 回归继续通过。**运行结果必须按最终 HEAD 的 CI 判断，不能引用旧 PR/之前通过的 Job 作为 M6 通过证据。**

目前未实现的生产能力：非模拟 LIS 来源签名/样本身份/跨 Run 更正通知，按影响范围重新生成正确的 125g/L 医学报告，自动打开并完成 I-06 有界复核，任意 LLM 语义 Grounding，业务通知队列、医院授权部署高可用 PostgreSQL、多 Worker 抢占/锁租约、电子签名或正式临床发布。后续可以在 M7 真实 LLM 评估及 M6 的可靠性加强单独推进；不能称为真正临床安全保障。
