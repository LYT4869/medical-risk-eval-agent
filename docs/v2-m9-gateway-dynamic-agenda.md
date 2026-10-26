# TreeSem V2 M9｜动态调查进入可信 C++/MySQL（SIM 隔离路径）

**设计依据**：Notion 02 ADR-009/011/013、03 当前业务/技术契约、07 Q05–Q08/Q14、P001 T00–T09；不把 Notion 08–09 的其他内容作为已实现证据。M9 是 M8 原型的 Gateway/MySQL 加固，而非真实医院部署或临床模型验证。

## 代码职责（相对仓库路径）

- db/migrations/012_v2_m9_dynamic_artifacts.sql：向 MySQL8 的现有 Artifact CHECK 加入 m8_seed/m8_evidence/m8_issue 三个新增类型，保留原 evidence/issue/draft 与 M5/M6 写入/审核表，不覆写历史迁移。
- WebApps/TreeSemServer/src/infrastructure/persistence/V2SimArtifactStore.cpp：在原 JWT Run/Doctor/Patient 权限模型和 MySQL 事务上新增 M8 类型准入。首条 m8_seed 要匹配当前 doctor-confirmed Contract Revision + Summary；m8_evidence 只允许当前 P001 合成 E-002(v1,250mL)、E-003(v1,800mL)；m8_issue 强制 closed_unverified、semantic_validation=not_evaluated、no_diagnosis=true，语句 ID 引用必须包含在该结果的 evidence_ids，且必须由本 Run 持久化过 m8_evidence。新增类型同样沿用 Operation Journal、不可变 Artifact Key、单事务与跨 Run 隔离。旧 issue 类型显式拒绝带 M8 unverified 标记的结果；不会将 candidate 自动转为 verified 或向 M5/M6 发布。
- PythonServices/TreeSemAgent/v2_m8/gateway_repository.py：将现有 M8 逻辑 kind 映射到独立的 m8_* 业务 kind；每次读写必须使用 Run-scoped Agent Capability，不能直接复用医生 JWT，也不能读写旧 M3 verified Issue 与 Report。
- PythonServices/TreeSemAgent/v2_m8/workflow.py：新增 M8GatewayWorkflow，真实目标从 C++ Gateway 已确认的 summary 读取，使用相同模型可替换 plan_task/decide/consolidate Fake 协议，经 Gateway 落地业务 Result；Graph Checkpoint 本阶段仍是独立的本地 SQLite SqliteSaver，恢复时读取已提交 MySQL 结果，不重复生成。
- PythonServices/TreeSemAgent/v2_m8/close.py：仅将真实 404（或本地明确缺失）视为首次生成，不吞 403/409/5xx；避免将权限失败误判为 Artifact 不存在。
- PythonServices/TreeSemAgent/v2_m8/coverage.py：动态 Issue/Objective 结构覆盖的只读预览。只检查哪些合成 Profile Objective 对应了 closed_unverified Issue、未处理语句数量、未决结果；**永远 publish_allowed=false**。现阶段未做正式的自然语言医生意图目标、源证据有效性重检、临床语义 Grounding，不应冒充 M5 Coverage 或报告准入。
- tests/test_v2_m9_gateway_integration.py：启动真实 C++ 服务器 + MySQL8，验证授权/DoctorConfirm 后动态 Agenda、E-002/E-003 跨 Issue 引用、增加第三个跟进 Issue、未验证 Close、只读 Coverage、Run 隔离、密钥作用域、非法旧 kind/伪造来源/晋升结果被拒、幂等 Operation 重放、第一次 Issue SQL 提交后但 Graph Checkpoint 前的强制故障与跨进程恢复，并确保 Draft 不创建。
- .github/workflows/ci.yml：复用现有 v2_m6_mysql_postgres_integration Job 的 C++ build、MySQL8 与 PostgreSQL16 容器，在原 M6+M7-A 集成之后执行 M9 独立纵向测试，避免重复编译原生工程。与上一场景复用同一测试数据库时，M6 最后会撤销测试医生分配，M9 明确重新启用自己的合成身份测试环境，不影响前一个已完成的授权回归。

## 复现及验收

前置：已启用 MySQL8、安装 SQL migrations 001–012、编译 C++ 开启 MySQL/Auth，且 Python PYTHONPATH=PythonServices/TreeSemAgent。运行：

~~~bash
python tests/test_v2_m9_gateway_integration.py build-v2-integration/treesem_server
~~~

CI 检查：v2_m6_mysql_postgres_integration → Run M9 native Gateway and MySQL dynamic IssueResult safety regression。所有老 M1–M8 CI Job 必须继续通过。本阶段主张仅限于 SIM 授权、结构协议、来源 ID 约束和崩溃重放，并没有运行真实 LLM、HIS/EMR/LIS 或通用医疗 NLI。

## 明确未实现

1. M9 Workflow 使用本地 SQLite Graph Checkpointer；M6 的真实 PostgreSQL Saver 仍在既有流程里，尚未作为 M9 的新进程图状态存储实测。
2. 目前合成 Profile 的 O-01/O-02 只是测试声明；无法声称模型已经从医生自然语言要求自主生成可信 objective coverage。
3. 语义正确性、异义同问题去重、任意来源版本及 late correction 的 M9 动态 statement-level invalidation 均尚未实现。旧 M4/M6 的固定 P001 版本/失效安全检查仍然生效，但不是自动泛化到 M9。
4. 未验证的动态 IssueResult **不能进入 M5/M6 Report Publisher**。必须先有可信的通用 Evidence/Statement 审查与 Source Freshness 门槛；当前只提供阻断的结构化 Coverage 预览。
5. 无云端百炼 API Key、无真实 Qwen 质量评测，后续要与工程回归分离记录 REAL/SIM/PLAN。
