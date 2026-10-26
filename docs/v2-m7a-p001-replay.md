# TreeSem V2 M7-A｜P001 SIM 集成回放与结构化 Trace

状态以当前 M7-A Draft PR 和 GitHub CI 为准；提交不代表验收。

## 目标

复用 M6 的真实 C++ HTTP/MySQL 8 + PostgreSQL 16 PostgresSaver 集成脚本，不新建业务 Graph。主场景检查 M1 Contract 确认、M3 Fake Agent 动态 I-04、M4 EV-001、M6 Writer Proposal 恢复与报告 v1、M5 Explain/Shorten 生成 v2/v3、M6 EV-002 晚到更正，以及三个历史报告版本 stale/禁止编辑。其他支线检查发布前更正、业务提交先于 Graph Checkpoint 的恢复、正常生成和重复执行、医生权限撤销。

## CI 复现

CI Job: v2_m6_mysql_postgres_integration。预先启动 MySQL 8/PostgreSQL 16，编译 C++ JWT Gateway，执行迁移并安装 M6 依赖。随后一条命令执行集成业务并产出 Trace：

~~~bash
python tests/test_v2_m6_gateway_integration.py \
  build-v2-integration/treesem_server \
  --trace-out artifacts/v2-m7a-p001-replay.json
~~~

本地复现之前，需要与 CI 相同的隔离数据库、迁移、测试账号环境（TREESEM_DB_PASSWORD，PostgreSQL Saver 时需 TREESEM_M6_POSTGRES_DSN）。上述一条命令不自动安装系统依赖或建立数据库。绝不可指向真实患者数据库。原来不带 --trace-out 的 M6 命令继续兼容。

纯标准库的 Trace Schema 单测：

~~~bash
python -m unittest discover -s tests -p 'test_v2_m7_trace.py' -v
~~~

## 结构化 Trace

文件 schema 为 treesem.v2.m7a.replay.v1，分类 SIM，明确声明 Fake Agent、Fake Writer 及不是模型质量实验。每条事件只存 seq/scenario/step/refs/metrics/result。事件必须符合四个预定义场景的准确顺序；ref 只接受 E/S/I/P/EV 编号，metrics 为预先允许的非负整数。不会保存患者文本、run_id、Token、JWT、Capability、操作密钥、原始报告或非结构化 Payload。所有 SQL/Gateway 断言和场景均通过后才原子写出成功 JSON；失败以 CI 日志为准，不可复用旧文件声称成功。上传 CI Artifact 用于复盘，Trace 不代替数据库审计。

| 场景 | 工程不变量 |
| --- | --- |
| late_after_delivery | 确认、调查、EV-001、Proposal、v1、Explain、v2/v3、EV-002；v1/v2/v3 历史可读但 stale，当前编辑拒绝 |
| correction_before_publish | Stage 后晚到 EV-002，发布拒绝、无报告 |
| report_commit_before_checkpoint | SQL 报告提交后恢复，报告与操作仍各一条 |
| fresh_generation_replay | Writer 只调用一次，跨进程重放不重写，撤权拒绝 |

未实现：真实本地 Qwen、通用语义 Grounding、真实临床资料、自动 I-06 复核/新 Hb125 报告、生产级多 Worker Lease/Fencing、临床签发。M7-B 另行进行真实模型可行性、错误归因和 Token/延迟测量；本阶段合成 CI 不是 Agent Task Success。
