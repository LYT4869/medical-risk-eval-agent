# M4：C++ 业务服务、Session 与 MySQL

## 实现结果

M4 将一次性预测升级成可持久化业务后端。模型结果只在创建预测时计算一次，之后的详情、解释、历史与比较均读取已保存快照。

```text
Muduo EventLoop
  -> Prediction Scheduler（推理 + 短事务）
     -> PredictionService -> IModelService -> ONNX
     -> ITreeSemStore -> MySQL
  -> Database Scheduler（查询、反馈、/ready）
     -> Explanation / History / Comparison / Feedback Service
     -> ITreeSemStore -> MySQL
```

两个有界队列隔离推理和查询。EventLoop 不执行推理、SQL 或连接池等待；预测期间不持有事务，只有“插入预测 + 更新 Session 当前预测”进入短事务。

## Session 语义

- Public API 从 `treeSemSession` HttpOnly Cookie 读取匿名 Session。
- Internal API 只信任 `X-TreeSem-Session-Id`，不从 Cookie 降级读取。
- Public 预测没有 Session、Cookie 未知或已过期时创建新 Session 并覆盖 Cookie。
- Public 查询遇到未知/过期 Cookie 也创建替代 Session；目标记录若不存在仍返回 404，同时设置新 Cookie。
- Internal 缺少或格式非法的 Header 返回 400；未知或过期 Session 返回 404。
- Session 使用滑动 TTL；MySQL 是真相源，不维护第二份进程内缓存。

该 Session 只是匿名业务上下文，不代表登录用户。医生、患者和管理员认证、RBAC 与审计仍属于 M6。

## 持久化与一致性

`ITreeSemStore` 隔离 Application 层和 SQL，包含 MySQL 正式实现与仅供测试/显式开发的内存实现。MySQL 连接池固定上限、限时获取，连接由 RAII Lease 独占；归还时回滚遗留事务并恢复 autocommit。SQL 全部使用 PreparedStatement。

预测事务：

```text
完成 ONNX 推理
  -> BEGIN / READ COMMITTED
  -> SELECT session FOR UPDATE 并检查未过期
  -> INSERT prediction + result_json 快照
  -> UPDATE current_prediction_id / version / expiry
  -> COMMIT
```

这样不会在昂贵推理期间占用行锁；事务失败时不返回一条无法查询的临时预测。History 使用 `(created_at, prediction_id)` keyset pagination，避免大 offset。

反馈是追加事实，`reviewer_verified` 在 M4 固定为 false。`(session_id, idempotency_key)` 唯一；相同 payload 重放返回原记录，不同 payload 返回 409。反馈不会覆盖模型输出，也不会自动进入训练。

## 数据与安全边界

- 外部 ID 使用 OpenSSL `RAND_bytes` 生成 128 bit 随机值。
- 跨 Session 查询统一返回 404，避免泄露资源是否存在。
- 数据库保存完整解释快照，但不保存完整 `raw_features` 请求。
- 解释中的重要特征原始值仍属于患者派生数据，因此当前数据库只适合本地演示。
- 日志和错误不包含 SQL、密码、数据库地址、绝对路径或患者输入。
- M4 没有身份认证、字段加密、正式数据保留策略或医疗合规声明。

## 数据库和测试

迁移文件是 `db/migrations/001_m4_core.sql`，包含 Session、Prediction、Feedback 与 `schema_migrations`。开发 MySQL 使用 `docker-compose.m4.yml`，并额外创建 `treesem_test` 测试库。

默认单元/进程测试使用内存 Store，不依赖外部服务。真实 MySQL 重启持久化测试是显式开启项：

```bash
TREESEM_DB_NAME=treesem_test ./scripts/migrate_treesem_db.sh

cmake -S . -B build \
  -DTREESEM_ENABLE_MYSQL_INTEGRATION_TESTS=ON \
  -DTREESEM_TEST_BUNDLE_DIR="$PWD/artifacts/treesem/pph/<model_version>"

TREESEM_DB_PASSWORD=treesem_dev_password \
ctest --test-dir build -R m4_mysql_integration_test --output-on-failure
```

该测试创建预测、停止 C++ Server、重新启动，然后用原 Cookie 查询同一预测和历史，证明状态不依赖进程内存。

## 当前验证结果

本轮完成 MySQL-enabled 和 MySQL-disabled 两种编译入口、内存 Store 完整进程 E2E，以及全部旧模块回归。由于 Docker socket 不可访问，验收使用普通用户启动的隔离 MySQL 8.0.42 临时实例，已验证：

- migration 从空库执行，并连续执行两次保持幂等；
- 创建预测后重启 C++ Server，原 Cookie 仍能查询预测和历史；
- MySQL 上的 feedback 首次写入、幂等重放、payload 冲突和跨 Session 隔离；
- MySQL 停止时 `/health` 保持 200、`/ready` 返回安全 503；
- MySQL 恢复后连接池淘汰坏连接并补建，Server 无需重启即可恢复 `/ready`。

这仍是本地工程验证，不代表已完成生产医疗合规、安全和高可用部署。
