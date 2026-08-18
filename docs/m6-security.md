# M6：身份、RBAC、Capability 与安全审计

## 身份和资源边界

| 角色 | 可访问范围 |
|---|---|
| Patient | 自己的预测、解释、历史、比较和对话 |
| Doctor | 仅 active assignment 患者；可预测、对话和提交 verified feedback |
| Admin | 创建医生、管理 assignment、查看审计元数据；不能读取临床内容 |

授权不信任请求中的 role、patient ID 或 reviewer。资源不属于 Patient、或 Doctor 没有 active assignment 时统一返回 404；角色本身不允许时返回 403。Admin 被显式阻止进入预测和聊天业务链。

## Token 设计

Access Token 是 HS256 Bearer JWT，默认 15 分钟，严格校验签名、算法、issuer、audience、subject、role、`iat`、`exp` 和 `jti`。Refresh Token 是 256 bit 不透明随机值，数据库只保存 SHA-256，使用 `HttpOnly; SameSite=Strict` Cookie，默认 7 天。

刷新时在数据库事务中消费旧 Token 并写入 replacement；并发刷新只允许一个成功。再次使用旧 Token 会撤销整个 Token Family，新 Token 同样失效。Access JWT 与 Internal Capability JWT 使用不同密钥和 audience。

密码使用 Argon2id：32-byte 随机盐、64 MiB memory、3 iterations、parallelism 1。连续 5 次登录失败锁定 15 分钟；未知邮箱也执行 dummy hash，返回相同错误文案，避免明显的账户枚举时序差异。

## Agent Capability

C++ 创建 Agent Run 后签发最多 2 分钟的 Capability JWT，其中绑定：

```text
actor / role / session / subject / run_id / allowed_tools / audience / exp
```

Python 调用 Internal Tool 时携带该 Token。C++ 中间件还检查 Session Header、Tool scope；业务层检查 Run 仍为 `running`，Doctor assignment 仍有效。Run 完成或失败后即使 JWT 尚未到期也不能继续访问 Tool。

C++ 调用 Python `/v1/agent/runs` 还携带独立 service secret，Python 使用常量时间比较。裸 `X-TreeSem-Session-Id` 在 required 模式下不再构成授权。

## 审计和敏感数据

迁移 `003_m6_auth_security.sql` 创建 User、Refresh Session、Doctor-Patient Assignment 和 Audit Event，并给 Session、Prediction、Agent Run 和 Chat Message 增加 actor/subject 归属字段。

审计只记录 actor、role、action、resource ID、outcome、reason 和 UTC 时间；不保存临床值、结果 JSON、聊天正文、密码、Token、Cookie 或未脱敏邮箱。认证、assignment、临床访问、医生反馈、Capability 拒绝和审计查询进入安全审计。安全相关写操作的审计失败会 fail closed。

## Web 安全

- 精确 Origin 白名单；credential 请求不允许 `*`。
- Refresh/Logout 校验浏览器 Origin，CLI 无 Origin 请求可用。
- Auth、Chat 和临床响应使用 `Cache-Control: no-store`。
- 增加 `X-Content-Type-Options: nosniff` 和 `Referrer-Policy: no-referrer`。
- Authorization、Cookie、密码、Token、临床输入和完整聊天内容不写日志。
- Header、JWT、ID、邮箱、消息、Tool 结果和请求体均有长度上限。

## 启动模式

`TREESEM_AUTH_MODE=required` 是默认值；旧匿名链只在显式 `development` 模式开放。生产环境会拒绝短密钥、相同密钥、非 Secure Refresh Cookie 和通配 Origin。

管理员不能通过公开注册创建。`treesem-admin bootstrap` 只在数据库尚无 Admin 时创建首个管理员，并从环境变量读取凭据，不输出密码或哈希。后续 Doctor 只能由 Admin API 创建。

## API

认证：`register`、`login`、`refresh`、`logout`、`me`。

Admin：创建 Doctor、创建/撤销/查询 assignment、分页查询 audit metadata。

Doctor：按 patient context 发起预测、对话、查询历史；已有 prediction/explanation/comparison 资源路由继续执行资源级授权；Public feedback 的 reviewer 由服务端认证上下文填写且 `reviewer_verified=true`。

## 已验证内容

- Argon2 hash/verify、Access JWT 篡改、错误 audience 和 Capability scope。
- Refresh rotation、旧 Token 重用和 Token Family 撤销。
- 未认证响应、Origin 拒绝、Header 大小写规范化和多 Set-Cookie。
- MySQL 002/003 migration 重复执行、M4 持久化回归、M5 actor/subject 落库以及 M6 用户跨进程重启登录。
- remote-only 19 项回归和 ONNX/MySQL 完整构建 25 项回归；异步、Agent、认证进程测试连续 20 轮通过。

当前不宣称 HIPAA、等保或正式医疗合规；尚未实现 KMS、字段级加密、mTLS、分布式限流和多实例 Token cache。这些是明确的生产化边界，不影响本项目展示资源级授权、最小权限、令牌轮换和安全审计的工程思路。
