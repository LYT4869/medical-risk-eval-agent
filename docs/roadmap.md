# treeSem 工程路线与当前状态

## 模块状态

| 模块 | 内容 | 状态 |
|---|---|---|
| M0 | 基线、架构契约、文档与检查点 | 已完成 |
| M1 | C++ Controller/Service/Model/Infrastructure 分层 | 已完成 |
| M2 | Serving Bundle、Python 黄金实现、反归一化 | 已完成 |
| M3 | C++ ONNX Runtime 默认模型主链 | 已完成 |
| M4 | C++ 业务服务、Session、MySQL、医生反馈 | 已完成 |
| M5 | Python Agent Core 与领域 Tool | 已完成 |
| M6 | Doctor/Patient/Admin 权限、安全和审计 | 已完成 |
| M7 | 医疗知识 RAG 与 MCP Server | 待执行 |
| M8 | 医疗领域 Skill 和渐进加载 | 待执行 |
| M9 | Trace、Evaluation、压测和 RabbitMQ 决策门 | 待执行 |
| M10 | Docker、Demo、项目文档和面试材料 | 待执行 |

## M0 Definition of Done

- [x] 当前 C++ 和 Python 测试全部通过。
- [x] 真实 PPH 模型端到端返回 200。
- [x] 非法请求返回 400。
- [x] Adapter 不可用返回 502。
- [x] Worker 过载返回 503。
- [x] Adapter 超时返回 504。
- [x] 日志不记录完整预测响应。
- [x] API、错误码、线程模型和能力归属有文档。
- [x] 新服务命名使用 `treeSem`，历史兼容包不改名。
- [x] 创建本地可回滚基线提交，不推送远程。

## M1 Definition of Done

- [x] `main.cpp` 只负责配置、依赖装配、路由、进程信号和服务启动。
- [x] API、Application、Model Abstraction、Infrastructure 四层落地。
- [x] `PredictionService` 不依赖 JSON、HTTP、libcurl 或 Python 名称。
- [x] 请求和模型响应均经过类型、维度、有限值和范围校验。
- [x] 错误响应统一为稳定 code 和安全 message，不回传内部异常正文。
- [x] EventLoop 不执行阻塞模型 HTTP，队列有界且过载返回 503。
- [x] 客户端提前断开、一次性 responder、慢模型健康检查和并发串线均有测试覆盖。
- [x] SIGINT/SIGTERM 停止接收后排空已接 Worker 任务。
- [x] 11 个 C++/Python/进程级测试全部通过，并对并发相关测试连续运行多轮。
- [x] 真实 `C++ -> Python Adapter -> treeSem` 与 M0 基线语义一致。
- [x] 编译警告选项开启且 `git diff --check` 通过。

## M2 Definition of Done

- [x] 确定性 Bundle Exporter 固定模型版本、文件 checksum 和跨语言 Schema。
- [x] Serving 不再从原始 CSV 动态重建 Scaler。
- [x] Python Adapter 仅凭 Bundle 启动并作为黄金参考实现。
- [x] 神经网络只执行 `mu`、分类和 cluster 的确定性子图；树由原生 evaluator 执行。
- [x] `sample_index`、预处理数组和 49 字段原始 object 三种输入结果一致。
- [x] 响应兼容扩展 `model_version`、backend、原始值、原始阈值和未知单位。
- [x] reference matrix 被显式标记为患者派生数据且仅用于本地测试/演示。
- [x] 全量 1489 样本相对历史实现的概率最大差异为 0，离散输出无差异。
- [x] C++/Python 单元回归、真实 Bundle 校验和真实端到端链路通过。

## M3 Definition of Done

- [x] opset 17 确定性 ONNX 子图经过 checker、shape inference 和 Python ORT 全量验证。
- [x] C++ Loader 启动时校验 Schema、SHA256、特征顺序、Scaler、树拓扑和 ONNX 契约。
- [x] C++ 原生执行预处理、反归一化、决策树、重要特征和决策路径。
- [x] ONNX Session 启动时创建一次，Worker 并发使用独立 Tensor，ORT 内部线程固定为 1+1。
- [x] `remote`、`onnx`、`onnx_fallback`、`shadow` 四种模式可配置并有测试。
- [x] Python Adapter 下线不影响 ONNX；运行异常只 fallback 一次；非法输入和损坏 Bundle 不 fallback。
- [x] 1489 样本 Python/C++ 最大概率差异 `1.79e-7`，label、cluster、leaf、path 和重要特征顺序 100% 一致。
- [x] 多 Worker 并发连续 20 轮、客户端断开、优雅停机和 EventLoop 非阻塞不变量保持。
- [x] 本机固定口径性能报告完成，性能不作为 CI 硬门槛。

## M4 Definition of Done

- [x] 异步动态路由支持命名参数、静态优先、模板校验与重复检测。
- [x] 推理和数据库查询分别使用有界 Worker Pool，EventLoop 不等待 SQL 或模型。
- [x] Public Cookie 与 Internal Header Session 链路已接入，MySQL 为默认配置。
- [x] Prediction、Explanation、History、Comparison、Feedback 由独立 C++ Service 提供。
- [x] 预测快照和 Session 当前预测在 `READ COMMITTED` 短事务中原子提交。
- [x] MySQL 连接池限时获取、RAII 归还、参数化 SQL、坏连接淘汰和安全停机已实现。
- [x] History 使用 keyset cursor；Feedback 使用追加记录和 payload 幂等校验。
- [x] `/health` 不访问数据库，异步 `/ready` 经数据库调度池执行 `SELECT 1`。
- [x] 内存业务 E2E 与 M0～M3 全量回归通过。
- [x] 隔离 MySQL 8.0.42 的 migration、重启持久化、故障恢复 E2E 与性能记录完成。

## M5 Definition of Done

- [x] Python Agent Core 可独立运行，LLM 使用 OpenAI-compatible 协议并可由 Fake Client 完整测试。
- [x] Agent Loop 具备 step、Tool call、重复调用和总 deadline 终止条件。
- [x] 五个结构化领域 Tool 只通过 C++ Internal API 获取预测事实。
- [x] C++ Agent Scheduler 与推理、数据库 Scheduler 隔离，EventLoop 不等待 LLM。
- [x] Agent Run、最终消息、Tool 摘要和幂等状态持久化，不保存思维链或完整 Tool Result。
- [x] grounding policy 拒绝引用本轮 Tool 未返回的 prediction ID。
- [x] Python Agent 故障不影响 ONNX 预测主链。

## M6 Definition of Done

- [x] required 为默认认证模式，development 显式保留匿名回归链。
- [x] Argon2id、Access JWT、Refresh Token 强制轮换与旧 Token 重用检测已实现。
- [x] Patient 资源所有权、Doctor active assignment 和 Admin 非临床边界在服务端执行。
- [x] Internal Tool 使用绑定 actor/session/subject/run/scope 的短期 Capability JWT。
- [x] 认证、assignment、临床访问和审计查询使用不含敏感正文的安全审计。
- [x] 002/003 migration、内存认证 E2E、MySQL 重启持久化和完整 M0～M4 回归通过。
- [x] CORS Origin、Cookie、no-store、安全 Header、日志敏感字段和长度上限有明确约束。

## 当前停止线

M0～M6 代码、测试和文档已完成。下一步是 M7 医疗知识 RAG/MCP；Skill、完整 Trace/Evaluation 和部署演示仍分别属于 M8～M10。模型质量问题继续作为独立工作流处理，不阻塞 Serving 与 Agent 工程架构。
