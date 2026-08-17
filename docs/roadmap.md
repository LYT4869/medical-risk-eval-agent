# treeSem 工程路线与当前状态

## 模块状态

| 模块 | 内容 | 状态 |
|---|---|---|
| M0 | 基线、架构契约、文档与检查点 | 已完成 |
| M1 | C++ Controller/Service/Model/Infrastructure 分层 | 已完成 |
| M2 | Serving Bundle、Python 黄金实现、反归一化 | 已完成 |
| M3 | C++ ONNX Runtime 默认模型主链 | 待执行 |
| M4 | C++ 业务服务、Session、MySQL、医生反馈 | 待执行 |
| M5 | Python Agent Core 与领域 Tool | 待执行 |
| M6 | Doctor/Patient/Admin 权限、安全和审计 | 待执行 |
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

## 当前停止线

M0/M1/M2 已完成。下一步实现 M3 的 ONNX 导出、C++ 原生 Bundle/Tree 和四种 backend mode；尚未加入 MySQL、Agent、MCP/RAG、Skill 或医疗权限。
